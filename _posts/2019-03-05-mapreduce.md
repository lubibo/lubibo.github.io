---
layout:     post   				    # 使用的布局（不需要改）
title:      mapreduce原理	     # 标题 
# subtitle:                            #副标题
date:       2019-03-05 				# 时间
author:     Lubibo 						# 作者
header-img:  	#这篇文章标题背景图片
catalog: true 						# 是否归档
tags:								#标签
    - 分布式计算
    - mapreduce
    - hadoop
---
## <center>mapreduce原理</center>
### 1. 简介
- 为什么需要mapreduce？
![jpg](/img/mapreduce/mapreduce背景.jpg)
- •MapReduce设计的一个理念就是“计算向数据靠拢”，而不是“数据向计算靠拢”，因为，移动数据需要大量的网络传输开销.
### 2. 体系结构
  1. MapReduce体系结构主要由四个部分组成，分别是：Client、JobTracker、TaskTracker以及Task。
  ![jpg](/img/mapreduce/架构图.jpg)
     - Client：用户编写的MapReduce程序通过Client提交到JobTracker端。用户可通过Client提供的一些接口查看作业运行状态。
     - JobTracker：JobTracker负责资源监控和作业调度JobTracker 监控所有TaskTracker与Job的健康状况，一旦发现失败，就将相应的任务转移到其他节点。JobTracker 会跟踪任务的执行进度、资源使用量等信息，并将这些信息告诉任务调度器（TaskScheduler），而调度器会在资源出现空闲时，选择合适的任务去使用这些资源。
     - TaskTracker：TaskTracker 会周期性地通过“心跳”将本节点上资源的使用情况和任务的运行进度汇报给JobTracker，同时接收JobTracker 发送过来的命令并执行相应的操作（如启动新任务、杀死任务等）。TaskTracker 使用“slot”等量划分本节点上的资源量（CPU、内存等）。一个Task 获取到一个slot 后才有机会运行，而Hadoop调度器的作用就是将各个TaskTracker上的空闲slot分配给Task使用。slot 分为Map slot 和Reduce slot 两种，分别供MapTask 和Reduce Task 使用。
     - TaskTask 分为Map Task 和Reduce Task 两种，均由TaskTracker 启动。
### 3. 工作流程
![jpg](/img/mapreduce/运行流程.jpg)
![jpg](/img/mapreduce/总体运行流程.jpg)
  1. 初始化
     - MapReduce程序创建新的JobClient实例
     - JobClient向JobTracker请求获得一个新的JobId标识本次作业
     - JobClient将运行作业需要的相关资源放入作业对应的HDFS目录、计算分片数量和map任务数量。（Map数=split数=FileInputFormat.getSplits()）。split计算方法：
         - 文件剩余字节数/splitSize>1.1，则创建一个split（字节数=splitSize），文件剩余字节数=文件大小-splitSize。
         - 文件剩余字节数/splitSize<1.1，剩余的部分作为一个split。
         - Splitsize=Math.max(minSize,Math.min(goalSize,blockSize))，通常Splitsize=blockSize,如输入的文件较小，文件字节数之和小于blockSize时，splitSize=输入文件字节数之和。
         - split 是一个逻辑概念，它只包含一些元数据信息，比如数据起始位置、数据长度、数据所在节点等。
     - RR（record read）：根据spilt的信息去HDFS读取相应的数据块。
     - 向JobTracker提交作业，并获得作业的状态对象句柄。
  1. map过程
     - 作业提交请求放入队列等待调度。
     - 从HDFS中取出作业分片信息，创建对应数量的TaskInProgress调度和监控Map任务。
     - 执行过程：
       - 从HDFS提取相关资源（Jar包、数据）
       - 创建TaskRunner运行Map任务
       - 在单独的JVM中启MapTask执行map函数
       - 中间结果数据定期存入缓存
       - 缓存写入磁盘
       - 定期报告进度

  2. shuffle过程
     - map端的shuffle
     ![jpg](/img/mapreduce/map端的shuffle.jpg)
       - 在初始化的时候，每个map任务会被分配到一个缓冲区（默认100M）
       - 当map任务执行过程中不断产生map的结果数据，这些数据最先写入缓冲区中，主要有两个原因：首先就是减少IO寻址的开销，其次就是为了对map之后的数据做进一步处理。
       - 分区：将map的结果进行分区，分区数一般等于reduce的数量。默认HashPartitioner(key.hashCode() & Integer.MAX_VALUE)%numReduceTasks。根据分区算法生成的partition属性值，跟<k,v>一起序列化成数组写到缓冲区。
       - 溢出比（默认0.8）：当缓冲区的数据量达到一定比例时就要将数据写入到临时文件当中。
       - 在写入文件之前，还要进行排序操作（这里是快排），然后进行combine。（combine，就是原来是<k,<1,1>>,合并之后就是<k,2>）。
       - 当所有的数据都map结束之后，产生了很多的临时文件，这时候就需要对这些文件的内容归并成一个大的文件（归并排序）。在归并的过程中,需要按照之前的partition属性值归并到一起，以便之后的reduce来将这个区的文件拉走。
     - reduce端的shuffle
     ![jpg](/img/mapreduce/reduce端的shuffle.jpg)
       - Reduce任务通过RPC向JobTracker询问Map任务是否已经完成，若完成，则领取数据Reduce领取数据先放入缓存，来自不同Map机器，先归并，再合并，写入磁盘
       - 多个溢写文件归并成一个或多个大文件，文件中的键值对是排序的
       - 当数据很少时，不需要溢写到磁盘，直接在缓存中归并，然后输出给Reduce。
  3. reduce过程
     - 分配Reduce任务
     - 创建TaskRunner运行Reduce任务
     - 在单独的JVM中启动ReduceTask执行reduce函数
     - 从Map节点下载中间结果数据
     - 输出结果临时文件
     - 定期报告进度

### 4. 编程案例
  1. wordCount
  ```java
  public static class WordCountMap extends Mapper<LongWritable, Text, Text, IntWritable> {
		private final IntWritable one = new IntWritable(1);
		private Text word = new Text();

		public void map(LongWritable key, Text value, Context context) throws IOException, InterruptedException {
			String line = value.toString();
			System.out.println("key:"+key.toString()+"  value:"+value.toString()+"   context:"+context.toString());
			StringTokenizer token = new StringTokenizer(line);
			while (token.hasMoreTokens()) {
				word.set(token.nextToken());
				context.write(word, one);
			}
		}
	}

	public static class WordCountReduce extends Reducer<Text, IntWritable, Text, IntWritable> {
		public void reduce(Text key, Iterable<IntWritable> values, Context context)
				throws IOException, InterruptedException {
			int sum = 0;
			System.out.print("key:"+key.toString()+"   values:");
			for (IntWritable val : values) {
				System.out.print(val.get()+",");
				sum += val.get();
			}
			System.out.println();
			context.write(key, new IntWritable(sum));
		}
	}

	public static void main(String[] args) throws Exception {
		System.setProperty("hadoop.home.dir", "E:\\developing_tools\\hadoop-2.7.7");
		Configuration conf = new Configuration();
		@SuppressWarnings("deprecation")
		Job job = new Job(conf);
		job.setJarByClass(WordCount.class);
		job.setJobName("wordcount");
		job.setOutputKeyClass(Text.class);
		job.setOutputValueClass(IntWritable.class);
		job.setMapperClass(WordCountMap.class);
		job.setReducerClass(WordCountReduce.class);
		job.setInputFormatClass(TextInputFormat.class);
		job.setOutputFormatClass(TextOutputFormat.class);
		FileInputFormat.addInputPath(job, new Path("hdfs://172.17.11.198:9000/input"));
		FileOutputFormat.setOutputPath(job, new Path("hdfs://172.17.11.198:9000/output"));
		job.waitForCompletion(true);
	}
  ```
  2. 将默认的降序改成升序。
  ```java
    public static class MyMapper extends Mapper<Object, Text, IntWritable, IntWritable> {
		public void map(Object key, Text value, Context context) throws IOException, InterruptedException {
			IntWritable data = new IntWritable(Integer.parseInt(value.toString()));
			System.out.println("value:"+value.toString());
			IntWritable random = new IntWritable(new Random().nextInt());
			context.write(data, random);
		}
	}

	public static class MyReducer extends Reducer<IntWritable, IntWritable, IntWritable, IntWritable> {
		public void reduce(IntWritable key, Iterable<IntWritable> values, Context context)
				throws IOException, InterruptedException {
			while (values.iterator().hasNext()) {
				context.write(key, null);
				values.iterator().next();
			}
		}
	}


	public static class IntWritableDecreasingComparator extends IntWritable.Comparator {
		public int compare(byte[] b1, int s1, int l1, byte[] b2, int s2, int l2) {
			return -super.compare(b1, s1, l1, b2, s2, l2);//将默认的降序改成升序
		}
	}

	public static void main(String[] args) throws Exception {
		System.setProperty("hadoop.home.dir", "E:\\developing_tools\\hadoop-2.7.7");
		Configuration conf = new Configuration();
		Job job = new Job(conf, "");
		job.setMapperClass(MyMapper.class);//设置map类
		job.setReducerClass(MyReducer.class);//设置reduce类
		
		job.setOutputKeyClass(IntWritable.class);//设置输出键类型
		job.setOutputValueClass(IntWritable.class);//设置输出value类型

		FileInputFormat.addInputPath(job, new Path("hdfs://172.17.11.198:9000/inputnum"));
		FileOutputFormat.setOutputPath(job, new Path("hdfs://172.17.11.198:9000/outputnum"));
		
		job.setSortComparatorClass(IntWritableDecreasingComparator.class);// 设置比较器
		System.exit(job.waitForCompletion(true) ? 0 : 1);
	}
  ```

