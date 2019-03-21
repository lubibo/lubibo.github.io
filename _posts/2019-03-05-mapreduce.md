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
     - reduce端的shuffle
  3. reduce过程
### 4. 编程案例

