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
### 4. 编程案例

