---
layout:     post   				    # 使用的布局（不需要改）
title:      航空客户价值分析 	     # 标题 
subtitle:                            #副标题
date:       2017-02-06 				# 时间
author:     BY 						# 作者
header-img: img/post-bg-2015.jpg 	#这篇文章标题背景图片
catalog: true 						# 是否归档
tags:								#标签
    - 数据挖掘
    - 客户价值分析
---

## <center>航空客户价值分析</center>

**1.客户价值分析**
- 为什么要进行客户价值分析？
    - 指导企业的营销。客户营销战略倡导者Jay & Adam Curry从对国外数百家公司进行的客户营销实施的经验中<br>
      提炼了以下经验：
        - 公司80%的收入来自于顶端的20%的客户（“二八”）
        - 20%的客户的利润率为100%
        - 90%以上的收入来自于现有的客户
        - 大部分的营销预算被用在非现有的客户上
        - 5%-30%的客户在客户金字塔中具有升级潜力
        - 客户金字塔中客户升级2%，意味着销售收入增加10%，利润增加50%
- 如何进行客户价值分析？
    - 客户终生价值理论
    - 客户价值金字塔模型
    - 策略评估矩阵分析法
    - RFM客户价值分析模型
- RFM模型介绍：
    - R(Recency)：最近一次消费时间与截止时间的间隔。
    - F(Frequency)：顾客在某段时间内所消费的次数。
    - M(Montery)：顾客在某段时间内所消费的金额。
    - R、F、M可以分别作为三个维度，对应一个三维空间，同时可以将每个维度划分为5个等级（或其它个数）则将<br>
    空间划分为了5x5x5个区域，构成了125个类别，然后再进行其它分析。

**2.航空客户价值分析**
- 数据字段介绍（数据集中有44个，只涉及到6个）：
    - FFP_DATE：入会时间（办理会员的时间）
    - LOAD_TIME:观测窗口结束时间（截止时间）
    - FLIGHT_COUNT:飞行次数
    - LAST_TO_END:最后一次消费时间到截止时间的时长
    - avg_discount：平均折扣系数
    - SEG_KM_SUM:总的飞行公里数
- 步骤：
    - 处理数据缺失值与异常值
    - 结合RFM模型筛选数据特征
    - 标准化筛选后的数据

**3.分析过程及其代码**

```python
import numpy as np
import pandas as pd
from sklearn import cluster, preprocessing
```


```python
f = open('...../data/air_data.csv','rb')
airline_data = pd.read_csv(f, encoding='gb18030')
```


```python
print(sorted(airline_data.columns))
```

    ['ADD_POINTS_SUM_YR_1', 'ADD_POINTS_SUM_YR_2', 'ADD_Point_SUM', 'AGE', 'AVG_BP_SUM', 'AVG_FLIGHT_COUNT', 'AVG_INTERVAL', 'BEGIN_TO_FIRST', 'BP_SUM', 'EP_SUM', 'EP_SUM_YR_1', 'EP_SUM_YR_2', 'EXCHANGE_COUNT', 'Eli_Add_Point_Sum', 'FFP_DATE', 'FFP_TIER', 'FIRST_FLIGHT_DATE', 'FLIGHT_COUNT', 'GENDER', 'L1Y_BP_SUM', 'L1Y_ELi_Add_Points', 'L1Y_Flight_Count', 'L1Y_Points_Sum', 'LAST_FLIGHT_DATE', 'LAST_TO_END', 'LOAD_TIME', 'MAX_INTERVAL', 'MEMBER_NO', 'P1Y_BP_SUM', 'P1Y_Flight_Count', 'Point_NotFlight', 'Points_Sum', 'Ration_L1Y_BPS', 'Ration_L1Y_Flight_Count', 'Ration_P1Y_BPS', 'Ration_P1Y_Flight_Count', 'SEG_KM_SUM', 'SUM_YR_1', 'SUM_YR_2', 'WEIGHTED_SEG_KM', 'WORK_CITY', 'WORK_COUNTRY', 'WORK_PROVINCE', 'avg_discount']
    
### 查看缺失值情况

```python
print('处理前的数据规模：{0}\n'.format(airline_data.shape), end='')
print('缺失值情况：\n',airline_data.isnull().sum())
```

    处理前的数据规模：(62988, 44)
    缺失值情况：
     MEMBER_NO                     0
    FFP_DATE                      0
    FIRST_FLIGHT_DATE             0
    GENDER                        3
    FFP_TIER                      0
    WORK_CITY                  2269
    WORK_PROVINCE              3248
    WORK_COUNTRY                 26
    AGE                         420
    LOAD_TIME                     0
    FLIGHT_COUNT                  0
    BP_SUM                        0
    EP_SUM_YR_1                   0
    EP_SUM_YR_2                   0
    SUM_YR_1                    551
    SUM_YR_2                    138
    SEG_KM_SUM                    0
    WEIGHTED_SEG_KM               0
    LAST_FLIGHT_DATE              0
    AVG_FLIGHT_COUNT              0
    AVG_BP_SUM                    0
    BEGIN_TO_FIRST                0
    LAST_TO_END                   0
    AVG_INTERVAL                  0
    MAX_INTERVAL                  0
    ADD_POINTS_SUM_YR_1           0
    ADD_POINTS_SUM_YR_2           0
    EXCHANGE_COUNT                0
    avg_discount                  0
    P1Y_Flight_Count              0
    L1Y_Flight_Count              0
    P1Y_BP_SUM                    0
    L1Y_BP_SUM                    0
    EP_SUM                        0
    ADD_Point_SUM                 0
    Eli_Add_Point_Sum             0
    L1Y_ELi_Add_Points            0
    Points_Sum                    0
    L1Y_Points_Sum                0
    Ration_L1Y_Flight_Count       0
    Ration_P1Y_Flight_Count       0
    Ration_P1Y_BPS                0
    Ration_L1Y_BPS                0
    Point_NotFlight               0
    
### 缺失值与异常值处理

```python

# 去除票价为空的记录
index1 = airline_data['SUM_YR_1'].notnull()
index2 = airline_data[ 'SUM_YR_2'].notnull()
data_notnull = airline_data.loc[index1 & index2, :]
print('去除票价为空后的数据规模：',data_nonull.shape, end='')
```

    去除票价为空后的数据规模： (62299, 44)


```python
# 异常值处理（票价、平均折扣率、总飞行公里数为0的数据）
index3 = data_notnull['SUM_YR_1'] != 0
index4 = data_notnull['SUM_YR_2'] != 0
index5 = (data_notnull['SEG_KM_SUM'] >0) & (data_notnull['avg_discount'] != 0)
data = data_notnull[(index3 | index4) & index5]
print('去除异常值后的数据规模：',data.shape, end='')
```

    去除异常值后的数据规模： (62044, 44)


### 特征处理
- 在这里我们除了利用RFM三个特征以外，还添加了L和C特征
    - R:最近一次乘坐飞机到截止时间的月数， R = LAST_TO_END
    - F：在观测窗口内乘坐本公司飞机的次数， F = FLIGHT_COUNT
    - M：客户在观测窗口时间内累计飞行的公里数， M = SEG_KM_SUM
    - L: 会员入会时间到截止时间的月数， L = LOAD_TIME-FFP_DATE
    - C：在观测窗口时间内乘坐仓位对应的折扣系数的平均值， C = avg_discount
```python
#选取特征
data_feature = data[['FFP_DATE', 'LOAD_TIME', 'FLIGHT_COUNT', 'LAST_TO_END', 'avg_discount', 'SEG_KM_SUM']]
# 构建特征L
L = pd.to_datetime(data_feature['LOAD_TIME']) - pd.to_datetime(data_feature['FFP_DATE'])
L = L.astype('str').str.split().str[0]
L = L.astype('int')/30  #最近一次购买到截止时间的月数
data_feature = pd.concat([L, data_feature.iloc[:, 2:]], axis=1)
data_feature.rename(columns={0:'L', 'FLIGHT_COUNT':'F', 'LAST_TO_END':'R', 'avg_discount':'C', 'SEG_KM_SUM':'M'}, inplace=True)
data_feature.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>L</th>
      <th>F</th>
      <th>R</th>
      <th>C</th>
      <th>M</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>90.200000</td>
      <td>210</td>
      <td>1</td>
      <td>0.961639</td>
      <td>580717</td>
    </tr>
    <tr>
      <th>1</th>
      <td>86.566667</td>
      <td>140</td>
      <td>7</td>
      <td>1.252314</td>
      <td>293678</td>
    </tr>
    <tr>
      <th>2</th>
      <td>87.166667</td>
      <td>135</td>
      <td>11</td>
      <td>1.254676</td>
      <td>283712</td>
    </tr>
    <tr>
      <th>3</th>
      <td>68.233333</td>
      <td>23</td>
      <td>97</td>
      <td>1.090870</td>
      <td>281336</td>
    </tr>
    <tr>
      <th>4</th>
      <td>60.533333</td>
      <td>152</td>
      <td>5</td>
      <td>0.970658</td>
      <td>309928</td>
    </tr>
  </tbody>
</table>
</div>




```python
# 数据标准化

print('各个特征的数值范围:')
print(data_feature.agg(np.ptp))
print('各个特征的最小值:')
print(data_feature.agg(np.min))
print('各个特征的最大值:')
print(data_feature.agg(np.max))
```

    各个特征的数值范围:
    L       102.400000
    F       211.000000
    R       730.000000
    C         1.363983
    M    580349.000000
    dtype: float64
    各个特征的最小值:
    L     12.166667
    F      2.000000
    R      1.000000
    C      0.136017
    M    368.000000
    dtype: float64
    各个特征的最大值:
    L       114.566667
    F       213.000000
    R       731.000000
    C         1.500000
    M    580717.000000
    dtype: float64
    


```python
data_scaler = preprocessing.StandardScaler().fit_transform(data_feature)
# 标准化后的前5行数据
data_scaler[:5, :]
```




    array([[ 1.43571897, 14.03412875, -0.94495516,  1.29555058, 26.76136996],
           [ 1.30716214,  9.07328567, -0.9119018 ,  2.86819902, 13.1269701 ],
           [ 1.32839171,  8.71893974, -0.88986623,  2.88097321, 12.65358345],
           [ 0.65848092,  0.78159082, -0.41610151,  1.99472974, 12.54072306],
           [ 0.38603481,  9.92371591, -0.92291959,  1.3443455 , 13.89884778]])




```python
model = cluster.KMeans(n_clusters=5, init='k-means++',tol=0.0001, random_state=123)
```


```python
model.fit(data_scaler)
```




    KMeans(algorithm='auto', copy_x=True, init='k-means++', max_iter=300,
        n_clusters=5, n_init=10, n_jobs=1, precompute_distances='auto',
        random_state=123, tol=0.0001, verbose=0)




```python
model.cluster_centers_
```




    array([[ 1.1606405 , -0.08693608, -0.37723405, -0.15582938, -0.09485838],
           [ 0.48333235,  2.48322162, -0.7993897 ,  0.30863251,  2.42474345],
           [-0.70023473, -0.16115738, -0.41487557, -0.25507456, -0.16097666],
           [-0.31357463, -0.5739782 ,  1.68623831, -0.17307006, -0.5368072 ],
           [ 0.05189204, -0.2266683 , -0.00324598,  2.19225735, -0.23105363]])


### 画出每个聚类中心的雷达图，便于分析聚类结果
```python
labels = data_feature.columns
centers = np.round(model.cluster_centers_, 4)
angles = np.linspace(0, 2*np.pi, 5, endpoint=False)

centers = np.concatenate((centers,centers[:,0].reshape(5,1)), axis=1)
angles = np.concatenate((angles, [angles[0]]))
fig = plt.figure()
ax = fig.add_subplot(111, polar=True)
colors = ['r','b','y','g','k']
for i in range(5):
    plt.plot(angles, centers[i,:],color=colors[i], linewidth=2)

ax.set_thetagrids(angles * 180/np.pi, labels, fontproperties="SimHei")
plt.show()
```


![png](output_12_0.png)


```python
model.labels_
```




    array([1, 1, 1, ..., 2, 3, 3])




```python
result = pd.Series(model.labels_).value_counts()
print(result)
```

    2    24661
    0    15741
    3    12125
    1     5336
    4     4181
    dtype: int64
    
