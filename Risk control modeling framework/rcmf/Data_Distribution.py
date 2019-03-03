# -*- coding: utf-8 -*-
# Author: tigflanker
# Date: 10 Dec 2018
# Desc: 本方法用于观察数据某一列变量的分布情况，用于策略定制时的节点划分

import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt

def Data_Distribution(
        serin = ''  # 输入序列
        ,bins = []  # 手动分箱list
        ,closed = 'left'
        ):
    
    serin = serin.dropna()

    bins.insert(0, serin.min())
    bins.append(serin.max())
    
    fig, (ax, ax0, ax1) = plt.subplots(ncols=3, figsize=(8, 8))
    
    ax0 = plt.subplot(221) # 第一行的左图
    ax1 = plt.subplot(222) # 第一行的右图
    ax  = plt.subplot(212) # 第二整行
    
    _value_counts = serin.value_counts().sort_index()
    ax.plot(_value_counts.index, list(map(lambda x:sum(_value_counts[_value_counts.index <= x])/sum(_value_counts),_value_counts.index)))
 
    ax.grid()
    ax.set_title('Cumu')
    ax.set_xticks(np.linspace(serin.min(),serin.max(),15))
    ax.set_xlabel('Value')
    ax.set_ylabel('Cumulative percentage')
 
    ax0.hist(serin, bins = bins, histtype='bar', rwidth=0.8, cumulative=False)
    ax0.set_xticks(bins)
    ax0.set_title('Hist') 
    
    # 计算分布矩阵
    _bins = list(map(lambda i:tuple([bins[i],bins[i+1]]),range(len(bins) - 1)))
     
    # table
    _bins = pd.IntervalIndex.from_tuples(_bins, closed = closed)
    _vc = pd.cut(serin, _bins).value_counts()

    _table = pd.DataFrame(list(zip(list(_vc.index),list(_vc))),columns=['interval','N'])
    _table = _table.sort_values('interval').reset_index(drop = True)
    
    # Calc
    _R, _cum_N, _cum_R = 0, 0, 0
    _calc_list = []
    for x in _table['N']:
        _cum_N += x
        _R = x * 100 / sum(_table['N'])
        _cum_R += _R
        
        _calc_list.append(['%.2f%%' % _R, _cum_N, '%.2f%%' % _cum_R])
        
    _calc_list = pd.DataFrame(_calc_list, columns=['rate','cum N','cum Rate'])
    _table = pd.merge(_table, _calc_list, left_index=True, right_index=True)  
    # 分布矩阵计算完毕

    ax1.pie(list(_table['N']),labels=list(_table['interval']),
            autopct='%1.1f%%',shadow=False,startangle=90) 
    ax1.axis('equal')  
    ax1.set_title('Pie')

    fig.tight_layout()
    plt.show()

    print(_table)
    
if __name__ == '__main__':
    datain_example = pd.read_csv('http://biostat.mc.vanderbilt.edu/wiki/pub/Main/DataSets/titanic.txt')
    
    Data_Distribution(serin = datain_example['age'], bins = [15, 60])  # 年龄
