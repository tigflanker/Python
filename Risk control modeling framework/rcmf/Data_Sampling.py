# -*- coding: utf-8 -*-
# Author: tigflanker
# Date: 31 Oct 2018
# Update: 04 Nov 2018：增加过采样 
# 抽样

import re
import pandas as pd 
import numpy as np

# 1. 单纯抽样：全部数据按照一定比例或指定数量进行抽样：sample_rule = 100; sample_rule = 0.3
# 2. 分层抽样：按照某个变量进行排序，按照等频分箱，每箱内抽取均匀样本量：stratify_by = ['age', 20] + sample_rule = 100 or 0.3
# 3. 非均衡抽样：① 设定某个分组变量，对其包含值个数取最小，按此最小值对所有分组 1:1 抽样：sample_rule = 'align' + group_by = 'survived'  
#              ② 指定规则分组抽样，按照分组变量的每个值设定规则（如：100、0.3、'max'）：sample_rule = {0:100, 1:'max'} + group_by = 'survived'
#              ③ 非均衡 + 分层抽样：sample_rule = {0:100, 1:0.3} + group_by = 'survived' + stratify_by = ['age', 20]
# 系数中若倍数大于1，或例数大于样本量本身，即为过采样 
# 用法举例请参见最下方 

def Data_Sampling(datain,
                  sample_rule = {},
                  stratify_by = [],
                  group_by = ''
                  ): 

    # 复制至最终数据集 
    sample_out = datain.copy()
    os_times = 0 # 初始化过采样轮数 
    
    # 分层抽样子宏 
    def stratify_sample(sample_set_in = 'sample_out', 
                        stratify_n = 'stratify_by[1]', 
                        _stratify_by = 'stratify_by[0]', 
                        stratify_rule = 'sample_rule'):
        sample_set_in = sample_set_in.sort_values(_stratify_by).reset_index(drop = True)
        
        def _stra(l):
            _sample_rate = stratify_rule if type(stratify_rule) is float \
                                        else min(stratify_rule / stratify_n / l, 1)
            return list(np.random.choice([True,False],size=l,p=[_sample_rate,1-_sample_rate]))
        
        _bkt_n = [0] * stratify_n
        for x in range(len(sample_set_in)):
            _bkt_n[x%stratify_n] += 1
    
        keep_index = list(map(_stra, _bkt_n))
        keep_index = eval('[' + re.sub('[\[\]]','',str(keep_index)) + ']')  # 数组扁平化
        while (type(stratify_rule) is not float) and (sum(keep_index) != stratify_rule):
            keep_index = list(map(_stra, _bkt_n))
            keep_index = eval('[' + re.sub('[\[\]]','',str(keep_index)) + ']')  # 数组扁平化
        
        return keep_index
    
    # 均衡抽样 
    if type(sample_rule) not in (dict, str):
        
        # 提取过采样需求 
        os_times = int(sample_rule) if type(sample_rule) is float else sample_rule // len(sample_out)
        sample_rule = sample_rule % 1 if type(sample_rule) is float else sample_rule % len(sample_out)
        
        # 单纯抽样
        if (type(sample_rule) is float) and (len(stratify_by) == 0):
            print('>>> 当前正在执行简单抽样，抽样比例为：',os_times + sample_rule,'（过采样）' if os_times > 0 else '')
            keep_index = np.random.choice([True,False],
                                          size=len(sample_out),p=[sample_rule,1-sample_rule])
        if (type(sample_rule) is not float) and (len(stratify_by) == 0):  
            print('>>> 当前正在执行简单抽样，定样数量为：',os_times * len(sample_out) + sample_rule,'（过采样）' if os_times > 0 else '')
            keep_index = []
            while sum(keep_index) != sample_rule:
                keep_index = np.random.choice([True,False],
                                              size=len(sample_out),p=[sample_rule/len(sample_out),
                                                      1-sample_rule/len(sample_out)])
        
        # 分层抽样 
        if len(stratify_by) > 0:        
            print('>>> 当前正在执行分层抽样，分层变量为：',stratify_by[0],'，层数：',stratify_by[1],
                  '，抽样比例为：' if type(sample_rule) is float else '，定样数量为：',
                  os_times + sample_rule if type(sample_rule) is float else os_times * len(sample_out) + sample_rule,
                  '（过采样）' if os_times > 0 else '')
            keep_index = stratify_sample(sample_set_in = sample_out, stratify_n = stratify_by[1], _stratify_by = stratify_by[0], stratify_rule = sample_rule)
            
        # 均衡抽样部分的抽样实施动作 
        _sample_out = sample_out[keep_index]
        for _os in range(os_times):
            _sample_out = pd.concat([_sample_out, sample_out], axis=0, ignore_index=True, sort=False)
            
        sample_out = _sample_out
        
    # 非均衡抽样
    else :
        if sample_rule == 'align':
            _group_vc = sample_out[group_by].value_counts()
            _group_vc[:] = _group_vc.min()
            sample_rule = dict(_group_vc)
            print('>>> 已选择对齐方式，将按照',group_by,'组值中最小例数标齐各组例数，此例数为：',_group_vc.min())
            
        _sample_out = pd.DataFrame()
        for _value in sample_rule:
            # 划分数据集
            _sub_sample_set = sample_out.loc[sample_out[group_by] == _value]
            
            # 提取过采样需求
            if sample_rule[_value] != 'max':
                os_times = int(sample_rule[_value]) if type(sample_rule[_value]) is float else sample_rule[_value] // len(_sub_sample_set)
                sample_rule[_value] = sample_rule[_value] % 1 if type(sample_rule[_value]) is float else sample_rule[_value] % len(_sub_sample_set)
            
            # case
            if sample_rule[_value] == 'max':
                print('>>> 当前正在执行分组抽样，当前组值为：',group_by,'=',_value,'，按全量保留')
                keep_index = [True] * len(_sub_sample_set)
            elif len(stratify_by) == 0:
                if type(sample_rule[_value]) is float:
                    print('>>> 当前正在执行分组抽样，当前组值为：',group_by,'=',_value,'，抽样比例为：',os_times + sample_rule[_value],
                          '（过采样）' if os_times > 0 else '')
                    keep_index = np.random.choice([True,False],
                                                  size=len(_sub_sample_set),p=[sample_rule[_value],1-sample_rule[_value]])
                if type(sample_rule[_value]) is not float:
                    print('>>> 当前正在执行分组抽样，当前组值为：',group_by,'=',_value,'，定样数量为：',
                          os_times * len(_sub_sample_set) + sample_rule[_value],'（过采样）' if (os_times > 0) & (sample_rule[_value] > 0) else '')
                    keep_index = []
                    while sum(keep_index) != sample_rule[_value]:
                        keep_index = np.random.choice([True,False],
                                                      size=len(_sub_sample_set),p=[sample_rule[_value]/len(_sub_sample_set),
                                                              1-sample_rule[_value]/len(_sub_sample_set)])
            elif len(stratify_by) > 0:
                print('>>> 当前正在执行分组 + 分层抽样，当前组值为：',group_by,'=',_value,
                      '，分层变量为：',stratify_by[0],'，层数：',stratify_by[1],
                      '，抽样比例为：' if type(sample_rule[_value]) is float else '，定样数量为：',
                      sample_rule[_value] + os_times if type(sample_rule[_value]) is float else os_times * len(_sub_sample_set) + sample_rule[_value],
                      '（过采样）' if os_times > 0 else '')
                keep_index = stratify_sample(sample_set_in = _sub_sample_set, stratify_n = stratify_by[1], 
                                             _stratify_by = stratify_by[0], stratify_rule = sample_rule[_value])
            
            __sub_sample_set = _sub_sample_set[keep_index]
            for _os in range(os_times):
                __sub_sample_set = pd.concat([__sub_sample_set, _sub_sample_set], axis=0, ignore_index=True, sort=False)
            _sub_sample_set = __sub_sample_set
            
            _sample_out = pd.concat([_sample_out, _sub_sample_set], axis=0, ignore_index=True, sort=False)
    
        sample_out = _sample_out
        
    return sample_out
    
if __name__ == '__main__':
    # 使用示例：
    datain = pd.read_csv('http://biostat.mc.vanderbilt.edu/wiki/pub/Main/DataSets/titanic.txt')
    
    #print(datain.describe())
    #print(datain['survived'].value_counts())
        
    # 1. 简单抽样：sample_rule
    sample_out = Data_Sampling(datain,
                    sample_rule = 2.3,
                    stratify_by = [],
                    group_by = '')
    
    print(sample_out.describe())
    print(sample_out['survived'].value_counts())
    
    # 1. 简单抽样：sample_rule
    sample_out = Data_Sampling(datain,
                    sample_rule = 2000,
                    stratify_by = [],
                    group_by = '')
    
    print(sample_out.describe())
    print(sample_out['survived'].value_counts())
    
    # 2. 分层抽样：sample_rule + stratify_by
    sample_out = Data_Sampling(datain,
                    sample_rule = 2.3,
                    stratify_by = ['age', 20],
                    group_by = '')
    
    print(sample_out.describe())
    print(sample_out['survived'].value_counts())
    
    # 3. 非均衡抽样：sample_rule + group_by
    sample_out = Data_Sampling(datain,
                    sample_rule = 'align',
                    stratify_by = [],
                    group_by = 'survived')
    
    print(sample_out.describe())
    print(sample_out['survived'].value_counts())
    
    # 3. 非均衡抽样：sample_rule + group_by
    sample_out = Data_Sampling(datain,
                    sample_rule = {0:'max',1:4000},
                    stratify_by = [],
                    group_by = 'survived')
    
    print(sample_out.describe())
    print(sample_out['survived'].value_counts())
    
    # 3. 非均衡抽样：sample_rule + group_by
    sample_out = Data_Sampling(datain,
                    sample_rule = {0:2.0,1:4000},
                    stratify_by = ['age', 20],
                    group_by = 'survived')
    
    print(sample_out.describe())
    print(sample_out['survived'].value_counts())