# -*- coding: utf-8 -*-
# Author: guoyang14
# Date: 13 Dec 2018
# 无监督分箱（等频、等宽）

# 导包/宏参设定
import numpy as np
import pandas as pd 
pd.set_option('display.max_columns', 500)

def cut_merge(
         cut_datain  # 输入主参数，可输入DataFrame或Series
        ,cut_way = 'ef'  # 等频: ef; 等宽: ew
        ,cut_y = ''  # 定义Y变量，用于分箱屏蔽
        ,cut_n = 3  # 分箱数（等宽模式下如果出现空箱，则逐步缩箱）
        ):
    
    # 主执行宏（数值型） 
    def _cut_merge(cut_var, cut_n = cut_n):
        if cut_way == 'ew':
            cut_min = cut_var.min()
            cut_unit = (cut_var.max() - cut_var.min()) / cut_n
    
            divide_rule = list(map(lambda x:(round(cut_min + x * cut_unit, 8) if x != 0 else -np.inf
                                             , round(cut_min + (x + 1) * cut_unit, 8) if x != cut_n - 1 else np.inf) 
                    ,range(cut_n)))
        elif cut_way == 'ef':
            divide_rule = list(map(lambda x:(cut_var.quantile(x / cut_n) if x != 0 else -np.inf
                                             , cut_var.quantile((x + 1) / cut_n) if x != cut_n - 1 else np.inf) 
                    ,range(cut_n)))
        
        def _cut_temp(x):
            group_x = 0
            for i, d in enumerate(divide_rule):
                if (d[0] < x) & (x <= d[1]):
                    group_x = i
            return group_x
        
        # 如果使用等宽分箱，避免出现空组
        while (cut_way == 'ew') & (len(set(map(_cut_temp,cut_var))) < cut_n):
            cut_n -= 1
            cut_unit = (cut_var.max() - cut_var.min()) / cut_n
            divide_rule = list(map(lambda x:(round(cut_min + x * cut_unit, 8) if x != 0 else -np.inf
                                             , round(cut_min + (x + 1) * cut_unit, 8) if x != cut_n - 1 else np.inf) 
                    ,range(cut_n)))
        
        return pd.Series(list(map(_cut_temp,cut_var)), name=cut_var.name), {cut_var.name: divide_rule}
    
    # 主参判断 
    if type(cut_datain) is pd.Series:
        cut_df, cut_divide_rule = _cut_merge(cut_datain)
    elif type(cut_datain) is pd.DataFrame:
        
        # 备份原数 
        cut_df = cut_datain.copy()
        
        # 数据更新 
        _datatp = cut_datain.dtypes                
        _map_result = list(map(lambda col:_cut_merge(cut_datain[col]), 
                               list(_datatp[(_datatp != 'object') & (_datatp.index != cut_y)].index)))
        cut_df.update(pd.DataFrame(list(map(lambda u:u[0], _map_result))).T)
        
        # 字典整理 
        cut_divide_rule = {}
        for x in list(map(lambda u:u[1],_map_result)):
            cut_divide_rule.update(x)
            
        ## 字符型处理
        if sum(_datatp == 'object') > 0:
            for _char_var in list(_datatp[(_datatp == 'object') & (_datatp.index != cut_y)].index):
                _sub_dict = {}
                for i, value in enumerate(set(cut_df[_char_var])):
                    _sub_dict[value] = i

                # 字典更新
                cut_divide_rule.update({_char_var:list(_sub_dict)})
                
                # 值更新
                cut_df = pd.concat([cut_df[list(set(cut_df.columns) - set([_char_var]))],
                                    pd.Series(list(map(lambda y:_sub_dict[y], cut_df[_char_var])), name=_char_var)], axis=1)
        
        
    else:
        print('Only pd.DataFrame or pd.Series data type supported in this version.')
        cut_df, cut_divide_rule = 0, 0
        
    # Feature Analysis Dataframe
    if type(cut_datain) is pd.DataFrame:
        # 箱内情况统计
        def _tmp_func(var):
            _agged = cut_df.groupby(var)[cut_y].agg([sum,'count',np.mean]) 
            _agged['Var'], _agged['Binn'] = var, _agged.index
            
            return _agged
        
        rate_block = pd.concat(list(map(_tmp_func, cut_divide_rule)))
        rate_block = rate_block.rename(index=str, columns={'sum':'bads','count':'N','mean':'bad rate'})
        rate_block['bad rate'] = ['%.2f%%' % (i * 100) for i in rate_block['bad rate']]
            
        # 框架 
        fa_df = pd.concat(list(map(lambda x:pd.Series(cut_divide_rule[x], index=[x] * len(cut_divide_rule[x])), list(cut_divide_rule))))    
        fa_df = pd.DataFrame({'Var':list(fa_df.index),'Bins':list(fa_df)})
        
        fa_df['Binn'] = [cut_divide_rule[row['Var']].index(row['Bins']) for index, row in fa_df.iterrows()]
    
        fa_df = pd.merge(fa_df, rate_block, how = 'left', on = ['Var', 'Binn'])
    else:
        fa_df = []

    return cut_df, cut_divide_rule, fa_df
    
# 使用
datain_example = pd.read_csv('http://biostat.mc.vanderbilt.edu/wiki/pub/Main/DataSets/titanic.txt')

#cut_df, cut_divide_rule, fa_df = cut_merge(datain_example['age'], cut_way = 'ef', cut_n = 5)
cut_df, cut_divide_rule, fa_df = cut_merge(datain_example, cut_way = 'ew', cut_n = 10, cut_y = 'survived')

# 测试
from sklearn.datasets import load_breast_cancer
datain_example = load_breast_cancer()

data_part = pd.DataFrame(datain_example.data, columns=datain_example.feature_names)
flag_part = pd.Series(datain_example.target, name='y_var')

datain_example = pd.concat([flag_part, data_part], axis=1)
cut_df, cut_divide_rule, fa_df = cut_merge(datain_example, cut_way = 'ew', 
                                          cut_n = 10, cut_y = 'y_var')