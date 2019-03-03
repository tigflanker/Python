# -*- coding: utf-8 -*-
# Author: tigflanker
# Date: 13 Dec 2018
# 无监督分箱（等频、等宽）

# 导包/宏参设定
import numpy as np
import pandas as pd 
from pandas._libs import lib
import math

def Cut_Merge(
        datain  # 输入主参数，可输入DataFrame或Series
        ,cut_way = 'ef'  # 等频: ef; 等宽: ew
        ,cut_y = ''  # 定义Y变量，用于分箱屏蔽
        ,cut_n = 3  # 分箱数（等宽模式下如果出现空箱，则逐步缩箱）
        ,calc_woe = True  # 连带计算WoE-IV，数据集直接以WoE转化后输出
        ):
    
    # 必须重置index
    datain = datain.reset_index(drop = True)
        
    # 主执行宏（数值型） 
    def _cut_merge(cut_var, cut_n = cut_n):       
        if cut_way == 'ew':
            cut_mass = pd.cut(np.array(cut_var), [-np.inf] + list(np.linspace(cut_var.min(), cut_var.max(), cut_n + 1))[1:cut_n] + [np.inf])
            divide_rule = cut_mass.categories
            
        elif cut_way == 'ef':
            divide_rule = [cut_var.quantile(x) for x in np.linspace(0, cut_n, cut_n + 1) / cut_n][1:cut_n]
            cut_mass = pd.cut(np.array(cut_var), [-np.inf] + sorted(set(divide_rule),key = divide_rule.index) + [np.inf])
            divide_rule = cut_mass.categories
        
        # 如果使用等宽分箱，避免出现空组
        while (cut_way == 'ew') & (len(set(cut_mass.codes) - {-1}) < cut_n - 1):
            cut_n -= 1
            cut_mass = pd.cut(np.array(cut_var), [-np.inf] + list(np.linspace(cut_var.min(), cut_var.max(), cut_n - 1)) + [np.inf])
            divide_rule = cut_mass.categories
        
        return pd.Series(cut_mass.codes, name=cut_var.name), {cut_var.name: divide_rule}
    
    # 主参判断 
    if type(datain) is pd.Series:
        cut_df, cut_divide_rule = _cut_merge(datain)
    elif type(datain) is pd.DataFrame:
        
        # 备份原数 
        cut_df = datain.copy()
        
        # 数据更新 
        print('Start cutting numerical variables...')
        _datatp = datain.dtypes    
        _map_result = list(map(lambda col:_cut_merge(datain[col]), 
                               set(_datatp[(_datatp != 'object') & (_datatp.index != cut_y)].index)))
        cut_df.update(pd.DataFrame(list(map(lambda u:u[0], _map_result))).T)
        
        # 字典整理 
        cut_divide_rule = {}
        for x in list(map(lambda u:u[1],_map_result)):
            cut_divide_rule.update(x)
        print('Numerical variables clear!')

        ## 字符型处理
        if sum(_datatp == 'object') > 0:
            print('Start cutting character variables...')
            cut_df_char = pd.DataFrame(index = cut_df.index)
            for _char_var in list(_datatp[(_datatp == 'object') & (_datatp.index != cut_y)].index):
                _sub_dict = {}
                for i, value in enumerate(sorted(list(set(cut_df[_char_var])),key=lambda x:x is not np.nan)):
                    _sub_dict[value] = i if np.nan not in set(cut_df[_char_var]) else i - 1

                # 字典更新
                cut_divide_rule.update({_char_var:list(_sub_dict)})
                
                # 值更新
                cut_df_char = pd.concat([cut_df_char, pd.Series(list(map(lambda y:_sub_dict[y], cut_df[_char_var])), name=_char_var)], axis=1)
            
            cut_df = pd.concat([cut_df[_datatp[(_datatp != 'object')].index], cut_df_char], axis=1)      
#            cut_df = pd.merge(cut_df[_datatp[(_datatp != 'object')].index], cut_df_char, left_index=True, right_index=True)
            print('Character variables clear!')
            
    else:
        print('Only pd.DataFrame or pd.Series data type supported in this version.')
        cut_df, cut_divide_rule = 0, 0
        
    # Feature Analysis Dataframe
    if type(datain) is pd.DataFrame:
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
        fa_df = fa_df.dropna()
        
        fa_df['Binn'] = fa_df.index
        fa_df['Binn'] = fa_df.Binn.groupby(fa_df.Var).rank(method='first').apply(lambda x:int(x - 1))
        
        # dummy出nan情况 
        if len(rate_block.loc[rate_block.Binn == -1,'Var']):
            _dummy_nan = pd.DataFrame(rate_block.loc[rate_block.Binn == -1,'Var'])
            _dummy_nan['Bins'], _dummy_nan['Binn'] = np.nan, -1
            fa_df = pd.concat([fa_df, _dummy_nan], axis=0, ignore_index=True)
        
        fa_df = pd.merge(fa_df, rate_block, how = 'right', on = ['Var', 'Binn'])
        fa_df = fa_df.sort_values(['Var', 'Binn']).reset_index(drop=True)
    else:
        fa_df = []
        
    # 是否连带计算WoE-IV情况，并将结果以WoE转化后输出
    if calc_woe:
        print('Calculating WoE & IV value...')
        print('!!!Notice: The value in output dataset will be transform into WoE.!!!')
        
        # 计算好坏总数
        _sum_set = fa_df.groupby('Var').agg({'bads':sum, 'N':sum}).rename(columns={'bads':'bads_sum','N':'N_sum'})
        fa_df = pd.merge(fa_df, _sum_set, how='left', left_on='Var', right_index=True)
        
        # WoE_IV计算主方法 
        def _WoE_IV(x):
            LS = int(x.bads in (0, x.N))
            
            PYi = (x.bads + LS) / (x.bads_sum + LS)
            PNi = (x.N - x.bads + LS) / (x.N_sum - x.bads_sum + LS)
                
            WoE = math.log(PYi / PNi)   
            return WoE, (PYi - PNi) * WoE
           
        fa_df = pd.concat([fa_df, pd.DataFrame(list(fa_df.apply(_WoE_IV, axis = 1)), columns=['Woe', 'Iv'])], axis=1) 
        _iv_set = pd.DataFrame(fa_df.groupby('Var').Iv.sum(), columns = ['Iv'])
        _iv_set = pd.concat([_iv_set, _iv_set.Iv.rank(ascending=False).rename('Iv_rank')], axis=1)
        fa_df = pd.merge(fa_df.drop(['Iv', 'bads_sum', 'N_sum'], axis=1), _iv_set, how='left', left_on='Var', right_index=True)    
        
        # 2.3.1.3 WOE数据集映射 
        wi_dataout = cut_df.copy()
        
        woe_dict = {}
        for col in set(fa_df.Var):
            woe_dict[col] = dict(zip(fa_df.Binn[fa_df.Var == col], fa_df.Woe[fa_df.Var == col]))
        
    #         Way 1:
    #         cut_df.update(wi_dataout[list(set(fa_df.Var))].apply(lambda x:x.map(woe_dict[x.name])))
    #         wi_dataout = wi_dataout[list(set(fa_df.Var))].apply(lambda x:x.replace(woe_dict[x.name]))
    
    #         Way 2:
    #         wi_dataout = []
    #         for col, value in cut_df[list(set(fa_df.Var))].iteritems():
    #             wi_dataout.append(value.map(woe_dict[col]))
    #         wi_dataout = pd.DataFrame(wi_dataout).T
            
    #         Way 3:
        def dictmap(self, dictx):
            def infer(x):
                dictx_sub = dictx[x.name]
                if x.empty:
                    return lib.map_infer(x, lambda d:dictx_sub.get(d))
                return lib.map_infer(x.astype(object).values, lambda d:dictx_sub.get(d))
        
            return self.apply(infer)
        
        wi_dataout = dictmap(cut_df[list(set(fa_df.Var))], woe_dict)
        
        # final erge
        print('Final merge step...')
        cut_df = pd.merge(cut_df.drop(list(wi_dataout.columns), axis = 1), wi_dataout, left_index = True, right_index = True)

    return cut_df, fa_df
    
if __name__ == '__main__':
    # 使用
    datain_example = pd.read_csv('http://biostat.mc.vanderbilt.edu/wiki/pub/Main/DataSets/titanic.txt')
    
    #cut_df, cut_divide_rule, fa_df = Cut_Merge(datain_example['age'], cut_way = 'ef', cut_n = 5)
    cut_df, fa_df = Cut_Merge(datain_example, cut_way = 'ew', cut_n = 10, cut_y = 'survived')
    
    import time
    start = time.clock()
    cut_df, fa_df = Cut_Merge(datain_example, cut_way = 'ew', cut_n = 10, cut_y = 'survived')
    end = time.clock()
    
    print (end-start)
    
    # 测试
    from sklearn.datasets import load_breast_cancer
    datain_example = load_breast_cancer()
    
    data_part = pd.DataFrame(datain_example.data, columns=datain_example.feature_names)
    flag_part = pd.Series(datain_example.target, name='y_var')
    
    datain_example = pd.concat([flag_part, data_part], axis=1)
    cut_df, fa_df = Cut_Merge(datain_example, cut_way = 'ef', cut_n = 10, cut_y = 'y_var')
    
    datain_c = pd.read_csv('D:/Desktop/for_test.csv',sep=',',encoding='utf-8',engine='python')
    cut_df, fa_df = Cut_Merge(datain_c, cut_way = 'ef', cut_n = 10, cut_y = 'y_tag')
    