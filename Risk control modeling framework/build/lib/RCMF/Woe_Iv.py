# -*- coding: utf-8 -*-
# Author: tigflanker
# Date: 04 Oct 2018
# Update: 01 Jan 2019: fa_df在本程序中列为必须，WoE-IV可直接依照fa_df统计内容算出 
#                      如无使用Cut_merge分箱，请采用之前版本

# 0. 导包/宏参设定
import pandas as pd 
import math

# 2.3.1 WOE-IV
def Woe_Iv(datain, fa_df):
    
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

    wi_dataout.update(wi_dataout[list(set(fa_df.Var))].apply(lambda x:x.map(lambda y:woe_dict[x.name][y])))
    
    # 2.3.1.5 收尾（结果update，输出）
    return wi_dataout, woe_dict, fa_df

if __name__ == '__main__':
    # 使用 
    
    # 1. 使用乳腺癌预测数据集 
    # 需调用cut_merge方法对数据集先行分箱 
    from sklearn.datasets import load_breast_cancer
    datain_example = load_breast_cancer()
    data_part = pd.DataFrame(datain_example.data, columns=datain_example.feature_names)
    flag_part = pd.Series(datain_example.target, name='y_var')
    datain_example = pd.concat([flag_part, data_part], axis=1)
    
    # 2. 使用泰坦尼克号数据 
    datain_example = pd.read_csv('http://biostat.mc.vanderbilt.edu/wiki/pub/Main/DataSets/titanic.txt')
    
    cut_df, fa_df = Cut_Merge(sample_out, cut_way = 'ef', cut_n = 5, cut_y = 'survived')
    
    # 测试
    wi_dataout, woe_dict, fa_df = Woe_Iv(cut_df, y_var = 'y_var', fa_df = fa_df)