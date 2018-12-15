# -*- coding: utf-8 -*-
# Author: guoyang14
# Date: 04 Oct 2018

# 0. 导包/宏参设定
import pandas as pd 
pd.set_option('display.max_columns', 500)
#import numpy as np
import math

# 2.3.1 WOE-IV
def WoE_IV(
        wi_datain 
        ,wi_yi = 1
        ,y_var = 'y_var'
        ,fa_df = 'feature_analysis_dataframe_in'
        ):

    # WOE-IV计算主程序 
    def _WoE_IV(wi_datain = 'datain'
               ,wi_var = 'wi_var'  # 针对某个特征进行WI计算 
               ,y_var = 'y_var'
               ,yi = wi_yi  # 目标变量风险值 
                ):  
        
        def _wi_calc(x):
            data_block = wi_datain[[y_var,wi_var]]
            LS = len(data_block.loc[data_block[wi_var] == x, y_var].unique()) == 1  # 是否自动设置Laplace Smoothing
            
            PYi = (sum((data_block[wi_var] == x) & (data_block[y_var] == yi)) + int(LS))/ (sum(data_block[y_var] == yi) + int(LS))
            PNi = (sum((data_block[wi_var] == x) & (data_block[y_var] != yi)) + int(LS))/ (sum(data_block[y_var] != yi) + int(LS))
            
            WoE = math.log(PYi / PNi)   
            return WoE, (PYi - PNi) * WoE
        
        IV = 0
        WoE_list = {}
        for x in wi_datain[wi_var].unique():
            IV += _wi_calc(x)[1]
            WoE_list[x] = _wi_calc(x)[0]
            
        return IV, WoE_list
    
    # 传值整合 
    woe_dict = {}
    def _WoE_IV_t(var_name):
        iv, woe_list = _WoE_IV(wi_datain = wi_datain ,wi_var = var_name ,y_var = y_var)
        woe_dict[var_name] = woe_list
    
        return [var_name, iv]
    
    _need_wi = wi_datain.dtypes                
    woe_iv_list = list(map(_WoE_IV_t, list(_need_wi[(_need_wi != 'object') & (_need_wi.index != y_var)].index)))
    
    # 2.3.1.2 IV权重列表 
    woe_iv_list = pd.DataFrame(list(woe_iv_list), columns = ['Var','Iv'])
    woe_iv_list = woe_iv_list.sort_values('Iv',ascending=False).reset_index(drop=True)
    
    # 2.3.1.3 WOE数据集映射 
    wi_dataout = wi_datain.copy()
    def _woe_apply(var_name):
        wi_dataout[var_name] = list(map(lambda x:woe_dict[var_name][x], wi_dataout[var_name]))
    
    list(map(_woe_apply, woe_dict))  # re = 
    
    # 2.3.1.4 箱内状况探查 
    # 如果给出特征分箱DF，则结果merge；否则仅给出特征级别的排序
    
    if type(fa_df) is pd.DataFrame:
        # 添加IV排序
        woe_iv_list['Iv_rank'] = pd.Series(woe_iv_list.index) + 1
        
        _with_iv = pd.merge(fa_df, woe_iv_list, on='Var', how='left')
        
        def _tmp_func(var):
            _woe_block = pd.DataFrame(list(woe_dict[var].values()), index=list(woe_dict[var]), columns=['Woe'])
            _woe_block['Var'], _woe_block['Binn'] = var, _woe_block.index
            
            return _woe_block    
        
        fa_df = pd.merge(_with_iv, pd.concat(list(map(_tmp_func, woe_dict))), on=['Var','Binn'], how='left')
    else:
        fa_df = woe_iv_list
    
    # 2.3.1.5 收尾（结果update，输出）
    return wi_dataout, woe_dict, fa_df


# 使用 
y_var = 'y_var'

# 使用乳腺癌预测数据集 
# 需调用cut_merge方法对数据集先行分箱 
from sklearn.datasets import load_breast_cancer
datain_example = load_breast_cancer()
data_part = pd.DataFrame(datain_example.data, columns=datain_example.feature_names)
flag_part = pd.Series(datain_example.target, name='y_var')
datain_example = pd.concat([flag_part, data_part], axis=1)

cut_df, cut_divide_rule, fa_df = cut_merge(datain_example, cut_way = 'ew', cut_n = 5, cut_y = 'y_var')

# 测试
wi_dataout, woe_dict, fa_df = WoE_IV(cut_df, y_var = 'y_var', fa_df = fa_df)
