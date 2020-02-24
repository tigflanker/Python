# -*- coding: utf-8 -*-
# Author: tigflanker
# Date: 17 Dec 2018
# 哑变量处理 

# 导包/宏参设定
import pandas as pd 

def Dummy(datain
          ,dummy_var = []  # 待处理变量，接受单str或list、set可循环内容 
          ,drop_one = True  # 是否将首值"第0个映射"删除，以防模型特征共线性 
          ,drop_orig = True  # 是否删除数据集原始变量 
          ):
          
    '''
    # Example:
    datain_test = pd.read_csv('http://biostat.mc.vanderbilt.edu/wiki/pub/Main/DataSets/titanic.txt')
    
    # 单序列dummy
    dummy_mapping_df, dummy_mapping_dict = Dummy(datain_test, dummy_var = 'pclass')  
    
    # 数据集dummy
    dummy_mapping_df, dummy_mapping_dict = Dummy(datain_test[['pclass','survived','age','sex']])  # dummy全集
    dummy_mapping_df, dummy_mapping_dict = Dummy(datain_test, dummy_var = ['pclass','survived','age','sex'])  # dummy指定特征（字符数值均可）
    
    # 参数测试
    dummy_mapping_df, dummy_mapping_dict = Dummy(datain_test, drop_orig=False, drop_one=False)
    '''
    
    # 参数解析 
    if type(dummy_var) == str:
        dummy_var = [dummy_var]
    elif len(dummy_var) == 0:
        dummy_var = list(datain.dtypes[datain.dtypes == 'object'].index)
        
    # dummy主程序 
    def _dummy_temp(x):
        dummy_value = [0] * len(dummy_value_list)
        if x in dummy_value_list:
            dummy_value[dummy_value_list.index(x)] = 1
        
        return dummy_value    
    
    # 生成参数数据和映射字典 
    dummy_mapping_df = datain.copy()
    dummy_mapping_dict = {}
    for i, dv in enumerate(dummy_var):
        dummy_value_list = list(datain[dv].unique())
        dummy_mapping_dict[dv] = [[dv+'_'+str(x), dummy_value_list[x]] for x in range(len(dummy_value_list))]
        dummy_mapping_df = pd.concat([dummy_mapping_df,pd.DataFrame(list(map(_dummy_temp, datain[dv]))
            , columns=[dv+'_'+str(x) for x in range(len(dummy_value_list))])], axis=1)
    
        if drop_orig:
            del dummy_mapping_df[dv]
        if drop_one:
            del dummy_mapping_df[dv+'_0']
            dummy_mapping_dict[dv] = dummy_mapping_dict[dv][1:]
                    
    return dummy_mapping_df, dummy_mapping_dict
