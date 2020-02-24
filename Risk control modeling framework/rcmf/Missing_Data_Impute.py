# -*- coding: utf-8 -*-
# Author: tigflanker
# Date: 21 Oct 2018
# Update: 18 Dec 2018
# 缺失值填补

# 导包/宏参设定
import pandas as pd 
import numpy as np

def Missing_Data_Impute(datain  # 需填补DF
                        ,imp_config = 'imp_mean'  # 填补方式，用法如上
                        ,define_key_parameter = ['imp_mean', 'imp_median', 'imp_mode', 'imp_knn']  # 手动设置关键值，如想填补的值和以上命令有重复
                        ):
    
    '''
    # 参数imp_config：
    # 0. 本宏采用的填补方法关键字如下：'imp_mean'/'imp_median'/'imp_mode'/'imp_knn'，'imp_knn'未启用
    # 1. 传入关键值，则数值型特征按照命令方式填补，字符型特征按照众数填补；如imp_config = 'imp_median'
    # 2. 如传入数值，则对输入数据集所有需要填补的连续性变量按照该数值填补，字符型变量按照众数填补；如imp_config = -999
    # 3. 如传入字典，则按照字典指定规则进行填补。
    #    最复杂情况例如：Age和Sex填均值、Income填中位数、Debt填-1，Class填'Unknow'
    #    则imp_config = {'imp_mean':['Age','Sex'], 'imp_median':'Income', -1:'Debt', 'Unknow':'Class'}
    
    # Example:
    datain_test = pd.DataFrame([[0,1,2,3,4,'a'],[5,6,7,8,9,'b'],[np.nan,np.nan,np.nan,np.nan,np.nan,np.nan]],
                        columns=['Index','Age','Sex','Income','Debt','Class'])
        
    imp_dataout = Missing_Data_Impute(datain_test)  
    imp_dataout = Missing_Data_Impute(datain_test, imp_config=-1)
    imp_dataout = Missing_Data_Impute(datain_test, imp_config={'imp_median':['Age','Sex'], 0:'Debt', 'Unknow':'Class'})
    '''
    
    # 字段类型列表
    datatp = datain.dtypes
    nume_cols = set(list(datatp[datatp != 'object'].index))
    char_cols = set(list(datatp[datatp == 'object'].index))
    
    # 参数解析
    # 解析出需填补字段
    need_to_imp = set(datain.columns[pd.isnull(datain).apply(sum) > 0])

    # 重写imp_config参数
    _imp_config = {}
    if imp_config in define_key_parameter:  # 默认全填补 
        if len(need_to_imp & nume_cols) > 0:
            _imp_config[imp_config] = need_to_imp & nume_cols
        if len(need_to_imp & char_cols) > 0:
            _imp_config[define_key_parameter[2]] = need_to_imp & char_cols
    elif type(imp_config) in (int, float):  # 指定值全填补
        if len(need_to_imp & nume_cols) > 0:
            _imp_config[imp_config] = need_to_imp & nume_cols
        if len(need_to_imp & char_cols) > 0:
            _imp_config[define_key_parameter[2]] = need_to_imp & char_cols
    if type(imp_config) != dict:
        imp_config = _imp_config
    print('本次填补规则如下：', imp_config)

    # 分析配置字典并开始填补   
    imp_dataout = datain.copy()
    
    # 填补
    for imp_value, imp_cols in imp_config.items():
        imp_cols = [imp_cols] if type(imp_cols) is str else imp_cols
        for imp_col in imp_cols:
           if imp_value == define_key_parameter[0]:
               imp_dataout[imp_col] = imp_dataout[imp_col].fillna(imp_dataout[imp_col].mean())  # 均值填补
           elif imp_value == define_key_parameter[1]:
               imp_dataout[imp_col] = imp_dataout[imp_col].fillna(imp_dataout[imp_col].median())  # 中位数填补
           elif imp_value == define_key_parameter[2]:
               imp_dataout[imp_col] = imp_dataout[imp_col].fillna(imp_dataout[imp_col].mode()[0])  # 众数填补
           else:
               imp_dataout[imp_col] = imp_dataout[imp_col].fillna(imp_value)  # 指定数填补
        
    return imp_dataout
