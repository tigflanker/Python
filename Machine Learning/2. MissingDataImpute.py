# -*- coding: utf-8 -*-
# Author: guoyang14
# Date: 21 Oct 2018
# 缺失值填补

# 0. 导包/宏参设定
import pandas as pd 
pd.set_option('display.max_columns', 500)
import numpy as np
np.seterr(divide='ignore', invalid='ignore')
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
#from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor

y_var = 'survived'

# 0.1 数据导入 
#datain = pd.read_csv('D:/data/kaggle/titanic/train.csv')
datain = pd.read_csv('http://biostat.mc.vanderbilt.edu/wiki/pub/Main/DataSets/titanic.txt')

# 1. 数据探索
def data_exp(datain  # 数据入
             ,y_var = y_var  # 定义Y变量
             ,ext_n = 3  # 按照Means ± N * std方式计数极端值，此为N值
             ,plt_out = ''  # 定义变量分布、相关描述图地址，若不定义输出路径则不输出
             ,plt_show = False  # 是否在output中输出分布图
             ,char_cate_threshold = 10  # 定义绘制字符型（分类型）饼图时，最大类数
             ,sug_drop_na_rate = 0.6  # 标识出变量缺失程度是否超过该阈值
             ,sug_drop_ct_rate = 0.2  # 标识出字符型变量类数是否超过阈值
                   ):
    # 1.1 数据全局印象
    datatp = datain.dtypes
    nume_cols = set(list(datatp[datatp != 'object'].index))
    char_cols = set(list(datatp[datatp == 'object'].index))

    Variable_pre_exp = []
    for x in datain.columns:
        Variable_pre_exp.append(
                    [x
                     ,'Numeric' if x in nume_cols else 'Character'
                     ,str(datain[x].min())+'|'+str(datain[x].max()) if x in nume_cols else \
                         str(min(map(lambda y:len(str(y)),datain[x])))+'|'+str(max(map(lambda y:len(str(y)),datain[x])))
                     ,sum(datain[x] == 0)
                     ,'%.2f%%' % (sum(datain[x] == 0) * 100 / datain.shape[0])
                     ,len(datain[x].value_counts())
                     ,'%.2f%%' % (len(datain[x].value_counts()) * 100 / datain.shape[0])
                     ,sum(datain[x].isna())
                     ,'%.2f%%' % (sum(datain[x].isna()) * 100 / datain.shape[0])
                     ,0 if x in char_cols else sum((datain[x] < datain[x].mean() - ext_n * datain[x].std()) \
                                                 | (datain[x] > datain[x].mean() + ext_n * datain[x].std())) 
                     ,(len(datain[x].value_counts()) / datain.shape[0] >= sug_drop_ct_rate) & (x in char_cols)
                     ,sum(datain[x].isna()) / datain.shape[0] >= sug_drop_na_rate
                     ]
                )

    Variable_pre_exp = pd.DataFrame(Variable_pre_exp, columns=['var','var_type','min_max','0_n','0_r','val_cate_n','val_cate_r'
                                                               ,'na_n','na_r','ext_'+str(ext_n),'sug_drop_cate','sug_drop_na'])
    print('>>>>>The preliminary exploration of dataset:<<<<<\n',Variable_pre_exp)
        
    return Variable_pre_exp
    
Variable_pre_exp = data_exp(datain)

# 2. 数据处理 
# 2.0 依照第一步数据探索结果，剔除缺失程度高和离散程度大的特征
for x in Variable_pre_exp.loc[Variable_pre_exp.sug_drop_na | Variable_pre_exp.sug_drop_cate, 'var']:
    del datain[x]
    
Variable_pre_exp = Variable_pre_exp[(~Variable_pre_exp.sug_drop_na) & (~Variable_pre_exp.sug_drop_cate)]\
    .reset_index(drop=True).drop(['sug_drop_na', 'sug_drop_cate'], 1)
    
# 2.1 缺失值填补 
# 依赖宏变量Variable_pre_exp
# 缺失值填补将留出3种填补方法，分别分为：简单填补（Simple：均值/众数）、
# 按特征建模填补（random Forest）、按相似观测填补（Knn）
# 目前Knn填补留空
    
# 参数impute_config：单一str则对所有需填补字段按同一方法填补、字典则按照字典指示填补
def IMP(imp_datain = 'datain'  # 需填补DF
        ,impute_config = {'embarked':'F', 'home.dest':'S'}  # 填补方式，如单一值，则对需填补变量通用；默认按简单填补填充
        ,n_estimators = 100  # 按特征填补树量或迭代次数
        ,mm = 'mean'  # 简单填补时，连续型变量填补方式：'mean'和'median'
        ):
    
    # 因随机森林填补需做离散型特征哑变量处理，提前编译哑变量方法
    # 返回值：dummy_mapping_df返回哑变量集, dummy_mapping_dict返回处理字段的前后映射规则 {'处理字段':[['哑变量1', '原值1'], ['哑变量2', '原值2']]}
    def dummy_value_func(dummy_datain = 'imp_datain'
                         ,dummy_var = ['var1', 'var2']  # 待处理变量，接受单str或list、set可循环内容
                         ):
        if type(dummy_var) == str:
            dummy_var = [dummy_var]
            
        def dummy_temp(x):
            dummy_value = [0] * len(dummy_value_list)
            if x in dummy_value_list:
                dummy_value[dummy_value_list.index(x)] = 1
            
            return dummy_value    
        
        for i, dv in enumerate(dummy_var):
            dummy_value_list = list(dummy_datain[dv].unique())
            if i == 0:
                dummy_mapping_dict = {dv : [[dv+'_'+str(x), dummy_value_list[x]] for x in range(len(dummy_value_list))]}  # 用于储存要dummy字段和dummy后字段键值对
                dummy_mapping_df = pd.DataFrame(list(map(dummy_temp, dummy_datain[dv]))
                                                , columns=[dv+'_'+str(x) for x in range(len(dummy_value_list))])
            else:
                dummy_mapping_dict[dv] = [[dv+'_'+str(x), dummy_value_list[x]] for x in range(len(dummy_value_list))]
                dummy_mapping_df = pd.concat([dummy_mapping_df,pd.DataFrame(list(map(dummy_temp, dummy_datain[dv]))
                                                , columns=[dv+'_'+str(x) for x in range(len(dummy_value_list))])], axis=1)
        
        return dummy_mapping_df, dummy_mapping_dict
    
    # 字段类型列表（以下填补方法均需要）
    nume_cols = list(Variable_pre_exp.loc[Variable_pre_exp.var_type == 'Numeric', 'var'])
    char_cols = list(Variable_pre_exp.loc[Variable_pre_exp.var_type == 'Character', 'var'])
    
    # 简单填补
    def Simple_impute(imp_datain = 'imp_datain', imp_var = 'imp_var', mm = 'mean'):
        if imp_var in nume_cols:
            imp_datain[imp_var] = imp_datain[imp_var].fillna(imp_datain[imp_var].mean() if mm == 'mean' else imp_datain[imp_var].median())  # 均值填
        else:
            simple_impute_dict = imp_datain[imp_var].dropna().value_counts().to_dict()
            imp_datain[imp_var] = imp_datain[imp_var].fillna(max(simple_impute_dict ,key=simple_impute_dict.get))  # 众数填

    # 随机森林填补法
    # 采用随机森林填补法：需对非目标连续型变量做简单填补，需对分类型变量做哑变量处理
    # 依赖函数：Simple_impute、dummy_value_func
    # 依赖宏变量：Variable_pre_exp、nume_cols、char_cols、y_var
    def Forest_impute(imp_datain = 'imp_datain'
                     ,imp_var = 'imp_var'
                     ,n_estimators = 100
                     ):    
        # 对全部char做哑变量处理  
        dummy_mapping_df, dummy_mapping_dict = dummy_value_func(dummy_datain = imp_datain ,dummy_var = char_cols)
        
        # 复制数据，以免污染源数据集
        imp_data_cp = imp_datain.copy()
        
        # 可用于建模的X序列
        imp_X_list = list(set(Variable_pre_exp['var']) - set([imp_var,y_var]))
        
        # 数据预处理：分类型哑变量处理、连续性缺失值简单填补
        # 依照哑变量列表做数据集处理
        for x in dummy_mapping_dict.keys():
            if x != imp_var:
                # 更新特征集
                imp_data_cp = pd.concat([imp_data_cp, dummy_mapping_df[[i[0] for i in dummy_mapping_dict[x]]]], axis=1)
                del imp_data_cp[x]
                
                # 更新特征列表
                imp_X_list = imp_X_list + [i[0] for i in dummy_mapping_dict[x]]
                imp_X_list.remove(x)
            
        # 对有缺失的连续性变量做简单填补
        for x in set(imp_X_list) & set(nume_cols) & set(Variable_pre_exp.loc[Variable_pre_exp.na_n > 0,'var']):
            Simple_impute(imp_datain = imp_data_cp, imp_var = x)
            
        # 对处理后的数据集做划分
        X_train = imp_data_cp.loc[imp_data_cp[imp_var].notna(), imp_X_list]
        Y_train = imp_data_cp.loc[imp_data_cp[imp_var].notna(), imp_var]
        # 判断待预测变量是不是Char，如果是的话，做mapping预处理
        if imp_var in char_cols:
            Y_mapping_list = list(Y_train.unique())
            Y_train = list(map(lambda x:Y_mapping_list.index(x),Y_train))
        X_impute = imp_data_cp.loc[imp_data_cp[imp_var].isna(), imp_X_list]
        
        # 训练 & 预测 
        imp_clf = (RandomForestClassifier if imp_var in char_cols else RandomForestRegressor)(n_estimators=n_estimators)
        imp_clf.fit(X_train, Y_train)  
        Y_impute = imp_clf.predict(X_impute)
        if imp_var in char_cols:
            Y_impute = list(map(lambda x:Y_mapping_list[x],Y_impute))
        
        imp_datain.loc[imp_datain[imp_var].isna(), imp_var] = Y_impute
    
    def Knn_impute():
        pass
    
    # 分析配置字典并开始填补   
    imp_dataout = imp_datain.copy()
    
    if type(impute_config) is str:
        need_to_impute = list(Variable_pre_exp.loc[Variable_pre_exp.na_n > 0 ,'var'])
        for x in need_to_impute:
            if impute_config == 'F':
                Forest_impute(imp_datain = imp_dataout, imp_var = x, n_estimators = n_estimators)  # 随机森林填补
            else:
                Simple_impute(imp_datain = imp_dataout, imp_var = x, mm = mm)  # 用简单填补
    else:
        for x in impute_config:
            if impute_config[x] == 'F':
                Forest_impute(imp_datain = imp_dataout, imp_var = x, n_estimators = n_estimators)  # 随机森林填补
            else:
                Simple_impute(imp_datain = imp_dataout, imp_var = x, mm = mm)  # 用简单填补
        
    return imp_dataout
    
imp_datain1 = IMP(imp_datain = datain, impute_config='F')
Variable_pre_exp1 = data_exp(imp_datain1)

imp_datain2 = IMP(imp_datain = datain, impute_config={'embarked':'F', 'age':'S'})
Variable_pre_exp2 = data_exp(imp_datain2)