# -*- coding: utf-8 -*-
# Author: guoyang14
# Date: 18 Dec 2018

# 导MLMF建模包 
import sys
sys.path.append(r'D:\Desktop\Machine learning modeling framework') 

import MLMF as mf

# 导官方包 
import pandas as pd 
pd.set_option('display.max_columns', 500)
from sklearn.model_selection import train_test_split
import xgboost as xgb
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier


# 0. 数据导入

datain = pd.read_csv('http://biostat.mc.vanderbilt.edu/wiki/pub/Main/DataSets/titanic.txt')
y_var = 'survived'

###################################### 通用部分
# 数据探索
Variable_pre_exp, obs_miss_rate = mf.data_exp(datain,calc_obs_na=True)

# 按缺失率、唯一值率剔除特征；按行缺失率剔除观测
datain = datain.drop(columns=Variable_pre_exp.loc[Variable_pre_exp.sug_drop_na | Variable_pre_exp.sug_drop_cate, 'var']) #  
Variable_pre_exp = Variable_pre_exp[(~Variable_pre_exp.sug_drop_na) & (~Variable_pre_exp.sug_drop_cate)]\
    .reset_index(drop=True).drop(['sug_drop_na', 'sug_drop_cate'], 1)

datain = datain[list(obs_miss_rate < 0.5)].reset_index(drop=True)

# 剔除无用特征
datain = datain.drop(columns=['row.names']) 

# 缺失值填补
imp_dataout = mf.IMP(datain)  

###################################### 机器学习模型

# 哑变量化
dummy_mapping_df, dummy_mapping_dict = mf.Dummy_char(imp_dataout, drop_one = False)  

# 训练样本拆分 
X_train, X_test, y_train, y_test = train_test_split(dummy_mapping_df.drop([y_var], axis=1), 
                                                    dummy_mapping_df[y_var], test_size = 0.2, random_state = 2018)

# 网格搜索调参
clf_cv = xgb.XGBClassifier()
param_dist = {
        'n_estimators':[50,100,150,200], # [50,100,150,200],
        'max_depth':[3,5,7], #range(2,8,1),
        'learning_rate':[0.01,0.05,0.1,0.5], #np.linspace(0.01,2,10),
        'subsample':[0.7, 0.8, 0.9], #[0.7, 0.8, 0.9],
        'gamma':[0, 1], #[0, 1],
        'reg_alpha':[0, 1], #[0, 1],
        'reg_lambda':[0, 1], #[0, 1],
        'colsample_bytree':[0.7, 0.8, 0.9], #[0.5, 0.6, 0.7, 0.8, 0.9],
        'min_child_weight':[1] #range(1,9,3)
#        'scale_pos_weight':[40,50,60]
        }

grid = GridSearchCV(clf_cv,param_dist,cv = 3,scoring = 'roc_auc',n_jobs = -1)
grid.fit(X_train,y_train)
clf_xgb = grid.best_estimator_  # 按照网格搜索最好case作为分类器
print(grid.best_score_)

#clf_xgb = xgb.XGBClassifier(
#        learning_rate=0.01,
#        n_estimators=50,
#        max_depth=7,
#        min_child_weight=1,
#        subsample=0.9,
#        colsample_bytree=0.7,
#        gamma=0,
#        reg_alpha=1,
#        reg_lambda=0,
#        scale_pos_weight=1
#        ).fit(X_train, y_train)

clf_lr = LogisticRegression().fit(X_train, y_train)
clf_rf = RandomForestClassifier(n_estimators=100).fit(X_train, y_train)

# 模型表现
mf.MLMP(clf_xgb, X_train, y_train, X_test, y_test, out_path = 'D:/xxx.xlsx', 
     calibration = {'Logistic Regression':clf_lr,'Random Forest':clf_rf,'XGboost':clf_xgb})


###################################### 评分卡模型 
# 分箱
cut_df, cut_divide_rule, fa_df = mf.cut_merge(imp_dataout, cut_way = 'ef', cut_n = 10, cut_y = y_var)

# WoE-IV
wi_dataout, woe_dict, fa_df = mf.WoE_IV(cut_df, y_var = y_var, fa_df = fa_df)

# 训练样本拆分 
X_train, X_test, y_train, y_test = train_test_split(wi_dataout.drop([y_var], axis=1), wi_dataout[y_var]
                                                    , test_size = 0.2, random_state = 2018)

# 逻辑回归模型
clf = LogisticRegression().fit(X_train, y_train)

# 模型表现
mf.MLMP(clf, X_train, y_train, X_test, y_test)

# PDO评分转换
fa_df, Basic_score = mf.PDO_Score_trans(X_train, y_train, clf, fa_df)


### 打分阶段
calc_score_data = mf.data_sampling(imp_dataout, group_by = y_var, sample_rule = 'align')
calc_score_data = mf.IMP(calc_score_data) 

# PDO打分
# 打分数据需提前填补 
PDO_Score_out = mf.PDO_Score(calc_score_data, y_var, fa_df, Basic_score)

###################################### 以下备用
# 3.2 TPOT探索*
def tpot_exp():
    tpot = TPOTClassifier(scoring='roc_auc')
    tpot.fit(X_train, y_train)
    tpot.score(X_test, y_test)
