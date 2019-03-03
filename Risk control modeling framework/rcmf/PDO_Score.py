# -*- coding: utf-8 -*-
# Author: tigflanker
# Date: 20 Oct 2018
# PDO转换: 由训练好的逻辑回归模型，按照PDO转化规则计算PDO评分 
# PDO_Score_trans宏用于计算每个特征分箱对应的PDO评分，需要输入为fa_df（包括分箱规则和对应WOE）、训练集（用途仅为计算ks最佳切点，用于计算好坏比）
# PDO_Score宏用于计算输入数据阵的每个特征的PDO评分和总评分，fa_df为必要参数（需提取分箱规则和PDO Score）

# 导包/宏参设定
import pandas as pd 
from pandas._libs import lib
import numpy as np
from scipy import stats 
import matplotlib.pyplot as plt
import math
from sklearn.metrics import roc_curve

# 4.3 PDO转换⁸ 
def PDO_Score(
        X_train  # 训练集
        ,y_train  # 训练集标签
        ,clf  # 模型
        ,fa_df
        ,pdo_p0 = 600  # pdo基线分 
        ,pdo_pdo = 20  # odds翻倍所需增加分值
        ,odds = 0  # odds部分讨论不完备，这里给出odds值用作直接定义
        ):

    # 1.设定总假设基准分 pdo_p0 pdo_pdo
    
    # 2.估算 AB
    # 2.1 按ks最佳切点计算总体odds值
    if not odds:  # 如果没有定义odds，将按照ks的百分位点区间内，去计算好坏比
        y_perdict = pd.Series(clf.predict_proba(X_train)[:,1])
        fpr, tpr, thresholds = roc_curve(y_train.reset_index(drop=True), y_perdict, pos_label = 1)
        w = tpr - fpr
        ks_cut = fpr[w.argmax()]  # ks_score = w.max()
        
        ks_true = y_train.reset_index(drop=True)[round(y_perdict, 2) == round(ks_cut, 2)]
        
        odds =  (1 - ks_true.mean()) / ks_true.mean()
    
    # 2.2 计算 AB值
    B = pdo_pdo / math.log(2)
    A = pdo_p0 - B * math.log(odds)
    
    # 2.3 计算每个变量、每个分箱的score 
    Basic_score = A + B * clf.intercept_[0]
    
    coef_dict = dict(zip(X_train.columns,clf.coef_[0]))
    
    fa_df['Score'] = list(map(lambda row:row[1] * B * coef_dict[row[0]], np.array(fa_df[['Var','Woe']])))

    return fa_df, Basic_score

# PDO打分
# 打分数据需提前填补 
def PDO_Score_Convert(
        PDO_datain
        ,fa_df
        ,y_var = ''
        ,Basic_score = 600
        ,num_bins = 50
        ):

    if set(fa_df.Var.unique()) != set(PDO_datain.columns) - set([y_var]):
        print('Columns not match, please check. This time use intersection set replace.')
    calc_cols = list(set(fa_df.Var.unique()) & set(PDO_datain.columns))
        
    def _sub_func(c):
        _var_score = np.array(fa_df[['Bins','Score']][fa_df.Var == c.name])   
        def _get_score(v):
            try:
                return list(filter(lambda x: v in x[0] if type(x[0]) is pd._libs.interval.Interval else v == x[0], _var_score))[0][1]
            except:
                return 0
            
        return pd.Series(lib.map_infer(c.astype(object).values, _get_score), name = c.name)
    
    PDO_Score_out = PDO_datain[calc_cols].apply(_sub_func)
    PDO_Score_out['PDO_Score_Convert'] = PDO_Score_out.apply(sum, axis = 1) + Basic_score
    
    # 频率分布图、T检验  
    if len(y_var) > 0:
        _pdo_y = pd.concat([PDO_datain[y_var], PDO_Score_out['PDO_Score_Convert']], axis=1, join='inner')
        
        x0 = _pdo_y.loc[_pdo_y[y_var] == 0,'PDO_Score_Convert']
        x1 = _pdo_y.loc[_pdo_y[y_var] == 1,'PDO_Score_Convert']
        
        P_value = stats.ttest_ind(x0,x1)[1]  
                
        fig, ax = plt.subplots()
        
        n, bins0, patches = ax.hist(x0, num_bins, density=1, histtype='step')
        n, bins1, patches = ax.hist(x1, num_bins, density=1, histtype='step')
        
        y0 = ((1 / (np.sqrt(2 * np.pi) * x0.std())) * np.exp(-0.5 * (1 / x0.std() * (bins0 - x0.mean()))**2))
        y1 = ((1 / (np.sqrt(2 * np.pi) * x1.std())) * np.exp(-0.5 * (1 / x1.std() * (bins1 - x1.mean()))**2))
        
        ax.plot(bins0, y0, '--')
        ax.plot(bins1, y1, '--')
        ax.set_xlabel('PDO Score')
        ax.set_ylabel('Probability density')
        ax.set_title('Histogram for 2 groups\nTtest P value:'+str(round(P_value,4)))
        
        fig.tight_layout()
        plt.show()   

    return PDO_Score_out

# 参考文章：https://blog.csdn.net/sscc_learning/article/details/78591210?utm_source=blogxgwz0

# 使用示例暂略，可参见Titanic测试实例