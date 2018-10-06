# -*- coding: utf-8 -*-
# Author: Tigflanker
# Date: 04 Oct 2018
# 本文将引用Titanic数据源做较为全面的预测建模

# 0. 导包/宏参设定
import os
import pandas as pd 
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import seaborn as sns
from scipy.stats import chi2

y_var = 'Survived'

# 0.1 数据导入 
datain = pd.read_csv('D:/Desktop/Projects/Data/titanic/train.csv')
#datain = pd.read_csv('http://biostat.mc.vanderbilt.edu/wiki/pub/Main/DataSets/titanic.txt')

# 1. 数据探索
def data_exp(datain, y_var = y_var, char_cate_threshold = 10, ext_n = 3, plt_out = 'data_pre_exp.pdf', plt_show = False):
    # 1.1 数据全局印象
    datatp = datain.dtypes
    nume_cols = set(list(datatp[datatp != 'object'].index))
    char_cols = set(list(datatp[datatp == 'object'].index))

    Variable_pre_exp = []
    for x in datain.columns:
        Variable_pre_exp.append(
                    [x
                     ,'Numeric' if x in nume_cols else 'Character'
                     ,len(datain[x].value_counts())
                     ,'%.2f%%' % (len(datain[x].value_counts()) * 100 / datain.shape[0])
                     ,sum(datain[x].isna())
                     ,'%.2f%%' % (sum(datain[x].isna()) * 100 / datain.shape[0])
                     ,0 if x in char_cols else sum((datain[x] < datain[x].mean() - ext_n * datain[x].std()) \
                                                 | (datain[x] > datain[x].mean() + ext_n * datain[x].std())) 
                     ]
                )

    Variable_pre_exp = pd.DataFrame(Variable_pre_exp, columns=['var','var_type','val_cate_n','val_cate_r','na_n','na_r','ext_'+str(ext_n)])
    print('>>>>>The preliminary exploration of dataset:<<<<<\n',Variable_pre_exp)

    # 1.2 特征分布、阈值¹
    mpl.rcParams['axes.unicode_minus'] = False 
    mpl.rcParams['font.sans-serif'] = ['SimHei'] 
    plt.rcParams['font.sans-serif'] = ['SimHei'] 
    
    with PdfPages(plt_out) as pdf:   
        for x in datain.columns:
            if x in nume_cols:
                datain[x].hist()
                plt.title('Numeric variable ' + x + "'s distribution.")
                pdf.savefig()
                if plt_show: plt.show()
                plt.close()
            elif Variable_pre_exp.val_cate_n[Variable_pre_exp['var'] == x].values[0] <= char_cate_threshold:
                char_vc = datain[x].value_counts()
                fig1, ax1 = plt.subplots()
                ax1.pie(char_vc.values, labels=char_vc.index, autopct='%1.1f%%', startangle=90)
                ax1.axis('equal') 
                plt.title('Character variable ' + x + "'s value counts.")
                pdf.savefig(fig1)
                if plt_show: plt.show() 
                plt.close()
      
        # 1.3 特征相关性矩阵
        sns.heatmap(datain.corr(),annot= True)
        plt.title('Character variable CORR heat map.')
        pdf.savefig()
        if plt_show: plt.show()
        plt.close()

        sns.pairplot(datain,hue= y_var,diag_kind='kde')
        plt.title('Character variable pair plot on' + y_var + '.')
        pdf.savefig()
        if plt_show: plt.show()
        plt.close()
    
data_exp(datain,plt_out = 'D:/Desktop/Projects/Data/titanic/data_pre_exp.pdf')

# 2. 数据处理 
# 2.1 连续型变量离散化或标准化


#n_std_threshold = 3
def calc_chi2(datain, group = 'order_group', y_var = y_var):
    N = len(datain)
    chi2 = 0
    for g in set(datain[group]):
        for y in set(datain[y_var]):
            Agj = sum((datain[group] == g) & (datain[y_var] == y))
            Egj = sum(datain[group] == g) * sum(datain[y_var] == y) / N
            chi2 += 0 if Egj == 0 else (Agj - Egj)**2/Egj

    return chi2

data_age = datain[['Survived','Age']].loc[datain.Age.notna()].sort_values(by='Age').reset_index()

data_age['order_group'] = data_age.index

data_age = data_age.loc[:500]

chi2_threshold = np.inf
group_threshold = 3
while(len(data_age['order_group'].unique()) > group_threshold):
    group_list = list(data_age['order_group'].unique())
    
    min_chi2 = [0, []]
    for i in range(len(group_list) - 1):
        chi2 = calc_chi2(data_age[(data_age.order_group == group_list[i]) | (data_age.order_group == group_list[i + 1])])
        if (chi2 < min_chi2[0]) | (i == 0):
            min_chi2 = [chi2, [group_list[i]]]
        elif chi2 == min_chi2[0]:
            min_chi2[1].append(group_list[i])
    
    def temp_func1(x):
        if x in min_chi2[1]:
            x = group_list[group_list.index(x) + 1]
            while(x in min_chi2[1]):
                x = group_list[group_list.index(x) + 1]
        return x 
    
    if min_chi2[0] < chi2_threshold:        
        data_age.loc[:,'order_group'] = list(map(temp_func1,data_age.order_group))
    else :
        break

# 2.2 缺失值填补
# 2.3 字符型变量/无序数值型变量独热处理
# 2.4 数据规约
# 3. 建模阶段
# 3.1 数据集划分
# 3.2 TPOT探索*
# 3.3 模型建立
# 3.3.1 网格搜索调参
# 3.3.2 交叉验证
# 4. 模型表现评定
# 4.1 AUC、F1-Score等评定
# 4.2 相关图
# info 
# info1: https://blog.csdn.net/henbile/article/details/79974037
# ⁰¹²³⁴⁵⁶⁷⁸⁹⁽⁾