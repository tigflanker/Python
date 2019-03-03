# -*- coding: utf-8 -*-
# Author: tigflanker
# Date: 29 Dec 2018

from statsmodels.stats.outliers_influence import variance_inflation_factor
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def Corr_Vif(
        datain
        ,corr_method = 'pearson'  # 'pearson', 'kendall', 'spearman'
        ,corr_plt_filter = 20
        ,font_size = 12
        ,figsize = (14, 10)
        ,rounds = 2
        ):

    # VIF(方差膨胀因子)
    _values = datain.values
    vif_list = list(map(lambda x: [datain.columns[x], variance_inflation_factor(
        _values, x)], range(len(datain.columns))))
    vif_list = pd.DataFrame(vif_list, columns=['Var', 'Vif'])

    # Corss table
    corr_table = datain.corr(method = corr_method).fillna(0)
    
    # 计算需要展示的变量 
    dict_x = {}
    for c, row in corr_table.iteritems():
        for v in row.index:
            if c != v and row[v] > (corr_plt_filter if type(corr_plt_filter) is float else 0):
                dict_x[(c, v)] = abs(row[v])
            
    show_cols = set()
    for x in sorted(dict_x.items(),key = lambda x:x[1],reverse = True):
        show_cols.add(x[0][0])
        show_cols.add(x[0][1])
        
        if len(show_cols) >= (corr_plt_filter if type(corr_plt_filter) is int else 100):
            break    
    
    if len(show_cols) >= 2:
        f, ax = plt.subplots(figsize = figsize)
        sns.heatmap(corr_table[list(show_cols)].T[list(show_cols)].applymap(lambda x: round(x, rounds)), annot=True,
                    annot_kws={'size': font_size}, linewidths=0.05)
        ax.tick_params(axis='x', labelsize=font_size)
        ax.tick_params(axis='y', labelsize=font_size)
        plt.title('Character variable CORR heat map('+corr_method+').')
        plt.show()

    return pd.merge(vif_list, corr_table, left_on='Var', right_index=True)

if __name__ == '__main__':
    # Example
    from sklearn.datasets import load_breast_cancer
        
    datain_example = load_breast_cancer()
    datain_example = pd.DataFrame(
        datain_example.data, columns=datain_example.feature_names)
    
    corr_df = Corr_Vif(datain_example, corr_plt_filter=0.95, figsize=(18,12))
    corr_df = Corr_Vif(datain_example, corr_plt_filter=15, figsize=(18,12))
            