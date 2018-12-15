# -*- coding: utf-8 -*-
# Author: guoyang14
# Date: 12 Dec 2018
# 数据探索

# 0. 导包/宏参设定
import numpy as np
import pandas as pd 
pd.set_option('display.max_columns', 500)
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import seaborn as sns

y_var = ''  # Y变量

# 0.1 数据导入 
datain_test = pd.read_csv('http://biostat.mc.vanderbilt.edu/wiki/pub/Main/DataSets/titanic.txt')
#datain = pd.read_csv('~/data/qyhx_jxd.txt', skiprows = [0,1,2], encoding='utf-8', sep='\t')

# 1. 数据探索
# 返回：
# Variable_pre_exp：数据探索列表 
# obs_miss_rate：数据集每条观测的缺失率（横向）
def data_exp(datain  # 数据入
             ,y_var = y_var  # 定义Y变量
             ,ext_n = 3  # 按照Means ± N * std方式计数极端值，此为N值
             ,plt_out = ''  # 定义变量分布、相关描述图地址，若不定义输出路径则不输出
             ,plt_show = False  # 是否在output中输出分布图
             ,char_cate_threshold = 10  # 定义绘制字符型（分类型）饼图时，最大类数
             ,sug_drop_na_rate = 0.6  # 标识出变量缺失程度是否超过该阈值
             ,sug_drop_ct_rate = 0.2  # 标识出字符型变量类数是否超过阈值
             ,cross_plt = False  # 输出交叉图
             ,calc_obs_na = False  # 计算观测缺失率 
             ,ecd = 'utf-8'
                   ):
    # 1.1 数据全局印象
    datatp = datain.dtypes
    nume_cols = set(list(datatp[datatp != 'object'].index))
    char_cols = set(list(datatp[datatp == 'object'].index))

    def vpe(col):
        vpe = [col
               ,'Numeric' if col in nume_cols else 'Character'
               ,str(datain[col].min())+'|'+str(datain[col].max()) if col in nume_cols else \
                str(min(map(lambda y:len(str(y)),datain[col])))+'|'+str(max(map(lambda y:len(str(y)),datain[col])))
               ,sum(datain[col] == 0)
               ,'%.2f%%' % (sum(datain[col] == 0) * 100 / datain.shape[0])
               ,datain[col].nunique()
               ,'%.2f%%' % (datain[col].nunique() * 100 / datain.shape[0])
               ,sum(datain[col].isnull())
               ,'%.2f%%' % (sum(datain[col].isnull()) * 100 / datain.shape[0])
               ,datain[col].mode()[0]
               ,'%.2f%%' % (sum(datain[col] == datain[col].mode()[0]) * 100 / datain.shape[0])               
               ,0 if col in char_cols else sum((datain[col] < datain[col].mean() - ext_n * datain[col].std()) \
                | (datain[col] > datain[col].mean() + ext_n * datain[col].std())) 
               ,(datain[col].nunique() / datain.shape[0] >= sug_drop_ct_rate) & (col in char_cols)
               ,sum(datain[col].isnull()) / datain.shape[0] >= sug_drop_na_rate
               ]
        return vpe
    
    Variable_pre_exp = pd.DataFrame(list(map(vpe, datain.columns)), columns=['var','var_type','min_max','0_n','0_r','val_cate_n','val_cate_r'
                                                               ,'na_n','na_r','md_v','md_r','ext_'+str(ext_n),'sug_drop_cate','sug_drop_na'])
    _rename = {'var':'字段名','var_type':'字段类型','min_max':'最小最大值（字符型为长度）','0_n':'0值数','0_r':'0值率',
               'val_cate_n':'值类型数','val_cate_r':'值类型率','na_n':'缺失数','na_r':'缺失率','md_v':'众数值','md_r':'众数率',
               'ext_'+str(ext_n):str(ext_n)+'倍标准差极端值数','sug_drop_cate':'值类数超过'+str(sug_drop_ct_rate),
               'sug_drop_na':'值缺失超过'+str(sug_drop_na_rate)}
    Variable_pre_exp_name = Variable_pre_exp.rename(index=str, columns=_rename)
    
    print('>>>>> 数据质量探查表 <<<<<\n',Variable_pre_exp_name)
    
    # 计算观测缺失率 
    if calc_obs_na:
        obs_miss_rate = datain.apply(lambda row:sum(row.isna()) / datain.shape[1],axis=1)
        _value_counts = obs_miss_rate.value_counts().sort_index()
        
        _labels = [round(x * 100,2) for x in _value_counts.index]
        _quants = [round(x * 100 / datain.shape[0], 2) for x in _value_counts]
        
        _width = 0.4
        _ind = np.linspace(0.5,9.5,len(_value_counts))
        
        fig = plt.figure(figsize=(12,6))
        ax  = fig.add_subplot(111)
        
        ax.bar(_ind - _width/2, _quants, _width)
        ax.set_xticks(_ind - _width/2)
        ax.set_xticklabels(_labels)
        
        ax.set_xlabel('Missing Rate(%)')
        ax.set_ylabel('Obs Rate(%)')
        ax.set_title('Observation missing rate')
        plt.grid()
        plt.show()
    else:
        obs_miss_rate = []

    # 1.2 特征分布、阈值¹
    if len(plt_out):
        Variable_pre_exp_name.to_csv(plt_out + '.csv',encoding=ecd,index=False)
        
        mpl.rcParams['axes.unicode_minus'] = False 
        mpl.rcParams['font.sans-serif'] = ['SimHei'] 
        plt.rcParams['font.sans-serif'] = ['SimHei'] 
        
        with PdfPages(plt_out + '.pdf') as pdf:   
            for x in datain.columns:
                if x in nume_cols:
                    datain[x].hist(bins = 25)
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
                    plt.show() 
                    plt.close()
          
            # 1.3 特征相关性矩阵
            # 20181203:将Cross矩阵限定输出变量数
            if cross_plt > 0:
                # Cross Keep
                cross_plt = 10 if type(cross_plt) is bool else cross_plt
                
                corr = datain.corr().fillna(0)
                corr_list = dict(list(map(lambda x:[x, max(abs(corr.loc[corr.index != x,x]))], corr.columns)))             
                cross_keep_cols = [x[0] for x in sorted(corr_list.items(), key = lambda k: k[1], reverse = True)[:cross_plt]]
                                                
                f, ax= plt.subplots(figsize = (14, 10))                    
                sns.heatmap(datain[cross_keep_cols].corr(),annot= True,annot_kws={'size':9}, linewidths = 0.05)
                ax.tick_params(axis='x',labelsize=8)
                ax.tick_params(axis='y',labelsize=8)
                plt.title('Character variable CORR heat map.')
                
                pdf.savefig()
                if plt_show: plt.show()
                plt.close()

#                if len(y_var):
#                    corr_list = corr[y_var].abs()
#                    cross_keep_cols = corr_list.sort_values(ascending=False)[:cross_plt + 1].index
#                    
#                    sns.pairplot(datain[cross_keep_cols],hue= y_var,diag_kind='kde')
#                    plt.title('Character variable pair plot on' + y_var + '.')
#                    pdf.savefig(dpi = 1000)
#                    if plt_show: plt.show()
#                    plt.close()
        
    return Variable_pre_exp, obs_miss_rate
    
#Variable_pre_exp = data_exp(datain,plt_out = 'D:/data/kaggle/titanic/data_pre_exp')
Variable_pre_exp, obs_miss_rate = data_exp(datain_test,calc_obs_na=True)
#Variable_pre_exp = data_exp(datain,plt_out = '~/result/qyhx_jxd_exp')

# 使用
# 删除缺失率或类别数过高的特征 
for x in Variable_pre_exp.loc[Variable_pre_exp.sug_drop_cate, 'var']: # Variable_pre_exp.sug_drop_na | 
    del datain_test[x]
    
Variable_pre_exp = Variable_pre_exp[(~Variable_pre_exp.sug_drop_na) & (~Variable_pre_exp.sug_drop_cate)]\
    .reset_index(drop=True).drop(['sug_drop_na', 'sug_drop_cate'], 1)
    
# 删除缺失率过高的观测
datain_test = datain_test[list(obs_miss_rate < 0.9)].reset_index(drop=True)