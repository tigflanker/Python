# -*- coding: utf-8 -*-
# Author: tigflanker
# Date: 12 Dec 2018
# 数据探索

# 0. 导包/宏参设定
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import pandas as pd
pd.set_option('display.max_columns', 500)

y_var = ''  # Y变量

# 0.1 数据导入
datain_test = pd.read_csv(
    'D:\Desktop\数据源\企业端\train.csv', encoding='gbk', sep=',', engine='python')
#datain_test = pd.read_csv('http://biostat.mc.vanderbilt.edu/wiki/pub/Main/DataSets/titanic.txt')
#datain = pd.read_csv('~/data/qyhx_jxd.txt', skiprows = [0,1,2], encoding='utf-8', sep='\t')

# 1. 数据探索
# 返回：
# Variable_pre_exp：数据探索列表
# obs_miss_rate：数据集每条观测的缺失率（横向）

def Data_Explore(datain  # 数据入
                 ,ext_n=3  # 按照Means ± N * std方式计数极端值，此为N值
                 ,plt_out=''  # 定义变量分布、相关描述图地址，若不定义输出路径则不输出
                 ,plt_show=False  # 是否在output中输出分布图
                 ,char_cate_threshold=10  # 定义绘制字符型（分类型）饼图时，最大类数
                 ,sug_drop_na_rate=0.6  # 标识出变量缺失程度是否超过该阈值
                 ,sug_drop_ct_rate=0.2  # 标识出字符型变量类数是否超过阈值
                 ,sps=4  # 频数分布图按4宫格or9宫格展示（9宫格参数尚未调好、勿使用）
                 ,calc_obs_na=False  # 计算观测缺失率
                 ,ecd='utf-8'
                 ):
    
    # 1.1 数据全局印象
    datatp = datain.dtypes
    nume_cols = set(list(datatp[datatp != 'object'].index))
    char_cols = set(list(datatp[datatp == 'object'].index))

    def vpe(col):
        _zero_n = sum(datain[col] == 0)  # 0值个数
        _N = datain.shape[0]  # 数据条数
        _nunique = datain[col].nunique()  # 唯一值数
        _isnull_n = sum(datain[col].isnull())  # 缺失值数
        _mode_v = datain[col].mode()[0] if _nunique > 0 else np.nan  # 众数

        vpe = [col  # 字段名
               , 'Numeric' if col in nume_cols else 'Character'  # 字段类型
               , str(datain[col].min())+'|'+str(datain[col].max()) if col in nume_cols else \
               str(min(map(lambda y:len(str(y)), datain[col])))+'|'+str(
                   max(map(lambda y:len(str(y)), datain[col])))  # 最大最小值
               , _zero_n  # 0值数
               , '%.2f%%' % (_zero_n * 100 / _N)  # 0值率
               , _nunique  # 唯一值数
               , '%.2f%%' % (_nunique * 100 / _N)  # 唯一值率
               , _isnull_n  # 缺失值数
               , '%.2f%%' % (_isnull_n * 100 / _N)  # 缺失值率
               , _mode_v  # 众数值
               , '%.2f%%' % (sum(datain[col] == _mode_v) * 100 / _N)  # 众数率
               , 0 if col in char_cols else sum((datain[col] < datain[col].mean() - ext_n * datain[col].std()) \
                                                | (datain[col] > datain[col].mean() + ext_n * datain[col].std()))   # 极端值率
               , (_nunique / _N >= sug_drop_ct_rate) & (col in char_cols)  # 唯一值率超过阈值
               , _isnull_n / _N >= sug_drop_na_rate  # 缺失值率超过阈值
               ]
        return vpe

    Variable_pre_exp = pd.DataFrame(list(map(vpe, datain.columns)), columns=[
                                    'var', 'var_type', 'min_max', '0_n', '0_r', 'val_cate_n', 'val_cate_r', 'na_n', 'na_r', 'md_v', 'md_r', 'ext_'+str(ext_n), 'sug_drop_cate', 'sug_drop_na'])
    _rename = {'var': '字段名', 'var_type': '字段类型', 'min_max': '最小最大值（字符型为长度）', '0_n': '0值数', '0_r': '0值率',
               'val_cate_n': '唯一值数', 'val_cate_r': '唯一值率', 'na_n': '缺失数', 'na_r': '缺失率', 'md_v': '众数值', 'md_r': '众数率',
               'ext_'+str(ext_n): str(ext_n)+'倍标准差极端值数', 'sug_drop_cate': '值类数超过'+str(sug_drop_ct_rate),
               'sug_drop_na': '值缺失超过'+str(sug_drop_na_rate)}
    Variable_pre_exp_name = Variable_pre_exp.rename(index=str, columns=_rename)

    print('>>>>> 数据质量探查表 <<<<<\n', Variable_pre_exp_name)

    # 计算观测缺失率
    if calc_obs_na:
        obs_miss_rate = datain.apply(lambda row: sum(
            row.isnull()) / datain.shape[1], axis=1)
        _value_counts = obs_miss_rate.value_counts().sort_index()

        _labels = [round(x * 100, 2) for x in _value_counts.index]
        _quants = [round(x * 100 / datain.shape[0], 2) for x in _value_counts]

        _width = 0.4
        _ind = np.linspace(0.5, 9.5, len(_value_counts))

        fig = plt.figure(figsize=(12, 6))
        ax = fig.add_subplot(111)

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
        Variable_pre_exp_name.to_csv(
            plt_out + '.csv', encoding=ecd, index=False)

        mpl.rcParams['axes.unicode_minus'] = False
        mpl.rcParams['font.sans-serif'] = ['SimHei']
        plt.rcParams['font.sans-serif'] = ['SimHei']

        with PdfPages(plt_out + '.pdf') as pdf:
            # 9宫格or4宫格
            sps = 4
            plt_control = {4: {'cols': 221, 'wspace': 0.4, 'hspace': 0.6, 'fontsize': 10},
                           9: {'cols': 331, 'wspace': 0.6, 'hspace': 0.8, 'fontsize': 8}}

            _cols = nume_cols | set(Variable_pre_exp.loc[(Variable_pre_exp.var_type == 'Character') & (
                Variable_pre_exp.val_cate_n <= char_cate_threshold), 'var'])

            for t, x in enumerate(_cols):
                plt.subplot(plt_control[sps]['cols'] + t % sps)
                plt.subplots_adjust(
                    hspace=plt_control[sps]['hspace'], wspace=plt_control[sps]['wspace'])

                if x in nume_cols:
                    datain[x].hist(bins=25)
                    plt.title('Num var ' + x + "'s distribution.",
                              fontsize=plt_control[sps]['fontsize'])
                else:
                    char_vc = datain[x].value_counts()
                    plt.pie(char_vc.values, labels=char_vc.index,
                            autopct='%1.1f%%', startangle=90)
                    plt.axis('equal')
                    plt.title('Char var ' + x + "'s value counts.",
                              fontsize=plt_control[sps]['fontsize'])

                if (t % sps == sps - 1) or (t == len(_cols) - 1):
                    pdf.savefig()
                    if plt_show:
                        plt.show()
                    plt.close()

    return Variable_pre_exp, obs_miss_rate


#Variable_pre_exp = Data_Explore(datain,plt_out = 'D:/data/kaggle/titanic/data_pre_exp')
#Variable_pre_exp = Data_Explore(datain_test,plt_out = 'D:/Desktop/数据源/企业端/aftagg',cross_plt=15,ecd='gbk')
Variable_pre_exp, obs_miss_rate = Data_Explore(
    datain_test, plt_out='D:/Desktop/xxx')
#Variable_pre_exp = Data_Explore(datain,plt_out = '~/result/qyhx_jxd_exp')

# 使用
# 删除缺失率或类别数过高的特征
# for x in Variable_pre_exp.loc[Variable_pre_exp.sug_drop_cate, 'var']: # Variable_pre_exp.sug_drop_na |
#    del datain_test[x]
#
#Variable_pre_exp = Variable_pre_exp[(~Variable_pre_exp.sug_drop_na) & (~Variable_pre_exp.sug_drop_cate)]\
#    .reset_index(drop=True).drop(['sug_drop_na', 'sug_drop_cate'], 1)

# 删除缺失率过高的观测
#datain_test = datain_test[list(obs_miss_rate < 0.9)].reset_index(drop=True)
