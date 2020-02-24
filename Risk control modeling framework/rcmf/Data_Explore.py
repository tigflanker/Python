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

# 1. 数据探索
# 返回：
# Variable_pre_exp：数据探索列表
# obs_miss_rate：数据集每条观测的缺失率（横向）

def Data_Explore(datain  # 数据入
                 ,qtl=[0.25, 0.5, 0.75]  # 计算分位数情况 
                 ,ext_t=3  # 按照 Mean ± t * std方式计数极端值
                 ,samples=5  # 携带实际观测值 
                 ,plt_out=''  # 定义变量分布、相关描述图地址，若不定义输出路径则不输出
                 ,plt_show=False  # 是否在output中输出分布图
                 ,char_cate_threshold=10  # 定义绘制字符型（分类型）饼图时，最大类数
                 ,sps=4  # 频数分布图按4宫格or9宫格展示（9宫格参数尚未调好、勿使用）
                 ,calc_obs_na=False  # 计算观测缺失率
                 ,ecd='utf-8'
                 ):
                 
    '''
    # 参数说明
    datain                 : 数据输入（必须）
    qtl=[0.25, 0.5, 0.75]  : 计算分位数情况 
    ext_t=3                : 按照 Mean ± t * std方式计数极端值
    samples=5              : 携带实际观测值 
    plt_out=''             : 定义变量分布、相关描述图地址，若不定义输出路径则不输出
    plt_show=False         : 是否在output中输出分布图
    char_cate_threshold=10 : 定义绘制字符型（分类型）饼图时，最大类数
    sps=4                  : 频数分布图按4宫格or9宫格展示（9宫格参数尚未调好、勿使用）
    calc_obs_na=False      : 计算观测缺失率
    ecd='utf-8'            : 结果输出字符类型
    
    # Example:
    datain_test = pd.read_csv('http://biostat.mc.vanderbilt.edu/wiki/pub/Main/DataSets/titanic.txt')
    
    Variable_pre_exp, obs_miss_rate = Data_Explore(datain_test, plt_out='D:/Desktop/xxx')
    '''
    
    # 1.1 数据全局印象
    datatp = datain.dtypes
    char_cols = set(list(datatp[datatp == 'object'].index))
    nume_cols = set(list(datatp[datatp != 'object'].index))
    
    # 通用：缺失、类数
    # 数值：均值/标准差、最大/最小值、0值、分布
    # 字符：众数
    
    # 数值型或通用
    _N = datain.shape[0]
    _isnull_n = datain.isnull().sum().rename('na_n')
    _nunique = datain.nunique().rename('val_cate_n')
    
    _zero_n = (datain == 0).sum().rename('0_n')
    
    # 数据分布描述（此处将混合型字段中的字符值call missing） 
    _num_df = datain.apply(lambda x:pd.to_numeric(x, errors='coerce'))
    _desc = _num_df.describe(percentiles = qtl, include = 'all').T
    
    _desc['mean_std'] = _desc['mean'].apply(lambda x:str(round(x, 4))) + '|' \
                      + _desc['std'].apply(lambda x:str(round(x, 4)))
    _desc['min_max'] = _desc['min'].apply(lambda x:str(x)) + '|' \
                     + _desc['max'].apply(lambda x:str(x))
    _desc['skew_kurt'] = _num_df.skew().apply(lambda x:str(round(x, 4))) + '|' \
                       + _num_df.kurt().apply(lambda x:str(round(x, 4)))
    _pvar = 'p_'+'_'.join([str(int(x * 100)) for x in qtl])   
    _desc[_pvar] = eval("+'|'+".join(["_desc['"+str(int(x * 100))+
         "%'].apply(lambda x:str(round(x, 4)))" for x in qtl]))

    # 计算极端值延伸的计算：两界极端值占比、去极端值均数/标准差、最大最小值
    _ms_dt = _desc[['mean', 'std']].dropna().T.to_dict()  # 将均数、标准差做成字典，供极端值计算 
    
    def ext_calc(col):
        _en = _desc['count'][col]    
        
        _elown = np.sum(_num_df[col] < _ms_dt[col]['mean'] - ext_t * _ms_dt[col]['std'])
        _euppn = np.sum(_num_df[col] > _ms_dt[col]['mean'] + ext_t * _ms_dt[col]['std'])
    
        _edesc = _num_df[col][_num_df[col].between(_ms_dt[col]['mean'] - ext_t * _ms_dt[col]['std'], 
                         _ms_dt[col]['mean'] + ext_t * _ms_dt[col]['std'])].describe()
    
        return (str(round(_elown * 100 / _en, 2)) + '%|' + str(round(_euppn * 100 / _en, 2)) + '%'
                ,str(round(_edesc['mean'], 4)) + '|' + str(round(_edesc['std'], 4))
                ,str(_edesc['min']) + '|' + str(_edesc['max']))
                
    # 框架、众数、极端值延伸 
    def vpe(col):
        vpe = [col  # 字段名
               , 'Character' if col in char_cols else 'Numeric'  # 字段类型
               , datain[col].mode()[0] if _nunique[col] > 0 else np.nan  # 众数
               
               , ext_calc(col) if col in _ms_dt else ('nan|nan', 'nan|nan', 'nan|nan')  # 极端值相关，待拆包
               ]
        return vpe
    
    Variable_pre_exp = pd.DataFrame(list(map(vpe, datain.columns)), columns=['var', 'var_type', 'md_v', 'ext_mass'])
    Variable_pre_exp['ext_low_upp_r'] = Variable_pre_exp.ext_mass.apply(lambda x:x[0])
    Variable_pre_exp['noext_mean_std'] = Variable_pre_exp.ext_mass.apply(lambda x:x[1])
    Variable_pre_exp['noext_min_max'] = Variable_pre_exp.ext_mass.apply(lambda x:x[2])
    
    # 拼接
    Variable_pre_exp = pd.concat([Variable_pre_exp.set_index('var', drop=False), _desc, 
                                  pd.concat([_isnull_n,_nunique,_zero_n], axis=1)], axis=1, sort=False)
            
    # 取样
    if samples > 0:
        datain_sample = datain.apply(lambda x:pd.Series([np.nan] * samples, name=x.name) if \
                        np.sum(x.notna()) == 0 else x.dropna().sample(samples, replace = True).reset_index(drop = True))
        datain_sample = datain_sample.T.rename({i:'sample_' + str(i + 1) for i in range(samples)}, axis = 1)    
        
        Variable_pre_exp = pd.concat([Variable_pre_exp, datain_sample], axis=1, sort=False)    
        
    # 率延伸
    Variable_pre_exp['0_r'] = round(Variable_pre_exp['0_n'] * 100 / _N, 2).apply(str) + '%'
    Variable_pre_exp['na_r'] = round(Variable_pre_exp['na_n'] * 100 / _N, 2).apply(str) + '%'
    Variable_pre_exp['val_cate_r'] = round(Variable_pre_exp['val_cate_n'] * 100 / _N, 2).apply(str) + '%'
    Variable_pre_exp['num_r'] = round(Variable_pre_exp['count'] * 100 / (_N - Variable_pre_exp['na_n']), 2).apply(str) + '%'
    
    # 整理输出
    Variable_pre_exp = Variable_pre_exp[['var', 'var_type', 'na_n', 'na_r', 'val_cate_n', 
                                         'val_cate_r', 'mean_std', 'min_max', '0_n', '0_r', 
                                         'skew_kurt', _pvar, 'num_r','md_v', 
                                         'ext_low_upp_r', 'noext_mean_std', 'noext_min_max'
                                         ]
            + ['sample_' + str(i + 1) for i in range(samples)] if samples > 0 else []  # 取样字段 
            ]

    _rename = {'var': '字段名', 'var_type': '字段类型', 'min_max': '最小最大值', 'mean_std': '均值标准差', 
               '0_n': '0值数', '0_r': '0值率', 'val_cate_n': '唯一值数', 'val_cate_r': '唯一值率', 'skew_kurt': '偏度峰度', 
               'na_n': '缺失数', 'na_r': '缺失率', 'md_v': '众数值', _pvar: '分布情况'+str(qtl), 'num_r': '字段数值占比',
               'ext_low_upp_r': '上下界极端值比例('+str(ext_t)+'倍标准差)', 'noext_mean_std': '去极端值均值标准差', 'noext_min_max': '去极端值最小最大值'
               }
    _rename.update({'sample_' + str(i + 1):'抽样' + str(i + 1) for i in range(samples)} if samples > 0 else {})

    Variable_pre_exp_name = Variable_pre_exp.rename(index=str, columns=_rename)

    print('>>>>> Data explore list(N = ' + str(_N) + ') <<<<<\n', Variable_pre_exp_name)

    # 计算观测缺失率
    if calc_obs_na:
        obs_miss_rate = datain.apply(lambda row: np.sum(
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

        try:
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
        except:
            print('Notice:No plotting frameworks exist.')

    return Variable_pre_exp, obs_miss_rate
