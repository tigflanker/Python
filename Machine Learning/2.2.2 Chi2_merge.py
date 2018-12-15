# -*- coding: utf-8 -*-
# Author: guoyang14
# Date: 04 Oct 2018
# Update: 10 Dec 2018：修正分箱边界问题 
# 卡方分箱

import re
import pandas as pd 
pd.set_option('display.max_columns', 500)
import numpy as np
from scipy import stats
import math
import matplotlib.pyplot as plt

def calc_chi2(datain, group = 'group', y_var = 'y_var'):
    N = len(datain)
    chi2 = 0
    for g in set(datain[group]):
        for y in set(datain[y_var]):
            Agj = sum((datain[group] == g) & (datain[y_var] == y))
            Egj = sum(datain[group] == g) * sum(datain[y_var] == y) / N
            chi2 += 0 if Egj == 0 else (Agj - Egj)**2/Egj

    return chi2

def chi2_merge(
    chi2_datain = 'datain'
    ,divided_var = 'divided_var'
    ,y_var = 'y_var'
    ,aim_groups = 'auto'  # 目标组数(手动指定卡方分箱的目标组数，不建议改动)
    ,step_size = 1  # 该参数用于定义每次迭代收集最小的N个卡方值对应的组标进行合并，用于提高合并速率，不建议大于5
    ,print_detail = False
    ,calc_entropy = False
    ,collection_point = 20  # 在分组降维至该参数以下时，收集卡方值/收集分桶规则
    ):

    chi2_datain = chi2_datain.loc[chi2_datain[divided_var].notna()].sort_values(by=[divided_var, y_var]).reset_index()
    chi2_datain[divided_var + '_chi2_group'] = chi2_datain.index
    Total_variable_length = len(chi2_datain)
    
    chi2_change = []  # 用于绘制卡方和值变化图
    divide_rule = []  # 用于收集区间规则
    spr_list = [] # 收集Spearman-R变化
    while(len(chi2_datain[divided_var + '_chi2_group'].unique()) > 2):
        group_list = list(chi2_datain[divided_var + '_chi2_group'].unique())
        
        min_chi2 = {}  # 用于收集卡方值及其组标
        chi2_avg = 0
        if len(chi2_datain[divided_var + '_chi2_group'].unique()) <= 30:  # 30以内降速为1，以防跨过目标组数
            step_size = 1
        for i in range(len(group_list) - 1):
            chi2 = calc_chi2(chi2_datain[(chi2_datain[divided_var + '_chi2_group'] == group_list[i]) \
                                         | (chi2_datain[divided_var + '_chi2_group'] == group_list[i + 1])]
                             , group = divided_var + '_chi2_group', y_var = y_var)  
            chi2 = round(chi2, 8)  # 将计算得到的卡方值按8位精度截尾
            chi2_avg += chi2
            if chi2 in min_chi2:
                min_chi2[chi2] = min_chi2[chi2] + [group_list[i]]  
            elif len(min_chi2) < step_size:
                min_chi2[chi2] = [group_list[i]]  
            elif chi2 < max(min_chi2):
                del min_chi2[max(min_chi2)]
                min_chi2[chi2] = [group_list[i]]
                
        if len(min_chi2) < step_size:
            min_chi2 = {min(min_chi2) : min_chi2[min(min_chi2)]}
        merge_group_list = set(eval('[' + re.sub('[\[\]]','',str(list(min_chi2.values()))) + ']'))  # 数组扁平化
                        
        # 变组
        def temp_func1(x):
            if x in merge_group_list:
                x = group_list[group_list.index(x) + 1]
                while(x in merge_group_list):
                    x = group_list[group_list.index(x) + 1]
            return x 
        
        chi2_datain.loc[:,divided_var + '_chi2_group'] = list(map(temp_func1,chi2_datain[divided_var + '_chi2_group']))
        
        # 计算Spearman相关系数
        calc_spm = chi2_datain.groupby(divided_var + '_chi2_group')
        spm_r, spm_p = stats.spearmanr(calc_spm[divided_var + '_chi2_group'].mean(), calc_spm[y_var].mean())
        
        # 熵和gini(gini暂略)⁴
        if calc_entropy:
            def chi2_entropy(y):
                data_part = chi2_datain.loc[chi2_datain[divided_var + '_chi2_group'] == y,[y_var,divided_var + '_chi2_group']]
            
                return (len(data_part) / Total_variable_length) * \
                       sum(map(lambda x: - (sum(data_part[y_var] == x) / len(data_part)) * \
                               math.log((sum(data_part[y_var] == x) / len(data_part)),2)
                               ,data_part[y_var].unique()))
        
            chi2_entropy = sum(map(chi2_entropy, chi2_datain[divided_var + '_chi2_group'].unique()))
        
        # 输出
        if print_detail:
            if calc_entropy:
                chi2_entropy = ', entropy ' + str(round(chi2_entropy, 8))
            else:
                chi2_entropy = ''
                
            print('Tot-len',Total_variable_length
                  ,', cur-gps',len(chi2_datain[divided_var + '_chi2_group'].unique())
                  ,', step size',step_size
                  ,', avg chi2',round(chi2_avg / len(group_list), 4)
                  ,', spr-r',round(spm_r, 4)
                  ,chi2_entropy)
            
            # 收集卡方值的变化  
            if len(chi2_datain[divided_var + '_chi2_group'].unique()) <= collection_point:
                chi2_change.append([len(chi2_datain[divided_var + '_chi2_group'].unique())
                                    ,round(chi2_avg / len(group_list), 8)
                                    ,round(chi2_avg / len(group_list), 8) - last_chi2
                                    ,round(spm_r, 8)])
                
        # 收集Spearman-R变化 
        if len(chi2_datain[divided_var + '_chi2_group'].unique()) <= collection_point:
            spr_list.append(spm_r)                
        last_chi2 = round(chi2_avg / len(group_list), 8)  # 记录上次(变组前)卡方均值    
                
        # 重新序列划分好的区间值
        if len(chi2_datain[divided_var + '_chi2_group'].unique()) <= collection_point:
            current_group_id = list(chi2_datain[divided_var + '_chi2_group'].unique())
            chi2_datain[divided_var + '_chi2_group'] = list(map(lambda x:current_group_id.index(x) + 1, chi2_datain[divided_var + '_chi2_group']))
    
            # 划分规则
            divide_rule_x = chi2_datain.groupby(divided_var + '_chi2_group')[divided_var].agg(['min', 'max'])
            divide_rule_x['groups'] = len(divide_rule_x)
            
            if len(divide_rule):  
                divide_rule = pd.concat([divide_rule, divide_rule_x])                    
            else:
                divide_rule = divide_rule_x
    
    # 绘制卡方变化图⁵ 
    if print_detail:
        x_group = list(map(lambda x:x[0], chi2_change))
        fig = plt.figure()
        
        ax1 = fig.add_subplot(111)
        ax1.plot(x_group, list(map(lambda x:x[3], chi2_change)))
        ax1.set_xlabel('Groups change')
        ax1.set_ylabel('Spearman R change(blue)')
        ax1.set_title('Spearman R change')
        
        ax2 = ax1.twinx()  # this is the important function
        ax2.plot(x_group, list(map(lambda x:x[1], chi2_change)), 'r')
        ax2.set_ylabel('Chi2 value change(red)')
        
        plt.grid(axis='x')  # 失效了
        plt.xticks(x_group)
        plt.show()        
        
    # 给出最优分箱组数 
    # 规则：首先筛选有效分箱组；其次在有效分箱组中从多至少遍历，
    #      直至Spearman最佳分箱点，或者是最小有效分箱组
    divide_rule['interval'] = divide_rule['max'] - divide_rule['min']
    available_groups =  list(filter(lambda x:
        len(set(divide_rule.loc[divide_rule.groups == x,'interval']) - set([0])) > 1
        ,set(divide_rule.groups)))
        
    # Spearman-R指示最佳分箱点 
    spr_index_bst = len(spr_list) - np.argwhere(abs(np.array(spr_list)) == max(abs(np.array(spr_list)[:-1])))[0,0] + 1  
    
    best_cut_point = max(min(available_groups), spr_index_bst) if aim_groups == 'auto' else aim_groups
    
    _best_cut_rule = divide_rule.loc[divide_rule.groups == best_cut_point,['min','max']]
    best_cut_rule = []
    
    _best_cut_rule = _best_cut_rule[_best_cut_rule['max'] != _best_cut_rule['min']].reset_index(drop = True)
    for i in _best_cut_rule.index:
        if i == 0:
            best_cut_rule.append(tuple([-np.inf, (_best_cut_rule.loc[i,'max'] + _best_cut_rule.loc[i + 1,'min']) / 2 ]))
        elif i == max(_best_cut_rule.index):
            best_cut_rule.append(tuple([(_best_cut_rule.loc[i,'min'] + _best_cut_rule.loc[i - 1,'max']) / 2, np.inf]))
        else: #if _best_cut_rule.loc[i,'min'] != _best_cut_rule.loc[i,'max']:
            best_cut_rule.append(tuple([(_best_cut_rule.loc[i,'min'] + _best_cut_rule.loc[i - 1,'max']) / 2, 
                                       (_best_cut_rule.loc[i,'max'] + _best_cut_rule.loc[i + 1,'min']) / 2]))
            
    if print_detail:
        print(divided_var + "'s " + 'best chi2-merge groups:', len(best_cut_rule))
        
    # 按最优分箱点分箱 
    def cut_temp(x):
        group_x = 0
        for i, d in enumerate(best_cut_rule):
            if (-np.inf == d[0]) & (d[1] == x):
                group_x = i
                break
            elif (d[0] <= x) & (x < d[1]):  # 左闭右开，边界值往后堆
                group_x = i
                break
        return group_x
        
    chi2_datain[divided_var] = list(map(cut_temp, chi2_datain[divided_var])) 
    chi2_datain = chi2_datain[['index', divided_var]]
    chi2_datain.set_index(['index'], inplace=True)       

    return chi2_datain, best_cut_rule # divide_rule

datain = pd.read_csv('http://biostat.mc.vanderbilt.edu/wiki/pub/Main/DataSets/titanic.txt')

chi2_df, chi2_divide_rule = chi2_merge(chi2_datain = datain[['survived','age']], y_var = 'survived'
                                       , divided_var = 'age'
                                       , print_detail = True, step_size = 3)

# ref 
# ref1: https://www.powershow.com/view1/1fa37b-ZDc1Z/ChiMerge_Discretization_powerpoint_ppt_presentation
# ref2: http://www.aaai.org/Papers/AAAI/1992/AAAI92-019.pdf
# ref3: https://blog.csdn.net/autoliuweijie/article/details/51594373
