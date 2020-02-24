# -*- coding: utf-8 -*-
# Author: Huyao
# Packing: GY
# Inc Date: 12 Dec 2018

import math
import numpy as np
import pandas as pd 

# 单独PSI计算程序
def psi(
        base = pd.Series()
        ,comp = pd.Series()
        ,threshold = [10, 100]
        ,check = False
        ,drop_na = False
        ):
    
    # 基本统计/参数分解
    groups = threshold[0]
    max_groups = threshold[1]
        
    union_values = set(base.dropna()) | set(comp.dropna()) | set({} if drop_na else {np.nan})
    union_groups = len(union_values)
    base_n, comp_n = len(base), len(comp)
    
    # 箱内计算
    def bin_psi_calc(x):
        
        if type(x) in (tuple, list):
            base_count = np.sum(base.between(x[0], x[1]))
            comp_count = np.sum(comp.between(x[0], x[1]))
        else:
            base_count = np.sum(base == x)
            comp_count = np.sum(comp == x)
        
        lpls = int(base_count * comp_count == 0)
        
        base_ratio = (base_count + lpls) / (base_n + lpls)
        comp_ratio = (comp_count + lpls) / (comp_n + lpls)
    
        if check:
            print('Var:',base.name,'Bin:',x, 'Bin PSI:',round((base_ratio - comp_ratio) * math.log(base_ratio / comp_ratio), 2))
    
        return (base_ratio - comp_ratio) * math.log(base_ratio / comp_ratio)
    
    if 'object' in (base.dtype, comp.dtype):
        if union_groups > max_groups:
            print('The character type variable "'+base.name+'" contain too many value types(',union_groups,').')
            return np.nan
        else:
            return np.sum(list(map(bin_psi_calc, union_values)))
    else:  # 纯数值型
        if union_groups <= groups:
            return np.sum(list(map(bin_psi_calc, union_values)))  # 直接按照字符型对待
        else:        
            cut_points = np.linspace(min(union_values - {np.nan}), max(union_values - {np.nan}), num=groups + 1)
                    
            return np.sum(list(map(lambda u:bin_psi_calc((cut_points[u], cut_points[u + 1])), range(groups))) 
                          + ([] if drop_na else [bin_psi_calc(np.nan)]))
        
# 数据集循环计算(调度)
def PSI(base
        ,comp
        ,threshold = [10, 100]
        ,check = False
        ,drop_na = False
        ):
        
    """
    PSI计算说明：
    1. 如果比较对象为字符型/分类型变量，则需要按照两边的类数并集进行切割（是否需要考虑设置上限？）
    2. 如果比较对象为连续型变量，并且两边值类型数大于待划分类数，则正常划分并计算PSI
    3. 如果比较对象为连续型变量，两边并集类数也小于待划分类数，则取每个类数单独成组进行计算
    
    *1. 两边值的缺失值单独计算一组
    *2. 如若出现空箱，采用拉普拉斯平滑处理：当前两边组的例数均加1（分母同时加1）
    
    参数说明：
    统一使用下方的PSI函数，psi为子函数
    Base/Comp：两侧待比对数据，输入类型为Dataframe或Series
    threshold：为元组或列表，第一位为分箱个数，第二位针对字符型变量，如果超过类数，则不计算该变量
    check：检查，若关闭则静默计算
    
    # Example:
    from sklearn.datasets import load_breast_cancer
    datain_example = load_breast_cancer()
    datain_example = pd.DataFrame(datain_example.data, columns=datain_example.feature_names)
    
    psi_value = PSI(datain_example['mean radius'], datain_example['mean texture'])
    psi_value = PSI(datain_example[datain_example.index < 300], datain_example[datain_example.index >= 300])
    """

    if type(base) is pd.DataFrame and type(comp) is pd.DataFrame:
        com_cols = set(base.columns) & set(comp.columns)
        psi_value = list(map(lambda col:[col, psi(base[col], comp[col], threshold, check, drop_na)], com_cols))
        psi_value = pd.DataFrame(psi_value, columns = ['Var', 'Psi'])
    else:
        psi_value = psi(base, comp,  threshold, check, drop_na)
        print('PSI value:', psi_value)
        
    return psi_value
