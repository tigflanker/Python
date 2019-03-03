# -*- coding: utf-8 -*-
# Author: Huyao
# Packing: GY
# Inc Date: 12 Dec 2018

import math
import numpy as np
import pandas as pd 

# Author:Huyao
def psi(bench, comp, group):
    
    if type(bench) is pd.Series:
        bench = np.array(bench)
    if type(comp) is pd.Series:
        comp = np.array(comp)
    
    ben_len = len(bench)
    comp_len = len(comp)
    bin_len = int(math.floor(ben_len/group))
    
    bench.sort()
    comp.sort()
    psi_cut=[]
    
    for i in range(1,group):
        lowercut=bench[(i-1) * bin_len + 1]
        if i != group:
            uppercut = bench[(i * bin_len)]
            ben_cnt = bin_len
        else:
            uppercut = bench[-1]
            ben_cnt = ben_len - group * (bin_len - 1)
            
        comp_cnt = len([j for j in comp if j > lowercut and j <= uppercut])
        ben_pct = (ben_cnt+0.0) / ben_len
        comp_pct = (comp_cnt+0.0) / comp_len
        
        if ben_pct == 0:
            ben_pct = 0.0001
        if comp_pct == 0:
            comp_pct = 0.0001
            
        psi_cut.append((ben_pct - comp_pct) * math.log(ben_pct / comp_pct))

    return sum(psi_cut)

def PSI(bench
        ,comp
        ,group = 10
        ):

    if type(bench) is pd.DataFrame and type(comp) is pd.DataFrame:
        com_cols = set(bench.columns) & set(comp.columns)
        psi_value = list(map(lambda col:[col, psi(bench[col], comp[col], group)], com_cols))
        psi_value = pd.DataFrame(psi_value, columns = ['Var', 'Psi'])
    else:
        psi_value = psi(bench, comp, group)
        print('PSI value:', psi_value)
        
    return psi_value

if __name__ == '__main__':
    #Example:
    from sklearn.datasets import load_breast_cancer
    datain_example = load_breast_cancer()
    datain_example = pd.DataFrame(datain_example.data, columns=datain_example.feature_names)
    
    psi_value = PSI(datain_example['mean radius'], datain_example['mean texture'])
    psi_value = PSI(datain_example[datain_example.index < 300], datain_example[datain_example.index >= 300])
    
