# -*- coding: utf-8 -*-
# Author: guoyang14
# Date: 27 May 2019
# 敏感值加密解密

import random 
import hashlib
import pandas as pd

# 函数用途为对数据集中敏感信息加密，例如对模型标签加密
# PSR参数为加密/解密字典，key必须包含：struct、r、s、p（p为解密过程必须参数）
#    struct：结构排列方式
#    r：random，为随机数位数，默认为千位内
#    s：salt，为密令
#    p：加密过程留空，解密过程指定，例如 p:[0, 1]
#    例：解密过程：PSR = {'struct':'psr','r':3,'s':'apple','p':['AS','DF','AD']}
def PSR_Crypto(
        Serin = ''  # 输入类型为序列型
        ,PSR = {'struct':'psr','r':3,'s':'apple'}  # P:password,R:random,S:salt；P定义则为解密过程
        ,Serout = ''  # 加密/解密后字段命名，留空则设置为原字段名加固定后缀
        ):
        
    '''
    # Example
    exam_ser = pd.Series(map(lambda x:random.choice(['X0', 'X1']), range(100000)), name = 'col')
    
    # 加密后
    %time exam_encrypt = PSR_Crypto(Serin = exam_ser, PSR = {'struct':'psr','r':3,'s':'apple'})
    
    # 解密后
    %time exam_origin = PSR_Crypto(Serin = exam_encrypt, PSR = {'struct':'psr','r':3,'s':'apple','p':['X0', 'X1']})
    
    # 检查加密前后是否一致
    exam_check = pd.concat([exam_ser, exam_encrypt, exam_origin], axis = 1)
    sum(exam_check.col != exam_check.col_encrypt_origin)
    '''

    def hm(s):
        hm = hashlib.md5()
        hm.update(s.encode('utf8'))
        
        return hm.hexdigest()

    if 'p' in PSR:
        # 解密过程
        _pl = PSR['p']
        
        def _psr(_p):
            _s = PSR['s']
            _e = ' + '.join(['str(_' + i + ')' for i in PSR['struct']])
            
            return pd.DataFrame(list(map(lambda _r:[hm(eval(_e, {'_s': _s, '_r': _r, '_p': _p})), _p], range(10 ** PSR['r'])))
                                , columns = [Serin.name, Serout if len(Serout) else Serin.name + '_origin'])
            
        return pd.merge(pd.DataFrame(Serin), pd.concat(map(_psr, _pl)), left_on = Serin.name, right_on = Serin.name, how = 'left')
     
    else:
        # 加密过程        
        def _psr(t):
            _p = t
            _s = PSR['s']
            _r = random.randint(0, 10 ** PSR['r'] - 1)
        
            return hm(eval(' + '.join(['str(_' + i + ')' for i in PSR['struct']])))
    
        print('Encrypt process, struct:'+PSR['struct']+', random digit:'+str(PSR['r'])+', salt:'+str(PSR['s']))
        return pd.Series(Serin.map(_psr), name = Serout if len(Serout) else Serin.name + '_encrypt')
    
