import pandas as pd

left = pd.DataFrame({'A': ['A0', 'A1', 'A2', 'A3'], 'B': ['B0', 'B1', 'B2', 'B3'], 'key': [1,2,2,3]}, index=[1,2,3,4])
right = pd.DataFrame({'A': ['C0', 'C1'], 'D': ['D0', 'D1'], 'key':[2,3]}, index=[2,4])
series = pd.Series(['S1','S2','S3','S4'])

'''
left:
    A   B  key
1  A0  B0    1
2  A1  B1    2
3  A2  B2    2
4  A3  B3    3    

right:
    A   D  key
2  C0  D0    2
4  C1  D1    3
'''

pd.merge(left, right, how='inner', on=None, left_on=None, right_on=None, 
        left_index=False, right_index=False, sort=False, 
        suffixes=(u'_x', u'_y'), copy=True, indicator=False, 
        validate=None)

pd.concat(objs, axis=0, join='outer', join_axes=None, ignore_index=False, 
         keys=None, levels=None, names=None, verify_integrity=False, 
         sort=None, copy=True)

# 按照某个字段做正常join 
pd.merge(left, right, on='key')  # inner join 
pd.merge(left, right, how='left', on='key')  # left outer join 
pd.merge(left, right, how='outer', on='key')  # full join 

# 按照index去接
pd.merge(left, right, left_index=True, right_index=True)
# 不完全等价于
pd.concat([left, right], axis=1, join='inner')

# 混合接：用左边的key字段去接右边的index
pd.merge(left, right, how='left', left_on='key', right_index=True)

# union：仅union推荐使用concat，其他情况推荐使用merge
pd.concat([left, right], axis=0, ignore_index=True)

# 复杂情况：df和series混合接
# index接：
pd.concat([right, series], axis=1)
# df的字段接series的index
pd.merge(right,pd.DataFrame(series, columns=['new']),left_on='key', right_index=True)

# 总结：merge方法比较通用，缺点之一是：没看到可以做union；
# 另一个缺点是：没法直接pd和series混接，series得中转。