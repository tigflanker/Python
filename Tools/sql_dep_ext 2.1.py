# -*- coding: utf-8 -*-
# Necessary python version 3.x
# Author：guoyang10@jd.com
# Usage:
#   1. Command Line: >>>python 'sql_dep_ext 2.0.py' filename
#   2. Current execution: sqlin = u"D:\Desktop\example sql code.sql"
#   3. Current execution: sqlin = u''' use dev; create table xxx as ... select ... from ...'''

import re
import sys

if len(sys.argv) == 1:
    sqlin = u''' '''  # Current execution 1
    sqlin = u"D:\\Desktop\\xxx\\xxx"  # Current execution 2
else:
    sqlin = sys.argv[1]  # Command Line 

# 第一步：根据每个SQL生成扁平依赖字典
def sql_dep_ext(sqlin, re_comm = '--'):
    sql_dep_ext_dict = dict()
    now_table = ''
    
    re_create = re.compile(r'^.*(create\s+table|insert\s+into)[\w\s]*?(\w+\.\w+)', re.I)
    re_from = re.compile(r'^.*(from|join)\s+(\w+\.\w+)', re.I)
    
    for x in sqlin.split('\n') if sqlin.count('\n') > 5 else open(sqlin, encoding='utf-8'):
        if len(re_comm):
            x = re.sub(re_comm+'.*$','', x, 1)
        rc_create = re.match(re_create, x)
        if rc_create:
            now_table = rc_create.group(2)
            if now_table not in sql_dep_ext_dict:
                sql_dep_ext_dict[rc_create.group(2)] = None
                
        rc_from = re.match(re_from, x)
        if rc_from and len(now_table) and rc_from.group(2).lower() != now_table.lower():
            if isinstance(sql_dep_ext_dict[now_table], set):
                sql_dep_ext_dict[now_table].add(rc_from.group(2))
            else:
                sql_dep_ext_dict[now_table] = set([rc_from.group(2)])
            
    return sql_dep_ext_dict
            
flat_dict = sql_dep_ext(sqlin)

# 第二步：扁平转层级树
for y in flat_dict: 
    if flat_dict[y] is None:
        flat_dict[y] = {'Insert or Dummy.'}
    flat_dict[y] = list(flat_dict[y])

for i, y in enumerate(flat_dict.keys()):
    value_match = '''["']'''+y+'''['"]'''+"(?!\: \[)"

    flat_dict2str = str(flat_dict)
    if re.match('.*?'+value_match, flat_dict2str):
        flat_dict2str = re.sub(value_match, "'"+y+"'"+', '+str(flat_dict[y]), flat_dict2str)
        flat_dict = eval(flat_dict2str)
        flat_dict.pop(y)

flat_dict = str(flat_dict).replace(':',',')
flat_dict = flat_dict.replace('{','[')
tree_stru = eval(flat_dict.replace('}',']'))

# 第三步：绘制依赖树图
blank=chr(32)  # chr(32)==' ' ;chr(183)=='·' ;chr(12288)=='　'
tabs=['']
def draw_tree(lst):
    l=len(lst)
    if l==0:print('─'*3)
    else:
        for i,j in enumerate(lst):
            if i!=0:print(tabs[0],end='')
            if l==1:s='─'*3
            elif i==0:s='┬'+'─'*2
            elif i+1==l:s='└'+'─'*2
            else:s='├'+'─'*2
            print(s,end='')
            if isinstance(j,list) or isinstance(j,tuple):
                if i+1==l:tabs[0]+=blank*3
                else:tabs[0]+='│'+blank*2
                draw_tree(j)
            else:print(j)
    tabs[0]=tabs[0][:-3]
    
draw_tree(tree_stru)

# draw_tree info: http://paste.ubuntu.org.cn/105151

'''
Example:
output:
┬──dev_tmp.final_table
├──┬──dwd.table_a
│  ├──app.table_b
│  ├──dev_tmp.table_c
│  └─────dwd.table_d
├──dev_tmp.table_e
└──┬──dwd.table_f
   ├──dev_tmp.table_j
   └──┬──dev_tmp.table_h
      ├──┬──dwd.table_i
      │  └──dwd.table_f
      ├──dev_tmp.tableg
   ... ...
'''
