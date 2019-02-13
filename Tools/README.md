# Python

Sql_dep_ext：通过输入的SQL文件或者语句段，绘制血缘图；输入SQL程序可以最终指向多个目标表，需要遵循SQL的标准语句格式。
```Sql_dep_ext示例
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
```

pd_merge：为python双表对接方式整理。
