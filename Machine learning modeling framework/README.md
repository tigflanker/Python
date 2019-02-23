# Machine learning modeling framework（风控建模工具集）
 
## 声明：本工具箱程序除模型训练部分调用sklearn、画图部分调用scikit-plot、PSI计算引用@H*Yao，其余均为自写。出于应用有限，暂于23Feb19停更

## 最新Note：
* MLMF使用说明已初步完成
* 2.2.2 Chi2_merge.py模块可以使用，但还需修改
* MLMF为各个模块的汇总，使用时调用即可；Example.x为测试实例，可参照（！！实际项目中的特征名和数据等保密信息已做处理！！）

## 需更新：
* 整个工具箱re-view：包括每个模块其他语言的double-check、每个模块参数名称标准化
* 特征筛选部分没有落地，仍在整理中，包括：集成模型筛选方法、逻辑回归L1正则化
* vintage、迁移率、pmml等
* 3个使用示例最好生成MD，增加可读性
* 2.2.2 Chi2_merge.py：去除卡方分箱的“最优分箱组数”判定，添加卡方值阈值约束
* 特征筛选：逐步回归

## 下图为本工具箱的使用流程
![Process](https://github.com/tigflanker/Python/blob/master/Machine%20learning%20modeling%20framework/process.jpg)

## 更新日志：
# 29Dec2018
* 添加特征筛选部分（2.x）：包含特征稳定性（PSI from *Yao）、特征共线性（方差膨胀因子和相关性矩阵）
* 1. Data_Explore.py：去除相关性矩阵、对分布图拼接按照4维或9维子图展示 
* 4.1 Model performance.py：添加准召图

## 更新日志：
# 13Jan2019
* 2.2.2 Cut_merge.py：① 优化分箱过程，直接从pd.cut解析结果； ② 优化从分箱数据集至WOE数据集的映射逻辑； ③ 将WOE计算并入分箱程序
* 4.1 Model performance.py：① 利用sklearn中混淆矩阵函数加快计算； ② 添加3线图（准确率、召回率、F1-Score）

## 更新日志：
# 25Jan2019
* 2.x Collinearity.py：相关性计算部分的热力图展示，可以控制哪些特征用于绘图；例如可以abs(corr) > 0.8的特征，或者是按相关性排序后取前20个特征
* 3.x PDO_Score_Convert.py：放活odds定义，如不定义，则按照如下规则计算：对train数据集计算ks切点，取切点的百分位点区间内的所有y_label，计算好坏比
* MLMF使用说明：完成初版
