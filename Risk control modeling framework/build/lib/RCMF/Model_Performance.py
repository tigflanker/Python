# -*- coding: utf-8 -*-
# Author: tigflanker
# Date: 19 Dec 2018
# Version: 1.0
# 说明：该宏为机器学习线，模型表现；输入对象为分隔好的训练和验证集，外加预测模型：
# X_train, X_test, y_train, y_test和clf 
# scikitplot为@Huyao推荐，方便绘制sklearn模型产出 

# 导包/宏参设定
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
import scikitplot as skplt #  pip install scikit-plot
from sklearn.metrics import auc,roc_curve
from sklearn.metrics import confusion_matrix

# 4. 机器学习模型表现评定
def Model_Performance(clf  # 输入模型
                      ,X_train  # 训练集特征
                      ,y_train  # 训练集标签
                      ,X_test = []  # 测试集特征
                      ,y_test = []  # 测试集标签
                      ,out_path = ''  # 特征重要性输出，xlsx格式 
                      ,calc_list = ['pr','roc','ks']  # 需计算的项目列表，分别为：准召列表、ROC/AUC、KS、特征重要性、模型学习率['cut','pr','roc','ks','fi','lc']
                      ,calibration = {}  # calibration图，如{'Random Forest':clf_rf,'Logistic Regression':clf_lr}
                      ):
                      
    '''
    # Example:
    Model_Performance(clf_xgb, X_train, y_train, X_test, y_test, out_path = 'D:/xxx.xlsx', calibration = {'Random Forest':clf_rf,'XGboost':clf_xgb})
    Model_Performance(clf_xgb, X_train, y_train, calc_list = ['pr', 'roc', 'ks', 'cut'])
    '''
    
    # 4.1 切点排序及准召、F1-Score
    def PRTF(X = 'X_train', y = 'y_train', clf = 'clf'):
        sum_y = sum(y)
        sum_op_y = sum(y != 1)

        def _PRTF(i):
            conf_mat = confusion_matrix(y, clf.predict_proba(X)[:,1] > i)

            precision = round(conf_mat[1][1] / (conf_mat[0][1] + conf_mat[1][1]), 4) if conf_mat[0][1] + conf_mat[1][1] else 0
            recall = round(conf_mat[1][1] / sum_y, 4)  # recall = TPR 
            FPR = round(conf_mat[0][1] / sum_op_y, 4)

            return [i, precision, recall, FPR, round(2 * precision * recall / (precision + recall),4)]

        pred_max = max(clf.predict_proba(X)[:,1])
        pred_min = min(clf.predict_proba(X)[:,1])
        cut_range = np.linspace(int(pred_min * 100), int(pred_max * 100), int(pred_max * 100) - int(pred_min * 100) + 1) / 100
        PRTF_table = pd.DataFrame(list(map(_PRTF,cut_range)), columns=['Cut point','Precision','Recall(TPR)','FPR','F1-score'])

        fig, ax = plt.subplots()
        ax.plot(PRTF_table['Cut point'], PRTF_table['Precision'],'--', label='Precision Curve')
        ax.plot(PRTF_table['Cut point'], PRTF_table['Recall(TPR)'],'--', label='Recall Curve')
        ax.plot(PRTF_table['Cut point'], PRTF_table['F1-score'],'k', label='F1-score Curve')

        max_f1 = max(PRTF_table['F1-score'])
        best_cut = np.array(PRTF_table.loc[PRTF_table['F1-score'] == max_f1, 'Cut point']).mean()

        ax.plot([best_cut, best_cut], [max_f1 - 0.15, max_f1 + 0.05], color='red')
        ax.set_title('Precision, Recall, and F1-Score curves.')
        plt.text(best_cut, max_f1 - 0.2, 'Max F1:'+str(round(max_f1, 2))+' Cut point:'+str(best_cut))
        ax.legend()
        plt.show()

        return PRTF_table
    
    if 'cut' in calc_list:
        prtf = PRTF(X = X_train, y = y_train, clf = clf)
        print('>>>>>Best cut point on MAX(F1-score):\n', prtf.loc[prtf['F1-score'] == max(prtf['F1-score'])])
        
        if len(out_path):
            prtf.to_csv(out_path+'_cut.csv',index=False)
    
    # 模型Score 
    print('>>>>>Train set score:{:.2%}'.format(clf.score(X_train, y_train)))
    if len(X_test):
        print('>>>>>Test set score:{:.2%}'.format(clf.score(X_test, y_test)))
    
    # 4.2 AUC、ROC曲线、KS图
    # scikitplot系列图   
    # 该函数图完整引用ref7内容并轻微整理
    def roc_fig(**kwargs):    
        fpr, tpr, thresholds = roc_curve(y_train, clf.predict_proba(X_train)[:,1], pos_label = 1)
        auc_score = auc(fpr, tpr)
        
        if len(X_test):
            fpr1, tpr1, thresholds1 = roc_curve(y_test, clf.predict_proba(X_test)[:,1], pos_label = 1)
            auc_score1 = auc(fpr1, tpr1)
        
        fig, ax = plt.subplots()
        ax.plot(fpr, tpr, label='Train set AUC = %.5f'% auc_score)
        if len(X_test):
            ax.plot(fpr1, tpr1, label='Test set AUC = %.5f'% auc_score1)
        ax.set_title('Receiver Operating Characteristic')
        ax.plot([0, 1], [0, 1], '--', color = (0.6, 0.6, 0.6))  # 中轴线
        ax.legend()
        plt.show()
    
    if ('roc' in calc_list) or ('auc' in calc_list):
        roc_fig()
    
    # PR图（准召图）
    if 'pr' in calc_list:
        skplt.metrics.plot_precision_recall(y_test if len(y_test) else y_train, 
                                            clf.predict_proba(X_test if len(X_test) else X_train),
                                            classes_to_plot = [1], plot_micro = False)
        plt.show()
        if len(y_test) == 0:
            print('>>>>> Notice:This PR curve from train set.')
        
    # 学习速率图
    if 'lc' in calc_list:
        skplt.estimators.plot_learning_curve(clf, X_train, y_train)
        plt.show()
    
    # ks图
    if 'ks' in calc_list:
        skplt.metrics.plot_ks_statistic(y_test if len(y_test) else y_train, clf.predict_proba(X_test if len(X_test) else X_train))
        plt.show()
        if len(y_test) == 0:
            print('>>>>> Notice:This KS curve from train set.')
        
    # 特征重要性图 
#    hasattr(clf, 'feature_importances_')
    if 'fi' in calc_list:
        try:
            skplt.estimators.plot_feature_importances(clf, x_tick_rotation = 90, feature_names=X_train.columns)
            plt.show()
        
            if len(out_path):
                result_save = pd.ExcelWriter(out_path+'.xlsx')
                pd.DataFrame(list(zip(X_train.columns, clf.feature_importances_)), columns=['Var','imp']).to_excel(result_save,'Feature importance',encoding='utf-8',index=False)
                if 'cut' in calc_list:
                    prtf.to_excel(result_save,'Cut point',encoding='utf-8',index=False)
                result_save.save()
                
                pd.DataFrame(list(zip(X_train.columns, clf.feature_importances_)), columns=['Var','imp']).to_csv(out_path+'_importance.csv',index=False)
        except:
            pass
    
    # Calibration Curve
    if len(calibration):
        skplt.metrics.plot_calibration_curve(y_test, 
                                             list(map(lambda x:x.predict_proba(X_test),calibration.values())),
                                             list(calibration))
        plt.show()
        
    # ref7：https://www.cnblogs.com/gasongjian/p/8159501.html
        