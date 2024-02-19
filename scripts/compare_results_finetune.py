import seaborn as sn
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from sklearn.metrics import confusion_matrix

datatable_path = '../../cell_segmentation/input/Ki67_datatable.xlsx'
ihcseg_path = r'C:\Users\M306410\Desktop\repos\DeepLIIF\results\logs\quantification_deepliif.log'
basemodel_path = 'results/ihc2maskvar_before2018/logs/quantification_40000.log'
ihc2maskvar_N5000_path = 'results/ihc2maskvar_before2018_N5000/logs/quantification_40000.log'
ihc2maskvar_N10000_path = 'results/ihc2maskvar_before2018_N10000/logs/quantification_40000.log'
ihc2maskvar_N15000_path = 'results/ihc2maskvar_before2018_N15000/logs/quantification_40000.log'
ihc2maskvar_N20000_path = 'results/ihc2maskvar_before2018_N20000/logs/quantification_40000.log'
ihc2maskvar_N25000_path = 'results/ihc2maskvar_before2018_N25000/logs/quantification_40000.log'
ihc2maskvar_N30000_path = 'results/ihc2maskvar_before2018_N30000/logs/quantification_40000.log'
ihc2maskvar_N35000_path = 'results/ihc2maskvar_before2018_N35000/logs/quantification_40000.log'
ihc2maskvar_N40000_path = 'results/ihc2maskvar_before2018_N40000/logs/quantification_40000.log'

datatable_df = pd.read_excel(datatable_path)
datatable_df = datatable_df[datatable_df.year >= 2018]
ihcseg_df = pd.read_csv(ihcseg_path, delimiter=',')
base_df = pd.read_csv(basemodel_path, delimiter=',')
ihc2maskvar_N5000_df = pd.read_csv(ihc2maskvar_N5000_path, delimiter=',')
ihc2maskvar_N10000_df = pd.read_csv(ihc2maskvar_N10000_path, delimiter=',')
ihc2maskvar_N15000_df = pd.read_csv(ihc2maskvar_N15000_path, delimiter=',')
ihc2maskvar_N20000_df = pd.read_csv(ihc2maskvar_N20000_path, delimiter=',')
ihc2maskvar_N25000_df = pd.read_csv(ihc2maskvar_N25000_path, delimiter=',')
ihc2maskvar_N30000_df = pd.read_csv(ihc2maskvar_N30000_path, delimiter=',')
ihc2maskvar_N35000_df = pd.read_csv(ihc2maskvar_N35000_path, delimiter=',')
ihc2maskvar_N40000_df = pd.read_csv(ihc2maskvar_N40000_path, delimiter=',')

merged_ihcseg_df = pd.merge(datatable_df, ihcseg_df, left_on='filepath', right_on='filepath', suffixes=['_datatable', '_model'])
merged_base_df = pd.merge(datatable_df, base_df, left_on='filepath', right_on='filepath', suffixes=['_datatable', '_model'])
merged_ihc2maskvar_N5000_df = pd.merge(datatable_df, ihc2maskvar_N5000_df, left_on='filepath', right_on='filepath', suffixes=['_datatable', '_model'])
merged_ihc2maskvar_N10000_df = pd.merge(datatable_df, ihc2maskvar_N10000_df, left_on='filepath', right_on='filepath', suffixes=['_datatable', '_model'])
merged_ihc2maskvar_N15000_df = pd.merge(datatable_df, ihc2maskvar_N15000_df, left_on='filepath', right_on='filepath', suffixes=['_datatable', '_model'])
merged_ihc2maskvar_N20000_df = pd.merge(datatable_df, ihc2maskvar_N20000_df, left_on='filepath', right_on='filepath', suffixes=['_datatable', '_model'])
merged_ihc2maskvar_N25000_df = pd.merge(datatable_df, ihc2maskvar_N25000_df, left_on='filepath', right_on='filepath', suffixes=['_datatable', '_model'])
merged_ihc2maskvar_N30000_df = pd.merge(datatable_df, ihc2maskvar_N30000_df, left_on='filepath', right_on='filepath', suffixes=['_datatable', '_model'])
merged_ihc2maskvar_N35000_df = pd.merge(datatable_df, ihc2maskvar_N35000_df, left_on='filepath', right_on='filepath', suffixes=['_datatable', '_model'])
merged_ihc2maskvar_N40000_df = pd.merge(datatable_df, ihc2maskvar_N40000_df, left_on='filepath', right_on='filepath', suffixes=['_datatable', '_model'])

# fig = plt.figure()
# fig.suptitle('IHC Slide Classification\n(Percent Positive Nuclei > 20%)', fontsize=15)
# ax1 = fig.add_subplot(211)
# ax2 = fig.add_subplot(212)
# # ax3 = fig.add_subplot(313)

# sn.heatmap(
#     confusion_matrix(merged_ihcseg_df.percent_positive_nuclei_datatable>20, merged_ihcseg_df.percent_positive_nuclei_model>20), 
#     annot=True, fmt="d", ax=ax1)
# ax1.set_ylabel('Diagnostic Data', fontsize = 10)
# ax1.set_xlabel('ihcseg', fontsize = 10)

# sn.heatmap(
#     confusion_matrix(merged_base_df.percent_positive_nuclei_datatable>20, merged_base_df.percent_positive_nuclei_model>20), 
#     annot=True, fmt="d", ax=ax2)
# ax2.set_ylabel('Diagnostic Data', fontsize = 10)
# ax2.set_xlabel('ihc2maskvar: before 2018', fontsize = 10)

# sn.heatmap(
#     confusion_matrix(merged_finetuned2019_df.percent_positive_nuclei_datatable>20, merged_finetuned2019_df.percent_positive_nuclei_model>20), 
#     annot=True, fmt="d", ax=ax3)
# ax3.set_ylabel('Diagnostic Data', fontsize = 10)
# ax3.set_xlabel('ihc2maskvar: finetuned2019', fontsize = 10)

# fig.tight_layout()
# plt.show()

error_metrics = {
        'Name'     : [],
        'FPR'      : [],
        'FNR'      : [],
        'PPV'      : [],
        'NPV'      : [],
        'Precision': [],
        'Recall'   : [],
        'Accuracy' : [],
        'F1'       : []
    }
for name, cm in [('ihcseg', confusion_matrix(merged_ihcseg_df.percent_positive_nuclei_datatable>20, merged_ihcseg_df.percent_positive_nuclei_model>20)),
                 ('ihc2maskvar_base', confusion_matrix(merged_base_df.percent_positive_nuclei_datatable>20, merged_base_df.percent_positive_nuclei_model>20)),
                 ('ihc2maskvar_N5000', confusion_matrix(merged_ihc2maskvar_N5000_df.percent_positive_nuclei_datatable>20, merged_ihc2maskvar_N5000_df.percent_positive_nuclei_model>20)),
                 ('ihc2maskvar_N10000', confusion_matrix(merged_ihc2maskvar_N10000_df.percent_positive_nuclei_datatable>20, merged_ihc2maskvar_N10000_df.percent_positive_nuclei_model>20)),
                 ('ihc2maskvar_N15000', confusion_matrix(merged_ihc2maskvar_N15000_df.percent_positive_nuclei_datatable>20, merged_ihc2maskvar_N15000_df.percent_positive_nuclei_model>20)),
                 ('ihc2maskvar_N20000', confusion_matrix(merged_ihc2maskvar_N20000_df.percent_positive_nuclei_datatable>20, merged_ihc2maskvar_N20000_df.percent_positive_nuclei_model>20)),
                 ('ihc2maskvar_N25000', confusion_matrix(merged_ihc2maskvar_N25000_df.percent_positive_nuclei_datatable>20, merged_ihc2maskvar_N25000_df.percent_positive_nuclei_model>20)),
                 ('ihc2maskvar_N30000', confusion_matrix(merged_ihc2maskvar_N30000_df.percent_positive_nuclei_datatable>20, merged_ihc2maskvar_N30000_df.percent_positive_nuclei_model>20)),
                 ('ihc2maskvar_N35000', confusion_matrix(merged_ihc2maskvar_N35000_df.percent_positive_nuclei_datatable>20, merged_ihc2maskvar_N35000_df.percent_positive_nuclei_model>20)),
                 ('ihc2maskvar_N40000', confusion_matrix(merged_ihc2maskvar_N40000_df.percent_positive_nuclei_datatable>20, merged_ihc2maskvar_N40000_df.percent_positive_nuclei_model>20)),
                ]:
    tn, fp, fn, tp = cm.ravel()
    p = tp + fn
    n = tn + fp

    error_metrics['Name']      += [name]
    error_metrics['FPR']       += [fp/n]
    error_metrics['FNR']       += [fn/p]
    error_metrics['PPV']       += [tp/(tp+fp)]
    error_metrics['NPV']       += [tn/(tn+fn)]
    error_metrics['Precision'] += [tp/(tp+fp)]
    error_metrics['Recall']    += [tp/p]
    error_metrics['Accuracy']  += [(tp+tn)/(p+n)]
    error_metrics['F1']        += [(2*tp)/(2*tp + fp + fn)]
    
pd.DataFrame.from_dict(error_metrics).to_csv('./results/logs/error_metrics_N.csv')