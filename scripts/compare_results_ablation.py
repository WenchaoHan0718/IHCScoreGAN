import seaborn as sn
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from sklearn.metrics import confusion_matrix

datatable_path = '../../cell_segmentation/input/Ki67_datatable.xls'
ihc2maskvar_alldata_path = 'results/ihc2maskvar_before2018/logs/quantification_40000.log'
ihc2maskvar_nodetach_path = 'results/ihc2maskvar_before2018_nodetach/logs/quantification_40000.log'
ihc2mask_alldata_path = 'results/ihc2mask_before2018/logs/quantification_40000.log'
ihc2maskvar_shuffle_path = 'results/ihc2maskvar_shuffle2/logs/quantification_40000.log'

datatable_df = pd.read_excel(datatable_path)
ihc2maskvar_alldata_df = pd.read_csv(ihc2maskvar_alldata_path, delimiter=',')
ihc2maskvar_nodetach_df = pd.read_csv(ihc2maskvar_nodetach_path, delimiter=',')
ihc2mask_alldata_df = pd.read_csv(ihc2mask_alldata_path, delimiter=',')
ihc2maskvar_shuffle_df = pd.read_csv(ihc2maskvar_shuffle_path, delimiter=',')

ihc2maskvar_alldata_df = pd.merge(datatable_df, ihc2maskvar_alldata_df, left_on='filepath', right_on='filepath', suffixes=['_datatable', '_model'])
ihc2maskvar_nodetach_df = pd.merge(datatable_df, ihc2maskvar_nodetach_df, left_on='filepath', right_on='filepath', suffixes=['_datatable', '_model'])
ihc2mask_alldata_df = pd.merge(datatable_df, ihc2mask_alldata_df, left_on='filepath', right_on='filepath', suffixes=['_datatable', '_model'])
ihc2maskvar_shuffle_df = pd.merge(datatable_df, ihc2maskvar_shuffle_df, left_on='filepath', right_on='filepath', suffixes=['_datatable', '_model'])

fig = plt.figure()
fig.suptitle('IHC Slide Classification\n(Percent Positive Nuclei > 20%)', fontsize=15)
ax1 = fig.add_subplot(311)
ax2 = fig.add_subplot(312)
ax3 = fig.add_subplot(313)

sn.heatmap(
    confusion_matrix(ihc2maskvar_alldata_df.percent_positive_nuclei_datatable>20, ihc2maskvar_alldata_df.percent_positive_nuclei_model>20), 
    annot=True, fmt="d", ax=ax1)
ax1.set_ylabel('Diagnostic Data', fontsize = 10)
ax1.set_xlabel('IHC to Variational Mask', fontsize = 10)

sn.heatmap(
    confusion_matrix(ihc2maskvar_nodetach_df.percent_positive_nuclei_datatable>20, ihc2maskvar_nodetach_df.percent_positive_nuclei_model>20), 
    annot=True, fmt="d", ax=ax2)
ax2.set_ylabel('Diagnostic Data', fontsize = 10)
ax2.set_xlabel('IHC to Variational Mask, No Detach', fontsize = 10)

sn.heatmap(
    confusion_matrix(ihc2mask_alldata_df.percent_positive_nuclei_datatable>20, ihc2mask_alldata_df.percent_positive_nuclei_model>20), 
    annot=True, fmt="d", ax=ax3)
ax3.set_ylabel('Diagnostic Data', fontsize = 10)
ax3.set_xlabel('IHC to Binary Mask', fontsize = 10)

fig.tight_layout()
# plt.show()

def get_error_metrics(data):
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
    for name, cm in data:
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
    return error_metrics

data = [('ihc2maskvar', confusion_matrix(ihc2maskvar_alldata_df.percent_positive_nuclei_datatable>20, ihc2maskvar_alldata_df.percent_positive_nuclei_model>20)),
                    ('ihc2maskvar_nodetach', confusion_matrix(ihc2maskvar_nodetach_df.percent_positive_nuclei_datatable>20, ihc2maskvar_nodetach_df.percent_positive_nuclei_model>20)),
                    ('ihc2mask', confusion_matrix(ihc2mask_alldata_df.percent_positive_nuclei_datatable>20, ihc2mask_alldata_df.percent_positive_nuclei_model>20))]
error_metrics = get_error_metrics(data)
    
pd.DataFrame.from_dict(error_metrics).to_csv('./results/logs/error_metrics_ablation.csv')

print()