import seaborn as sn
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from sklearn.metrics import confusion_matrix

datatable_path = 'input/Ki67_datatable.xls'
ihcseg_path = 'results/logs/quantification_ihcseg.log'
ihc2kp_path = 'C:\\Users\\M306410\\Desktop\\repos\\UGATIT-pytorch\\results\\ihc_to_mask_variation_old\\logs\\quantification_60000.log'

datatable_df = pd.read_excel(datatable_path)
ihcseg_df = pd.read_csv(ihcseg_path, delimiter=',')
ihc2kp_df = pd.read_csv(ihc2kp_path, delimiter=',')

merged_ihcseg_df = pd.merge(datatable_df, ihcseg_df, left_on='filepath', right_on='filepath', suffixes=['_datatable', '_ihcseg'])
merged_ihc2kp_df = pd.merge(datatable_df, ihc2kp_df, left_on='filepath', right_on='filepath', suffixes=['_datatable', '_ihc2kp'])
merged_ihc2kp_df = pd.merge(merged_ihcseg_df, ihc2kp_df, left_on='filepath', right_on='filepath', suffixes=['_datatable', '_ihc2kp'])

merged_results_df = pd.merge(ihcseg_df, ihc2kp_df, left_on='filepath', right_on='filepath', suffixes=['_ihcseg', '_ihc2kp'])

fig = plt.figure()
fig.suptitle('IHC Slide Classification\n(Percent Positive Nuclei > 20%)', fontsize=15)
ax1 = fig.add_subplot(311)
ax2 = fig.add_subplot(312)
ax3 = fig.add_subplot(313)

sn.heatmap(
    confusion_matrix(merged_ihcseg_df.percent_positive_nuclei_datatable>20, merged_ihcseg_df.percent_positive_nuclei_ihcseg>20), 
    annot=True, fmt="d", ax=ax1)
ax1.set_ylabel('Diagnostic Data', fontsize = 10)
ax1.set_xlabel('Supervised Model', fontsize = 10)

sn.heatmap(
    confusion_matrix(merged_ihc2kp_df.percent_positive_nuclei_datatable>20, merged_ihc2kp_df.percent_positive_nuclei>20), 
    annot=True, fmt="d", ax=ax2)
ax2.set_ylabel('Diagnostic Data', fontsize = 10)
ax2.set_xlabel('Proposed Model', fontsize = 10)

sn.heatmap(
    confusion_matrix(merged_results_df.percent_positive_nuclei_ihcseg>20, merged_results_df.percent_positive_nuclei_ihc2kp>20), 
    annot=True, fmt="d", ax=ax3)
ax3.set_ylabel('Supervised Model', fontsize = 10)
ax3.set_xlabel('Proposed Model', fontsize = 10)

fig.tight_layout()
plt.show()

error_cases = merged_ihc2kp_df[
    ((merged_ihc2kp_df.percent_positive_nuclei>20) != (merged_ihc2kp_df.percent_positive_nuclei_datatable>20)) 
    | ((merged_ihc2kp_df.percent_positive_nuclei_ihcseg>20) != (merged_ihc2kp_df.percent_positive_nuclei_datatable>20))
    ][[ 'filepath', 
        'row_datatable', 
        'percent_positive_nuclei_datatable', 
        'total_nuclei_datatable', 
        'percent_positive_nuclei', 
        'total_nuclei', 
        'percent_positive_nuclei_ihcseg', 
        'total_nuclei_ihcseg']]

error_cases.to_csv('./results/logs/error_cases.csv')

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
for name, cm in [('ihcseg', confusion_matrix(merged_ihcseg_df.percent_positive_nuclei_datatable>20, merged_ihcseg_df.percent_positive_nuclei_ihcseg>20)),
                 ('ihc2kp', confusion_matrix(merged_ihc2kp_df.percent_positive_nuclei_datatable>20, merged_ihc2kp_df.percent_positive_nuclei>20))]:
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
    
pd.DataFrame.from_dict(error_metrics).to_csv('./results/logs/error_metrics.csv')

print()