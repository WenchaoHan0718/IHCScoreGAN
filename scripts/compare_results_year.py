import seaborn as sn
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

import os, time
from tqdm import tqdm

from sklearn.metrics import confusion_matrix

datatable_path = '../../cell_segmentation/input/Ki67_datatable.xlsx'
ihc2maskvar_2014_path = 'results/ihc2maskvar_2014/logs/quantification_40000.log'
ihc2maskvar_2016_path = 'results/ihc2maskvar_2016/logs/quantification_40000.log'
ihc2maskvar_2017_path = 'results/ihc2maskvar_2017/logs/quantification_40000.log'
ihc2maskvar_2018_path = 'results/ihc2maskvar_2018/logs/quantification_40000.log'
ihc2maskvar_2019_path = 'results/ihc2maskvar_2019/logs/quantification_40000.log'

datatable_df = pd.read_excel(datatable_path)
# df = [time.struct_time(os.path.getctime(filepath)) for filepath in tqdm(datatable_df.filepath[0])]
# input_dir   = r'C:\Users\M306410\Desktop\cell_segmentation\results\tiles_png'
# counts_per_tile = {}
# for year in [datatable_df.iloc[int(file.split('-')[0])].year for root, _, file_list in os.walk(input_dir) for file in file_list]:
#     if year not in counts_per_tile: counts_per_tile[year] = 0
#     counts_per_tile[year] += 1
ihc2maskvar_2014_df    = pd.read_csv(ihc2maskvar_2014_path, delimiter=',')
ihc2maskvar_2016_df    = pd.read_csv(ihc2maskvar_2016_path, delimiter=',')
ihc2maskvar_2017_df    = pd.read_csv(ihc2maskvar_2017_path, delimiter=',')
ihc2maskvar_2018_df    = pd.read_csv(ihc2maskvar_2018_path, delimiter=',')
ihc2maskvar_2019_df    = pd.read_csv(ihc2maskvar_2019_path, delimiter=',')

merged_ihc2maskvar_2014_df = pd.merge(datatable_df, ihc2maskvar_2014_df, left_on='filepath', right_on='filepath', suffixes=['_datatable', '_model'])
merged_ihc2maskvar_2016_df = pd.merge(datatable_df, ihc2maskvar_2016_df, left_on='filepath', right_on='filepath', suffixes=['_datatable', '_model'])
merged_ihc2maskvar_2017_df = pd.merge(datatable_df, ihc2maskvar_2017_df, left_on='filepath', right_on='filepath', suffixes=['_datatable', '_model'])
merged_ihc2maskvar_2018_df = pd.merge(datatable_df, ihc2maskvar_2018_df, left_on='filepath', right_on='filepath', suffixes=['_datatable', '_model'])
merged_ihc2maskvar_2019_df = pd.merge(datatable_df, ihc2maskvar_2019_df, left_on='filepath', right_on='filepath', suffixes=['_datatable', '_model'])

# fig = plt.figure()
# fig.suptitle('IHC Slide Classification\n(Percent Positive Nuclei > 20%)', fontsize=15)
# ax1 = fig.add_subplot(611)
# ax2 = fig.add_subplot(612)
# ax3 = fig.add_subplot(613)
# ax4 = fig.add_subplot(614)
# ax5 = fig.add_subplot(615)
# ax6 = fig.add_subplot(616)

# sn.heatmap(
#     confusion_matrix(merged_ihc2maskvar_df.percent_positive_nuclei_datatable>20, merged_ihc2maskvar_df.percent_positive_nuclei_model>20), 
#     annot=True, fmt="d", ax=ax1)
# ax1.set_ylabel('Diagnostic Data', fontsize = 10)
# ax1.set_xlabel('ours', fontsize = 10)

# sn.heatmap(
#     confusion_matrix(merged_ihcseg_df.percent_positive_nuclei_datatable>20, merged_ihcseg_df.percent_positive_nuclei_model>20), 
#     annot=True, fmt="d", ax=ax2)
# ax2.set_ylabel('Diagnostic Data', fontsize = 10)
# ax2.set_xlabel('supervised', fontsize = 10)

# sn.heatmap(
#     confusion_matrix(merged_results_df.percent_positive_nuclei_model1>20, merged_results_df.percent_positive_nuclei_model2>20), 
#     annot=True, fmt="d", ax=ax3)
# ax3.set_ylabel('ours', fontsize = 10)
# ax3.set_xlabel('supervised', fontsize = 10)

# sn.heatmap(
#     confusion_matrix(merged_ihc2maskvarN10000_df.percent_positive_nuclei_datatable>20, merged_ihc2maskvarN10000_df.percent_positive_nuclei_model>20), 
#     annot=True, fmt="d", ax=ax4)
# ax4.set_ylabel('Diagnostic Data', fontsize = 10)
# ax4.set_xlabel('ours', fontsize = 10)

# sn.heatmap(
#     confusion_matrix(merged_ihc2maskvarN20000_df.percent_positive_nuclei_datatable>20, merged_ihc2maskvarN20000_df.percent_positive_nuclei_model>20), 
#     annot=True, fmt="d", ax=ax5)
# ax5.set_ylabel('Diagnostic Data', fontsize = 10)
# ax5.set_xlabel('ours', fontsize = 10)

# sn.heatmap(
#     confusion_matrix(merged_ihc2maskvarN30000_df.percent_positive_nuclei_datatable>20, merged_ihc2maskvarN30000_df.percent_positive_nuclei_model>20), 
#     annot=True, fmt="d", ax=ax6)
# ax6.set_ylabel('Diagnostic Data', fontsize = 10)
# ax6.set_xlabel('ours', fontsize = 10)

# fig.tight_layout()
# plt.show()

# print()

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
for name, cm in [('ihc2maskvar_2014', confusion_matrix(merged_ihc2maskvar_2014_df.percent_positive_nuclei_datatable>20, merged_ihc2maskvar_2014_df.percent_positive_nuclei_model>20)),
                 ('ihc2maskvar_2016', confusion_matrix(merged_ihc2maskvar_2016_df.percent_positive_nuclei_datatable>20, merged_ihc2maskvar_2016_df.percent_positive_nuclei_model>20)),
                 ('ihc2maskvar_2017', confusion_matrix(merged_ihc2maskvar_2017_df.percent_positive_nuclei_datatable>20, merged_ihc2maskvar_2017_df.percent_positive_nuclei_model>20)),
                 ('ihc2maskvar_2018', confusion_matrix(merged_ihc2maskvar_2018_df.percent_positive_nuclei_datatable>20, merged_ihc2maskvar_2018_df.percent_positive_nuclei_model>20)),
                 ('ihc2maskvar_2019', confusion_matrix(merged_ihc2maskvar_2019_df.percent_positive_nuclei_datatable>20, merged_ihc2maskvar_2019_df.percent_positive_nuclei_model>20)),]:
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
    
pd.DataFrame.from_dict(error_metrics).to_csv('./results/logs/error_metrics_year.csv')

print(error_metrics)