import pandas as pd
import numpy as np
import os, time, sys
from glob import glob
from tqdm import tqdm
import matplotlib.pyplot as plt

sys.path.append('../cell_segmentation')
from preprocessing import mask
from input.get_from_store import *

datatable_path = './input/Ki67_datatable.xls'
datatable_df = pd.read_excel(datatable_path)

years_df = datatable_df.filepath.apply(lambda x: time.gmtime(os.path.getmtime(x)).tm_year)
# years_df.hist(bins=years_df.max()-years_df.min(), xrot=90, xlabelsize=10, ax=ax)

# ihc2kp_path = 'C:\\Users\\M306410\\Desktop\\repos\\UGATIT-pytorch\\results\\ihc_to_mask_variation\\logs\\quantification_60000.log'
# ihc2kp_df = pd.read_csv(ihc2kp_path, delimiter=',')
# merged_ihc2kp_df = pd.merge(datatable_df, ihc2kp_df, left_on='filepath', right_on='filepath', suffixes=['_datatable', '_ihc2kp'])
# years_df = merged_ihc2kp_df.filepath.apply(lambda x: time.gmtime(os.path.getmtime(x)).tm_year)

stats_list = []
for n in tqdm(range(len(datatable_df))):
    row = datatable_df.iloc[n]
    slide_file, annot_file = row.filepath, row.annotation_filepath
    slide_year = time.gmtime(os.path.getmtime(slide_file)).tm_year
    
    msk = mask.xml_to_mask(annot_file, 1, None, get_counts_from_store(n))
    if msk is None: continue
    annot_size = (np.array(msk) == 255).sum()

    stats_list += [(slide_year, annot_size)]

stats_df = pd.DataFrame.from_dict({'year': [x[0] for x in stats_list], 'annotation_size': [x[1] for x in stats_list]})
# stats_df.hist(bins=stats_df.year.max()-stats_df.year.min(), xrot=90, xlabelsize=10, ax=ax)

years = [len(stats_df.year[stats_df.year==year]) for year in range(2012, 2024)]
# years = [len(stats_df.year[(stats_df.year==year) & (stats_df.annotation_size > 10000000)]) for year in range(2012, 2024)]

years = [35, 40, 333, 2, 434, 695, 195, 129, 95, 85, 86, 14] # annot_size < 10m
years = [143, 72, 253, 11, 553, 918, 655, 621, 476, 486, 416, 89] # annot_size > 10m
fig, ax = plt.subplots()
plt.bar(range(2012, 2024), years)
ax.set_xticks(range(2011, 2025))
ax.set_title('Number of Samples by Year')
ax.set_ylabel('# Samples')
ax.set_xlabel('Year')
plt.show()

print()