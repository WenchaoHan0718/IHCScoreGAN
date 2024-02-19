import json
import os
import pandas as pd
from pathlib import Path
from tqdm import tqdm

n_types = 5
type2label = {0: 'unlabeled', 1: '0+', 2: '1+', 3: '2+', 4: '3+', None: 'unlabeled'}
datatable_path = 'input/Ki67_datatable.xls'
df = pd.read_excel(datatable_path)

def quantify(json_dir, log_file):
    with open(log_file, 'w') as fp: 
        fp.write('row,filepath,record_type,percent_positive_nuclei,3+_cells_%,2+_cells_%,1+_cells_%,0+_cells_%,3+_nuclei,2+_nuclei,1+_nuclei,0+_nuclei,total_nuclei,class\n')

    counts = {}
    for json_path in tqdm([os.path.join(json_dir, file) for file in os.listdir(json_dir)]):
        sample_name = Path(json_path).stem.split('-')[0]
        if sample_name not in counts.keys(): 
            counts[sample_name] = {}
            for i in range(n_types): counts[sample_name][type2label[i]] = 0

        with open(json_path) as fp:
            j = json.load(fp)
            for value in j['nuc'].values():
                counts[sample_name][type2label[value['type']]] += 1
        fp.close()

    for sample_name in counts.keys(): counts[sample_name]['total'] = sum([value for value in counts[sample_name].values()])

    with open(log_file, 'a') as fp:
        for sample_name in counts.keys():
            sample_counts = counts[sample_name]
            record = {
                'row':sample_name,
                'filepath': df.iloc[int(sample_name)].filepath,
                'record_type': 'ihcseg',
                'percent_positive_nuclei': round((sample_counts['3+']+sample_counts['2+']+sample_counts['1+'])*100/sample_counts['total'], 4),
                '3+_cells_%': round((sample_counts['3+']/sample_counts['total'])*100, 4),
                '2+_cells_%': round((sample_counts['2+']/sample_counts['total'])*100, 4),
                '1+_cells_%': round((sample_counts['1+']/sample_counts['total'])*100, 4),
                '0+_cells_%': round((sample_counts['0+']/sample_counts['total'])*100, 4),
                '3+_nuclei': sample_counts['3+'],
                '2+_nuclei': sample_counts['2+'],
                '1+_nuclei': sample_counts['1+'],
                '0+_nuclei': sample_counts['0+'],
                'total_nuclei': sample_counts['total'],
                'class': 'positive' if (sample_counts['3+']+sample_counts['2+']+sample_counts['1+'])/sample_counts['total'] > .2 else 'negative'
            }

            fp.write(','.join([str(v) for v in record.values()]) + '\n')

if __name__ == '__main__':
    quantify('results/segmentation/ihcseg/json', 'results/logs/quantification_ihcseg.log')