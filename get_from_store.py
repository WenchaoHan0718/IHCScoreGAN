import os, shutil
import pandas as pd

dst_path = './input'
os.makedirs(dst_path, exist_ok=True)
datatable_path = 'input/Ki67_datatable.xlsx'
df = pd.read_excel(datatable_path)

dst_slides_dir = os.path.join(dst_path, 'slides')
dst_annots_dir = os.path.join(dst_path, 'annotations')
dst_masks_dir = os.path.join(dst_path, 'masks')

# os.makedirs(dst_slides_dir, exist_ok=True)
# os.makedirs(dst_annots_dir, exist_ok=True)
# os.makedirs(dst_masks_dir, exist_ok=True)

def get_files_from_store(n=0):
    row = df.iloc[n]
    filename = row.filepath.split('\\')[-1]
    dst_filepath = os.path.join(dst_slides_dir, filename)
    print(f'Copying file {row.filepath} to {dst_filepath}')
    shutil.copyfile(row.filepath, dst_filepath)

    annotation_filename = row.annotation_filepath.split('\\')[-1]
    dst_annotation_filepath = os.path.join(dst_annots_dir, annotation_filename)
    print(f'Copying file {row.annotation_filepath} to {dst_annotation_filepath}')
    shutil.copyfile(row.annotation_filepath, dst_annotation_filepath)

    return dst_filepath, dst_annotation_filepath

def get_counts_from_store(n=0):
    counts = {'(3+) Nuclei':0,'(2+) Nuclei':0,'(1+) Nuclei':0,'(0+) Nuclei':0,'Total Nuclei':0}
    row = df.iloc[n]
    for key in counts.keys():
        row_key = '_'.join(key.split(' ')).replace(')', '').replace('(', '').lower()
        counts[key] += row[row_key]
    return counts

if __name__ == '__main__':
    get_files_from_store(1)