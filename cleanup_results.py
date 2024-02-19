import os, shutil
from tqdm import tqdm

src_dir = './results/segmentation/ihcseg/overlay'

for file in tqdm(os.listdir(src_dir)):
    src_file = os.path.join(src_dir, file)
    if os.path.isdir(src_file): continue
    sample_name = file.split('-')[0]
    dst_dir = os.path.join(src_dir, sample_name)
    dst_file = os.path.join(dst_dir, file)
    os.makedirs(dst_dir, exist_ok=True)
    shutil.move(src_file, dst_file)