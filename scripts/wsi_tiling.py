from preprocessing import slide, tiles, mask, filter, util

OPENSLIDE_PATH = 'C:\\Temp\\openslide-win64-20231011\\bin'
import os, shutil, glob
with os.add_dll_directory(OPENSLIDE_PATH):
    import openslide

src_dir = 'C:\\Users\\M306410\\Downloads\\TCGA\\40x\\slides'
dst_dir = 'C:\\Users\\M306410\\Downloads\\TCGA\\preprocessing'
os.makedirs(dst_dir, exist_ok=True)

for i, file in enumerate(glob.glob(src_dir + '\\*')):
    sample_name = str(i+1).zfill(3)
    os.rename(file, os.path.join(src_dir, sample_name + '.svs'))
slide.singleprocess_training_slides_to_images()
filter.singleprocess_apply_filters_to_images()
tiles.singleprocess_filtered_images_to_tiles()