OPENSLIDE_PATH = 'C:\\Temp\\openslide-win64-20231011\\bin'
import os, shutil
with os.add_dll_directory(OPENSLIDE_PATH):
    import openslide

import numpy as np
from PIL import Image
import pandas as pd

from preprocessing import slide, tiles, mask, filter, util
from scripts.result_quantification import quantify
from get_from_store import *
from run_infer_hovernet import *

json_base = './results/segmentation/ihcseg/'
log_path = './results/logs/'

datatable_path = 'input/Ki67_datatable.xlsx'
datatable_df = pd.read_excel(datatable_path)

# ihc2maskvar_shuffle_df = pd.read_csv('../repos/UGATIT-pytorch/results/ihc2maskvar_shuffle2/logs/quantification_40000.log')

def cleanup():
    for file in [os.path.join('results', file) for file in os.listdir('results') if file not in ['segmentation', 'logs', 'tiles_png', 'tiles512_png', 'tiles512n_png']]: shutil.rmtree(file)

if __name__ == '__main__':
    # os.makedirs(log_path, exist_ok=True)
    # log_file = os.path.join(log_path, 'quantification.log')

    # if not os.path.exists(log_file): 
    #     with open(log_file, 'w') as fp: 
    #         fp.write('row,filepath,record_type,percent_positive_nuclei,3+_cells_%,2+_cells_%,1+_cells_%,0+_cells_%,3+_nuclei,2+_nuclei,1+_nuclei,0+_nuclei,total_nuclei,class\n')

    # datatable_df = datatable_df[(datatable_df.samples>0)]

    for row_index in [row_index for row_index in list(datatable_df['Unnamed: 0'])]:

        row = df.iloc[row_index]
        slide_file, annot_file = row.filepath, row.annotation_filepath#get_files_from_store(n)


        sample_name = str(row_index).zfill(3)

        if sample_name in [x for x in os.listdir('results/tiles_png')]: continue
        cleanup()

        s = openslide.OpenSlide(slide_file)
        
        msk = mask.xml_to_mask(annot_file, 1, s.dimensions, get_counts_from_store(row_index))
        if msk is None: continue
        
        if (np.array(msk) == 255).sum() > 100000000: 
            print(f'Annotation too large ({(np.array(msk) == 255).sum()}); skipping slide {df.iloc[row_index].filepath}.')
            continue


        level = s.get_best_level_for_downsample(slide.SCALE_FACTOR)
        img = s.read_region((0, 0), level, s.level_dimensions[level])
        img_size = tuple([d//32 for d in s.dimensions])
        img = img.resize(img_size, Image.BILINEAR)

        img, large_w, large_h, new_w, new_h = slide.slide_to_scaled_pil_image(row_index, slide_file)
        img_path = slide.get_training_image_path(row_index, large_w, large_h, new_w, new_h)
        os.makedirs(slide.DEST_TRAIN_DIR, exist_ok=True)
        img.save(img_path)
        
        msk_size = tuple([d//32 for d in msk.size])
        resized_msk = np.array(msk.resize(msk_size, Image.BILINEAR))
        grays_msk = filter.filter_grays(np.array(img.convert('RGB')), tolerance=5)
        total_msk = resized_msk
        masked_path = slide.get_filter_image_result(row_index)
        masked_image = Image.fromarray(img * np.dstack([total_msk, total_msk, total_msk]))#Image.composite(img, Image.fromarray(np.zeros(shape=img_size).T).convert('RGB'), total_msk)
        dst_dir = slide.FILTER_DIR
        os.makedirs(dst_dir, exist_ok=True)
        masked_image.save(masked_path)

        tiles.singleprocess_filtered_images_to_tiles(image_num_list=[row_index], mask=msk, datatable=df, html=False)

    # WSI Args
    # argv = ['--gpu','1',
    #         '--nr_types','5',
    #         '--type_info_path','segmentation/type_info_ihcseg.json',
    #         '--batch_size','12',
    #         '--model_mode','original',
    #         '--model_path','segmentation/pretrained/hovernet_original_ihcseg_type_tf2pytorch.tar',
    #         '--nr_inference_workers','1',
    #         '--nr_post_proc_workers','1',
    #         'wsi',
    #         '--input_dir',row.filepath,
    #         '--output_dir','results/segmentation/ihcseg',
    #         '--input_mask_dir',row.annotation_filepath,
    #         '--proc_mag','20',
    #         '--save_thumb',
    #         '--save_mask']

    # Tiles Args
    # argv = ['--gpu','1',
    #         '--nr_types','5',
    #         '--type_info_path','segmentation/type_info_ihcseg.json',
    #         '--batch_size','100',
    #         '--model_mode','original',
    #         '--model_path','segmentation/pretrained/hovernet_original_ihcseg_type_tf2pytorch.tar',
    #         '--nr_inference_workers','8',
    #         '--nr_post_proc_workers','8',
    #         'tile',
    #         '--input_dir','results/tiles_png',
    #         '--output_dir','results/segmentation/ihcseg',
    #         '--mem_usage','0.1',
    #         '--draw_dot']

    # run_infer(argv)
    # quantify(os.path.join(json_base, 'json'), log_file)
    # cleanup()
