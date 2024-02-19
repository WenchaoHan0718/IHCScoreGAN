OPENSLIDE_PATH = 'C:\\Temp\\openslide-win64-20231011\\bin'
import xml.etree.ElementTree as et
import os
from pathlib import Path
import numpy as np
from tqdm import tqdm
from PIL import Image
import pandas as pd
from glob import glob
import cv2

from scipy.ndimage import maximum_filter
import torch

def quantify_keypoints(counts, K, path, neighborhood_size=30, threshold=0.8, return_coords=False):
    width, height, _ = K.shape

    type2label = {0:'positive', 1:'negative'}
    sample_name = Path(path).stem.split('-')[0]

    if sample_name not in counts.keys(): 
        counts[sample_name] = {'positive':0, 'negative':0, 'total':0}

    keypoints = K[:,:,0]
    maxima = (keypoints == maximum_filter(keypoints, neighborhood_size)) & (keypoints > threshold)
    classes = K[:,:,1:][maxima].argmax(axis=-1)
    counts[sample_name]['total'] += len(classes)
    for i in range(2):
        counts[sample_name][type2label[i]] += (classes==i).sum()

    if return_coords:
        coords = torch.stack(torch.meshgrid([torch.linspace(0, width-1, width), torch.linspace(0, height-1, height)], indexing='ij'), -1).to(torch.int)[maxima]
        return counts, zip(coords, classes)
    else: 
        return counts

with os.add_dll_directory(OPENSLIDE_PATH):
    import openslide

annotation_dir = r'\\mfad.mfroot.org\rchapp\aperio_prod\Annotations'
datatable_path = 'input/Ki67_datatable.xls'
df = pd.read_excel(datatable_path)
tile_size = 256

def stitch_markups(sample_name):
    filepath = df.iloc[sample_name].filepath
    slide_dir = Path(filepath).parent
    markup_dir = os.path.join(slide_dir, '_Markup_')
    annot_path = os.path.join(annotation_dir, Path(filepath).name.replace('.svs', '.xml'))
    tree = et.parse(annot_path)
    root = tree.getroot()
    region_ids = [x.attrib['InputRegionId'] for x in root.iter('Region') if x.attrib['InputRegionId'] != '0']
    markups = [os.path.join(markup_dir, x) for x in sorted(os.listdir(markup_dir)) if any([y in x for y in region_ids])]

    print(f'Opening slide {filepath}...')
    dimensions = np.array(openslide.OpenSlide(filepath).dimensions)
    dimensions -= dimensions % tile_size
    print(f'Creating blank image of dimensions {dimensions}...')
    img = Image.new('RGB', tuple(dimensions))
    for markup in tqdm(markups):
        s = openslide.OpenSlide(markup)
        x, y = [int(v) for k, v in s.properties.items() if 'Offset' in k]
        tile = s.read_region((0, 0), 0, s.dimensions)
        img.paste(tile, (x, y))

    return img

def stitch_tiles(tiles_dir, sample_name, tile_size=128):
    tiles_dir = os.path.join(tiles_dir, str(sample_name).zfill(3))
    filename = df.iloc[int(sample_name)].filepath
    print(f'Opening slide {filename}...')
    dimensions = np.array(openslide.OpenSlide(filename).dimensions)
    dimensions -= dimensions % tile_size
    print(f'Creating blank image of dimensions {dimensions}...')
    img = Image.new('RGB', tuple(dimensions))

    pbar = tqdm(sorted(glob(tiles_dir + '/*.png')))
    assert len(pbar)>0, 'Tiles do not exist for this slide.'
    pbar.set_description('Adding patches to blank image...')
    for tilename in pbar:
        r, c, x, y, w, h = [int(item[1:]) for item in Path(tilename).name.split('.')[0].split('-')[2:]]

        tile = Image.open(tilename).resize((w, h))
        img.paste(tile, (tile_size * c, tile_size * r))

    return img

if __name__=='__main__':
    Image.MAX_IMAGE_PIXELS = None
    sample_name = 3352#[2, 135, 2538]

    # img1 = stitch_tiles(r'C:\Users\M306410\Desktop\repos\UGATIT-pytorch\results\ihc2maskvar_before2018\test_40000', sample_name, 128)
    img1_ti = stitch_tiles(r'C:\Users\M306410\Desktop\cell_segmentation\results\tiles_png', sample_name, 128)
    img1_cp = stitch_tiles(r'C:\Users\M306410\Desktop\repos\UGATIT-pytorch\results\ihc2maskvar_before2018\test_cp_40000', sample_name, 128)
    # img2 = stitch_tiles(r'C:\Users\M306410\Desktop\cell_segmentation\results\segmentation\ihcseg\overlay', sample_name)
    img3 = stitch_tiles(r'C:\Users\M306410\Desktop\cell_segmentation\results\segmentation\DeepLIIF_new\overlay', sample_name, 512)
    # img4 = stitch_markups(sample_name)
    # img5 = stitch_tiles(r'C:\Users\M306410\Desktop\cell_segmentation\results\tiles512n_png', sample_name, 512)

    # display = Image.new('RGB', (w*3, h))

    # crop_region = (1000, 4000, 2000, 5000)
    x, y, w, h = cv2.boundingRect(cv2.cvtColor(np.array(img1_ti), cv2.COLOR_RGB2GRAY))
    # img1_crop = img1.crop((x, y, x+w, y+h))#.crop(crop_region)
    img1_ti_crop = img1_ti.crop((x, y, x+w, y+h))#.crop(crop_region)
    img1_cp_crop = img1_cp.crop((x, y, x+w, y+h))#.crop(crop_region)

    arr = np.array(img1_cp_crop)/255.
    img1_cp_crop = Image.fromarray(np.array(img1_cp_crop)[:,:,0,None].repeat(3, -1))
    counts, coords = quantify_keypoints({}, arr+(np.random.rand(*arr.shape)/255.), str(sample_name), return_coords=True, neighborhood_size=25/2, threshold=0.5)
    img = np.array(img1_ti_crop)/255.
    for coord, type in coords:
        coord = np.array([coord[1], coord[0]])
        img = cv2.circle(img, coord, 3, [1, 0, 0] if type==0 else [0, 0, 1], thickness=-1)
    img1_kp_crop = Image.fromarray((img*255).astype(np.uint8))
    # [x.show() for x in [img1_crop, img1_kp_crop, img1_cp_crop]]

    # img2_crop = img2.crop((x, y, x+w, y+h))#.crop(crop_region)
    # Image.fromarray((img*255).astype(np.uint8)).show()

    x, y, w, h = cv2.boundingRect(cv2.cvtColor(np.array(img3), cv2.COLOR_RGB2GRAY))
    img3_crop = img3.crop((x, y, x+w, y+h))#.crop(crop_region)

    # x, y, w, h = cv2.boundingRect(cv2.cvtColor(np.array(img4), cv2.COLOR_RGB2GRAY))
    # img4_crop = img4.crop((x-125, y-112, x+w, y+h))#.crop(crop_region)

    # x, y, w, h = cv2.boundingRect(cv2.cvtColor(np.array(img5), cv2.COLOR_RGB2GRAY))
    # img5_crop = img5.crop((x, y, x+w, y+h))#.crop(crop_region)

    # mask = Image.fromarray(np.array(img1).sum(axis=-1)==(255*3))
    # img3 = Image.composite(Image.fromarray(np.ones(shape=(crop_region[3]-crop_region[1], crop_region[2]-crop_region[0]))*255).convert('RGB'), img3, mask)

    [x.show() for x in [img1_ti_crop, img1_kp_crop, img3_crop]]

    # for i, img in enumerate([img1, img2, img3]):
    #     img = img.crop((x, y, x+w, y+h))
    #     img = img.crop(crop_region)
        # display.paste(img, (w*i, 0))
    # display.show()

    print()