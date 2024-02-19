import os
from glob import glob
import numpy as np
from tqdm import tqdm
from PIL import Image

ihcseg_dir = '../results/segmentation/ihcseg/overlay'
ihc2kp_dir = r'C:\Users\M306410\Desktop\repos\UGATIT-pytorch\results\ihc_to_mask_variation\test_60000'

img_size = 256

all_ihcseg = []
all_ihcseg_roots = []
for root, _, fnames in tqdm(os.walk(ihcseg_dir)):
    root = root.split('\\')[-1]
    if root not in all_ihcseg_roots: all_ihcseg_roots.append(root)
    all_ihcseg += [os.path.join(root, fname) for fname in fnames if fname.endswith('.png')]

all_ihc2kp = []
for root, _, fnames in tqdm(os.walk(ihc2kp_dir)):
    root = root.split('\\')[-1]
    if root not in all_ihcseg_roots: continue
    for fname in [os.path.join(root, fname) for fname in fnames if fname.endswith('.png')]:
        if fname not in all_ihcseg: print(fname)
        else: all_ihc2kp += [fname]

dst = Image.new('RGB', (img_size * 8, img_size * 2))
random_indices = np.random.randint(len(all_ihcseg), size=(8,))
for i, idx in enumerate(random_indices):
    ihcseg_file = os.path.join(ihcseg_dir, all_ihcseg[idx])
    ihc2kp_file = os.path.join(ihc2kp_dir, all_ihc2kp[idx])

    dst.paste(Image.open(ihcseg_file), (img_size * i, 0))
    dst.paste(Image.open(ihc2kp_file), (img_size * i, img_size))

dst.show(title='Top: IHC Segmentation Model. Bottom: GAN Model.')
