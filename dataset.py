import torch
import torch.utils.data as data
from torchvision.transforms.functional import adjust_contrast

from PIL import Image

import random
import os
import os.path

import json
import glob
from tqdm import tqdm
from scipy.ndimage import center_of_mass

import cv2
import numpy as np

import pandas as pd
import time

def has_file_allowed_extension(filename, extensions):
    """Checks if a file is an allowed extension.

    Args:
        filename (string): path to a file

    Returns:
        bool: True if the filename ends with a known image extension
    """
    filename_lower = filename.lower()
    return any(filename_lower.endswith(ext) for ext in extensions)


def find_classes(dir):
    classes = [d for d in os.listdir(dir) if os.path.isdir(os.path.join(dir, d))]
    classes.sort()
    class_to_idx = {classes[i]: i for i in range(len(classes))}
    return classes, class_to_idx


def make_dataset(dir, extensions):
    images = []
    for root, _, fnames in tqdm(sorted(os.walk(dir))):
        for fname in sorted(fnames):
            if has_file_allowed_extension(fname, extensions):
                path = os.path.join(root, fname)
                item = (path, 0)
                images.append(item)
    return images

class DatasetFolder(data.Dataset):
    def __init__(self, root, loader, extensions, transform=None, target_transform=None, return_path=False):
        samples = make_dataset(root, extensions)
        if len(samples) == 0:
            raise(RuntimeError("Found 0 files in subfolders of: " + root + "\n"
                               "Supported extensions are: " + ",".join(extensions)))

        self.root = root
        self.kp_root = None

        self.loader = loader
        self.extensions = extensions
        self.samples = samples

        self.transform = transform
        self.target_transform = target_transform

        self.return_path = return_path

    def __getitem__(self, index):
        path, target = self.samples[index]
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)

        if self.kp_root is not None:
            keypoint = self.get_instance_dist_map(sample, os.path.split(path)[-1][:-4])
            keypoint = (keypoint - 0.5) / 0.5
            return (sample, target, keypoint) if not self.return_path else (sample, target, keypoint, path)
        return (sample, target) if not self.return_path else (sample, target, path)

    def __len__(self):
        return len(self.samples)

    def __repr__(self):
        fmt_str = 'Dataset ' + self.__class__.__name__ + '\n'
        fmt_str += '    Number of datapoints: {}\n'.format(self.__len__())
        fmt_str += '    Root Location: {}\n'.format(self.root)
        tmp = '    Transforms (if any): '
        fmt_str += '{0}{1}\n'.format(tmp, self.transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        tmp = '    Target Transforms (if any): '
        fmt_str += '{0}{1}'.format(tmp, self.target_transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        return fmt_str
    
    def get_instance_dist_map(self, sample, filename):
        mask = torch.zeros((3, 256, 256))
        with open(self.kp_root / f'{filename}.json') as f:
            j = json.load(f)
        f.close()

        for nuc in [nuc for nuc in j['nuc'].values()]:
            centroid = round(nuc['centroid'][1]), round(nuc['centroid'][0])
            blob = torch.zeros((256, 256))
            cv2.drawContours(blob.numpy(), [np.array(nuc['contour'])], -1, [1], thickness=cv2.FILLED)
            coords = torch.stack(torch.meshgrid(
                [torch.linspace(0, 255, 256), 
                torch.linspace(0, 255, 256)], indexing='ij'), -1)
            dist = ((coords - torch.tensor(centroid)).norm(dim=-1) / torch.tensor([20., 20.]).norm(dim=-1))
            mask[1:][sample.permute(1, 2, 0)[centroid].argmax()] += blob
            blob *= 1 - dist
            mask[0] += blob.clip(min=0, max=1)
        return mask

IMG_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif']

def default_loader(path): 
    with open(path, 'rb') as f: 
        img = Image.open(f).convert('RGB')
    f.close()
    return img

class ImageFolder(DatasetFolder):
    def __init__(self, root, transform=None, target_transform=None, return_path=False,
                 loader=default_loader, allowed_sample_names=None):
        super(ImageFolder, self).__init__(root, loader, IMG_EXTENSIONS,
                                          transform=transform,
                                          target_transform=target_transform,
                                          return_path=return_path)
        self.imgs = self.samples


class ImageAndKeypointFolder(DatasetFolder):
    def __init__(self, root, transform=None, target_transform=None, return_path=False,
                 loader=default_loader, allowed_sample_names=None):
        super(ImageAndKeypointFolder, self).__init__(root / 'overlay', loader, IMG_EXTENSIONS,
                                          transform=transform,
                                          target_transform=target_transform,
                                          return_path=return_path)
        self.imgs = self.samples
        self.kp_root = root / 'json'