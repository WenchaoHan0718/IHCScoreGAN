import torch
import torch.utils.data as data
from PIL import Image
import os, json, cv2
from tqdm import tqdm
import numpy as np

from utils import norm, has_file_allowed_extension

IMG_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif']

def make_dataset(dir, extensions):
    '''
    Aggregates nested dataset files within a directory.

    Args:
        dir (string):        Path to the root dataset directory.
        extensions (string): List of acceptable file extensions.

    Returns:
        images (list):       List of image filepaths.
    '''
    images = []
    for root, _, fnames in tqdm(sorted(os.walk(dir))):
        for fname in sorted(fnames):
            if has_file_allowed_extension(fname, extensions):
                path = os.path.join(root, fname)
                item = (path, 0)
                images.append(item)
    return images

class DatasetFolder(data.Dataset):
    ''' Defines a dataset of images aggregated within a given root folder. '''

    def __init__(self, root, loader, extensions, transform=None, target_transform=None, return_path=False):
        samples = make_dataset(root, extensions)
        if len(samples) == 0:
            raise(RuntimeError("Found 0 files in subfolders of: " + root + ". Supported extensions are: " + ",".join(extensions) + '.'))

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
            # Here, the target dataset calculates a center point distance map and cell type mask and returns it.
            keypoint = norm(self.get_instance_dist_map(sample, os.path.split(path)[-1][:-4]))
            return (sample, target, keypoint) if not self.return_path else (sample, target, keypoint, path)
        return (sample, target) if not self.return_path else (sample, target, path)

    def __len__(self):
        return len(self.samples)
    
    def get_instance_dist_map(self, sample, filename):
        '''
        Creates the cell center point distance maps and cell type binary segmentation masks.

        Args:
            sample (tensor):    The synthetic colored and shared target segmentation masks.
            filename (string):  The corresponding image filenames, which allows recovery of the 
                                cell center point file.

        Returns:
            mask (tensor):      The cell center point distance maps and cell type binary segmentation masks.
        ''' 

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

def default_loader(path): 
    with open(path, 'rb') as f: 
        img = Image.open(f).convert('RGB')
    f.close()
    return img

class ImageFolder(DatasetFolder):
    def __init__(self, root, transform=None, target_transform=None, return_path=False, loader=default_loader):
        super(ImageFolder, self).__init__(root, 
                                          loader, 
                                          IMG_EXTENSIONS,
                                          transform=transform,
                                          target_transform=target_transform,
                                          return_path=return_path)
        self.imgs = self.samples


class ImageAndKeypointFolder(DatasetFolder):
    def __init__(self, root, transform=None, target_transform=None, return_path=False, loader=default_loader):
        super(ImageAndKeypointFolder, self).__init__(root / 'overlay', 
                                                     loader, 
                                                     IMG_EXTENSIONS,
                                                     transform=transform,
                                                     target_transform=target_transform,
                                                     return_path=return_path)
        self.imgs = self.samples
        self.kp_root = root / 'json'