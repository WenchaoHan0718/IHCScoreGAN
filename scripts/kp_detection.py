from PIL import Image
import numpy as np
import torch
from scipy.ndimage.filters import maximum_filter
from scipy.ndimage.morphology import generate_binary_structure
import cv2
import skimage

keypoints = np.ascontiguousarray(np.array(Image.open('kp_test.png')))[:,:,0]#/255.


# keypoints *= skimage.morphology.remove_small_objects(keypoints>100, min_size=10)
# contours, hiers = cv2.findContours(keypoints[:,:,0], cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
# img = keypoints
# for contour, hier in zip(contours, hiers[0]):
#     if hier[-1]!=-1: continue
#     mask = np.zeros_like(img[:,:,0])
#     cv2.drawContours(mask, contour, -1, [1], thickness=-1)
#     masked_kp = mask * keypoints[:,:,0]
#     max_coord = (mask * keypoints[:,:,0]).argmax(dim=-1)
#     img = cv2.circle(img, max_coord, 3, [255, 0, 0], thickness=-1)

max_filter = maximum_filter(keypoints, 15)
maxima = (keypoints == max_filter) & (keypoints > 100)
coords = torch.stack(torch.meshgrid([torch.linspace(0, 255, 256), torch.linspace(0, 255, 256)]), -1).to(torch.uint8)[maxima]

img = keypoints[:,:,None].repeat(3, -1)
for coord in coords:
    coord = np.array([coord[1], coord[0]])
    img = cv2.circle(img, coord, 3, [255, 0, 0], thickness=-1)

Image.fromarray(img).show()