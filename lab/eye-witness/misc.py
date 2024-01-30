
"""
object detection
"""

import os
import torch
import cv2
import numpy as np
import pandas as pd
from torchvision.models import detection


# sample image
fn = r'/mnt/home/russell.burdt/data/object-detection/imgs/sign.jpg'
assert os.path.isfile(fn)
img = cv2.imread(fn)
height, width = img.shape[:2]

# convert image (batch x channel x width x height)
tx = torch.tensor(img).permute((2, 1, 0)).unsqueeze(dim=0)
assert (tx.shape[2], tx.shape[3]) == (width, height)
tx = tx.type(dtype=torch.FloatTensor)
tx = tx / tx.max()

# ...
model = detection.fasterrcnn_resnet50_fpn(weights=detection.FasterRCNN_ResNet50_FPN_Weights.COCO_V1)
model.eval()




# """
# convert mp4 to frames
# """

# import os
# import cv2
# from shutil import rmtree
# from glob import glob


# # datadir, mp4 file, frames dir
# datadir = r'c:/Users/russell.burdt/Data/eye-witness'
# assert os.path.isdir(datadir)
# mp4 = os.path.join(datadir, '6-8-23, I5 and Cristianitos Rd', 'QM00030828.mp4')
# assert os.path.isfile(mp4)
# fdir = os.path.join(datadir, 'frames')
# assert not os.path.isdir(fdir)
# os.mkdir(fdir)

# # mp4 to frames
# cmd = f"""ffmpeg -i "{mp4}" {os.path.join(fdir, r'%04d.png')}"""
# os.system(cmd)
# frames = sorted(glob(os.path.join(fdir, '*.png')))
# assert frames

# # new video from subset of cropped frames
# fns = frames[160:190]
# imgs = [cv2.imread(fn)[120:200, :400, :] for fn in fns]
# rmtree(fdir)
# os.mkdir(fdir)
# assert all([cv2.imwrite(os.path.join(fdir, f'{x:04d}.png'), img) for x, img in enumerate(imgs)])
# cmd = f"""ffmpeg -y -framerate 2 -i {os.path.join(fdir, r'%04d.png')} {os.path.join(datadir, 'video.mp4')}"""
# os.system(cmd)
# rmtree(fdir)
