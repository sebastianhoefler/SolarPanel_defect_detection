from torch.utils.data import Dataset
from torchvision import datasets
import torch
from pathlib import Path
from skimage.io import imread
from skimage.color import gray2rgb
import numpy as np
import torchvision as tv
import pandas as pd
import cv2
import os
import random
from sklearn.preprocessing import LabelEncoder


train_mean = [0.59685254, 0.59685254, 0.59685254]
train_std = [0.16043035, 0.16043035, 0.16043035]

os.chdir('/Users/sebh/Developer/Pytorch_Challenge/src_to_implement')

class ChallengeDataset(Dataset):
    
    def __init__(self, data, mode: str):
        self.data = data
        self.mode = mode

        self._transform_train = tv.transforms.Compose([
            tv.transforms.ToPILImage(), 
            tv.transforms.RandomHorizontalFlip(p=0.5),
            tv.transforms.RandomVerticalFlip(p=0.5),
            RandomRotate180(),
            tv.transforms.RandomAffine(degrees=0, translate=(0.02, 0.02)),
            tv.transforms.transforms.RandomRotation(degrees=(-3, 3)),
            #tv.transforms.RandomResizedCrop(300, scale=(0.98, 1.02)),
            tv.transforms.RandomEqualize(p=0.5),
            tv.transforms.ToTensor(),
            tv.transforms.Normalize(train_mean, train_std)
        ])
        
        self._transform_val = tv.transforms.Compose([
            tv.transforms.ToPILImage(), 
            tv.transforms.ToTensor(),
            tv.transforms.Normalize(train_mean, train_std)
        ])

    
    def __len__(self):
        return len(self.data)


    def __getitem__(self, index):
        gray_img = imread(self.data.iloc[index]['filename'])

        # Convert the grayscale image to an RGB image and rearrange the dimensions for C, H, W
        rgb_img = torch.from_numpy(np.transpose(gray2rgb(gray_img), (2, 0, 1)))
        labels = torch.tensor([int(self.data.iloc[index]["crack"]), int(self.data.iloc[index]["inactive"])])

        if self.mode == 'train':
            img = self._transform_train(rgb_img)
        else:
            img = self._transform_val(rgb_img)

        return img, labels.float()
    

class RandomRotate180:
    def __init__(self, degrees=[0, 180]):
        self.degrees = degrees

    def __call__(self, img):
        degree = random.choice(self.degrees)
        return tv.transforms.functional.rotate(img, degree)
    
class Equalize(object):
    def __call__(self, img):
        return tv.transforms.functional.equalize(img)