import os

import logging
import pandas as pd 
import numpy as np
# import cv2

import torch
from torch import nn

from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingWarmRestarts

import torch.nn.functional as F

import matplotlib.pyplot as plt
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2

IM_FOLDER = 'train/images'
ROOT_FOLDER = '/app'
RUN_FOLDS = [0]
SEED = 67
DIM = (512, 512, 3)
BATCH_SIZE = 32
BASE_LR = 1e-4
NUM_EPOCHS = 30
PATIENT = 7
SAMPLE = None

# DEVICE = torch.device('cuda:0')
DEVICE = torch.device('cpu')

THRESHOLD = 0.5

PARENT_OUT_FOLDER = f'{ROOT_FOLDER}/models/'    

LABELS = ['Negative for Pneumonia', 'Typical Appearance', 'Indeterminate Appearance', 'Atypical Appearance']

CANDIDATES = [
    {
        'backbone_name':'eca_nfnet_l0',
        'model_path':'/',
        'batch_size':4,
    },
]

class ClfDataset(torch.utils.data.Dataset):
    
    def __init__(self, csv, im_folder, transforms=None):
        self.csv = csv.reset_index()
        self.im_folder = im_folder
        self.augmentations = transforms

    def __len__(self):
        return self.csv.shape[0]

    def __getitem__(self, index):
        row = self.csv.iloc[index]
        image = plt.imread(os.path.join(self.im_folder, row.fname))
        # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        if self.augmentations:
            augmented = self.augmentations(image=image)
            image = augmented['image']
        
        if('5k' in self.csv.columns):
            # return image, torch.tensor(row['5k'])
            return image, torch.tensor(row[['mask', 'distancing']])
        
        return image
    

def get_valid_transforms(candidate):
    dim = candidate.get('dim', DIM)
    return A.Compose(
        [
            A.Resize(dim[0],dim[1],always_apply=True),
            A.Normalize(),
            ToTensorV2(p=1.0)
        ]
    )


test_df = pd.read_csv('/data/demo_private_test/private_test_meta.csv')