import os

import logging
import pandas as pd 
import numpy as np
# import cv2
from tqdm import tqdm

import torch
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingWarmRestarts
import torch.nn.functional as F
import timm

import matplotlib.pyplot as plt
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2

TEST_FOLDER = '/data/private_test/'

ROOT_FOLDER = '/'

SEED = 67
DIM = (512, 512, 3)
BATCH_SIZE = 4
SAMPLE = None

# DEVICE = torch.device('cuda:0')
DEVICE = torch.device('cpu')

THRESHOLD = 0.3

LABELS = ['Negative for Pneumonia', 'Typical Appearance', 'Indeterminate Appearance', 'Atypical Appearance']

CANDIDATES = [
    {
        'backbone_name':'eca_nfnet_l0',
        'model_path':'/model/Fold0_eca_nfnet_l0_v7_2labels_dedup_relabel_dist_ValidLoss0.356_ValidF10.716_Ep04.pth',
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

class ClfModel(nn.Module):
    def __init__(self, backbone_name, n_outs=2):
        super(ClfModel, self).__init__()
        self.backbone = timm.create_model(backbone_name, pretrained=False)
        
        if('nfnet' in backbone_name):
            clf_in_feature = self.backbone.head.fc.in_features
            self.backbone.head.fc = nn.Linear(clf_in_feature, n_outs)
        else:
            clf_in_feature = self.backbone.classifier.in_features
            self.backbone.classifier = nn.Linear(clf_in_feature, n_outs)
        
    def forward(self, x):
        batch_size = x.shape[0]
        x = self.backbone(x)

        return x

def get_valid_transforms(candidate):
    dim = candidate.get('dim', DIM)
    return A.Compose(
        [
            A.Resize(dim[0],dim[1],always_apply=True),
            A.Normalize(),
            ToTensorV2(p=1.0)
        ]
    )

def predict_fn(test_loader, model, threshold=None, device='cuda:0'):
    model.eval()
    tk0 = tqdm(enumerate(test_loader), total=len(test_loader))
    batch_preds=[]
    for i, batch in tk0:
        if(len(batch)==2):
            images, labels = batch
        else:
            images = batch
        images = images.to(device)
        with torch.cuda.amp.autocast(), torch.no_grad():
            logits = model(images)
            probs = torch.sigmoid(logits)

            probs = probs[:,0]*probs[:,1]
        batch_preds.append(probs.detach().cpu().numpy())
        
        del batch, images, logits, probs
        torch.cuda.empty_cache()
        
    predictions = np.concatenate(batch_preds)
    if(threshold):
        predictions = (predictions>=threshold).astype(int)
    return predictions

if __name__ == '__main__':
    test_df = pd.read_csv(f'{TEST_FOLDER}/private_test_meta.csv')

    test_df['5K_prob'] = 0
    for candidate in CANDIDATES:
        print(f"=========== Candidate: {candidate['model_path']} ===========")

        model = ClfModel(candidate['backbone_name'])
        model.load_state_dict(torch.load(candidate['model_path'], map_location=DEVICE))
        model.to(DEVICE)

        batch_size = candidate.get('batch_size', BATCH_SIZE)

        test_ds = ClfDataset(test_df, f'{TEST_FOLDER}/images/', get_valid_transforms(candidate))
        test_loader = torch.utils.data.DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=4)
        test_df['5K_prob'] += predict_fn(test_loader, model, device=DEVICE)

    test_df['5K_prob'] /= len(CANDIDATES)

    test_df['5K'] = (test_df['5K_prob'] >= THRESHOLD).astype(int)
    print(test_df['5K'].value_counts())

    sub_out_path = '/result/submission.csv'
    print(sub_out_path)

    test_df[['image_id', 'fname', '5K']].to_csv(sub_out_path, index=False)