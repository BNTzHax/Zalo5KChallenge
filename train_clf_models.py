import os

import logging
import pandas as pd 
import numpy as np
import cv2
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from torch import nn

from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingWarmRestarts

import torch.nn.functional as F

IM_FOLDER = '/data/train/images'
ROOT_FOLDER = '/'

RUN_FOLDS = [0,1,2,3,4]
SEED = 67
DIM = (512, 512, 3)
BATCH_SIZE = 16
BASE_LR = 1e-4
NUM_EPOCHS = 30
PATIENT = 3
SAMPLE = None
DEVICE = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
THRESHOLD = 0.5

PARENT_OUT_FOLDER = f'{ROOT_FOLDER}/trained_models/'    

CANDIDATES = [
    {
        'backbone_name':'eca_nfnet_l0',
        'ver_note':'v7_2labels_dedup_relabel_dist',
    },
]

import sys
sys.path.append(f'/app/utils/')
from general import seed_torch
from general import seed_torch, init_progress_dict,\
             log_to_progress_dict, save_progress, log_and_print, get_logger

# seed every thing
seed_torch(SEED)

class ClfDataset(torch.utils.data.Dataset):
    
    def __init__(self, csv, im_folder, transforms=None):
        self.csv = csv.reset_index()
        self.im_folder = im_folder
        self.augmentations = transforms

    def __len__(self):
        return self.csv.shape[0]

    def __getitem__(self, index):
        row = self.csv.iloc[index]
        image = cv2.imread(os.path.join(self.im_folder, row.fname))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        if self.augmentations:
            augmented = self.augmentations(image=image)
            image = augmented['image']
        
        if('5k' in self.csv.columns):
            # return image, torch.tensor(row['5k'])
            return image, torch.tensor(row[['mask', 'distancing']])
        
        return image
    
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2

def get_train_transforms(candidate):
    dim = candidate.get('dim', DIM)
    return A.Compose(
        [
            A.HorizontalFlip(p=0.5),
            # A.VerticalFlip(p=0.5),
            A.ShiftScaleRotate(p=0.5, rotate_limit=20),
           
            A.Cutout(max_h_size=20, max_w_size=20, p = 0.5),
            A.ToGray(p=0.5),
            A.GaussNoise(p=0.5),
            A.MedianBlur(p=0.5),
            A.CLAHE(p=0.5),
            A.Sharpen(p=0.5),
            A.RandomBrightnessContrast(p=0.5),
            A.Downscale(p=0.5, scale_min=0.25, scale_max=0.75),
            
            A.Resize(dim[0],dim[1],always_apply=True),
            A.Normalize(),
            ToTensorV2(p=1.0)
        ]
    )

def get_valid_transforms(candidate):
    dim = candidate.get('dim', DIM)
    return A.Compose(
        [
            A.Resize(dim[0],dim[1],always_apply=True),
            A.Normalize(),
            ToTensorV2(p=1.0)
        ]
    )


import timm

class ClfModel(nn.Module):
    def __init__(self, backbone_name, n_outs=2):
        super(ClfModel, self).__init__()
        self.backbone = timm.create_model(backbone_name, pretrained=True)
        
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


import pandas as pd
from tqdm import tqdm
import numpy as np
import torch
from torch import nn
import gc
from sklearn.metrics import roc_auc_score, accuracy_score

class AverageMeter(object):
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    
    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
        
def train_valid_fn(dataloader,model,criterion,optimizer=None,scaler=None,device='cuda:0',scheduler=None,epoch=0,mode='train'):
    '''Perform model training'''
    if(mode=='train'):
        model.train()
    elif(mode=='valid'):
        model.eval()
    else:
        raise ValueError('No such mode')
        
    loss_score = AverageMeter()
    
    tk0 = tqdm(enumerate(dataloader), total=len(dataloader))
    for i, batch in tk0:
        if(mode=='train'):
            optimizer.zero_grad()
            
        # input, gt
        images, labels = batch
        images = images.to(DEVICE)
        labels = labels.to(DEVICE)

        # prediction
        with torch.cuda.amp.autocast():
            logits = model(images)
            # compute loss
            loss = criterion(logits, labels)
        
        if(mode=='train'):
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        
        loss_score.update(loss.clone().detach().cpu().item(), dataloader.batch_size)
        
        if(mode=='train'):
            tk0.set_postfix(Loss_Train=loss_score.avg, Epoch=epoch, LR=optimizer.param_groups[0]['lr'])
        elif(mode=='valid'):
            tk0.set_postfix(Loss_Valid=loss_score.avg, Epoch=epoch)
        
        del batch, images, labels, logits, loss
        torch.cuda.empty_cache()
        
    if(mode=='train'):
        if(scheduler.__class__.__name__ == 'CosineAnnealingWarmRestarts'):
            scheduler.step(epoch=epoch)
        elif(scheduler.__class__.__name__ == 'ReduceLROnPlateau'):
            scheduler.step(loss_score.avg)
    
    return loss_score.avg

from sklearn.metrics import f1_score

def compute_f1(dataloader, model, threshold=0.5, device='cuda:0'):
    model.eval()
    tk0 = tqdm(enumerate(dataloader), total=len(dataloader))
    batch_preds=[]
    batch_labels=[]
    for i, batch in tk0:
        images, labels = batch
        images = images.to(device)

        with torch.cuda.amp.autocast(), torch.no_grad():
            logits = model(images)
            probs = torch.sigmoid(logits)

            probs = probs[:,0]*probs[:,1]
        batch_preds.append(probs.detach().cpu().numpy())
        labels = labels[:,0]*labels[:,1]
        batch_labels.append(labels.numpy())
        
        del batch, images, labels, logits, probs
        torch.cuda.empty_cache()
        
    predictions = np.concatenate(batch_preds)
    predictions = (predictions>=threshold).astype(int)
    labels = np.concatenate(batch_labels)
    f1 = f1_score(y_true=labels, y_pred=predictions)
    return f1


if __name__ == '__main__':
    df = pd.read_csv(f'/fold_meta/train_resolve_duplicates_relabel_dist_v1_fold_split.csv')

    for candidate in CANDIDATES:
        print(f"######################### Candidate: {candidate['backbone_name']} ############################")
        run_folds = candidate.get('run_folds', RUN_FOLDS)
        
        parent_out_folder = candidate.get('parent_out_folder', PARENT_OUT_FOLDER)
        ver_note = candidate['ver_note']
        out_folder_name = f"{candidate['backbone_name']}_{ver_note}"
        out_folder = os.path.join(parent_out_folder, out_folder_name)

        os.makedirs(out_folder, exist_ok=True)
        
        for valid_fold in run_folds:
            # Read data
            if(SAMPLE):
                df = df.sample(SAMPLE, random_state=SEED)

            train_df = df[df.fold!=valid_fold]
            valid_df = df[df.fold==valid_fold]

            print(f'\n\n================= Fold {valid_fold} ==================')
            print(f'Number of training images: {len(train_df)}. Number of valid images: {len(valid_df)}')

            # create data loader
            train_dataset = ClfDataset(train_df, IM_FOLDER, get_train_transforms(candidate))
            valid_dataset = ClfDataset(valid_df, IM_FOLDER, get_valid_transforms(candidate))

            batch_size = candidate.get('batch_size', BATCH_SIZE)
            train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
            valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
            
            # Model
            model = ClfModel(candidate['backbone_name'])
            warm_start_weight = candidate.get('warm_start_weight')
            if(warm_start_weight):
                print('Load warm start weight:', warm_start_weight)
                model.load_state_dict(torch.load(warm_start_weight, map_location='cpu'))
            print('DEVICE:', DEVICE)
            model.to(DEVICE)
            print()
            
            # use amp to accelerate training
            scaler = torch.cuda.amp.GradScaler()

            # Optimizer and scheduler
            base_lr = candidate.get('base_lr', BASE_LR)
            optim = AdamW(model.parameters(), lr=BASE_LR)

            num_training_steps = NUM_EPOCHS * len(train_loader)
            lr_scheduler = ReduceLROnPlateau(optimizer=optim)

            # loss
            bce = nn.BCEWithLogitsLoss()
            
            # Logging
            logger = get_logger(
                name = f'training_log_fold{valid_fold}.txt',
                path=os.path.join(out_folder, f'training_log_fold{valid_fold}.txt')
            )

            best_valid_loss = 9999
            best_valid_ep = 0
            patient = PATIENT

            progress_dict = init_progress_dict(['loss', 'f1'])
            
            start_ep = candidate.get('warm_start_ep', 1)
            print('Start ep:', start_ep)

            for epoch in range(start_ep, NUM_EPOCHS+1):

                # =============== Training ==============
                train_loss = train_valid_fn(train_loader,model,bce,optimizer=optim, scaler=scaler, device=DEVICE,
                                            scheduler=lr_scheduler,epoch=epoch,mode='train')
                valid_loss = train_valid_fn(valid_loader,model,bce,scaler=scaler,device=DEVICE,epoch=epoch,mode='valid')

                # =============== Evaluation =================
                train_f1 = compute_f1(train_loader, model, THRESHOLD, DEVICE)
                valid_f1 = compute_f1(valid_loader, model, THRESHOLD, DEVICE)
                
                current_lr = optim.param_groups[0]['lr']
                log_line = f'Model: {out_folder_name}. Epoch: {epoch}. '
                log_line += f'Train loss:{train_loss} - Valid loss: {valid_loss}. '
                log_line += f'Train F1:{train_f1} - Valid F1: {valid_f1}. '
                log_line += f'Lr: {current_lr}.'

                log_and_print(logger, log_line)

                metric_dict = {'train_loss':train_loss,'valid_loss':valid_loss,
                            'train_f1':train_f1, 'valid_f1':valid_f1, 
                            }
                
                progress_dict = log_to_progress_dict(progress_dict, metric_dict)

                # plot figure and save the progress chart
                save_progress(progress_dict, out_folder, out_folder_name, valid_fold, show=False)

                if(valid_loss < best_valid_loss):
                    best_valid_loss = valid_loss
                    best_valid_ep = epoch
                    patient = PATIENT # reset patient

                    # save model
                    name = os.path.join(out_folder, 'Fold%d_%s_ValidLoss%.03f_ValidF1%.03f_Ep%02d.pth'%(valid_fold, out_folder_name, valid_loss, valid_f1, epoch))
                    log_and_print(logger, 'Saving model to: ' + name)
                    torch.save(model.state_dict(), name)
                else:
                    patient -= 1
                    log_and_print(logger, 'Decrease early-stopping patient by 1 due valid loss not decreasing. Patient='+ str(patient))

                if(patient == 0):
                    log_and_print(logger, 'Early stopping patient = 0. Early stop')
                    break