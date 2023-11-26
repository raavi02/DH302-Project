from tqdm import tqdm
import os
import time
from random import randint
 
import gc 
import numpy as np
from scipy import stats
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.model_selection import KFold

import nibabel as nib
import pydicom as pdm
import nilearn as nl
import nilearn.plotting as nlplt
import h5py

import matplotlib.pyplot as plt
from matplotlib import cm
import matplotlib.animation as anim
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
# !pip install seaborn
import seaborn as sns
import imageio
from skimage.transform import resize
from skimage.util import montage

from IPython.display import Image as show_gif
from IPython.display import clear_output
from IPython.display import YouTubeVideo

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F

from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.nn import MSELoss

# !pip install albumentations==0.4.6
import albumentations as A
# from albumentations.pytorch import ToTensor, ToTensorV2


from albumentations import Compose, HorizontalFlip
# from albumentations.pytorch import ToTensor, ToTensorV2 

import warnings
warnings.simplefilter("ignore")

sample_filename = 'BraTS2020_TrainingData/MICCAI_BraTS2020_TrainingData/BraTS20_Training_001/BraTS20_Training_001_flair.nii'
sample_filename_mask = 'BraTS2020_TrainingData/MICCAI_BraTS2020_TrainingData/BraTS20_Training_001/BraTS20_Training_001_seg.nii'
sample_img = nib.load(sample_filename)
sample_img = np.asanyarray(sample_img.dataobj)
sample_img = np.rot90(sample_img)
sample_mask = nib.load(sample_filename_mask)
sample_mask = np.asanyarray(sample_mask.dataobj)
sample_mask = np.rot90(sample_mask)
print("img shape ->", sample_img.shape)
print("mask shape ->", sample_mask.shape)

class GlobalConfig:
    root_dir = ''
    train_root_dir = 'BraTS2020_TrainingData/MICCAI_BraTS2020_TrainingData'
    test_root_dir = 'BraTS2020_ValidationData/MICCAI_BraTS2020_ValidationData'
    # path_to_csv = './train_data.csv'
    # pretrained_model_path = '../input/brats20logs/brats2020logs/unet/last_epoch_model.pth'
    # train_logs_path = '../input/brats20logs/brats2020logs/unet/train_log.csv'
    # ae_pretrained_model_path = '../input/brats20logs/brats2020logs/ae/autoencoder_best_model.pth'
    # tab_data = '../input/brats20logs/brats2020logs/data/df_with_voxel_stats_and_latent_features.csv'
    seed = 55
    
def seed_everything(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    
config = GlobalConfig()
seed_everything(config.seed)

survival_info_df = pd.read_csv('BraTS2020_TrainingData/MICCAI_BraTS2020_TrainingData/survival_info.csv')
name_mapping_df = pd.read_csv('BraTS2020_TrainingData/MICCAI_BraTS2020_TrainingData/name_mapping.csv')

name_mapping_df.rename({'BraTS_2020_subject_ID': 'Brats20ID'}, axis=1, inplace=True) 


df = survival_info_df.merge(name_mapping_df, on="Brats20ID", how="right")

paths = []
for _, row  in df.iterrows():

    id_ = row['Brats20ID']
    phase = id_.split("_")[-2]
    if phase == 'Training':
        path = os.path.join(config.train_root_dir, id_)
    else:
        path = os.path.join(config.test_root_dir, id_)
    paths.append(path)

df['path'] = paths

# Data cleaning - removing all null age entries
train_data = df.loc[df['Age'].notnull()].reset_index(drop=True)

# Calculating Age rank for the basis of K - Fold stratification
train_data["Age_rank"] =  train_data["Age"] // 10 * 10
train_data = train_data.loc[train_data['Brats20ID'] != 'BraTS20_Training_355'].reset_index(drop=True, )

len(df)

skf = StratifiedKFold(
    n_splits=7, random_state=config.seed, shuffle=True
)

# enumeratng all entries for defining the fold number
# assigning the fold number in increment order
for i, (train_index, val_index) in enumerate(
        skf.split(train_data, train_data["Age_rank"])
        ):
        train_data.loc[val_index, "fold"] = i

# splitting of the data wasn't done for train , test &  validation data
train_df = train_data.loc[train_data['fold'] != 0].reset_index(drop=True)
val_df = train_data.loc[train_data['fold'] == 0].reset_index(drop=True)

# selecting the rows where the AGE col. is null --> test_df
test_df = df.loc[~df['Age'].notnull()].reset_index(drop=True)
print("train_df ->", train_df.shape, "val_df ->", val_df.shape, "test_df ->", test_df.shape)

train_data.to_csv("train_data.csv", index=False)
test_df.to_csv("test_df.csv", index=False)
train_df.to_csv("train_df.csv", index=False)

class BratsDataset(Dataset):
    def __init__(self, df: pd.DataFrame, phase: str="test", is_resize: bool=False):
        self.df = df
        self.phase = phase
        self.augmentations = get_augmentations(phase)
        self.data_types = ['_flair.nii', '_t1.nii', '_t1ce.nii', '_t2.nii']
        self.is_resize = is_resize

    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, idx):
        # at a specified index ( idx ) select the value under 'Brats20ID' & asssign it to id_
        id_ = self.df.loc[idx, 'Brats20ID']

        # As we've got the id_ , now find the path of the entry by asserting the Brats20ID to id_
        root_path = self.df.loc[self.df['Brats20ID'] == id_]['path'].values[0]

        # load all modalities
        images = []

        for data_type in self.data_types:
            # here data_type is appended to the root path, as it only contains the name without the datatype such as .nii etc
            img_path = os.path.join(root_path, id_ + data_type)
            img = self.load_img(img_path)#.transpose(2, 0, 1)

            if self.is_resize:
                img = self.resize(img)

            img = self.normalize(img)
            images.append(img)

        # stacking all the t1 , t1ce , t2 , t2 flair files of a single ID in a stack
        img = np.stack(images)
        img = np.moveaxis(img, (0, 1, 2, 3), (0, 3, 2, 1))

        if self.phase != "test":
            mask_path =  os.path.join(root_path, id_ + "_seg.nii")
            mask = self.load_img(mask_path)

            if self.is_resize:
                mask = self.resize(mask)
                # mask --> conversion to uint8 --> normalization / clipping ( 0 to 1 ) --> conversion to float32
                mask = np.clip(mask.astype(np.uint8), 0, 1).astype(np.float32)
                # again clipping ( 0 to 1 )
                mask = np.clip(mask, 0, 1)

            # setting the mask labels 1 , 2 , 4 for the mask file ( _seg.ii )
            mask = self.preprocess_mask_labels(mask)

            augmented = self.augmentations(image=img.astype(np.float32),
                                           mask=mask.astype(np.float32))
            # Several augmentations / transformations like flipping, rotating, padding will be applied to both the images
            img = augmented['image']
            mask = augmented['mask']


            return {
                "Id": id_,
                "image": img,
                "mask": mask,
            }

        return {
            "Id": id_,
            "image": img,
        }

    def load_img(self, file_path):
        data = nib.load(file_path)
        data = np.asarray(data.dataobj)
        return data

    def normalize(self, data: np.ndarray):
        data_min = np.min(data)
        # normalization = (each element - min element) / ( max - min )
        return (data - data_min) / (np.max(data) - data_min)

    def resize(self, data: np.ndarray):
        data = resize(data, (78, 120, 120), preserve_range=True)
        return data

    def preprocess_mask_labels(self, mask: np.ndarray):

        # whole tumour
        mask_WT = mask.copy()
        mask_WT[mask_WT == 1] = 1
        mask_WT[mask_WT == 2] = 1
        mask_WT[mask_WT == 4] = 1
        # include all tumours

        # NCR / NET - LABEL 1
        mask_TC = mask.copy()
        mask_TC[mask_TC == 1] = 1
        mask_TC[mask_TC == 2] = 0
        mask_TC[mask_TC == 4] = 1
        # exclude 2 / 4 labelled tumour

        # ET - LABEL 4
        mask_ET = mask.copy()
        mask_ET[mask_ET == 1] = 0
        mask_ET[mask_ET == 2] = 0
        mask_ET[mask_ET == 4] = 1
        # exclude 2 / 1 labelled tumour

        # ED - LABEL 2
        # mask_ED = mask.copy()
        # mask_ED[mask_ED == 1] = 0
        # mask_ED[mask_ED == 2] = 1
        # mask_ED[mask_ED == 4] = 0


        # mask = np.stack([mask_WT, mask_TC, mask_ET, mask_ED])
        mask = np.stack([mask_WT, mask_TC, mask_ET])
        mask = np.moveaxis(mask, (0, 1, 2, 3), (0, 3, 2, 1))

        return mask
def get_augmentations(phase):
    list_transforms = []

    # Does data augmentations & tranformation required for IMAGES & MASKS
    # they include cropping, padding, flipping , rotating
#     list_trfms = Compose(list_transforms, is_check_shapes=False)
    list_trfms = Compose(list_transforms)
    return list_trfms


def get_dataloader(
    dataset: torch.utils.data.Dataset,
    path_to_csv: str,
    phase: str,
    fold: int = 0,
    batch_size: int = 1,
    num_workers: int = 4 ):

    '''Returns: dataloader for the model training'''
    df = pd.read_csv(path_to_csv)

    # selecting train_df to be all the entries EXCEPT the mentioned fold while calling dataloader
    train_df = df.loc[df['fold'] != fold].reset_index(drop=True)

    # selection a particluar fold while calling the get_dataloader function
    val_df = df.loc[df['fold'] == fold].reset_index(drop=True)
#     test_df = df.loc[~df['Age'].notnull()].reset_index(drop=True)
#     print(len(train_df) , len(val_df), len(test_df))


    # read csv --> train & validation df splitting --> assigning train_df / val_df to df based on phase --> returning dataloader
    # how does val_df / train_df got converted to ( id , image tensor , mask tensor )

    if phase == "train" :
        df = train_df
    elif phase == "valid" :
        df = val_df
#     else:
#         df = test_df
    dataset = dataset(df, phase)
    """
    DataLoader iteratively goes through every id in the df & gets all the individual tuples for individual ids & appends all of them
    like this :
    { id : ['BraTS20_Training_235'] ,
      image : [] ,
      tensor : [] ,
    }
    { id : ['BraTS20_Training_236'] ,
      image : [] ,
      tensor : [] ,
    }
    { id : ['BraTS20_Training_237'] ,
      image : [] ,
      tensor : [] ,
    }
    """
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
#         num_workers=num_workers,
        pin_memory=True,
        shuffle=True,
    )
    return dataloader


import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import sys
sys.path.append('SemanticGenesis/pytorch')  # Add the directory to the Python path
from models.ynet3d import *


#Declare the Dice Loss
def torch_dice_coef_loss(y_true,y_pred, smooth=1.):
    y_true_f = torch.flatten(y_true)
    y_pred_f = torch.flatten(y_pred)
    intersection = torch.sum(y_true_f * y_pred_f)
    return 1. - ((2. * intersection + smooth) / (torch.sum(y_true_f) + torch.sum(y_pred_f) + smooth))

# prepare your own data
train_loader = get_dataloader(dataset=BratsDataset, path_to_csv='train_data.csv', phase='train', fold=0)
# print(enumerate(train_loader))
# prepare the 3D model

model = UNet3D()

#Load pre-trained weights
weight_dir = 'semantic_genesis_chest_ct.pt'
checkpoint = torch.load(weight_dir,  map_location=torch.device('cuda:6'))
state_dict = checkpoint['state_dict']
#Load pre-trained weights
# checkpoint = torch.load(weight_dir)
# state_dict = checkpoint['state_dict']
state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
delete = [key for key in state_dict if "projection_head" in key]
for key in delete: del state_dict[key]
delete = [key for key in state_dict if "prototypes" in key]
for key in delete: del state_dict[key]
for key in state_dict.keys():
    if key in model.state_dict().keys():
        model.state_dict()[key].copy_(state_dict[key])
        print("Copying {} <---- {}".format(key, key))
    elif key.replace("classficationNet.", "") in model.state_dict().keys():
        model.state_dict()[key.replace("classficationNet.", "")].copy_(state_dict[key])
        print("Copying {} <---- {}".format(key.replace("classficationNet.", ""), key))
    else:
        print("Key {} is not found".format(key))
device = 'cuda:6'
model.to(device)
# model = nn.DataParallel(model, device_ids = [i for i in range(torch.cuda.device_count())])
criterion = torch_dice_coef_loss
optimizer = torch.optim.SGD(model.parameters(), 0.001, momentum=0.9, weight_decay=0.0, nesterov=False)

# train the model
initial_epoch = 0
for epoch in range(initial_epoch, 70):
    #print(epoch)
    model.train()
    for id, my_tensor in enumerate(train_loader):
        x = my_tensor['image']
        x = torch.mean(x, dim=1, keepdim=True)
        dim_to_drop = 2
        num_cols = x.size(dim_to_drop)
        x = torch.narrow(x, dim_to_drop, 0, num_cols - 3)
        # print("x ", x.shape)
        y = my_tensor['mask']
        y = torch.mean(y, dim=1, keepdim=True)
        num_cols = y.size(dim_to_drop)
        y = torch.narrow(y, dim_to_drop, 0, num_cols - 3)
        x, y = x.float().to(device), y.float().to(device)
        # print("y", y.shape)
        pred = model(x)
        # print("pred", pred.shape)
        loss = criterion(pred, y)
        print(epoch, loss)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
torch.save(model.state_dict(), 'my_model.pt')
print("final loss", loss)