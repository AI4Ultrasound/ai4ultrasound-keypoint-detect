#########Imports#########

import torch
from torch import optim
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset
import os
import json
import random
import cv2
import numpy as np

import datasplitter
from AIUSDataset import AIUSDataset
import utils
from Models import HRNet_UDP


if __name__=='__main__':
    ##############Setup hyperparameters#############

    hyperparameters={
        #Keypoint display parameters
        'display_data': 'test',
        'display_return_mode': 'clip', #We want to iterate through all frames in a clip, our AIUSDataset class already does this
        'base_model_name_load': 'model_July20',

        #Datasplitter params (Should be same on what we trained with for reproducibility)
        'dataset_root': '../../../Data/Keypoint_Detect_Data',
        'train_split': 0.7,
        'val_split': 0.15,
        'test_split': 0.15,
        'k_folds': None,
        'equal_prop_tags': ['site','annotator'],
        'metadata_tags': 
        {
            'site': ['All'],
            'annotator': ['All'],
            'zone_label': ['All'],
            'patient_id': ['All'],
            'time': ['All'],
            'transducer_type': ['All'],
            'manufacturer_name': ['All'],
            'coordinate_space': 'scanline',
        },
        'random_state': 42,
        'datasplitter_verbose': True,
        'outputdata_format': 'COCO_like',

        #Dataset params (Should be same on what we trained with for reproducibility)
        'us_framesize': (128,128),
        'line_type': 'bline',
        'normalize_method': 'basic',
        'resampling_freq': None,
        'convert_to_gray': True,
        'img_augmentation': None,
        'aug_prob': 0.1,

        #Model Parameters (Should be same on what we trained with for reproducibility)
        'return_mode': 'frame',
        'heatmap_head_dropout': 0.25,
        'model_type': 'pose_hrnet_w48_udp',
        'mmpose_config': '../mmpose_model_cache/td-hm_hrnet-w48_udp-8xb32-210e_coco-256x192.py', #Set to None on first run: mim downloads the model config into the mmpose_cache_dir automatically
        'pretrained_backbone': '../mmpose_model_cache/td-hm_hrnet-w48_udp-8xb32-210e_coco-256x192-3feaef8f_20220913.pth', #Set to None on first run
        'mmpose_cache_dir': '../mmpose_model_cache',
    }
    model_name_load=f"{hyperparameters['base_model_name_load']}"

    #Category clolours per category
    GT_CAT_COLORS={utils.CLASS_ID_PLEURAL_LINE: (255,0,0),utils.CLASS_ID_B_LINE: (0,255,0)} #Blue, green
    PRED_CAT_COLORS={utils.CLASS_ID_PLEURAL_LINE: (0,0,255),utils.CLASS_ID_B_LINE: (0,255,255)} #Red, yellow


    ##############Random Seed (should match what was done in training so that we can display results on test)##############
    torch.manual_seed(hyperparameters['random_state'])
    torch.cuda.manual_seed_all(hyperparameters['random_state'])
    random.seed(hyperparameters['random_state'])
    np.random.seed(hyperparameters['random_state'])

    ######################Setup Datasets#################
    ####Split dataset (get train,val,test strings)
    train_paths,val_paths,test_paths=datasplitter.datasplitter(dataset_root=hyperparameters['dataset_root'],
                        outdata_format=hyperparameters['outputdata_format'],
                        train_split=hyperparameters['train_split'],
                        val_split=hyperparameters['val_split'],
                        test_split=hyperparameters['test_split'],
                        k_folds=hyperparameters['k_folds'],
                        equal_prop_tags=hyperparameters['equal_prop_tags'],
                        metadata_tags=hyperparameters['metadata_tags'],
                        random_state=hyperparameters['random_state'],
                        verbose=hyperparameters['datasplitter_verbose'])
    

    if hyperparameters['display_data']=='train':
       disp_data=AIUSDataset(
                json_paths=train_paths,
                us_framesize=hyperparameters['us_framesize'],
                line_type=hyperparameters['line_type'],
                normalize_method=hyperparameters['normalize_method'],
                resampling_freq=hyperparameters['resampling_freq'],
                convert_to_gray=hyperparameters['convert_to_gray'],
                img_augmentations=hyperparameters['img_augmentation'],
                aug_prob=hyperparameters['aug_prob'],
                return_mode=hyperparameters['display_return_mode'])
    elif hyperparameters['display_data']=='valid':
        disp_data=AIUSDataset(
                json_paths=val_paths,
                us_framesize=hyperparameters['us_framesize'],
                line_type=hyperparameters['line_type'],
                normalize_method=hyperparameters['normalize_method'],
                resampling_freq=hyperparameters['resampling_freq'],
                convert_to_gray=hyperparameters['convert_to_gray'],
                img_augmentations=hyperparameters['img_augmentation'],
                aug_prob=hyperparameters['aug_prob'],
                return_mode=hyperparameters['display_return_mode'])

    else: #Using testing data
        disp_data=AIUSDataset(
                        json_paths=test_paths,
                        us_framesize=hyperparameters['us_framesize'],
                        line_type=hyperparameters['line_type'],
                        normalize_method=hyperparameters['normalize_method'],
                        resampling_freq=hyperparameters['resampling_freq'],
                        convert_to_gray=hyperparameters['convert_to_gray'],
                        img_augmentations=hyperparameters['img_augmentation'],
                        aug_prob=hyperparameters['aug_prob'],
                        return_mode=hyperparameters['display_return_mode'])

    ########################Setup Device####################
    device = torch.device(0 if torch.cuda.is_available() else 'cpu')
    print('using device: '+str(device))

    #Cast the stats of the dataset objects to tensors and GPU for unnormalization
    disp_data.cast_stat_dicts_to_tensor_and_device(device)

    #############Init Model###########
    ##################Initialize model, optimizer and lr scheduler########
    #####Create model
    if hyperparameters['line_type'] in ('pleuraline','bline'):
        num_categories=1
    else:
        num_categories=2

    in_channels=1 if hyperparameters['convert_to_gray'] else 3

    model=HRNet_UDP(
        model_type=hyperparameters['model_type'],
        device=device,
        return_mode=hyperparameters['return_mode'],
        num_categories=num_categories,
        in_channels=in_channels,
        mmpose_config=hyperparameters['mmpose_config'],
        pretrained_backbone=hyperparameters['pretrained_backbone'],
        mmpose_cache_dir=hyperparameters['mmpose_cache_dir'],
        heatmap_head_dropout=hyperparameters['heatmap_head_dropout']
    ).to(device=device)

    #Load in the model
    checkpoint_path=os.path.join('../models',model_name_load+'.pt')
    checkpoint_model=torch.load(checkpoint_path,map_location=device, weights_only=False) #Use best model on validation set for testing
    model.load_state_dict(checkpoint_model['model_state_dict'])
    model=model.to(device)