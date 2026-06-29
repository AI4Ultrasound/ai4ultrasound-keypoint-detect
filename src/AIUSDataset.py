import os
import json
import random
import hashlib
from collections import defaultdict
from typing import Dict,List,Optional,Tuple,Union

import cv2
import numpy as np
import torch
from torch.utils.data.dataset import Dataset

#Init constants
from utils import CLASS_ID_B_LINE, CLASS_ID_PLEURAL_LINE
_VALID_LINE_TYPES=frozenset({'bline','pleuraline','both'})
_VALID_NORM_METHODS=frozenset({None,'basic','lin','stand','clip+lin','stand+clip','zero_back'})
_VALID_AUGMENTATIONS=frozenset({'medianblur','brightnesscontrast','gaussnoise'})
_CASH_DIR_NAME='../usframecache' #Where we store the cashed, processed, us frames for quicker loading with __getitem__


class AIUSDataset(Dataset):
    def __init__(self,json_paths,device,us_framesize=None,line_type='both',normalize_method='basic',
                 resampling_freq=None,convert_to_gray=True,img_augmentations=None):
        """
        json_paths (list of str): paths to annotation files for each clip in this data split
        us_framesize (None or tuple): resize US frame to these dimensions, or None means no resizing
        line_type (str): 'bline','pleuraline','both'
        normalize_method (None or str): how we normalize the ultrasound image. Options: 
            - None = no normalization
            - 'basic' = divide by 255
            - 'lin'= subtract each pixel by image minimum, multiple by 255/(max-min) range, so: (I-Imin)*(255/(Imax-Imin))
            - 'stand'= subtract mean & divide by std
            - 'clip+lin'= clip values higher/less than 0.995*min and 0.995*max
            - 'stand+clip'=standardize, then clip values outside +-3*std
            - 'zero_back' = (I-127.5)/127.5 (ensures black background remains zero)
        resampling_freq (None or int): Resampling frequency (samples/sec) if we want to downsample the ultrasound images
        convert_to_gray: Whether we convert the us images to gray
        img_augmentation (None or list of str): Implements clip-level augmentations to us images. Options include: 'medianblur','brightnesscontrast','gaussnoise'

        Description: init method does the following operations in following order:
            - Load in annotations from json_path
            - Loop for each annotation path, and extract corresponding ultrasound images
            - Resize images
            - Convert to gray
            - Compute sample statistics for given normalization method
            - Perform normalization and save ultrasound frames in usframecache folder where each subfolder corresponds to the json_path     
            (Augmentations are applied in __getitem__ method)   
        """ 

        #Validate the class input
        if line_type not in _VALID_LINE_TYPES:
            raise ValueError(f"line_type='{line_type}' is invalid. "
                f"Valid options: {sorted(_VALID_LINE_TYPES)}.")
        if normalize_method not in _VALID_NORM_METHODS:
            raise ValueError(f"normalize_method='{normalize_method}' is invalid. "
                f"Valid options: {sorted(str(x) for x in _VALID_NORM_METHODS)}.")
        
        #Enforce augmentation methods to a list
        if img_augmentations is None:
            img_augmentations=[]
        elif isinstance(img_augmentations,str):
            img_augmentations=[img_augmentations]
        
        #Validate image augmentations selection
        for aug in img_augmentations:
            if aug not in _VALID_AUGMENTATIONS:
                raise ValueError(
                    f"augmentation='{aug}' is invalid. "
                    f"Valid options: {sorted(_VALID_AUGMENTATIONS)}."
                )
        
        #Setup variables    
        self.json_paths=json_paths
        self.device=torch.device(device) if isinstance(device,str) else device
        self.us_framesize=us_framesize
        self.line_type=line_type
        self.normalize_method=normalize_method
        self.resampling_freq=resampling_freq
        self.convert_to_gray=convert_to_gray
        self.img_augmentations=img_augmentations

        #Convert the line type to the class ID number
        if self.line_type=='bline':
            self.class_filter={CLASS_ID_B_LINE}
        elif self.line_type=='pleuraline':
            self.class_filter={CLASS_ID_PLEURAL_LINE}
        else: #Both
            self.class_filter={CLASS_ID_B_LINE,CLASS_ID_PLEURAL_LINE}

