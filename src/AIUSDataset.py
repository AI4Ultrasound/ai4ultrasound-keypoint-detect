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
import albumentations as A

#Init constants
from utils import CLASS_ID_B_LINE, CLASS_ID_PLEURAL_LINE
_VALID_LINE_TYPES=frozenset({'bline','pleuraline','both'})
_VALID_NORM_METHODS=frozenset({None,'basic','lin_perframe','lin_global','stand','clip+lin','stand+clip','zero_back'})
_VALID_AUGMENTATIONS=frozenset({'medianblur','brightnesscontrast','gaussnoise'})
_VALID_MODE_TYPES=frozenset({'frame','clip'})
_CASH_DIR_ROOT_NAME='../../../Data/usframecache' #Where we store the cashed, processed, us frames for quicker loading with __getitem__


class AIUSDataset(Dataset):
    def __init__(self,json_paths,us_framesize=None,line_type='both',normalize_method='basic',
                 resampling_freq=None,convert_to_gray=True,img_augmentations=None,aug_prob=0.1, return_mode='frame'):
        """
        json_paths (list of str): paths to annotation files for each clip in this data split
        us_framesize (None or tuple): resize US frame to these dimensions, or None means no resizing
        line_type (str): 'bline','pleuraline','both'
        normalize_method (None or str): how we normalize the ultrasound image. Options: 
            - None = no normalization
            - 'basic' = divide by 255
            - 'lin_perframe'= subtract each pixel by image minimum, multiple by 1/(max-min) range, so: (I-Imin)*(1/(Imax-Imin))
            - 'lin_global'= subtract each pixel by global imahe minimum, multiple by 1/(max-min) range, so: (I-IGlobalmin)*(1/(IGlobalmax-IGlobalmin))
            - 'stand'= subtract mean & divide by std
            - 'clip+lin'= clip values higher/less than mean +/-3*std (global min and max) then do lin_perframe
            - 'stand+clip'=standardize, then clip values outside +-3*std
            - 'zero_back' = (I-127.5)/127.5 (ensures black background remains zero)
        resampling_freq (None or int): Resampling frequency (samples/sec) if we want to downsample the ultrasound images
        convert_to_gray: Whether we convert the us images to gray
        img_augmentation (None or list of str): Implements on-the-fly augmentations to us images during __getitem__. Options include: 'medianblur','brightnesscontrast','gaussnoise'
        aug_prob (float): Probability that a particular frame (or clip) has the augmentation applied to it. Default is 0.1 = 10% of data will have augmentation applied.
        return_mode (str): 
            - 'frame' __getitem__ returns a single (C,H,W) image + its annotations
            - 'clip' __getitem__ returns the full sequence for a clip (T,C,H,W) + all per-frame annotations. Used for RNN/temporal models. Clips have variable T
            so use the collate_fn when constructing dataloader.


        Description: init method does the following operations in following order:
            1. Load in annotations from json_path
            2. Compute configuration cache (us_framesize,normalize_method,resampling_freq,convert_to_gray), if config
            matches stored cache, all image loading is skipped
            3a. All caches valid -> recover norm_stats from disk (0 images read)
            3b. cache invalid, global stats needed, norm_stats on disk then load+preprocess+normalize+cache missing frames only
            3c. cache invalid, global stats needed, no norm_stats on disk then:
                sub-pass a: load all frames => preprocess => accumulate stats => store images to cache
                sub-pass b: load cached frames ->normalize -> store images to cache
            3d. cache invalid, no global stats needed then: load + preprocess + normalize + cache missing frames only
            4. Build a flat sample_index array for __getitem__ access
            5. augmentations on the fly in __getitem__
        """ 

        #Validate the class input
        if line_type not in _VALID_LINE_TYPES:
            raise ValueError(f"line_type='{line_type}' is invalid. "
                f"Valid options: {sorted(_VALID_LINE_TYPES)}.")
        if normalize_method not in _VALID_NORM_METHODS:
            raise ValueError(f"normalize_method='{normalize_method}' is invalid. "
                f"Valid options: {sorted(str(x) for x in _VALID_NORM_METHODS)}.")
        
        #Validate the return mode
        if return_mode not in _VALID_MODE_TYPES:
            raise ValueError(f"return_mode='{return_mode}' is invalid. "
                f"Valid options: {sorted(_VALID_MODE_TYPES)}.")
        if not 0.0 <= aug_prob <= 1.0:
            raise ValueError(f"aug_prob={aug_prob} must be in [0.0, 1.0].")
        
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
        #self.device=torch.device(device) if isinstance(device,str) else device
        self.us_framesize=us_framesize
        self.line_type=line_type
        self.normalize_method=normalize_method
        self.resampling_freq=resampling_freq
        self.convert_to_gray=convert_to_gray
        self.img_augmentations=img_augmentations
        self.return_mode=return_mode
        self.stats_for_normalization_gpu={} #This will hold stats as tensors that are used on the GPU for unormalization
        self.aug_prob=aug_prob

        #Convert the line type to the class ID number
        if self.line_type=='bline':
            self.class_filter={CLASS_ID_B_LINE}
        elif self.line_type=='pleuraline':
            self.class_filter={CLASS_ID_PLEURAL_LINE}
        else: #Both
            self.class_filter={CLASS_ID_B_LINE,CLASS_ID_PLEURAL_LINE}

        #Cache the configuration parameters
        _cfg = {
            'us_framesize':    str(us_framesize),
            'normalize_method': str(normalize_method),
            'resampling_freq': str(resampling_freq),
            'convert_to_gray': str(convert_to_gray),
        }

        self._cache_cfg_hash = hashlib.md5(
            json.dumps(_cfg, sort_keys=True).encode()
        ).hexdigest()[:12]

        #These are containers populated by _build_dataset
        self.sample_index: List[Dict]=[] #This is a flat list, one dict per included US frame
        self.norm_stats: Dict={} #Dataset-level normalization statistics

        #Initialize the augmentation pipeline
        self._aug_pipeline=self._build_augmentation_pipeline() #Sets up the augmentation pipeline

        self._build_dataset() #Run the full preprocessing pipeline



    #####################Preprocessing pipeline#################
    def _build_dataset(self):
        """
        Three-pass pipeline:
            Pass 1: Parse all the JSON annotations into clip_records (per-clip)
            Pass 2: Image loading:
                - If all the frames cahce are valid (we haven't changed the cache configuration parameters) we recover the norm_stats from disk
                - Else, some/all frames need caching, and the global stats are needed. Then the norm_stats are already on disk from an interrupted
                previous run, and we load/normalize/cache only the missing frames, or the norm_stats are not on disk (first/full rebuild), then:
                    - Sub-pass A: load all the frames, preprocess => accumulate the stats => save image to disk => compute norm_stats
                    - Sub-pass B: read in frames => normalize => re-cache the farme
                - If frames need caching, but global sats are not needed, then we load => preprocess => normalize => cache 
            Pass 3: _index_clip() builds sample_index from cache paths
        """

        #Pass 1: Parse the annotation json's
        clip_records=[] #Holds the annotations from the json's
        for jp in self.json_paths:
            rec=self._parse_clip(jp) #Loads in the json file
            if rec is not None:
                clip_records.append(rec)
        
        if not clip_records: #No JSON annotations
            raise ValueError(
                "No valid clips found in the provided json_paths. "
                "Check that the paths exist and that annotations are non-empty "
                "for the requested line_type."
            )

        #Pass 2: build/verify frame cache
        needs_global_stats=self.normalize_method in ('stand','lin_global','stand+clip','clip+lin')

        if self._all_caches_valid(clip_records):
            #We already have correct cache configuration and us iamges stored in cache
            print("[AIUSDataset] All frame caches valid — skipping image loading.")
            if needs_global_stats:
                self.norm_stats=self._load_norm_stats_from_cache(clip_records[0]['json_path'])
                if not self.norm_stats:
                    #All .npy files are present, but the norm_stats file is deleted
                    self.norm_stats=self._compute_global_stats(clip_records)
                    self._save_norm_stats_to_cache(clip_records[0]['json_path'])
        else:
            #We don't have correct cache configuration, must load in images
            if needs_global_stats:
                saved_stats=self._load_norm_stats_from_cache(clip_records[0]['json_path'])
                if saved_stats:
                    #The norm_stats are on disk, but there was an interruption, 
                    print("[AIUSDataset] Resuming interrupted cache build "
                          "(norm_stats recovered from disk).")
                    self.norm_stats=saved_stats
                    for rec in clip_records:
                        self._process_and_cache_clip(rec) #Processes the clip and caches it
                else:
                    #First/full rebuild
                    #sub-pass A: PNG => preprocessing => stats => save preprocessed frames to _raw.npy
                    self.norm_stats=self._stats_pass_and_cache_raw(clip_records)

                    #Save the norm_stats incase sub-pass A is interrupted
                    self._save_norm_stats_to_cache(clip_records[0]['json_path'])
                    
                    #sub-pass B: _raw.npy -> normalize -> final .npy -> delete _raw.npy
                    for rec in clip_records:
                        self._normalize_and_recache_clip(rec)
            else:
                #No global stats are needed, 
                for rec in clip_records:
                    self._process_and_cache_clip(rec)
        #Pass 3, buil the flat sample_index for rapid _getitem_
        for rec in clip_records:
            self._index_clip(rec)
        if not self.sample_index:
            raise ValueError(
                "No samples remain after filtering. Verify that annotations of "
                "the requested line_type exist in the provided clips."
            )

        n_clips  = len(clip_records)
        n_frames = len(self.sample_index)
        print(
            f"[AIUSDataset] Ready — {n_frames} frames from {n_clips} clips "
            f"| line_type='{self.line_type}' "
            f"| normalize='{self.normalize_method}' "
            f"| augmentations={self.img_augmentations}"
        )
    
    
    def _parse_clip(self,json_path):
        """
        Reads in the JSON file, applies temporal resampling and line_type filtering (b-line or pleural line).
        Returns a clip record dict, or none if the file is invalid 
        """ 

        #Read in the json file
        try:
            with open(json_path, 'r', encoding='utf-8') as fh:
                data = json.load(fh)
        except Exception as exc:
            print(f"[AIUSDataset] Warning: could not read '{json_path}': {exc}")
            return None
        
        #Read in the meta_data
        meta_list = data.get('metadata', [])
        if not meta_list:
            print(f"[AIUSDataset] Warning: no metadata block in "
                  f"'{json_path}' — skipping.")
            return None
        
        #Extracts first dictionary of metadata (only one dict in metadata)
        clip_meta=meta_list[0]
        raw_freq=clip_meta.get('sampling_rate')
        original_sampling_freq=float(clip_meta.get('sampling_rate')) #original sampling rate in (milliseconds/frame)
        original_sampling_freq = 1 / (float(raw_freq) / 1000) if raw_freq is not None else None #original sampling rate in frames/second

        #Build frame_num => image-info lookup
        images_by_frame: Dict[int,Dict] ={
            img['frame_num']: img for img in data.get('images',[])
        }

        #Get the annotations with corresponding frame_num
        anns_by_frame: Dict[int,List[Dict]]=defaultdict(list)
        for ann in data.get('annotations',[]):
            anns_by_frame[ann['frame_num']].append(ann)
        
        #Determine which frame numbers to consider (temporal resampling)
        all_frame_nums=sorted(images_by_frame.keys())
        if self.resampling_freq is not None: #resampling_freq is in frames/sec (samples/sec)
            #We are re-sampling the ultrasound frames
            if original_sampling_freq is not None:
                all_frame_nums=self._resample_frame_indices(all_frame_nums,original_sampling_freq) #Resamples the ultrasound frames based on current sampling rate
            else:
                print(
                    f"[AIUSDataset] Warning: resampling_freq is set but "
                    f"no sampling_rate found in "
                    f"'{os.path.basename(json_path)}' metadata — "
                    f"skipping temporal resampling for this clip."
                )
        
        #Build per-frame entries and filter by line_type
        frame_entries=[]
        for fn in all_frame_nums:
            img_info=images_by_frame.get(fn) #Get the image info
            if img_info is None: #Fallback
                continue
            filtered_anns=[a for a in anns_by_frame.get(fn,[]) if a['category_id'] in self.class_filter] #Get annotations only for specific line type anf frame number

            if not filtered_anns:
                continue #Frame has no annotations for requested category type

            #Append to the frame entries list
            frame_entries.append({
                'frame_num': fn,
                'file_name':   img_info['file_name'],
                'height':      img_info['height'],
                'width':       img_info['width'],
                'px_mul_x':    img_info.get('px_mul_x', 1.0),
                'px_mul_y':    img_info.get('px_mul_y', 1.0),
                'annotations': filtered_anns,
            })
        
        #Checks that frame_entries is not empty
        if not frame_entries:
            return None
        
        #Extracts the file name
        stem=os.path.splitext(os.path.basename(json_path))[0]
        
        return {
            'json_path':     json_path,
            'stem':          stem,
            'clip_metadata': clip_meta,
            'frame_entries': frame_entries,
        }
    
    def _resample_frame_indices(self,frame_nums,original_freq):
        """
        subsamples frame_nums by self.resampling_freq if resampling_freq<original_freq 
        !!!Both original_freq and self.resampling_freq must be in samples (frames)/second
        """
        if self.resampling_freq>=original_freq or original_freq<=0:
            return frame_nums
        step=original_freq/self.resampling_freq #How many samples to step by
        selected,t,n=[],0.0,len(frame_nums) #Init variables

        while t<n: #Lopos while the time is smaller than the number of frames
            idx=int(round(t)) #round new sample index to integer
            if idx < n:
                selected.append(frame_nums[idx])
            t += step
        return selected

    def _load_and_preprocess_frame(self,file_name):
        """
        Loads a PNG from the disk -> then does colour-space conversion -> then resizes
        Returns a float32 numpy array in [0,255]        
        This is the un-normalized intermediate used for stats accumulation and raw caching
        """
        img = cv2.imread(file_name,cv2.IMREAD_UNCHANGED)

        if img is None: #Check that image exists
            raise FileNotFoundError(f"Cannot load image: '{file_name}'") 

        if self.convert_to_gray: #Convert the image to grayscale
            if img.ndim == 3 and img.shape[2] == 3:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            elif img.ndim == 3 and img.shape[2] == 4: #We have the opacity channel as well
                img = cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY)
            # Already 2-D → leave as is
        else: #Use 3-channel RGB image
            if img.ndim == 2:
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
            elif img.shape[2] == 4:
                img = cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)
            else:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        if self.us_framesize is not None:
            #Reshape the image if we have that set
            H,W=self.us_framesize
            img_height,img_width=img.shape[:2]
            if H!=img_height or W!=img_width: #Check that we actually have to resize the image
                img = cv2.resize(img, (W, H), interpolation=cv2.INTER_LINEAR) #Resize
            
        return img.astype(np.float32) 
    

    ##########Statistics Handling Functions#########
    def _accumulate_stats(self,img,acc):
        #Flatten the image
        flat=img.ravel().astype(np.float32)

        #Update the accumulator
        acc['sum']+=float(flat.sum()) #The sum of pixel values
        acc['sum_sq']+=float((flat**2).sum()) #Sum of squared pixel values
        acc['count']+=flat.size #Number of pixels per frame
        acc['min']=min(acc['min'],float(flat.min()))
        acc['max']=max(acc['max'],float(flat.max()))
        return acc
    
    def _compute_stats_from_accumulator(self,acc):
        mean_val=acc['sum']/acc['count']
        std_val=float(np.sqrt(np.maximum(acc['sum_sq']/acc['count']-(mean_val**2),0.0)))
        stats={'mean': float(mean_val), 'std': std_val,'min': float(acc['min']), 'max': float(acc['max'])}

        return stats

    def _compute_global_stats(self,clip_records):
        """
        Compute dataset-level stats (mean,std,min,max) by streaming all preprocessed frames through an accumulator.  
        
        Used ONLY for the edge case where all frame caches are valid but the
        norm_stats file was manually deleted.  In a normal first/full rebuild
        _stats_pass_and_cache_raw is used instead.
        """

        #Initialize the accumulator dictionary
        acc = {'sum': 0.0, 'sum_sq': 0.0, 'count': 0,'min': np.inf, 'max': -np.inf}

        #Loop for each of the clips:
        for rec in clip_records:
            for entry in rec['frame_entries']:
                try:
                    img = self._load_and_preprocess_frame(entry['file_name'])
                except FileNotFoundError as exc:
                    print(f"[AIUSDataset] Warning: {exc} — skipping for stats.")
                    continue
                #Accumulate stats for this image
                acc=self._accumulate_stats(img,acc)
        
        if acc['count']==0: #Checks that we have more than 0 pixels
            raise ValueError("No pixels accumulated — check that all image files exist.")
        
        stats=self._compute_stats_from_accumulator(acc)

        return stats
    
    def _stats_pass_and_cache_raw(self,clip_records):
        """
        Sub-pass A for first/full rebuild: load all frames => preprocess => accumulate stats => store images to cache
        """

        #Initialize the accumulator dictionary
        acc = {'sum': 0.0, 'sum_sq': 0.0, 'count': 0,'min': np.inf, 'max': -np.inf}

        #Loop for each of the clips:
        for rec in clip_records:
            #Create a cache directory
            cache_dir=self._get_cache_dir(rec['json_path'],rec['stem'])

            #Loop for each of the frames
            for entry in rec['frame_entries']:
                fn=entry['frame_num']
                raw_cache_path=os.path.join(cache_dir,f'frame_{fn:05d}_raw.npy') #Where we save the raw image before normalization

                #Load the US frame:
                try:
                    img = self._load_and_preprocess_frame(entry['file_name'])
                except FileNotFoundError as exc:
                    print(f"[AIUSDataset] Warning: {exc} — skipping frame.")
                    continue
                
                #Accumulate the statistics
                acc=self._accumulate_stats(img,acc)

                #Save the unormlized frame to raw cache and release
                self._array_to_npy_and_save(img,raw_cache_path)
        
        if acc['count']==0: #Checks that we have more than 0 pixels
            raise ValueError("No pixels accumulated — check that all image files exist.")
        
        #Compute the stats
        norm_stats=self._compute_stats_from_accumulator(acc)

        return norm_stats
    
    def _normalize_and_recache_clip(self,record):
        """
        sub-pass B: _raw.npy -> normalize -> final .npy -> delete _raw.npy
        This is called a clip record
        No PNG reads — all data is read from the raw disk cache written by
        Sub-pass A.  Writes cache_config.json after processing the clip.
        Requires self.norm_stats to be populated for global-stats methods.
        """
        cache_dir = self._get_cache_dir(record['json_path'], record['stem'])
        wrote_any = False #Boolean to check that we wrote some cache

        for entry in record['frame_entries']:
            fn=entry['frame_num']
            raw_cache_path=os.path.join(cache_dir,f'frame_{fn:05d}_raw.npy')

            cache_path=os.path.join(cache_dir, f'frame_{fn:05d}.npy') #Where we store the final ultrasound image

            #Checks that the raw file exists:
            if not os.path.isfile(raw_cache_path):
                print(f"[AIUSDataset] Warning: expected raw cache not found for "
                      f"frame {fn} of '{record['stem']}' — skipping.")
                continue

            #Load the raw (un-normalized) tensor
            img_np=self._load_frame_np(raw_cache_path)
            if img_np is None: #Check that exists
                continue
            img_np=self._array_channelconvert(img_np) #Gets it as a numpy image
            img_norm = self._normalize_frame(img_np) #Normalizes the US frame
            self._array_to_npy_and_save(img_norm, cache_path) #Saves the image

            #Remove the intermediate raw file
            os.remove(raw_cache_path)
            wrote_any=True
        
        if wrote_any:
            self._write_cache_config(cache_dir) #Write the cache configuration

    def _normalize_frame(self,img):
        """
        Normalize a float32 frame (values in [0,255]) according to the self.normalize_method
        """
        if self.normalize_method is None: #Don't normalize the image
            return img

        if self.normalize_method == 'basic':
            return (img / 255.0).astype(np.float32) #Divide image by 255

        #(I-Imin)/(Imax-Imin)
        if self.normalize_method == 'lin_perframe':
            imin, imax = float(img.min()), float(img.max())
            denom = imax - imin
            if denom < 1e-8:
                return np.zeros_like(img)
            return ((img - imin) / denom).astype(np.float32)
        
        if self.normalize_method == 'lin_global':
            imin, imax = self.norm_stats['min'], self.norm_stats['max']
            denom = imax - imin
            if denom < 1e-8:
                return np.zeros_like(img)
            return ((img - imin) / denom).astype(np.float32)

        if self.normalize_method == 'stand':
            mean_val, std_val = self.norm_stats['mean'], self.norm_stats['std']
            if std_val < 1e-8:
                return np.zeros_like(img)
            return ((img - mean_val) / std_val).astype(np.float32)

        if self.normalize_method == 'clip+lin':
            # Dataset-level clipping bounds ensure consistent scaling across all frames
            mean_val = self.norm_stats['mean']
            std_val  = self.norm_stats['std']
            lo   = mean_val - 3.0 * std_val
            hi   = mean_val + 3.0 * std_val
            img  = np.clip(img, lo, hi)
            denom = hi - lo
            if denom < 1e-8:
                return np.zeros_like(img)
            return ((img - lo) / denom).astype(np.float32)

        if self.normalize_method == 'stand+clip':
            mean_val, std_val = self.norm_stats['mean'], self.norm_stats['std']
            if std_val < 1e-8:
                return np.zeros_like(img)
            return np.clip(((img - mean_val) / std_val), -3.0, 3.0).astype(np.float32)

        if self.normalize_method == 'zero_back':
            return ((img - 127.5) / 127.5).astype(np.float32)

        return img  # Unreachable after __init__
    
    def cast_stat_dicts_to_tensor_and_device(self,device):
        #This casts the dict used for unormalization to the device and converts to tensor

        #Second number is default values in case that the norm_stats are empty
        mean_val=self.norm_stats.get('mean',0.0)
        std_val=self.norm_stats.get('std',1.0)
        max_val=self.norm_stats.get('max',255.0)
        min_val=self.norm_stats.get('min',0.0) 

        #Converts the values to tensors and casts to device
        mean_val=torch.tensor(mean_val,dtype=torch.float32,device=device)
        std_val=torch.tensor(std_val,dtype=torch.float32,device=device)
        max_val=torch.tensor(max_val,dtype=torch.float32,device=device)
        min_val=torch.tensor(min_val,dtype=torch.float32,device=device)

        self.stats_for_normalization_gpu={
            'mean':mean_val,
            'std':std_val,
            'max':max_val,
            'min':min_val
        }
    
    def unnormalize_image(self,image_tensor,is_gpu=True,return_oncpu=False):
        """
        Unnormalizes the image applied during preprocessing to product a uint8 numpy array.
        is_gpu: Whether we want to do processing on the gpu (image_tensor is always on the GPU). If true, should call cast_stats_to_tensor_and_device first to move the stats to the GPU.
        return_oncpu: Whether to return the unnormalized image on the CPU (as a numpy array) or on the GPU (as a tensor)

        Note: 'lin_perframe', the per-image min/max are not stored at dataset level. So not the exact pixel values are recovered.
        Similarly, 'stand+clip' loses information because of clip.
        Similarly, 'clip+lin' loses information because of clip.
        """
        if not is_gpu: #Run processing on the CPU
            mean_val=self.norm_stats['mean']
            std_val=self.norm_stats['std']
            max_val=self.norm_stats['max']
            min_val=self.norm_stats['min']
            img=image_tensor.detach().cpu().float().numpy() #Move the image tensor to the CPU and convert to numpy array, and detach from the computation graph
        else: #Run processing on the GPU
            mean_val=self.stats_for_normalization_gpu['mean']
            std_val=self.stats_for_normalization_gpu['std']
            max_val=self.stats_for_normalization_gpu['max']
            min_val=self.stats_for_normalization_gpu['min']
            img=image_tensor.detach().float() #Keep the image tensor on the GPU, detach from the computaiton graph and convert to float
        
        #Unnormalize image based on normalization method
        img_unnorm=None
        if self.normalize_method is None: #We did no normalization
            img_unnorm=img
        
        elif self.normalize_method == 'basic':
            img_unnorm=img*255.0
        
        elif self.normalize_method == 'lin_perframe': #Cannot recover original pixel values because we don't know original min/max for each frame
            img_unnorm=img*255.0

        elif self.normalize_method == 'lin_global':
            img_unnorm=img*(max_val-min_val)+min_val
        
        elif self.normalize_method == 'stand':
            img_unnorm=img*std_val+mean_val
        
        elif self.normalize_method == 'clip+lin': #Cannot recover original pixel values because clamp was applied previously
            lo=mean_val-3.0*std_val
            hi=mean_val+3.0*std_val
            img_unnorm=img*(hi-lo)+lo

        elif self.normalize_method == 'stand+clip': #Cannot recover original pixel values because clamp was applied previously
            img_unnorm=img*std_val+mean_val
        
        elif self.normalize_method == 'zero_back':
            img_unnorm=img*127.5+127.5
        else:
            img_unnorm=img
        
        #Clip the image to [0,255] and convert to uint8
        if is_gpu: #Process on the GPU with torch
            img_unnorm=torch.clamp(img_unnorm,0.0,255.0).to(torch.uint8) #Clamps the image to [0,255] and converts to uint8
            #Remove channel dimension for display
            if img_unnorm.ndim==4: #If we have clip dimension then (T,C,H,W)
                if img_unnorm.shape[1]==1: #If we have a channel dimension of 1, remove it
                    img_unnorm=img_unnorm.squeeze(1) #Remove channel dimension for display
                else: #Colour image
                    img_unnorm=img_unnorm.permute(0,2,3,1) #Move channel dimension to the end for display
            elif img_unnorm.ndim==3:            
                if img_unnorm.shape[0]==1: #If we have a channel dimension of 1, remove it
                    img_unnorm=img_unnorm.squeeze(0) #Remove channel dimension for display
                else: #Colour image
                    img_unnorm=img_unnorm.permute(1,2,0) #Move channel dimension to the end for display
            
            if return_oncpu: #Case where processing on the GPU, but move the image to the CPU and convert to numpy array
                img_unnorm=img_unnorm.detach().cpu().numpy() #Converts to cpu

        else: #Process on the CPU with numpy
            img_unnorm=np.clip(img_unnorm,0.0,255.0).astype(np.uint8) #Clamps the image to [0,255] and converts to uint8  
            # Remove channel dimension for display
            if img_unnorm.ndim == 4: # clip (T,C,H,W)
                if img_unnorm.shape[1] == 1:
                    img_unnorm = img_unnorm.squeeze(axis=1)          # (T, H, W)
                else:
                    img_unnorm = np.moveaxis(img_unnorm, 1, -1) # (T, H, W, C)
            elif img_unnorm.ndim == 3: # Single image (C,H,W)
                if img_unnorm.shape[0] == 1:
                    img_unnorm = img_unnorm[0]                    # (H, W)
                else:
                    img_unnorm = np.moveaxis(img_unnorm, 0, -1)   # (H, W, C)      

        
        return img_unnorm
        

    
    ###############Array Conversion Helpers################

    def _array_to_npy_and_save(self,img,path):
        """
        Converts a (H,W) or (H,W,C) numpy array to correct dimensions (based on colour conversion) and saves it to path.
        Works for un-normalized (_raw.npy) or normalized (.npy) images.  Saves as float32.
        """
        if self.convert_to_gray:
            img_array=img[np.newaxis,:,:] #Add a channel dimension for grayscale (1,H,W)
        else:
            img_array=np.moveaxis(img,-1,0) #Move channel dimension to front for RGB (3,H,W)
        
        np.save(path,img_array.astype(np.float32)) #Save the image array as float32
    
    def _tensor_to_array(self, tensor):
        """
        Convert a (C, H, W) float32 tensor back to a (H, W) or (H, W, 3)
        float32 numpy array for normalization or display.
        """
        if self.convert_to_gray:
            return tensor.squeeze(0).numpy()          # (H, W)
        else:
            return tensor.permute(1, 2, 0).numpy()    # (H, W, 3)
    
    def _array_channelconvert(self, array):
        """
        Convert a (C, H, W) float32 tensor back to a (H, W) or (H, W, 3)
        float32 numpy array for normalization or display.
        """
        if self.convert_to_gray:
            return array.squeeze(0)          # (H, W)
        else:
            return np.moveaxis(array,0,-1)    # (H, W, 3)
    
    def _load_frame_tensor(self, path):
        """
        Load a cached .npy file from *path* as a torch tensor (C,H,W).
        Returns None and prints a warning on failure.
        """
        try:
            img_array = np.load(path)                        # (C, H, W) float32
            return torch.tensor(img_array).clone()       # clone → tensor owns memory
        except Exception as exc:
            print(f"[AIUSDataset] Warning: could not load '{path}': {exc}")
            return None
    
    def _load_frame_np(self,path):
        """
        Load a cached .npy file from *path* as a numpy tensor (C,H,W).
        Returns None and prints a warning on failure.
        """
        try:
            img_array = np.load(path)                        # (C, H, W) float32
            return img_array.copy()       # Return image array
        except Exception as exc:
            print(f"[AIUSDataset] Warning: could not load '{path}': {exc}")
            return None
    
    ################Cache Interface Helpers################
    def _get_cache_dir(self,json_path,stem):
        """
        Create per-clip cache directory with the following structure:
        _USCACHE_ROOT/<coord_space>/<stem>/ 
        Where <coord_space> is 'sector' or 'scanline' and <stem> is the base name of the json file (<annotator>_<clip_hash>) with annotations
        """
        annotation_dir=os.path.dirname(os.path.abspath(json_path)) #Gets the path of the json file
        coord_space=os.path.basename(annotation_dir) #Gets the name of the directory, which is either 'sector' or 'scanline'
        cache_dir=os.path.join(_CASH_DIR_ROOT_NAME,coord_space,stem) #Creates the cache directory path
        os.makedirs(cache_dir,exist_ok=True) #Creates the cache directory if it doesn't exist
        return cache_dir
    
    def _cache_is_valid(self,cache_dir):
        """
        Checks if the stored config matches the current configuration for the ultrasound cache, so that we don't need to
        reload and preprocess the images if they are already cached with the same configuration.
        """
        cfg_path=os.path.join(cache_dir,'cache_config.json')
        if not os.path.isfile(cfg_path):
            return False
        
        try:
            with open(cfg_path,'r') as fh:
                return json.load(fh).get('hash') == self._cache_cfg_hash
        except Exception:
            return False
        
    def _write_cache_config(self,cache_dir):
        """
        Write the current config hash to the cache_dir/cache_config.json
        """
        cfg_path=os.path.join(cache_dir,'cache_config.json')
        with open(cfg_path,'w') as fh:
            json.dump({'hash': self._cache_cfg_hash},fh)
    
    def _all_caches_valid(self,clip_records):
        """
        Returns true only when every frame across all clips has a valid final cache .npy file.
        """
        for rec in clip_records:
            cache_dir=self._get_cache_dir(rec['json_path'],rec['stem'])
            if not self._cache_is_valid(cache_dir):
                return False
            for entry in rec['frame_entries']:
                fn=entry['frame_num']
                cache_path=os.path.join(cache_dir,f'frame_{fn:05d}.npy')
                if not os.path.isfile(cache_path):
                    return False
        return True

    def _process_and_cache_clip(self,record):
        """
        Cache missing frames for one clip. Used for:
            - No global stats case (normalize_method not in global stats group), so we don't need to compute global stats
            - Interrupted sub-pass B (norm_stats in cache, some frames are missing)
        
        Per-frame priority order:
          1. frame_{fn:05d}.npy valid → skip.
             Clean up any orphaned _raw.npy left by an interrupted Sub-pass B.
          2. frame_{fn:05d}_raw.npy exists → load raw tensor → normalize →
             save final .npy → delete _raw.npy.
          3. Neither cached → load PNG → preprocess → normalize → save final .npy.

        Writes cache_config.json if any frame was written.
        """
        cache_dir=self._get_cache_dir(record['json_path'],record['stem']) #Gets the cache directory for this clip
        cache_valid=self._cache_is_valid(cache_dir) #Checks if the cache directory is valid
        wrote_any=False #Boolean to check that we wrote some cache

        for entry in record['frame_entries']:
            fn=entry['frame_num']
            cache_path=os.path.join(cache_dir,f'frame_{fn:05d}.npy') #Where we store the final ultrasound image
            raw_cache_path=os.path.join(cache_dir,f'frame_{fn:05d}_raw.npy') #Where we save the raw image before normalization

            #Priority 1: final .npy already cached
            if cache_valid and os.path.isfile(cache_path):
                #Clean up any orphaned _raw.npy left by an interrupted Sub-pass B
                if os.path.isfile(raw_cache_path):
                    os.remove(raw_cache_path)

                continue #Skip to next frame

            #Priority 2: un-normalized _raw.npy exists, so we can normalize now
            if os.path.isfile(raw_cache_path):
                img_np=self._load_frame_np(raw_cache_path)
                if img_np is not None:
                    img_np=self._array_channelconvert(img_np) #Gets it as a numpy image
                    img_norm=self._normalize_frame(img_np) #Normalizes the US frame
                    self._array_to_npy_and_save(img_norm,cache_path) #Saves the image
                    os.remove(raw_cache_path) #Remove the intermediate raw file
                    wrote_any=True
                    continue
            
            #Priority 3: neither cached, so we load from original PNG
            try:
                img=self._load_and_preprocess_frame(entry['file_name']) #Load the US frame and preprocess it
            except FileNotFoundError as exc:
                print(f"[AIUSDataset] Warning: {exc} — skipping frame.")
                continue

            img_norm=self._normalize_frame(img) #Normalizes the US frame
            self._array_to_npy_and_save(img_norm,cache_path) #Saves the image
            wrote_any=True

        if wrote_any:
            self._write_cache_config(cache_dir) #Write the cache configuration file
    

    def _index_clip(self,record):
        """
        Append entries to the self.sample_index list
        mode='frame' => One entry per annotated frame
        mode='clip' => one entry per clip with all frames and annotations
        """
        cache_dir=self._get_cache_dir(record['json_path'],record['stem']) #Gets the cache directory for this clip

        if self.return_mode=='frame':
            #Frame level, so we append individual entries for each frame in the clip

            #Loops for all the entries (frames) in the clip
            for entry in record['frame_entries']:
                fn=entry['frame_num']
                cache_path=os.path.join(cache_dir,f'frame_{fn:05d}.npy') #Where we store the final ultrasound image
                if not os.path.isfile(cache_path):
                    continue
                
                #Store keypoints and categories as tensors for quicker extraction in _getitem_
                keypoints  = [
                    torch.tensor(np.array(a['keypoints'], dtype=np.float32))
                    for a in entry['annotations']
                ]
                categories = torch.tensor([a['category_id'] for a in entry['annotations']],dtype=torch.long) 

                #Saves the results, sample_index is the main object that we call from in _getitem_
                self.sample_index.append({
                    'cache_path':    cache_path,
                    'keypoints':     keypoints,   # List[ndarray (N_i, 2)] — variable length
                    'categories':    categories,  # List[int]
                    'frame_num':     fn,
                    'px_mul_x':      entry['px_mul_x'],
                    'px_mul_y':      entry['px_mul_y'],
                    'clip_id':       record['stem'],
                    'clip_metadata': record['clip_metadata'],
                })

        else:
            #clip level, so we collect all valid frames and append as a single entry to sample_index

            #Init the lists to hold the per-frame data for this clip
            clip_cache_paths = []
            clip_keypoints   = []
            clip_categories  = []
            clip_frame_nums  = []
            clip_px_mul_x    = []
            clip_px_mul_y    = []

            #Loop for each frame and append to accumulator lists
            for entry in record['frame_entries']:
                fn=entry['frame_num']
                cache_path=os.path.join(cache_dir,f'frame_{fn:05d}.npy') #Where we store the final ultrasound image
                if not os.path.isfile(cache_path):
                    continue
                keypoints  = [
                    torch.tensor(np.array(a['keypoints'], dtype=np.float32))
                    for a in entry['annotations']
                ]
                categories = torch.tensor([a['category_id'] for a in entry['annotations']],dtype=torch.long)

                clip_cache_paths.append(cache_path)
                clip_keypoints.append(keypoints)
                clip_categories.append(categories)
                clip_frame_nums.append(fn)
                clip_px_mul_x.append(entry['px_mul_x'])
                clip_px_mul_y.append(entry['px_mul_y'])

            #Append the clip-level entry to sample_index
            if not clip_cache_paths:
                return #No valid frames for this clip
            
            #T=number of frames in clip
            self.sample_index.append({
            'cache_paths':   clip_cache_paths,   # List[str] length T
            'keypoints':     clip_keypoints,     # List[List[ndarray]] (T, K_t, N_i, 2)
            'categories':    clip_categories,    # List[List[int]] (T, K_t)
            'frame_nums':    clip_frame_nums,    # List[int] length T
            'px_mul_x':      clip_px_mul_x,      # List[float] length T
            'px_mul_y':      clip_px_mul_y,      # List[float] length T
            'clip_id':       record['stem'],
            'clip_metadata': record['clip_metadata'],
            })

    
    ##################Norm Stats Cache Helpers################
    def _norm_stats_cache_path(self,json_path):
        """
        Returns the path for the dataset-level norm_stats JSON file.
        Stored at: _CASH_DIR_ROOT_NAME/<coord_space>/norm_stats_<hash>.json
        Where <coord_space> is 'sector' or 'scanline' and <hash> is the first 12 characters of the md5 hash of the current configuration.
        """
        annotation_dir=os.path.dirname(os.path.abspath(json_path)) #Gets the path of the json file
        coord_space=os.path.basename(annotation_dir) #Gets the name of the subdirectory one level up, which is either 'sector' or 'scanline'
        cache_dir=os.path.join(_CASH_DIR_ROOT_NAME,coord_space) #Creates the cache directory path
        os.makedirs(cache_dir,exist_ok=True) #Creates the cache directory if it doesn't exist
        return os.path.join(cache_dir,f'norm_stats_{self._cache_cfg_hash}.json')
    
    def _save_norm_stats_to_cache(self,json_path):
        """
        Saves the dataset-level norm_stats to a json file cache directory
        """
        norm_path=self._norm_stats_cache_path(json_path)
        with open(norm_path, 'w', encoding='utf-8') as fh:
            json.dump(self.norm_stats, fh, indent=2)
    
    def _load_norm_stats_from_cache(self,json_path):
        """
        Loads the dataset-level norm stats from the cached JSON file.
        Rerturns {} if absent/unreadable, signalling to recompute the norm stats for this configuration
        """
        norm_path=self._norm_stats_cache_path(json_path)
        if not os.path.isfile(norm_path):
            return {}
        
        try:
            with open(norm_path, 'r', encoding='utf-8') as fh:
                return json.load(fh)
        except Exception as exc:
            print(f"[AIUSDataset] Warning: could not read norm_stats cache "
                  f"'{norm_path}': {exc}")
            return {}
        
        

    

    ####################Dataset Interface####################

    def __len__(self):
        return len(self.sample_index)
    
    def __getitem__(self, idx):
        """
        if return_mode=='frame' => returns a single frame dict with:
            'image'      : FloatTensor (C, H, W)
            'keypoints'  : list[FloatTensor (N_i, 2)]
            'categories' : LongTensor (K,)
            'frame_num'  : int
            'px_mul_x'   : float
            'px_mul_y'   : float
            'clip_id'    : str
            'metadata'   : dict

        if return_mode=='clip' => returns a single clip dict with:
            'images'     : FloatTensor (T, C, H, W)  — padded if using collate_fn
            'keypoints'  : list[list[FloatTensor (N_i, 2)]]  — shape (T, K_t)
            'categories' : list[LongTensor (K_t,)]  — one tensor per frame
            'frame_nums' : list[int] length T
            'px_mul_x'   : list[float] length T
            'px_mul_y'   : list[float] length T
            'clip_len'   : int  — actual (un-padded) number of frames T
            'clip_id'    : str
            'metadata'   : dict

        T=number of frames in the clip, K_t=number of annotations in frame t, N_i=number of keypoints in annotation i
        !!!!!!!!!!!!No tensors are on device, handle this in model trainer!!!!!!!!!!
        """
        if self.return_mode == 'frame':
            return self._getitem_frame(idx)
        else:
            return self._getitem_clip(idx)
        
    
    def _getitem_frame(self,idx):
        """
        Returns a single annotated frame
        """
        sample=self.sample_index[idx] #Gets the current sample

        image=self._load_frame_tensor(sample['cache_path']) #Loads the image from cache

        if image is None:
            raise RuntimeError(
            f"Could not load cached frame '{sample['cache_path']}'."
            )

        if self.img_augmentations:
            image=self._apply_augmentations(image) #Applies augmentations to this image on the fly

        #Returns the dictionary with the frame data
        return {
        'image':      image,
        'keypoints':  sample['keypoints'],
        'categories': sample['categories'],
        'frame_num':  sample['frame_num'],
        'px_mul_x':   sample['px_mul_x'],
        'px_mul_y':   sample['px_mul_y'],
        'clip_id':    sample['clip_id'],
        'metadata':   sample['clip_metadata'],
        }
    
    def _getitem_clip(self,idx):
        """
        Returns the full clip as (T,C,H,W) tensor + per-frame annotations
        """
        sample=self.sample_index[idx] #Gets the current sample

        images=[]
        replay_params=None #Sample the augmentations once at the start of the clip, replay augmentation on rest of clip
        for cache_path in sample['cache_paths']:
            img=self._load_frame_tensor(cache_path) #Loads the image from cache
            if img is None:
                raise RuntimeError(
                    f"Could not load cached frame '{cache_path}'."
                )
            if self.img_augmentations:
                img,replay_params=self._apply_augmentations_cliplevel(img,replay_params) #Applies the augmentations at a clip level

            images.append(img)
        
        images=torch.stack(images,dim=0) #Stacks the images into a (T,C,H,W) tensor

        #Returns the dictionary with the clip data
        return {
            'images':     images,
            'keypoints':  sample['keypoints'],
            'categories': sample['categories'],
            'frame_nums': sample['frame_nums'],
            'px_mul_x':   sample['px_mul_x'],
            'px_mul_y':   sample['px_mul_y'],
            'clip_len':   len(images),
            'clip_id':    sample['clip_id'],
            'metadata':   sample['clip_metadata'],
        }
    
    ################Augmentation Helperes#############
    @property
    def _aug_range(self):
        """
        Approximate value range of a normalised frame.
        Used to scale augmentation intensities consistently across all
        normalization methods.
        """
        if self.normalize_method is None: #Range of values is 0-255.0
            return 255.0
        if self.normalize_method in ('stand', 'stand+clip'):
            return 6.0      # roughly [-3, +3]
        if self.normalize_method == 'zero_back':
            return 2.0      # [-1, +1]
        if self.normalize_method in ('lin_perframe', 'lin_global','clip+lin','basic'):
            return 1.0      # [0, 1]
        
        return 1.0 #Fallback
    

    def _build_augmentation_pipeline(self):
        """
        Convert self.img_augmentations dict into a callable Albumentations Compose pipeline. Call once at the end of __init__
        """
        if not self.img_augmentations:
            return None

        aug_range=self._aug_range #Range of values for the normalized image
        transforms: List[A.BasicTransform]=[] #Initialize the transform pipeline to an empty list

        for aug in self.img_augmentations:
            if aug=='medianblur':
                transforms.append(A.MedianBlur(blur_limit=(3,5),p=1.0)) #Setup the median blur augmentation
            elif aug=='brightnesscontrast':
                transforms.append(
                    A.RandomBrightnessContrast(
                        contrast_limit=0.2,
                        brightness_limit=0.1*aug_range,
                        brightness_by_max=True,
                        p=1.0,
                    )
                )
            elif aug=='gaussnoise':
                transforms.append(
                    A.GaussNoise(
                        var_limit=(0.0,(0.02*aug_range)**2),
                        mean=0.0,
                        per_channel=True,
                        p=1.0,
                    )
                )
            #Skip other options
        
        #Here, p=probability that the augmentation will be applied to the frame or the clip
        if self.return_mode=='clip':
            #Return a replay compose object, so we can apply same augmentation across the clip if we are doing clip-level __getitem__
            return A.ReplayCompose(transforms,p=self.aug_prob) if transforms else None
        
        return A.Compose(transforms,p=self.aug_prob) if transforms else None
    
    def _apply_augmentations(self,image_tensor):
        """
        Applies the augmentation pipeline to a single (C,H,W) image tensor. Returns a (C,H,W) tensor.
        """
        if self._aug_pipeline is None:
            return image_tensor #No augmentations to apply
        
        #Convert the tensor to a numpy array (H,W,C) for Albumentations
        img_hwc=image_tensor.numpy().transpose(1,2,0) #(H,W,C)
        img_aug=self._aug_pipeline(image=img_hwc)['image'] #Applies the augmentation pipeline to the image and extracts the image
        if img_aug.ndim==2: #If the image returned only has 2 dimensions (the channel dimension was dropped) add in an extra dimension at end
            img_aug=img_aug[:,:,np.newaxis]
        img_aug_tensor=torch.tensor(np.ascontiguousarray(img_aug.transpose(2,0,1)))

        return img_aug_tensor
    
    def _apply_augmentations_cliplevel(self,img,replay_params):
        #Applies augmentations to this image on the fly, but applies same augmentation across clip
        if self._aug_pipeline is None:
            return img #No augmentations to apply
        
        img_hwc=img.numpy().transpose(1,2,0)
        if replay_params is None: #First frame of clip
            result=self._aug_pipeline(image=img_hwc)
            replay_params=result['replay'] #Gets the augmentation paramaters for this clip
        else:
            result=A.ReplayCompose.replay(replay_params,image=img_hwc)
        img_aug=result['image']
        #Add axis (channel) in case of grayscale image
        if img_aug.ndim==2:
            img_aug=img_aug[:,:,np.newaxis]
        img=torch.tensor(np.ascontiguousarray(img_aug.transpose(2, 0, 1)))
        
        return img,replay_params
    




    

