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
_VALID_NORM_METHODS=frozenset({None,'basic','lin_perframe','lin_global','stand','clip+lin','stand+clip','zero_back'})
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
            - 'lin_perframe'= subtract each pixel by image minimum, multiple by 1/(max-min) range, so: (I-Imin)*(1/(Imax-Imin))
            - 'lin_global'= subtract each pixel by global imahe minimum, multiple by 1/(max-min) range, so: (I-IGlobalmin)*(1/(IGlobalmax-IGlobalmin))
            - 'stand'= subtract mean & divide by std
            - 'clip+lin'= clip values higher/less than 0.995*min and 0.995*max (global min and max) then do lin_perframe
            - 'stand+clip'=standardize, then clip values outside +-3*std
            - 'zero_back' = (I-127.5)/127.5 (ensures black background remains zero)
        resampling_freq (None or int): Resampling frequency (samples/sec) if we want to downsample the ultrasound images
        convert_to_gray: Whether we convert the us images to gray
        img_augmentation (None or list of str): Implements on-the-fly augmentations to us images during __getitem__. Options include: 'medianblur','brightnesscontrast','gaussnoise'

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
                    #All .pt files are present, but the norm_stats file is deleted
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
                    #sub-pass A: PNG => preprocessing => stats => save preprocessed frames to _raw.pt
                    self.norm_stats=self._stats_pass_and_cache_raw(clip_records)

                    #Save the norm_stats incase sub-pass A is interrupted
                    self._save_norm_stats_to_cache(clip_records[0]['json_path'])
                    
                    #sub-pass B: _raw.pt -> normalize -> final .pt -> delete _raw.pt
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
        original_sampling_freq=float(clip_meta.get('sampling_rate')) #original sampling rate in (milliseconds/frame)
        original_sampling_freq=1/(original_sampling_freq/1000) #original sampling rate in frames/second

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
    
    def _accumulate_stats(self,img,acc):
        #Flatten the image
        flat=img.ravel().astype(np.float32)

        #Update the accumulator
        acc['sum']+=float(flat.sum()) #The sum of pixel values
        acc['sum_sq']+=float((flat**2).sum()) #Sum of squared pixel values
        acc['count']+=flat.size #Number of pixels per frame
        acc['min']=min(acc['min'],float(flat.min()))
        acc['max']=max(acc['max'],float(flat.max()))
    
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
                raw_cache_path=os.path.join(cache_dir,f'frame_{fn:05d}_raw.pt') #Where we save the raw image before normalization

                #Load the US frame:
                try:
                    img = self._load_and_preprocess_frame(entry['file_name'])
                except FileNotFoundError as exc:
                    print(f"[AIUSDataset] Warning: {exc} — skipping frame.")
                    continue
                
                #Accumulate the statistics
                acc=self._accumulate_stats(img,acc)

                #Save the unormlized frame to raw cache and release
                self._array_to_tensor_and_save(img,raw_cache_path)
        
        if acc['count']==0: #Checks that we have more than 0 pixels
            raise ValueError("No pixels accumulated — check that all image files exist.")
        
        #Compute the stats
        norm_stats=self._compute_stats_from_accumulator(acc)

        return norm_stats
    
    def _normalize_and_recache_clip(self,record):
        """
        sub-pass B: _raw.pt -> normalize -> final .pt -> delete _raw.pt
        This is called a clip record
        No PNG reads — all data is read from the raw disk cache written by
        Sub-pass A.  Writes cache_config.json after processing the clip.
        Requires self.norm_stats to be populated for global-stats methods.
        """
        cache_dir = self._get_cache_dir(record['json_path'], record['stem'])
        wrote_any = False #Boolean to check that we wrote some cache

        for entry in record['frame_entries']:
            fn=entry['frame_num']
            raw_cache_path=os.path.join(cache_dir,f'frame_{fn:05d}_raw.pt')

            cache_path=os.path.join(cache_dir, f'frame_{fn:05d}.pt') #Where we store the final ultrasound image

            #Checks that the raw file exists:
            if not os.path.isfile(raw_cache_path):
                print(f"[AIUSDataset] Warning: expected raw cache not found for "
                      f"frame {fn} of '{record['stem']}' — skipping.")
                continue

            #Load the raw (un-normalized) tensor
            raw_tensor=self._load_tensor(raw_cache_path)
            if raw_tensor is None: #Check that exists
                continue
            img_np=self._tensor_to_array(raw_tensor) #Gets it as a numpy image
            img_norm = self._normalize_frame(img_np) #Normalizes the US frame
            self._array_to_tensor_and_save(img_norm, cache_path) #Saves tbe image

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

        #(I-Imin)*(255/(Imax-Imin))
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
            if std < 1e-8:
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
    
    
        



    

