"""
datasplitter.py
---------------
Splits a processed COCO-like keypoint-detection dataset into train / val / test
lists of annotation JSON paths, with optional equal-proportion enforcement
across one or two metadata tags, and optional k-fold cross-validation.
"""

import glob
import json
import os
import random
from collections import defaultdict
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
from sklearn.model_selection import KFold


#################Constants#################
_VALID_METADATA_TAGS = frozenset({
    'site', 'annotator', 'zone_label', 'patient_id',
    'time', 'transducer_type', 'manufacturer_name',
})





def datasplitter(dataset_root: str='../../../Data/Keypoint_Detect_Data',
                outdata_format: str='COCO_like',
                train_split: float=0.7,
                val_split: float=0.15,
                test_split: float=0.15,
                k_folds: Optional[int]=None,
                equal_prop_tags: Optional[List[str]]=None,
                metadata_tags: Optional[Dict[str, List[str]]] = None,
                random_state: int=42,
                verbose: bool=False,
                ) -> Union[Tuple[List, List, List], List[Tuple[List, List]]]:
    """
    Returns: separate lists of .json file paths for train/val/test or folds (which is list of tupples of train and valid lists) for k-folds.
    Input: 
    - dataset_root
    - outputdata_format (default is COCO_like, only one supported right now)
    - train_split, val_split, test_split => split ratio for each dataset
    - k_folds: None not doing kfold, else number of k_folds (int)
    - euqal_prop_tags: list containing which metadata_tags we want to enforce to have an equal proportion of in each dataset split
    - metadata_tags:
        - site: ['All'] or list
        - annotator:['All'] or list
        - zone_label:['All'] or list
        - patient_id: ['All'] or list
        - time:['All'] or list
        - transducer_type:['All'] or list
        - manufacturer_name:['All'] or list
        - coordinate_space
        
        

    **Data handling note: The metadata_tags allows inclusion of different data-types. 
    equal_prop_tags: ensures that for those metadata_tags we have the same proportion of clips for 
    each of the metadata in the different splits (same ratio). We can only do up to 2 tags for equal proprtions.
    This also ensures that all clips cooresponding to a given patient_id are in the same split (can't have the same patient in different splits)
    If 'annotations' is ==0, we ignore that clip.
    """
    
    if outdata_format != 'COCO_like':
        raise ValueError(
            f"outdata_format='{outdata_format}' is not supported. "
            "Only 'COCO_like' is currently implemented."
        )
    
    dataset_root=os.path.join(dataset_root,'COCO_Data')

    #################Validate and seed the RNG

    random.seed(random_state)
    np.random.seed(random_state)

    #Checking that equal_prop_tags are not more than 2, and that the tags exist:
    if equal_prop_tags is None:
        equal_prop_tags = []
    if len(equal_prop_tags) > 2:
        raise ValueError(
            f"equal_prop_tags supports at most 2 tags; "
            f"got {len(equal_prop_tags)}: {equal_prop_tags}."
        )
    for tag in equal_prop_tags:
        if tag not in _VALID_METADATA_TAGS:
            raise ValueError(
                f"'{tag}' is not a recognised metadata tag. "
                f"Valid options: {sorted(_VALID_METADATA_TAGS)}."
            )
        
    
    #Checks to see if metadata_tags is none:
    if metadata_tags is None:
        metadata_tags = {}
    
    #Checks if k_folds is none, and then validates that the train/valid/test splits sum to 1

    if k_folds is None:
        total = train_split + val_split + test_split
        if abs(total - 1.0) > 1e-6:
            raise ValueError(
                f"train_split + val_split + test_split must equal 1.0; "
                f"got {total:.6f}."
            )
    

    ################Locate the annotation directories###############

    ann_dir=None
    if metadata_tags['coordinate_space'] =='scanline':
        ann_dir=os.path.join(dataset_root, 'annotations', 'scanline')
    if metadata_tags['coordinate_space'] == 'sector':
        ann_dir=os.path.join(dataset_root, 'annotations', 'sector')
    
    ##############Loop through the annotation director reading in annotations and metadata
   #Dict of lists sorted by patient id
    patient_clips: Dict[str, List[dict]] = defaultdict(list) #string is patient_id, list is the clip data

    if not os.path.isdir(ann_dir):
            raise ValueError(f"[datasplitter] Warning: annotation directory not found — "
                  f"'{ann_dir}'.")
    
    #Looping for the json annotation files
    for json_path in sorted(glob.glob(os.path.join(ann_dir, '*.json'))):

        #Read in the annotations
        try:
            with open(json_path, 'r', encoding='utf-8') as fh:
                data = json.load(fh)
        except Exception as exc:
            if verbose:
                print(f"[datasplitter] Warning: could not read "
                        f"'{json_path}': {exc}")
            continue

        #Skip the clips that carry no annotations
        if len(data.get('annotations', [])) == 0:
            if verbose:
                print(f"  [skip] 0 annotations — "
                        f"{os.path.basename(json_path)}")
            continue

        #Extract the metadata for each clip:
        meta_clip_list = data.get('metadata', [])
        if not meta_clip_list:
            print(f"[datasplitter] Warning: no metadata in "
                    f"'{json_path}' — skipping.")
            continue
        meta_clip = meta_clip_list[0]

        #We get the annotator from the file name part before the '_'-delimiter
        stem = os.path.splitext(os.path.basename(json_path))[0]
        annotator_name = stem.split('_')[0]

        clip_data: dict = {
                'json_path':         json_path,
                'site':              meta_clip.get('site'),
                'annotator':         annotator_name,
                'zone_label':        meta_clip.get('zone_label'),
                'patient_id':        str(meta_clip.get('patient_id', 'UNKNOWN')),
                'time':              meta_clip.get('time'),
                'transducer_type':   meta_clip.get('transducer_type'),
                'manufacturer_name': meta_clip.get('manufacturer_name'),
            }

        if not _passes_filter(clip_data, metadata_tags):
            continue

        patient_clips[clip_data['patient_id']].append(clip_data) #Add annotations to dict organized per patient_id

    if not patient_clips:
        raise ValueError(
            "No valid clips found after applying filters. "
            "Check dataset_root, coordinate_space, and metadata_tags."
        )
    
    if verbose:
        n_total = sum(len(v) for v in patient_clips.values())
        print(f"\n[datasplitter] Loaded {n_total} clips from "
              f"{len(patient_clips)} patients "
              f"(after filtering and dropping zero-annotation clips).\n")
    
    #Return data split depending on which type of splitting we are using:
    if k_folds is not None:
        return _kfold_split(
            patient_clips, k_folds, equal_prop_tags, random_state, verbose
        )

    return _train_val_test_split(
        patient_clips, train_split, val_split, test_split,
        equal_prop_tags, verbose,
    )



##################################Private Helper Functions########################
def _passes_filter(clip_data: dict, metadata_tags: dict) -> bool:
    """Return True when *clip* satisfies every inclusion filter."""
    for tag in _VALID_METADATA_TAGS:
        allowed = metadata_tags.get(tag)
        if allowed is None or allowed == ['All']:
            continue
        if clip_data.get(tag) not in allowed:
            return False
    return True

def _patient_group_key(clips: List[dict], tags: List[str])-> tuple:
    """
    For each clip, we build a sortable key describing the patient's group.

    """
    parts = []
    for tag in tags:
        vals={c.get(tag) for c in clips} #All unique tag values across all clips
        if len(vals)==1:
            v=vals.pop()
            parts.append(str(v) if v is not None else 'UNKNOWN')
        else:
            parts.append('mixed') #In case that multiple tags for same patient
    return tuple(parts)

def _group_patients(patient_clips:Dict[str,List[dict]],equal_prop_tags: List[str],verbose: bool=False)-> Dict[tuple,List[str]]:
    """
    Group patients depending on the equal_prop_tags. If two tags in equal_prop_tags, we do cartesian product of every permutation of the equal_prop_tags.
    When equal_prop_tags is empty, all patients land in the same single group.
    When equal_propr_tags has one-two tags, patients a grouped by tag-value combination, enabling independent per-group splitting that preserves equal proportions.

    If there are multiple tags for a given patient (e.g. multiple annotators for a given patient) then the annotator will fall into the 'mixed' tag    
    """
    groups: Dict[tuple,List[str]]=defaultdict(list)
    for p_id,clips in patient_clips.items(): #Looping through all patient clips, getting the patient ID and the clip data
        key=(_patient_group_key(clips,equal_prop_tags) if equal_prop_tags else ('all',))
        groups[key].append(p_id) #For this key pair tuple, we append the corresponding patient id's to the corresponding list
    groups=dict(groups)

    # ── Sanity check: every patient_id must appear in exactly one group ──────
    if verbose:
        all_ids_in_groups = [pid for pids in groups.values() for pid in pids]
        assert len(all_ids_in_groups) == len(set(all_ids_in_groups)), (
            "BUG: duplicate patient_id found across groups."
        )
        assert len(all_ids_in_groups) == len(patient_clips), (
            "BUG: patient count mismatch between patient_clips and groups."
        )

        # ── Warn about mixed-bucket dilution (if there are multiple annotators for same patient, then all annotators go into mixed bucket and we can loose distribution)
        if equal_prop_tags:
            _warn_mixed_groups(groups, patient_clips, equal_prop_tags)
    
    return dict(groups)

def _split_list(patient_ids:List[str], train_frac: float,val_frac:float,test_frac:float)->Tuple[List[str],List[str],List[str]]:
    """
    Randomly split patient ids into train/val/test sets using integer rounding.
    """
    patient_ids=list(patient_ids)
    random.shuffle(patient_ids)  #Randomly shuffles the initial passed patient ids
    n=len(patient_ids) #number of patients
    n_test=round(n*test_frac) #Round to integer
    n_val=round(n*val_frac)
    n_train=n-n_test-n_val #Remainder go to n_train

    #Ensure a non-empty training set
    while n_train<=0 and n>0: #Loop while n_train==0
        if n_test>=n_val and n_test>0:
            n_test-=1 #Subtract from n_test

        elif n_val>0:#subtract from n_val
            n_val-=1
        else:
            break
        n_train=n-n_test-n_val #update n_train
    #Slice the patient IDS from the randomly shuffled id list
    test_ids=patient_ids[:n_test]
    val_ids=patient_ids[n_test:n_test+n_val]
    train_ids=patient_ids[n_test+n_val:]
    return train_ids,val_ids,test_ids

def _paths_for(patient_clips:Dict[str,List[dict]],patient_ids:List[str])->List[str]:
    """
    Returns list of strings of json paths for all patient id passed
    """

    return [
        clip['json_path']
        for pid in patient_ids
        for clip in patient_clips.get(pid,[])
    ]

#Splitting based on pre-set train/val/test fractions
def _train_val_test_split(patient_clips:Dict[str,List[dict]],
                          train_split: float,
                          val_split: float,
                          test_split: float,
                          equal_prop_tags: List[str],
                          verbose: bool,
                          )-> Tuple[List[str],List[str],List[str]]:
    """
    Returns a tuple of three lists countaining the training/validation/test paths to the json_path for that clip
    """

    groups=_group_patients(patient_clips,equal_prop_tags,verbose=verbose) #Returns dictionary of [tuples,list[str]] where the tuple is the equal_prop_tag(s) and the list is the patient ids

    all_train_ids: List[str] = []
    all_val_ids:   List[str] = []
    all_test_ids:  List[str] = []

    if verbose:
        if equal_prop_tags:
            print(f"[datasplitter] Equal-proportion tags : {equal_prop_tags}")
            print(f"[datasplitter] Number of groups      : {len(groups)}")
            _print_groups(groups, patient_clips)
        print()
    
    #Iterate through the groups in sorted order
    for key in sorted(groups): #Iterate through equal_prop_tags
        patient_ids=groups[key] #Get the patient ids for this group
        tr_sp,va_sp,test_sp=_split_list(patient_ids,train_split,val_split,test_split)

        #Extend the aggregate train,val,and test lists
        all_train_ids.extend(tr_sp)
        all_val_ids.extend(va_sp)
        all_test_ids.extend(test_sp)
    
    #Get the training, validation and test json paths
    train_paths=_paths_for(patient_clips,all_train_ids)
    val_paths=_paths_for(patient_clips,all_val_ids)
    test_paths=_paths_for(patient_clips,all_test_ids)

    #Show the summary of the splits
    if verbose:
        print(f"\n[datasplitter] Overall split summary:")
        _print_split_summary(
            patient_clips,
            all_train_ids, all_val_ids, all_test_ids,
            equal_prop_tags,
        )
    #Return results
    return train_paths,val_paths,test_paths


#k-fold splitting
def _kfold_split(
                patient_clips:Dict[str,List[dict]],
                k_folds: int,
                equal_prop_tags: List[str],
                random_state: int,                
                verbose: bool,
                )-> List[Tuple[List[str],List[str]]]:
    """
    Splits the patient_ids into list of k-folds (each fold is a touble with the training paths and the validation paths of json files for those clips)
    Extracts the groups of patients based on the equal_prop_tags, and ensures that the validation and training paths have equal proportions of clips based on these tags.
    """
    groups=_group_patients(patient_clips,equal_prop_tags,verbose=verbose) #Extract groups, which are dict of tuples,strings, where tuples are the equal_prop_tag combo and strings are the patient_ids
    #Initialize aggregate list of list of patient ids for train/validation splits
    fold_train_ids: List[List[str]]=[[] for _ in range(k_folds)]
    fold_val_ids: List[List[str]]=[[] for _ in range(k_folds)]

    kfold=KFold(n_splits=k_folds,shuffle=True,random_state=random_state) #Initialize the KFold object 
    for key in sorted(groups):
        patient_ids=np.array(groups[key])
        if len(patient_ids) < k_folds: #We have less patient ids than k_folds for this group, so cannot have at least one of each id in each fold
            print(
                f"[datasplitter] Warning: group {key} has only "
                f"{len(patient_ids)} patient(s), which is fewer than "
                f"k_folds={k_folds}.  Some folds may have no validation "
                f"data for this group."
            )
        
        for fold_idx,(tr_idx,val_idx) in enumerate(kfold.split(patient_ids)): #Splits the patient ids into the k-folds for this group
            fold_train_ids[fold_idx].extend(patient_ids[tr_idx].tolist())
            fold_val_ids[fold_idx].extend(patient_ids[val_idx].tolist())
    
    #Prepare the final folds object to return (which returns the json paths for the corresponding training and validation paths)
    folds = []
    for fold_idx in range(k_folds):
        tr_paths=_paths_for(patient_clips,fold_train_ids[fold_idx])
        val_paths=_paths_for(patient_clips,fold_val_ids[fold_idx])

        #Randomly shuffle these paths
        random.shuffle(tr_paths)
        random.shuffle(val_paths)
        folds.append((tr_paths,val_paths)) #Append tuple of paths to the final folds object (which is a list of tuples)
    
    if verbose:
        tag_str = (f" (equal-prop tags: {equal_prop_tags})"
                   if equal_prop_tags else "")
        print(f"\n[datasplitter] {k_folds}-fold CV complete{tag_str}.")
    return folds





#########################Printing function helpers:###############################
def _print_groups(
        groups: Dict[tuple, List[str]],
        patient_clips: Dict[str, List[dict]],
) -> None:
    """Print a one-line summary per group."""
    for key, pids in sorted(groups.items()):
        n_clips = sum(len(patient_clips[p]) for p in pids)
        print(f"    {str(key):<45}  "
              f"{len(pids):4d} patients  |  {n_clips:5d} clips")


def _print_split_summary(
        patient_clips: Dict[str, List[dict]],
        train_ids:       List[str],
        val_ids:         List[str],
        test_ids:        List[str],
        equal_prop_tags: List[str],
) -> None:
    """Print overall patient/clip counts and per-tag clip distributions."""
    labels   = ('Train', 'Val', 'Test')
    id_lists = (train_ids, val_ids, test_ids)

    for lbl, ids in zip(labels, id_lists):
        n_clips = sum(len(patient_clips[p]) for p in ids)
        print(f"  {lbl:<6}: {len(ids):4d} patients  |  {n_clips:5d} clips")

    # Per-tag distribution table
    for tag in equal_prop_tags:
        all_vals = sorted({
            str(c.get(tag, 'N/A'))
            for ids in id_lists
            for pid in ids
            for c in patient_clips.get(pid, [])
        })
        print(f"\n  '{tag}' clip distribution across splits:")
        header = f"    {'Value':<30}" + "".join(
            f"  {name:>8}" for name in labels
        )
        print(header)
        print("    " + "─" * (30 + 3 * 10))
        for val in all_vals:
            row = f"    {val:<30}"
            for ids in id_lists:
                cnt = sum(
                    1 for pid in ids
                    for c in patient_clips.get(pid, [])
                    if str(c.get(tag, 'N/A')) == val
                )
                row += f"  {cnt:>8}"
            print(row)


def _warn_mixed_groups(
        groups: Dict[tuple, List[str]],
        patient_clips: Dict[str, List[dict]],
        equal_prop_tags: List[str],
) -> None:
    """
    Warn when a significant fraction of patients fall into the 'mixed' bucket.

    'annotator' is singled out with a stronger warning because unlike
    site/time it is a clip-level property and mixed patients are common.
    """
    total_patients = sum(len(v) for v in groups.values())
    total_clips    = sum(len(c) for c in patient_clips.values())

    for tag_pos, tag in enumerate(equal_prop_tags):
        # Identify keys where this tag's position is 'mixed'
        mixed_patients = [
            pid
            for key, pids in groups.items()
            if key[tag_pos] == 'mixed'
            for pid in pids
        ]
        mixed_clips = sum(len(patient_clips[p]) for p in mixed_patients)

        if not mixed_patients:
            continue  # No mixed patients for this tag — all good

        pct_patients = 100.0 * len(mixed_patients) / total_patients
        pct_clips    = 100.0 * mixed_clips          / total_clips

        # Stronger warning for 'annotator' because it is a clip-level property
        if tag == 'annotator':
            print(
                f"\n[datasplitter] WARNING — 'annotator' is a clip-level "
                f"property, not a patient-level property.\n"
                f"  {len(mixed_patients)} / {total_patients} patients "
                f"({pct_patients:.1f}%) have clips from more than one "
                f"annotator and fall into the 'mixed' bucket.\n"
                f"  This accounts for {mixed_clips} / {total_clips} clips "
                f"({pct_clips:.1f}%).\n"
                f"  These patients' clips (from ALL their annotators) are "
                f"distributed randomly, reducing annotator proportion "
                f"enforcement.\n"
                f"  Consider filtering to a single annotator via:\n"
                f"    metadata_tags={{'annotator': ['<name>']}}\n"
                f"  before using 'annotator' in equal_prop_tags."
            )
        else:
            print(
                f"\n[datasplitter] Note — tag '{tag}': "
                f"{len(mixed_patients)} patient(s) ({pct_patients:.1f}%) "
                f"have mixed values and are distributed randomly "
                f"({pct_clips:.1f}% of clips)."
            )
            