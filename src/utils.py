import json
import re
import os
import glob
from typing import Any, Dict, List, Optional, Tuple
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.lines import Line2D

import torch
import torch.nn as nn
import numpy as np
import pydicom
from scipy.ndimage import map_coordinates
from PIL import Image
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment

#Class ID numbers
CLASS_ID_PLEURAL_LINE=1
CLASS_ID_B_LINE=2
CATEGORY_NAMES = {CLASS_ID_PLEURAL_LINE: 'pleural', CLASS_ID_B_LINE: 'bline'}
LINE_TYPE_TO_IDS = {
    'pleuraline': [CLASS_ID_PLEURAL_LINE],
    'bline':      [CLASS_ID_B_LINE],
    'both':       [CLASS_ID_PLEURAL_LINE, CLASS_ID_B_LINE],
}

#These are data keys that are needed for processing
REQUIRED_SCAN_KEYS = (
    "angle1",
    "angle2",
    "center_rows_px",
    "center_cols_px",
    "radius1",
    "radius2",
    "image_size_rows",
    "image_size_cols",
)




########################Preprocessing.py Helper Functions#########################

def find_json_for_dicom(dicom_path: str, json_dir: str) -> Optional[str]:
    """Match DICOM to JSON by SOPInstanceUID when possible, else by filename stem."""
    
    #Strategy 1: Read DICOM Header and compare the UID against the SOPInstanceUID field stored in each JSON
    ds = None
    try:
        ds = pydicom.dcmread(dicom_path, stop_before_pixels=True, force=True)
    except Exception:
        pass

    if ds is not None and getattr(ds, "SOPInstanceUID", None):
        uid = str(ds.SOPInstanceUID)
        for p in glob.glob(os.path.join(json_dir, "*.json")):
            try:
                data = load_annotation_json(p)
            except (json.JSONDecodeError, OSError):
                continue
            if str(data.get("SOPInstanceUID", "")) == uid:
                return p

    #Strategy 2: filename-stem match
    stem = os.path.splitext(os.path.basename(dicom_path))[0]  # Filename without extension
    if stem.lower().endswith('.dcm'):                           # strip second '.dcm' if it exists
        stem = os.path.splitext(stem)[0]

    candidates = sorted(glob.glob(os.path.join(json_dir, f"{stem}*.json")))
    if candidates:
        return candidates[0]

    #Strategy 3: Any JSON whose filename contains the stem
    for p in sorted(glob.glob(os.path.join(json_dir, "*.json"))):
        p_name = os.path.basename(p)
        p_stem = os.path.splitext(p_name)[0]
        if stem in p_name or re.match(re.escape(stem), p_stem):
            return p
    return None

def os_make_dir(folder_path: str) -> None:
    if not os.path.exists(folder_path):
        os.makedirs(folder_path, exist_ok=True)


def export_clip_to_png_and_json(input_dicom_path: str,
                                input_json_path: str,
                                output_annotation_dir: str,
                                output_image_dir: str,
                                filename_prefix: str,
                                file_metadata: dict={'site': None, 'patient_id': None, 'time': None},
                                site_str: str=None,
                                coordinate_space: str='scanline',
                                num_lines: int=128,
                                num_samples_per_line: int=128,
                                ) -> None:
    """ Converts a dicom clip (scan) and json annotation to individual images and annotations in the
      desired output format (only .png and .json supported now). 
      filename_prefix is a string with: <annotator>_<scan_id>
      file_metadata is dict: {site, patient_id, time}
      """
    
    if file_metadata is None:
        file_metadata={'site': None, 'patient_id': None, 'time': None}
    if site_str is None:
        raise ValueError(f"site_str must be passed to export_clip_to_png_and_json")
    
    #Validate coordinate space immediately 
    if coordinate_space not in ('sector','scanline'):
        raise ValueError(f"coordinate_space must be 'sector' or 'scanline', got '{coordinate_space}'.")
    
    #########Loading data and initializing parameters############
    json_data=load_annotation_json(input_json_path) #Loading json annotations
    scan_params = scanconversion_params_from_annotation(json_data) #Checks for any missing keys in the annotations, and returns annotations that have complete keys

    #Init dictionaries that will contain bline and pleura line annotations in mm
    frame_to_pleura_sector: Dict[int, List[np.ndarray]] = {}
    frame_to_blines_sector: Dict[int, List[np.ndarray]] = {}

    #Fill the annotation dictionaries (in original sector space)
    frame_nos, pleura_list, blists, pleura_ok, _b_ok = parse_all_lines_from_annotation(json_data)
    for fn, pts in zip(frame_nos, pleura_list):
        good = [p for p in pts if p.size > 0]
        if good:
            frame_to_pleura_sector[int(fn)] = good
    for fn, b_list in zip(frame_nos, blists):
        good = [b for b in b_list if b.size > 0]
        if good:
            frame_to_blines_sector[int(fn)] = good


    ################Loading in the ultrasound images and getting metadata###############
    frames_sector, psx, psy, ds = load_ultrasound_frames_from_dicom(input_dicom_path)
    transducer_metadata=extract_transducer_metadata(ds,site_str=site_str)
    ###############Converting images and annotations to requested coordinate space#######
    # After this block we have:
    #   num_frames              – total frames to iterate over
    #   frame_to_pleura_out     – {frame_num: (N,2) array} in output coordinate space
    #   frame_to_blines_out     – {frame_num: [(N,2), ...]} in output coordinate space
    # For scanline:  scan_hwc  – (N, H, W, C) numpy array of converted frames
    # For sector:    frames_sector is used directly (after axis reordering at save time)

    if coordinate_space=='scanline': #Coordinate space is scanline (rectangular)
        scan_tensor, scan_config = convert_to_scanlines(frames_sector, scan_params, num_lines=num_lines, num_samples_per_line=num_samples_per_line)
        if scan_tensor is None:
            raise ValueError("Empty frame array from DICOM.")

        #Convert scan tensor to np frames
        scan_hwc_scanline = np.moveaxis(scan_tensor, 1, 3)
        num_frames = scan_hwc_scanline.shape[0]

        #Init the dictionary to hold the annotations in the scanline space
        frame_to_pleura_out: Dict[int,List[np.ndarray]] = {}
        frame_to_bline_out: Dict[int,List[np.ndarray]]={}

        #Read in annotations in scanline space
        for fn, mm_list in frame_to_pleura_sector.items():
            if fn < 0 or fn >= num_frames:
                continue
            converted: List[np.ndarray] = [pleura_points_curvilinear_mm_to_scanlines(mm, psx, psy, scan_config) for mm in mm_list if mm.size>0]
            
            if converted:
                frame_to_pleura_out[fn] = converted
        
        for fn, mm_list in frame_to_blines_sector.items():
            if fn < 0 or fn >= num_frames:
                continue
            converted: List[np.ndarray] = [pleura_points_curvilinear_mm_to_scanlines(mm, psx, psy, scan_config) for mm in mm_list if mm.size>0]
            if converted:
                frame_to_bline_out[fn] = converted

    else: #Coordinate space is 'sector'
        num_frames=frames_sector.shape[0]

        frame_to_pleura_out: Dict[int,List[np.ndarray]] = {}
        frame_to_bline_out: Dict[int,List[np.ndarray]]={}
        for fn, mm_list in frame_to_pleura_sector.items():
            if fn < 0 or fn >= num_frames:
                continue
            converted = []
            for mm in mm_list:
                if mm.size>0:
                    #Convert annotations to pixel space, so divide by pixel spacing
                    pts_px = mm.astype(float).copy()
                    pts_px[:, 0] /= psx
                    pts_px[:, 1] /= psy
                    converted.append(pts_px)
            if converted:
                frame_to_pleura_out[fn] = converted
        for fn, mm_list in frame_to_blines_sector.items():
            if fn < 0 or fn >= num_frames:
                continue
            converted = []
            for mm in mm_list:
                if mm.size>0:
                    #Convert annotations to pixel space, so divide by pixel spacing
                    pts_px = mm.astype(float).copy()
                    pts_px[:, 0] /= psx
                    pts_px[:, 1] /= psy
                    converted.append(pts_px)
            if converted:
                frame_to_bline_out[fn]=converted
       

    #Looping through all the image frames in this clip and saves them to separate files
    lbl_filename=os.path.join(output_annotation_dir,f'{filename_prefix}.json')
    output_json={
        "metadata":[],
        "images": [],
        "annotations": [],
    }
    #Metadata contains: {site, patient_id, time, probe_orient, probe_type}
    output_json["metadata"].append({
        "site": file_metadata["site"],
        "patient_id": file_metadata["patient_id"],
        "time": file_metadata["time"],
        "transducer_type": transducer_metadata["transducer_type"],
        "manufacturer_name": transducer_metadata["manufacturer_name"],
        "zone_label": transducer_metadata["zone_label"],
        "sampling_rate": transducer_metadata["sampling_rate"],

    })


    ann_id=0
    for f in range(num_frames):
        img_filename=os.path.join(output_image_dir,f'{filename_prefix}_{f}.png')
        #Get the frame as (H,W,C) so PIL can save it correctly
        im=np.asarray(scan_hwc_scanline[f]) if coordinate_space=='scanline' else np.moveaxis(np.asarray(frames_sector[f]),0,-1) #This is in (H,W,C)
        

        #Enforce image to be 0->255 format:
        if im.dtype == np.uint8:
            im_u8 = im
        elif np.issubdtype(im.dtype, np.integer):
            im_u8 = np.clip(im, 0, 255).astype(np.uint8)
        else:
            im_f = im.astype(float)
            im_f = im_f - im_f.min()
            mx = float(im_f.max()) or 1.0
            im_u8 = np.clip(255.0 * im_f / mx, 0, 255).astype(np.uint8)
        
        #Get image height and width
        img_h,img_w=im_u8.shape[0],im_u8.shape[1]
        #Saves image:
        Image.fromarray(im_u8).save(img_filename)

        #Update the image in the json:
        output_json["images"].append({
                    "frame_num": f,
                    "file_name": str(img_filename),
                    "height": im.shape[0],
                    "width": im.shape[1],
                    "px_mul_x": psx,
                    "px_mul_y": psy,
                })

        #Handling the annotations and image labels in the output_json file:
        #Handling pleural keypoint annotations
        pleural_pts_list=frame_to_pleura_out.get(f,[])

        #First, compute the deepest pleural line (the line whose y-value is the largest) which will be passed to the b-line bonding box calculater to set the top of the b-line
        if pleural_pts_list:
            deepest_pleural_pts = max(pleural_pts_list,key=lambda pts: float(np.max(pts[:, 1])))
        else:
            deepest_pleural_pts=None

        for pleural_pts in pleural_pts_list:
            if pleural_pts is not None and pleural_pts.size>0:
                #Get the bounding box list ([xmin,ymin,w,h])
                bboxes=pleural_scanline_to_bbox(pleural_pts,img_w,img_h,min_side_px=5.0) #Creates a bounding box based on a min_side_px 
                if bboxes:
                    output_json["annotations"].append({
                        "id":ann_id, #Unique annotation ID via running counter
                        "frame_num": f,
                        "category_id": CLASS_ID_PLEURAL_LINE,
                        "keypoints": pleural_pts.tolist(),
                        "bboxes":bboxes,
                        })   
                    ann_id+=1             

        #Handling bline keypoint annotations
        bline_pts_list=frame_to_bline_out.get(f,[]) 
        for pts_b in bline_pts_list:
            if pts_b is not None and pts_b.size>0:
                bboxes_bline=bline_scanline_to_bbox(pts_b,deepest_pleural_pts,img_w,img_h,gap_below_pleura_px=1.0)
                if bboxes_bline:
                    output_json["annotations"].append({
                        "id": ann_id, #Just the frame number
                        "frame_num": f,
                        "category_id": CLASS_ID_B_LINE,
                        "keypoints": pts_b.tolist(),
                        "bboxes":bboxes_bline,
                    })
                    ann_id+=1
        
    #Save the json to its annotation folder
    with open(lbl_filename,"w",encoding="utf-8") as file:
        json.dump(output_json,file,indent=4)
    
    

        
def pleural_scanline_to_bbox(pleural_pts,img_w,img_h,min_side_px=5.0):
    '''
    Reads in the set of pleural_pts for a given frame, and returns a bounding box for the pleural line.
    Input:
        - pleural_pts: pleural points for a given frame. Size (N,2) Where N is number of pleural points in a given frame, and 2 is for x and y axis
        - img_w, img_h: image width and height
        - min_size_px: Minimum size of the bounding box in pixels
    Output:
        - pleura_bbox=[xmin,ymin,w,h], where xmin and ymin is the corner pixel
    '''
    #Gets the min and max values for the pleural points for both the x and y dimensions
    lines = pleural_pts[:, 0].astype(float)
    samples = pleural_pts[:, 1].astype(float)
    xmin, xmax = float(np.min(lines)), float(np.max(lines))
    ymin, ymax = float(np.min(samples)), float(np.max(samples))

    w,h=xmax-xmin,ymax-ymin
    if w<0 or h<0:
        return []
    
    #Adjusts width if the width or height is smaller than the minimum pixel size
    if w < min_side_px:
        pad= (min_side_px - w) / 2
        xmin -= pad
        xmax += pad
    if h < min_side_px:
        pad = (min_side_px - h) / 2
        ymin -= pad
        ymax += pad
    
    #Clips bbox based on image size
    img_max_x = float(img_w - 1)
    img_max_y = float(img_h - 1)
    xmin = float(np.clip(xmin, 0.0, img_max_x))
    xmax = float(np.clip(xmax, 0.0, img_max_x))
    ymin = float(np.clip(ymin, 0.0, img_max_y))
    ymax = float(np.clip(ymax, 0.0, img_max_y))
    
    #Checks that max is always bigger than min
    if xmax < xmin or ymax < ymin:
        return []
    
    #Compute new bbox width and height
    w=xmax-xmin
    h=ymax-ymin

    return [xmin,ymin,w,h] #COCO bbox format

def bline_scanline_to_bbox(pts_b,deepest_pleural_pts,img_w,img_h,gap_below_pleura_px=1.0):
    '''
    Takes in the B-line keypoints (either side of the b-line) and computes a bounding box for the B-line.
    Input:
        - pts_b: b-line points for a given frame. Size (M,2) Where M is number of bline points in a given frame, and 2 is for x and y axis
        - deepest_pleural_pts: pleural points for a given frame. Size (N,2) Where N is number of pleural points in a given frame, and 2 is for x and y axis
        - img_w, img_h: image width and height
    Output:
        - bline_bbox=[xmin,ymin,w,h] 
    '''
    if pts_b.size==0:
        return []
    
    lines = pts_b[:, 0].astype(float)
    xmin, xmax = float(np.min(lines)), float(np.max(lines))

    #Compute the top of the b-line bounding box, which is just below the pleura's deepest point on the frame
    if deepest_pleural_pts is not None and deepest_pleural_pts.size>0: 
        pleura_samples=deepest_pleural_pts[:,1].astype(float) #Gets the y elements of the pleural line
        pleura_ymax=float(np.max(pleura_samples)) #Gets the max pleural height
        y_top=pleura_ymax+float(gap_below_pleura_px) #Computes the top as the max pleura plus the gap
    else:
        y_top=float(np.min(pts_b[:,1])) #Fallback is the top of the bounding box
    
    if y_top>=img_h: #If the top of the bline bbox is heigher than the image height, then return empty list
        return []
    
    #Along horizontal, clip to the image widht
    img_max_x=float(img_w - 1)
    xmin = float(np.clip(xmin, 0.0, img_max_x))
    xmax = float(np.clip(xmax, 0.0, img_max_x))

    y_top=min(y_top,img_h-1)

    w=xmax-xmin
    h=img_h-y_top #Bounding box goes all the way to bottom of image
    if w<1e-9 or h<1e-9:
        return []
    return [xmin,y_top,w,h]



def load_annotation_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)
def _validate_scan_geometry(data: Dict[str, Any]) -> None:
    missing = [k for k in REQUIRED_SCAN_KEYS if k not in data]
    if missing:
        raise ValueError(f"Annotation JSON missing scan-conversion keys: {missing}")


def scanconversion_params_from_annotation(data: Dict[str, Any]) -> Dict[str, Any]:
    """Subset of JSON fields needed for scan conversion (no frame lists)."""
    _validate_scan_geometry(data)
    return {k: data[k] for k in REQUIRED_SCAN_KEYS if k in data}

def _polylines_mm_from_lines_field(
    lines_raw: Any,
    *,
    require_non_empty_line: bool,
) -> List[np.ndarray]:
    """Parse all ``{ "line": { "points": [[x,y],...] } }`` polylines from a ``pleura_lines``-style list."""
    out: List[np.ndarray] = []
    if not lines_raw or not isinstance(lines_raw, list):
        return out
    for item in lines_raw:
        if not isinstance(item, dict) or "line" not in item:
            continue
        pts = item["line"].get("points")
        if not pts or len(pts) < 2:
            continue
        arr = np.asarray([[float(p[0]), float(p[1])] for p in pts], dtype=float)
        if require_non_empty_line:
            if not (np.isfinite(arr).all() and arr.shape[0] >= 2):
                continue
        out.append(arr)
    return out


def _first_polyline_mm_from_entry(
    entry: Dict[str, Any],
    field: str,
    *,
    require_non_empty_line: bool,
) -> Tuple[np.ndarray, bool]:
    polys = _polylines_mm_from_lines_field(
        entry.get(field) or [], require_non_empty_line=require_non_empty_line
    )
    if not polys:
        return np.zeros((0, 2), dtype=float), False
    arr = polys[0]
    if require_non_empty_line:
        ok = bool(arr.size) and np.isfinite(arr).all() and arr.shape[0] >= 2
    else:
        ok = True
    return arr, ok


def parse_all_lines_from_annotation(
    data: Dict[str, Any],
    *,
    require_non_empty_line: bool = True,
) -> Tuple[List[int], List[List[np.ndarray]], List[List[np.ndarray]], List[bool], List[bool]]:
    """
    Pleura (first line only) plus **all** B-line polylines per frame.

    Returns:
        frame_numbers
        pleura_points_mm: list per frame of zero or more (N,2) array per frame
        b_lines_points_mm: list per frame of zero or more (N,2) arrays
        pleura_valid: first pleura line usable
        b_lines_valid: at least one usable B-line on that frame
    """
    frames = data.get("frame_annotations")
    if not isinstance(frames, list):
        raise ValueError("Annotation JSON has no list 'frame_annotations'.")

    frame_numbers: List[int] = []
    pleura_mm: List[List[np.ndarray]] = []
    blines_mm: List[List[np.ndarray]] = []
    pleura_ok: List[bool] = []
    b_ok: List[bool] = []

    for entry in frames:
        if not isinstance(entry, dict):
            raise ValueError("frame_annotations entries must be objects.")
        if "frame_number" not in entry:
            raise ValueError("frame_annotations entry missing 'frame_number'.")
        fn = int(entry["frame_number"])
        frame_numbers.append(fn)

        p_list = _polylines_mm_from_lines_field(
            entry.get("pleura_lines") or [], require_non_empty_line=require_non_empty_line
        )
        pleura_mm.append(p_list)
        pleura_ok.append(len(p_list) > 0)

        b_list = _polylines_mm_from_lines_field(
            entry.get("b_lines") or [], require_non_empty_line=require_non_empty_line
        )
        blines_mm.append(b_list)
        b_ok.append(len(b_list) > 0)

    return frame_numbers, pleura_mm, blines_mm, pleura_ok, b_ok


def load_ultrasound_frames_from_dicom(
    dicom_path: str,
) -> Tuple[np.ndarray, float, float, Any]:
    """
    Read a cine / multi-frame ultrasound DICOM.

    Returns:
        frames: float or uint array shaped (num_frames, num_channels, height, width)
        pixel_spacing_x, pixel_spacing_y: row/column spacing in mm (IEC 60601)
        ds: pydicom dataset (caller may read SOPInstanceUID, etc.)
    """
    ds = pydicom.dcmread(dicom_path, force=True)

    arr = ds.pixel_array
    # Expected: (frames, rows, cols, channels) per notebook reshape
    if arr.ndim != 4:
        raise ValueError(
            f"DICOM pixel_array has shape {arr.shape}; expected 4D (frames, H, W, C). "
            "If your data differ, adjust load_ultrasound_frames_from_dicom."
        )

    frames = np.moveaxis(arr, 3, 1)

    spacing = getattr(ds, "PixelSpacing", None)
    if spacing is None or len(spacing) < 2:
        raise ValueError("DICOM has no PixelSpacing with two values; cannot convert annotations from mm.")
    pixel_spacing_x = float(spacing[0])
    pixel_spacing_y = float(spacing[1])

    return frames, pixel_spacing_x, pixel_spacing_y, ds
def extract_transducer_metadata(ds: Any,site_str: str) -> Dict[str,Any]:
    """
    Extracts the probe metadata from a pydicom Dataset
    Input:
    ds: datastructure from the dcm file
    site_str: the site from which we are getting data

    Return dict with keys:
    transducer_type
    manufacturer_name
    zone_label (in R1-4/L1-4 format)
    sampling_rate (milliseconds/frame)
    """
    if site_str=='CARVD':
        zone_label=str(getattr(ds,"SeriesDescription",None) or "").strip()
        manufacturer_name=str(getattr(ds,"Manufacturer",None) or "").strip()
        transducer_type=getattr(ds, "TransducerType", None)
        transducer_type=str(transducer_type[0] if transducer_type else "").strip()
        sampling_rate=str(getattr(ds,"FrameTime",None) or "").strip()

    elif site_str=='Lahey':
        zone_label=str(getattr(ds,"SeriesDescription",None) or "")
        zone_label=zone_label.split()[-1] if zone_label.strip() else ""
        manufacturer_name=str(getattr(ds,"Manufacturer",None) or "").strip()

        transducer_type=str(getattr(ds,"ManufacturerModelName",None) or "").strip()
        sampling_rate=str(getattr(ds,"FrameTime",None) or "").strip()
    else:
        # Safe fallback for any unrecognised site
        transducer_type   = ""
        manufacturer_name = ""
        zone_label        = ""
        sampling_rate     = ""
    
    return{
        "transducer_type": transducer_type,
        "manufacturer_name": manufacturer_name,
        "zone_label": zone_label,
        "sampling_rate": sampling_rate,
    }



def convert_to_scanlines(
    frames: np.ndarray,
    config_dict: Dict[str, Any],
    num_lines: int = 128,
    num_samples_per_line: int = 128,
) -> Tuple[Optional[np.ndarray], Dict[str, Any]]:
    """Convert (num_frames, num_channels, H, W) curvilinear frames to scanline tensor."""
    num_frames = frames.shape[0]
    num_channels = frames.shape[1]

    if num_frames == 0:
        return None, {}

    num_rows_curvilinear = frames.shape[2]
    num_columns_curvilinear = frames.shape[3]

    scanconversion_config: Dict[str, Any] = {
        "num_samples_along_lines": num_samples_per_line,
        "num_lines": num_lines,
        "num_cartesian_image_rows": num_rows_curvilinear,
        "num_cartesian_image_cols": num_columns_curvilinear,
    }

    angle1 = float(config_dict["angle1"])
    angle2 = float(config_dict["angle2"])
    scanconversion_config["angle_min_degrees"] = min(angle1, angle2)
    scanconversion_config["angle_max_degrees"] = max(angle1, angle2)

    scanconversion_config["center_coordinate_pixel"] = [
        int(config_dict["center_rows_px"]),
        int(config_dict["center_cols_px"]),
    ]
    radius1 = float(config_dict["radius1"])
    radius2 = float(config_dict["radius2"])
    scanconversion_config["radius_min_px"] = min(radius1, radius2)
    scanconversion_config["radius_max_px"] = max(radius1, radius2)

    x_cart, y_cart = cartesian_coordinates(scanconversion_config)

    scanlines_data = np.zeros(
        (num_frames, num_channels, num_samples_per_line, num_lines),
        dtype=frames.dtype,
    )

    for frame_idx in range(num_frames):
        for channel_idx in range(num_channels):
            scanlines_data[frame_idx, channel_idx, :, :] = curvilinear_to_scanlines(
                frames[frame_idx, channel_idx, :, :],
                scanconversion_config,
                x_cart,
                y_cart,
            )

    return scanlines_data, scanconversion_config

def cartesian_coordinates(scanconversion_config: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray]:
    angle_min_deg = np.deg2rad(scanconversion_config["angle_min_degrees"])
    angle_max_deg = np.deg2rad(scanconversion_config["angle_max_degrees"])
    radius_min_px = scanconversion_config["radius_min_px"]
    radius_max_px = scanconversion_config["radius_max_px"]

    theta, r = np.meshgrid(
        np.linspace(angle_min_deg, angle_max_deg, scanconversion_config["num_lines"]),
        np.linspace(radius_min_px, radius_max_px, scanconversion_config["num_samples_along_lines"]),
    )

    x_cart = r * np.cos(theta) + scanconversion_config["center_coordinate_pixel"][1]
    y_cart = r * np.sin(theta) + scanconversion_config["center_coordinate_pixel"][0]

    return x_cart, y_cart

def curvilinear_to_scanlines(
    image: np.ndarray,
    scanconversion_config: Dict[str, Any],
    x_cart: np.ndarray,
    y_cart: np.ndarray,
    interpolation_order: int = 1,
) -> np.ndarray:
    num_samples = scanconversion_config["num_samples_along_lines"]
    num_lines = scanconversion_config["num_lines"]
    if len(image.shape) == 2:
        converted_image = np.zeros((num_samples, num_lines))
        converted_image[:, :] = map_coordinates(
            image, [y_cart, x_cart], order=interpolation_order, mode="constant", cval=0.0
        )
    else:
        num_channels = image.shape[0]
        converted_image = np.zeros((num_channels, num_samples, num_lines), dtype=image.dtype)
        for channel in range(num_channels):
            converted_image[channel, :, :] = map_coordinates(
                image[channel, :, :],
                [y_cart, x_cart],
                order=interpolation_order,
                mode="constant",
                cval=0.0,
            )

    return converted_image

def pleura_points_curvilinear_mm_to_scanlines(
    points_mm: np.ndarray,
    pixel_spacing_x: float,
    pixel_spacing_y: float,
    scanconversion_config: Dict[str, Any],
) -> np.ndarray:
    """Annotation points (mm) -> curvilinear pixels -> scanline (line, sample) indices."""
    if points_mm.size == 0:
        return np.zeros((0, 2), dtype=float)
    processed = points_mm.astype(float).copy()
    processed[:, 0] = processed[:, 0] / pixel_spacing_x
    processed[:, 1] = processed[:, 1] / pixel_spacing_y
    return curvilinear_to_scanlines_coordinates(processed, scanconversion_config)

def curvilinear_to_scanlines_coordinates(
    points_curvilinear: np.ndarray, scanconversion_config: Dict[str, Any]
) -> np.ndarray:
    """Map curvilinear pixel (col, row) to scanline indices (line, sample)."""
    points_scanlines = np.zeros_like(points_curvilinear, dtype=float)

    for i in range(points_curvilinear.shape[0]):
        x = float(points_curvilinear[i, 0])
        y = float(points_curvilinear[i, 1])
        cy, cx = scanconversion_config["center_coordinate_pixel"]
        angle = np.rad2deg(np.arctan2(y - cy, x - cx))
        if angle < 0:
            angle += 360
        angle_span = scanconversion_config["angle_max_degrees"] - scanconversion_config["angle_min_degrees"]
        angle_span = angle_span if abs(angle_span) > 1e-9 else 1.0
        line = (angle - scanconversion_config["angle_min_degrees"]) / angle_span * scanconversion_config["num_lines"]
        points_scanlines[i, 0] = line
        radius = np.sqrt((y - cy) ** 2 + (x - cx) ** 2)
        r_span = scanconversion_config["radius_max_px"] - scanconversion_config["radius_min_px"]
        r_span = r_span if abs(r_span) > 1e-9 else 1.0
        sample = (radius - scanconversion_config["radius_min_px"]) / r_span * scanconversion_config["num_samples_along_lines"]
        points_scanlines[i, 1] = sample

    return points_scanlines









####################Clip Sequence Handling Functions#############
def clip_collate_fn_base(clip_batch):
    """
    !!!Depricated, not using right now!!!!!!!!!
    Collate function for when return_mode=='clip' in the AIUSDataset class.
    Images are padded to T_max, keypoints and categories are kept as ragged Python lists.
    e.g., there is no K_max padding. Use pad_keypoints_for_loss() before calling the loss function
    
    Output:
        images: (B,max_t,C,H,W) - zero-padded
        keypoints: list[list[list[FloatTensor]]] (B,T,N_t,K_i,2) - ragged
        categories: list[list[LongTensor]] (B,T,N_t) - ragged
        padding_mask: BoolTensor (B,max_t)
    """
    #Finds the longest clip length in this batch for padding
    max_t=max(item['clip_len'] for item in clip_batch)
    B=len(clip_batch) #Number of batches

    #Get the shape of the first frame of the first clip
    C,H,W=clip_batch[0]['images'].shape[1:]

    #Allocate output tensors
    images_padded=torch.zeros(B,max_t,C,H,W,dtype=torch.float32)
    padding_mask=torch.zeros(B,max_t,dtype=torch.bool)



    for b,item in enumerate(clip_batch): #Loops for all batches and items in this batch of clips
        T=item['clip_len'] #Gets the clip length

        #Fill in with real frames and mark mask true where frames exist (we are padding at the end of the time dimension)
        images_padded[b,:T]=item['images'] 
        padding_mask[b,:T]=True
    
    return {
        'images': images_padded,
        'padding_mask': padding_mask,
        'keypoints': [item['keypoints']  for item in clip_batch],
        'categories': [item['categories']  for item in clip_batch],
        'frame_nums': [item['frame_nums'] for item in clip_batch],
        'px_mul_x': [item['px_mul_x']   for item in clip_batch],
        'px_mul_y': [item['px_mul_y']   for item in clip_batch],
        'clip_len': [item['clip_len']    for item in clip_batch],
        'clip_id': [item['clip_id']    for item in clip_batch],
        'metadata': [item['metadata']    for item in clip_batch],
    }

def clip_collate_fn_fullpad(clip_batch):
    """
    To be used when loading in data with a torch dataloader for clip-based prediction.
    Collate function that pads to T_max (max time dimension) and to max keypoints (K_max).
    """
    B=len(clip_batch) #Num of batches
    max_t=max(item['clip_len'] for item in clip_batch)
    #Get the shape of the first frame of the first clip
    C,H,W=clip_batch[0]['images'].shape[1:]

    #Compute number of keypoints across all frames in the batch
    batch_counts=[count for item in clip_batch for count in item['kp_counts']] #Loops for all keypoint counts across all frames across all batches
    K_max=max(batch_counts)
    K_max=max(K_max,1)

    #Pre-allocate tensors
    images=torch.zeros(B,max_t,C,H,W,dtype=torch.float32)
    keypoints=torch.zeros(B,max_t,K_max,2,dtype=torch.float32)
    bboxes=torch.zeros(B,max_t,K_max,4,dtype=torch.float32)
    areas=torch.zeros(B, max_t, K_max,dtype=torch.float32)
    visibility=torch.zeros(B,max_t,K_max,dtype=torch.bool) #Visibility of keypoints
    categories=torch.full((B, max_t, K_max), -1, dtype=torch.long)
    padding_mask=torch.zeros(B,max_t,dtype=torch.bool)

    #Fill the data tensors:
    for b,item in enumerate(clip_batch):
        T=item['clip_len'] #Gets the clip length

        #Fill in with real frames and mark mask true where frames exist (we are padding at the end of the time dimension)
        images[b,:T]=item['images'] 
        padding_mask[b,:T]=True

        for t in range(T): #Loops for all frames in this clip
            k_t=item['kp_counts'][t] #Gets number of keypoints for this frame
            if k_t==0:
                categories[b,t,:k_t]=item['categories'][t] #Categories without keypoints is -1
                continue #Empty frame, skip
            keypoints[b,t,:k_t]=item['keypoints'][t]
            bboxes[b,t,:k_t]=item['bboxes'][t]
            areas[b,t,:k_t]=item['areas'][t]
            visibility[b,t,:k_t]=True
            categories[b,t,:k_t]=item['categories'][t]
    
    return {
        'images': images,
        'padding_mask': padding_mask,
        'keypoints': keypoints,
        'bboxes': bboxes,
        'areas':areas,
        'visibility': visibility,
        'categories': categories,
        'kp_counts': [item['kp_counts'] for item in clip_batch],
        'frame_nums': [item['frame_nums'] for item in clip_batch],
        'px_mul_x': [item['px_mul_x']   for item in clip_batch],
        'px_mul_y': [item['px_mul_y']   for item in clip_batch],
        'clip_len': [item['clip_len']    for item in clip_batch],
        'clip_id': [item['clip_id']    for item in clip_batch],
        'metadata': [item['metadata']    for item in clip_batch],
    }


   

def frame_collate_fn(frame_batch):
    '''
    Adds the visibility variable to the return list
    and pads to max keypoints (K_max) in the batch.
    frame_batch has shape: (B,C,H,W)
    To be used when loading in data with a torch dataloader for frame-based prediction.
    '''
    B=len(frame_batch) #Num of batches

    #Compute number of keypoints across all frames in the batch
    batch_counts=[item['kp_counts'] for item in frame_batch] #Loops for all keypoint counts across batch
    K_max=max(batch_counts)
    K_max=max(K_max,1)

    keypoints=torch.zeros(B,K_max,2,dtype=torch.float32)
    bboxes=torch.zeros(B,K_max,4,dtype=torch.float32)
    areas=torch.zeros(B,  K_max,dtype=torch.float32)
    visibility=torch.zeros(B,K_max,dtype=torch.bool) #Visibility of keypoints
    categories=torch.full((B,K_max), -1, dtype=torch.long)

    for b,item in enumerate(frame_batch):
        k_t=item['kp_counts'] #Gets number of keypoints for this frame
        if k_t==0:
            continue #Empty frame, skip
        keypoints[b,:k_t]=item['keypoints']
        bboxes[b,:k_t]=item['bboxes']
        areas[b,:k_t]=item['areas']
        visibility[b,:k_t]=True
        categories[b,:k_t]=item['categories']
    return {
        'images': torch.stack([item['image'] for item in frame_batch]),
        'keypoints': keypoints,
        'bboxes': bboxes,
        'areas':areas,
        'visibility': visibility,
        'categories': categories,
        'kp_counts': [item['kp_counts'] for item in frame_batch],
        'frame_nums': [item['frame_num'] for item in frame_batch],
        'px_mul_x': [item['px_mul_x']   for item in frame_batch],
        'px_mul_y': [item['px_mul_y']   for item in frame_batch],
        'clip_id': [item['clip_id']    for item in frame_batch],
        'metadata': [item['metadata']    for item in frame_batch],
    }


#############################Matching, Heatmap Decoding and Error Computation Functions######################

#Hungarian Matching Algorithm Functions:
def hungarian_match_single(pred_kps, pred_cats_logits, target_kps, target_cats, vis_mask):
    """
    Solve the assignment problem for one image (finds closest matching keypoints).

    pred_kps         : (K_pred, 2)
    pred_cats_logits : (K_pred, num_classes)  or  None
    target_kps       : (K_tgt,  2)
    target_cats      : (K_tgt,) long   1=pleural  2=bline  -1=padded
    vis_mask         : (K_tgt,) bool

    Returns
    -------
    pred_idx   : (N_matched,) long — matched row indices in pred
    target_idx : (N_matched,) long — matched row indices in target
    """
    real_tgt_idx = vis_mask.nonzero(as_tuple=False).squeeze(1)   # (N_real,)
    N_real = real_tgt_idx.shape[0]
    K_pred = pred_kps.shape[0]
    if N_real == 0 or K_pred==0:
        return (torch.empty(0, dtype=torch.long,device=pred_kps.device),
                torch.empty(0, dtype=torch.long,device=pred_kps.device)) #Returns empty indexes if there are no visible or predicted keypoints

    

    # Spatial cost: squared L2  (K_pred × N_real)
    p    = pred_kps.unsqueeze(1)                       # (K_pred, 1,      2)
    t    = target_kps[real_tgt_idx].unsqueeze(0)       # (1,      N_real, 2)
    cost = torch.sum((p - t) ** 2, dim=-1)             # (K_pred, N_real)

    # Optional category cost: CE for every pred–target combination
    # This gives the category head gradient signal through the cost matrix
    # NOTE: linear_sum_assignment uses .detach() so gradients do NOT flow
    # through the matching decision itself — only through the loss on matched pairs.
    # Add an explicit category CE loss term in your training loop if you want
    # stronger gradient signal to the category head.
    if pred_cats_logits is not None:
        cat_cost = torch.zeros(K_pred, N_real, device=pred_kps.device)
        for j, tgt_j in enumerate(real_tgt_idx):
            # Category labels are 1-indexed; CE expects 0-indexed
            label          = (target_cats[tgt_j] - 1).clamp(min=0).expand(K_pred)
            cat_cost[:, j] = F.cross_entropy(
                pred_cats_logits, label, reduction='none'
            )
        cost = cost + cat_cost                          # (K_pred, N_real)

    # Solve on CPU (scipy requirement)
    pred_i, tgt_j = linear_sum_assignment(cost.detach().cpu().numpy())

    pred_i = torch.tensor(pred_i, dtype=torch.long,device=pred_kps.device)
    tgt_j  = real_tgt_idx[torch.tensor(tgt_j, dtype=torch.long,device=pred_kps.device)]
    return pred_i, tgt_j

def apply_hungarian_matching( pred, pred_cats_logits, target, visibility, areas, categories):
    """
    Solve the optimal 1-to-1 assignment per image, then repack the
    matched pairs into padded tensors so _compute_keypoint_loss can be
    applied identically to the fixed-matching path.

    Inputs
    ------
    pred             : (B, K_pred, 2)
    pred_cats_logits : (B, K_pred, num_classes)  or  None
    target           : (B, K_tgt,  2)
    visibility       : (B, K_tgt)   bool
    areas            : (B, K_tgt)   float  or  None
    categories       : (B, K_tgt)   long

    Returns  (padded to K_max = largest matched-pair count in the batch)
    -------
    matched_pred  : (B, K_max, 2)
    matched_tgt   : (B, K_max, 2)
    matched_vis   : (B, K_max)    bool   True=real pair, False=padding
    matched_areas : (B, K_max)    float  or  None
    matched_cats  : (B, K_max)    long   category of the matched target
    """

    B=pred.shape[0]
    device=pred.device

    # ---- solve per image -------------------------------------------
    all_pred_idx, all_tgt_idx = [], []
    #Loop through the batch
    for b in range(B):
        pi, tj = hungarian_match_single(
            pred_kps         = pred[b],
            pred_cats_logits = (pred_cats_logits[b]
                                if pred_cats_logits is not None else None),
            target_kps  = target[b],
            target_cats = categories[b],
            vis_mask    = visibility[b],
        )

        #Append to list containing all the matching indices
        all_pred_idx.append(pi)
        all_tgt_idx.append(tj)

    # ---- pad to largest matched count ------------------------------
    counts = [idx.shape[0] for idx in all_pred_idx]
    K_max  = max(counts) if any(c > 0 for c in counts) else 1

    #Pre-allocate tensors
    matched_pred=torch.zeros(B, K_max, 2, dtype=pred.dtype,   device=device)
    matched_tgt=torch.zeros(B, K_max, 2, dtype=target.dtype, device=device)
    matched_vis=torch.zeros(B, K_max,    dtype=torch.bool,   device=device)
    matched_areas=(torch.zeros(B, K_max, dtype=areas.dtype, device=device)
                        if areas is not None else None)
    matched_cats=torch.full((B, K_max), -1, dtype=torch.long, device=device)

    for b in range(B):
        pred_i=all_pred_idx[b].to(device)
        tgt_j=all_tgt_idx[b].to(device)
        N=pred_i.shape[0]
        if N == 0:
            continue
        matched_pred[b, :N]=pred[b][pred_i] #Gets the matching pred and target up to max size of pred
        matched_tgt[b,  :N]=target[b][tgt_j]
        matched_vis[b,  :N]=True
        if areas is not None:
            matched_areas[b, :N]=areas[b][tgt_j]
        matched_cats[b, :N]=categories[b][tgt_j]

    return matched_pred, matched_tgt, matched_vis, matched_areas, matched_cats

#Keypoint Heatmap Handling Functions
def make_target_heatmaps(keypoints, visibility, categories, H_out, W_out, H_in, W_in,line_type,heatmap_sigma):
    """
    !!!!!line_type must the same one passed to decode_heatmaps!!!!!!
    Creates Gaussian heatmaps from target keypoint coordinates.
    keypoints  : (B, K, 2)  pixel coords in input image space (x, y)
    visibility : (B, K)   bool
    categories : (B, K)   long  1=pleural  2=bline  -1=padded
    H_out/W_out : heatmap spatial dimensions
    H_in/W_in   : input image spatial dimensions  (for coordinate scaling)
    line_type: 'bline','pleuraline',or 'both'

    Returns
    -------
    heatmaps      : (B, num_categories, H_out, W_out)  float32
    target_weight : (B, num_categories)                 float32
                    1.0 if channel has >= 1 visible keypoint, else 0.0
    """

    B, K, _ = keypoints.shape
    device  = keypoints.device
    #Get the category id list, and num_categories
    category_ids=LINE_TYPE_TO_IDS[line_type]
    num_categories=len(category_ids)

    scale_x = W_out / W_in
    scale_y = H_out / H_in

    cx = keypoints[:, :, 0] * scale_x  # (B, K)
    cy = keypoints[:, :, 1] * scale_y  # (B, K)

    yy = torch.arange(H_out, dtype=torch.float32, device=device)
    xx = torch.arange(W_out, dtype=torch.float32, device=device)
    grid_y, grid_x = torch.meshgrid(yy, xx, indexing='ij')   # (H_out, W_out)

    heatmaps      = torch.zeros(B, num_categories, H_out, W_out,
                                dtype=torch.float32, device=device)
    target_weight = torch.zeros(B, num_categories,
                                dtype=torch.float32, device=device)
    
    vis_mask = visibility.bool() & (categories > 0)

    for cat_idx,cat_tag in enumerate(category_ids):
        # Boolean mask for keypoints belonging to this category: (B, K)
        cat_mask = vis_mask & (categories == cat_tag)
        if not cat_mask.any():
            continue

        # Broadcast: cx/cy (B,K,1,1) vs grid (1,1,H,W) → Gaussian (B,K,H,W)
        gauss = torch.exp(
            -(  (grid_x[None, None] - cx[:, :, None, None]) ** 2
              + (grid_y[None, None] - cy[:, :, None, None]) ** 2 )
            / (2.0 * heatmap_sigma ** 2)
        )  # (B, K, H_out, W_out)

        # Zero out keypoints not in this category, then max-blend over K
        gauss = gauss * cat_mask[:, :, None, None].float()  # (B, K, H_out, W_out)
        heatmaps[:, cat_idx] = gauss.max(dim=1).values           # (B, H_out, W_out)

        # Channel is active if any keypoint in that image belongs to this category
        target_weight[:, cat_idx] = cat_mask.any(dim=1).float()  # (B,)

    return heatmaps, target_weight

def decode_heatmaps(heatmaps, H_in, W_in,
                    detection_threshold=0.3, nms_kernel=5,line_type='pleuraline'):
    """
    !!!!!line_type must the same one passed to make_target_heatmaps!!!!!!
    Converts spatial heatmaps to keypoint detections via NMS + subpixel refinement.
    heatmaps : (B, C, H', W')
    Returns three lists of length B:
        - pred_kps_list    — each (N_b, 2) float  (x, y) in input-image pixels
        - pred_cats_list   — each (N_b,)   long   1-indexed category
        - pred_scores_list — each (N_b,)   float  peak heatmap score
    """
    B, C, H_out, W_out = heatmaps.shape
    category_ids=LINE_TYPE_TO_IDS[line_type] #Converts line_type to list of id intergers
    # Option A: simple consistent scaling (no +0.5)
    scale_x = W_in / W_out
    scale_y = H_in / H_out

    # NMS via max-pool
    pad = nms_kernel // 2
    hmap_max = F.max_pool2d(
        heatmaps,                        # (B, C, H', W')
        kernel_size=nms_kernel,
        stride=1,
        padding=pad
    )
    peak_mask = (heatmaps == hmap_max) & (heatmaps > detection_threshold)

    pred_kps_list, pred_cats_list, pred_scores_list = [], [], []
    for b in range(B):
        kps_b, cats_b, scores_b = [], [], []
        for c in range(C):
            ys, xs = peak_mask[b, c].nonzero(as_tuple=True)
            for py, px in zip(ys.tolist(), xs.tolist()):
                score = heatmaps[b, c, py, px].item()

                # Subpixel refinement
                rx, ry = float(px), float(py)
                hmap = heatmaps[b, c]
                if 0 < px < W_out - 1: 
                    dx = hmap[py, px + 1] - hmap[py, px - 1]
                    rx += 0.25 * dx.sign().item()
                if 0 < py < H_out - 1:
                    dy = hmap[py + 1, px] - hmap[py - 1, px]
                    ry += 0.25 * dy.sign().item()
                kps_b.append(
                    torch.tensor([[rx * scale_x, ry * scale_y]], dtype=torch.float32,device=heatmaps.device)
                )
                cats_b.append(torch.tensor([category_ids[c]], dtype=torch.long,device=heatmaps.device))
                scores_b.append(score)


        if kps_b:
            pred_kps_list.append(torch.cat(kps_b,    dim=0))
            pred_cats_list.append(torch.cat(cats_b,  dim=0))
            pred_scores_list.append(
                torch.tensor(scores_b, dtype=torch.float32,device=heatmaps.device)   # (N,)
            )
        else:
            # Image has zero detections — return empty tensors so indexing is safe
            pred_kps_list.append(torch.zeros(0, 2,device=heatmaps.device))
            pred_cats_list.append(torch.zeros(0, dtype=torch.long,device=heatmaps.device))
            pred_scores_list.append(torch.zeros(0,device=heatmaps.device))

    return pred_kps_list,pred_cats_list,pred_scores_list

def detect_one_image(pr_kps,gt_kps,threshold):
    """
    Runs Hungarian matching on a single predicted (pr_kps) and ground truth (gt_kps) pair and classifies 
    whether we are keeping the match or not based on a pixel distance threshold
    Returns:
        - n_tp: true positives (predictions matched to a ground truth keypoint within the threshold)
        - n_fp: false positives (predictions not matched or matched too far from a ground truth keypoint within the threshold)
        - n_fn: false negatives (ground truth keypoints unmatched or matched too far)
    """
    N_pred=pr_kps.shape[0] #Number of predictions
    N_gt=gt_kps.shape[0] #Number of ground truth predictions

    if N_pred==0 and N_gt==0: #No predictions or ground truth in this frame
        return 0,0,0
    if N_gt==0:
        return 0,N_pred,0 #second return is the number of false positives, which is just number of predicted
    if N_pred==0:
        return 0,0,N_gt #False negatives is just number of ground truth
    
    diff=pr_kps.unsqueeze(1)-gt_kps.unsqueeze(0) #(N_pred,N_gt,2) difference between predicted and ground truth keypoints
    dist_mat=torch.sqrt((diff**2).sum(dim=-1)) #Euclidean distance matrix (N_pred,N_gt)
    pred_i,tgt_j=linear_sum_assignment(dist_mat.cpu().numpy()) #Gets closest distances indexes in the dist_mat
    n_tp=int((dist_mat[pred_i,tgt_j]<threshold).sum().item()) #Gets the number of true positives (predictions which are mathched to ground truth within the given threshold)
    return n_tp,N_pred-n_tp,N_gt-n_tp   #n_fp=N_pred-n_tp and n_fn=N_gt-n_tp
#Error computation functions

def calculateError(pred,pred_categories,target_keypoints,visibility,areas,categories,
                   return_mode,matching_strategy,line_type,image_shape,px_mul_x,px_mul_y,match_threshold,max_diagnoal):
    """
    Input:
        - pred: output of model (either keypoints or heatmap). (B, K_pred, 2) when keypoints, (B, num_cat, H', W') when heatmap
        - pred_categories: The predicted categories (pleural vs. B-line). (B, K_pred, num_classes)
        - target_keypoints: (B, K, 2)    ground-truth keypoint pixel coordinates
        - visibility: (B, K)  bool, which ground-truth keypoints are visible
        - categories: (B, K)  long    1=pleural  2=bline  -1=padded (ground truth categories)
        - return_mode: 'clip' or 'frame'
        - matching_strategy: 'fixed','hungarian' or 'heatmap'. What is used by the loss function.
        - line_type: 'bline','pleuraline',or 'both'
        - image_shape: tuple of image dimension 
        - match_threshold: maximum distance between keypoints to count as a true positive
    Return:
        - localization_dict:
            - Contains: euc_dist_mm_avg,euc_dist_mm_std,peraxis_mm_err_avg,peraxis_mm_err_std,euc_dist_px_avg,euc_dist_px_std,peraxis_px_err_avg,peraxis_px_err_std,perc_err
        - detect_dict: Dictionary with metrics reflecting how good the dtection was (right number of keypoints)
            - Contains Precision, Recall, F1 and count error (per frame)
    ***Note: in the case when return_mode=='clip' it is (B,T..) dimensions above
    """
    ######If we are doing 'clip' mode
    if return_mode=='clip':
        if matching_strategy in ('fixed','hungarian'):
            if pred.dim() != 4:
                raise ValueError(
                    f"clip + {matching_strategy}: expected 4-D pred "
                    f"(B,T,K,2), got {pred.dim()}-D"
                )
            #convert from (B,T,K,2) to (B*T,K,2)
            B,T,K_pred,_=pred.shape
            pred   = pred.reshape(B * T, K_pred, 2)
            target_keypoints = target_keypoints.reshape(B * T, target_keypoints.shape[2], 2)
            if visibility  is not None: visibility  = visibility.view(B * T, -1)
            if areas       is not None: areas       = areas.view(B * T, -1)
            if categories  is not None: categories  = categories.view(B * T, -1)
            if pred_categories is not None: pred_categories = pred_categories.reshape(B * T, K_pred, -1)
        else: #Doing heatmap matching strategy
            # (B, T, C, H', W') → (B*T, C, H', W')
            if pred.dim() != 5:
                raise ValueError(
                    f"clip + heatmap: expected 5-D pred "
                    f"(B,T,C,H',W'), got {pred.dim()}-D"
                )
            B, T, C, H_out, W_out = pred.shape
            pred   = pred.reshape(B * T, C, H_out, W_out)
            target_keypoints = target_keypoints.reshape(B * T, target_keypoints.shape[2], 2)
            if visibility  is not None: visibility  = visibility.reshape(B * T, -1)
            if areas       is not None: areas       = areas.reshape(B * T, -1)
            if categories  is not None: categories  = categories.reshape(B * T, -1)
        B_eff = B * T

    elif return_mode=='frame':
        expected = 3 if matching_strategy in ('fixed', 'hungarian') else 4
        if pred.dim() != expected:
            raise ValueError(
                f"frame + {matching_strategy}: expected {expected}-D pred, "
                f"got {pred.dim()}-D"
            )
        B_eff = pred.shape[0] #Effective number of batches
    else:
        raise ValueError(f"Error with return_mode, got pred dim: {pred.dim()}D and return_mode: {return_mode}")

    H_in, W_in   = image_shape #Gets the image shape
    device     = pred.device #Gets the device

    #Single scaling tensor that is used to convert pixels to mm
     # ── Step 2: Build per-image scale tensor ──────────────────────────────
    #
    # Result: (B_eff, 1, 2) tensor so it broadcasts over (B_eff, K, 2).
    # Each image keeps its own pixel→mm conversion — no averaging.
    if len(px_mul_x) > 0 and isinstance(px_mul_x[0], (list, tuple)):
        # clip mode: flatten (B, T) → (B*T,)
        sx_flat = [float(v) for sublist in px_mul_x for v in sublist]
        sy_flat = [float(v) for sublist in px_mul_y for v in sublist]
    else:
        # frame mode: already length B
        sx_flat = [float(v) for v in px_mul_x]
        sy_flat = [float(v) for v in px_mul_y]

    # (B_eff, 2) → unsqueeze → (B_eff, 1, 2)
    scale = torch.tensor(
        list(zip(sx_flat, sy_flat)),
        dtype=torch.float32,
        device=device,
    ).unsqueeze(1)  # (B_eff, 1, 2)
    
    #Must get keypoints from the heatmap if model returns a heatmap
    if matching_strategy=='heatmap':
        #Convert predicted heatmap to keypoint coordinates        
        pred_list,pred_categories_list,_=decode_heatmaps(pred, H_in, W_in,detection_threshold=0.3, nms_kernel=5,line_type=line_type)

        # Pad variable-length per-image detections into (B, K_max, 2) tensors
        K_max = max((p.shape[0] for p in pred_list), default=0)
        K_max = max(K_max, 1)

        pred_kps_batch  = torch.zeros(B_eff, K_max, 2,dtype=torch.float32,device=device)
        pred_cats_batch = torch.full((B_eff, K_max), -1, dtype=torch.long,device=device)
        pred_vis_batch  = torch.zeros(B_eff, K_max, dtype=torch.bool,device=device)

        for b in range(B_eff):
            n = pred_list[b].shape[0]
            if n > 0:
                pred_kps_batch[b,  :n] = pred_list[b]
                pred_cats_batch[b, :n] = pred_categories_list[b]
                pred_vis_batch[b,  :n] = True

        #Update the variables
        pred=pred_kps_batch
        pred_cats_labels=pred_cats_batch
        pred_vis=pred_vis_batch
        
    else: #fixed or hungarian (model predicts keypoints)
        if pred_categories is not None:
            pred_cats_labels=(pred_categories.argmax(dim=-1) + 1
                                if pred_categories.dim() == 3
                                else pred_categories)
        else:
            pred_cats_labels=None
        
        #All prediction slots are considered visible
        pred_vis=torch.ones(pred.shape[:2], dtype=torch.bool, device=device)

    
    #########Apply the hungarian matching############# or not if we are using 'fixed'
    if matching_strategy=='fixed':
        # Model is Index-aligned by design (K_pred == K_tgt); no matching needed
        matched_pred  = pred
        matched_tgt   = target_keypoints
        matched_vis   = (visibility > 0)
        matched_areas = areas
        matched_cats  = categories
    else:
        # Hungarian matching for both 'hungarian' and decoded 'heatmap' coords
        logits_for_match = (
            pred_categories
            if matching_strategy == 'hungarian'
               and pred_categories is not None
               and pred_categories.dim() == 3
            else None
        )

        matched_pred, matched_tgt, matched_vis, matched_areas, matched_cats=apply_hungarian_matching(pred,logits_for_match,target_keypoints,visibility,areas,categories)


    #########Compute the localization accuracy (how close are keypoints to ground truth)##########
    #Create a vis_mask that invisible/padded keypoints are zeroid out
    vis_mask = matched_vis.unsqueeze(-1).float()        # (B_eff, K, 1)
    vis_flat = matched_vis.flatten()                     # (B_eff*K,)  bool
    n_vis    = matched_vis.sum().float().clamp(min=1.0) # scalar

    #All errors are computed as average across B_eff and K (where B_eff=B in frame, or B_eff=B*T in clip mode)
    #matched_pred and matched_tgt have dimensions: (B_eff, K, 2)
    #Compute euclidean distance and std in pixel
    euc_dist_px=torch.sqrt(
        (((matched_pred-matched_tgt)**2)*vis_mask).sum(dim=2)
        ) #Euclidean dist across (B_eff,K)
    euc_dist_px_avg=euc_dist_px.sum()/n_vis
    euc_dist_px_std=euc_dist_px.flatten()[vis_flat].std(correction=0) #Gets the standard deviation of only visible points

    #Compute per-axis pixel error
    peraxis_px_err=torch.abs(matched_pred-matched_tgt)*vis_mask #(B_eff,K,2)
    peraxis_px_err_avg = peraxis_px_err.sum(dim=(0, 1)) / n_vis
    peraxis_flat=peraxis_px_err.reshape(-1, 2)     # (B_eff*K, 2)
    visible_errors   = peraxis_flat[vis_flat] #(n_vis,2)
    peraxis_px_err_std=visible_errors.std(dim=0,correction=0) #Per-axis std (2,)

    #Errors in mm
    matched_pred_mm=matched_pred*scale
    matched_tgt_mm=matched_tgt*scale

    #Euc distance in mm
    euc_dist_mm=torch.sqrt(
        (((matched_pred_mm-matched_tgt_mm)**2)*vis_mask).sum(dim=2)
        ) #Euclidean dist across (B_eff,K)
    euc_dist_mm_avg=euc_dist_mm.sum()/n_vis
    euc_dist_mm_std=euc_dist_mm.flatten()[vis_flat].std(correction=0)

    #Compute per-axis mm error
    peraxis_mm_err= torch.abs(matched_pred_mm - matched_tgt_mm) * vis_mask  # (B_eff, K, 2)
    peraxis_mm_err_avg= peraxis_mm_err.sum(dim=(0, 1)) / n_vis                  # (2,)
    peraxis_mm_flat= peraxis_mm_err.reshape(-1, 2)                            # (B_eff*K, 2)
    visible_mm_errors= peraxis_mm_flat[vis_flat]                                # (n_vis, 2)
    peraxis_mm_err_std= visible_mm_errors.std(dim=0,correction=0)                             # (2,)
    #Compute the percentage error (euclidean pixel error divided by maximum diagonal)

    perc_err=(float(euc_dist_px_avg)/max_diagnoal)*100.0

    #Assign localization dict:
    localization_dict={
        'euc_dist_mm_avg':euc_dist_mm_avg,
        'euc_dist_mm_std':euc_dist_mm_std,
        'peraxis_mm_err_avg':peraxis_mm_err_avg,
        'peraxis_mm_err_std':peraxis_mm_err_std,
        'euc_dist_px_avg':euc_dist_px_avg,
        'euc_dist_px_std':euc_dist_px_std,
        'peraxis_px_err_avg':peraxis_px_err_avg,
        'peraxis_px_err_std':peraxis_px_err_std,
        'perc_err':perc_err,
    }


    ##################Compute the detection accuracy (did we estimate the right number of keypoints)######################
    """
    We complete a per-category hungarian matching
    The following numbers are computed for each (image,category/class):
    - TP = pred keypoints matched to ground truth keypoints and the distance is < matched_threshold
    - FP = pred keypoints are unmatched or too far awar from the ground truth keypoint
    - FN = ground truth is unmatched or matches are above threshold

    count_error_overall tracks |N_pred_total-N_gt_total| per image
    """

    has_cat_info=pred_cats_labels is not None #See if we have category (class) information

    #Overall count error (total preds vs total ground thruth per image, not accounting for categories)
    overall_count_errors = []
    for b in range(B_eff):
        N_pred_b = int(pred_vis[b].sum().item())
        N_gt_b   = int((visibility[b] > 0).sum().item())
        overall_count_errors.append(abs(N_pred_b - N_gt_b))

    #Per-category metrics
    cat_results={}
    all_tp,all_fp,all_fn=0,0,0 #tp,fp,fn for this batch


    category_ids = LINE_TYPE_TO_IDS[line_type]
    for cat in category_ids: #Loops for each category
        tp_c,fp_c,fn_c=0,0,0 #results for this category
        count_errors_c=[] #

        for b in range (B_eff): #Loops for each batch for this category
            #Ground truth keypoints of this category
            gt_mask = (visibility[b] > 0) & (categories[b] == cat) #Mask of visible GT points for this category
            gt_kps  = target_keypoints[b][gt_mask]   # (N_gt, 2)
            N_gt    = gt_kps.shape[0] #Number of ground truth keypoints

            #Predicted keypoints of this category
            if has_cat_info:
                pr_mask=pred_vis[b] & (pred_cats_labels[b]==cat) #Get the visible mask for predicted keypoints for this category
            else:
                #No category predictions, 
                pr_mask=pred_vis[b]
            pr_kps=pred[b][pr_mask] #(N_pred,2)
            N_pred=pr_kps.shape[0]

            count_errors_c.append(abs(N_pred-N_gt)) #Difference between predicted and ground truth for this category
            n_tp,n_fp,n_fn=detect_one_image(pr_kps,gt_kps,match_threshold) #Compute the number of tp, fp, and fn for this category predictions
            
            #Updates accumulators for the B_eff loop
            tp_c+=n_tp
            fp_c+=n_fp
            fn_c+=n_fn
        #Updates accumulators for overall stats across categories
        all_tp += tp_c
        all_fp += fp_c
        all_fn += fn_c

        #Compute the precision, recall and f1 values per category
        precision=tp_c/(tp_c+fp_c) if (tp_c+fp_c) > 0 else float('nan')
        recall=tp_c/(tp_c+fn_c) if (tp_c+fn_c) > 0 else float('nan') 
        f1=(2*precision*recall)/(precision+recall) if not any(np.isnan([precision,recall])) and (precision+recall) > 0 else float('nan') 

        #Get the category name and store the per-category results
        cat_name = CATEGORY_NAMES.get(cat, f'cat_{cat}')
        cat_results[cat_name] = {
            'precision':        precision,
            'recall':           recall,
            'f1':               f1,
            'count_error_mean': float(np.mean(count_errors_c)),
            'count_error_std':  float(np.std(count_errors_c)),
            # Fraction of frames where the model predicted exactly the right
            # number of keypoints for this category
            'count_exact_acc':  float(np.mean([e == 0 for e in count_errors_c])),
            #Add the raw totals for per-frame aggregation
            'count_error_sum':  float(np.sum(count_errors_c)),
            'count_exact_sum':  float(sum(e == 0 for e in count_errors_c)),
            'n_frames':         len(count_errors_c),
            'tp': tp_c,
            'fp': fp_c,
            'fn': fn_c,
        }
    
    #Compute the overall recall,precision and f1 and then save in the detect_dict
    overall_prec = all_tp / (all_tp + all_fp) if (all_tp + all_fp) > 0 else float('nan')
    overall_rec  = all_tp / (all_tp + all_fn) if (all_tp + all_fn) > 0 else float('nan')
    overall_f1   = (2 * overall_prec * overall_rec / (overall_prec + overall_rec)
                    if not any(np.isnan([overall_prec, overall_rec]))
                       and (overall_prec + overall_rec) > 0
                    else float('nan'))

    detect_dict = {
        'per_category': cat_results,
        'overall': {
            'precision':        overall_prec,
            'recall':           overall_rec,
            'f1':               overall_f1,
            # Overall count error = |total preds − total GT| per frame
            # NOT the sum of per-category errors
            'count_error_mean': float(np.mean(overall_count_errors)),
            'count_error_std':  float(np.std(overall_count_errors)),
            'count_exact_acc':  float(np.mean([e == 0 for e in overall_count_errors])),
            #Raw totals for per-frame aggregation
            'count_error_sum':  float(np.sum(overall_count_errors)),
            'count_exact_sum':  float(sum(e == 0 for e in overall_count_errors)),
            'n_frames':         len(overall_count_errors),
            'tp': all_tp,
            'fp': all_fp,
            'fn': all_fn,
        }
    }

    return localization_dict,detect_dict


##########################Helpers to Aggregate and Average Error Measures##############
def _localization_dict_to_serializable(d):
    """Convert any tensors in a localization_dict to plain Python types."""
    out = {}
    for k, v in d.items():
        if isinstance(v, torch.Tensor):
            out[k] = v.detach().cpu().tolist()   # scalar → float, (2,) → [x, y]
        else:
            out[k] = v
    return out

def average_localization_dict_overbatches(localization_dict_list):
    """
    localization_dict_list is a list of localization_dict's from calculateError, we want to get average of each value across all the batches that we accumulate this list of dicts
    """
    if not localization_dict_list:
        return {}
    #init the localization dict batch avg
    result = {}
    keys = localization_dict_list[0].keys()

    for key in keys:
        values = [d[key] for d in localization_dict_list]

        if isinstance(values[0], torch.Tensor):
            # peraxis_* fields are (2,) tensors — stack to (N_batches, 2) then nanmean
            stacked    = torch.stack([v.float() for v in values])  # (N_batches, ...)
            nan_mask   = torch.isnan(stacked)
            zeroed     = stacked.clone()
            zeroed[nan_mask] = 0.0
            valid_count = (~nan_mask).float().sum(dim=0).clamp(min=1.0)
            avg = zeroed.sum(dim=0) / valid_count
            # Restore NaN where every batch was NaN
            avg[nan_mask.all(dim=0)] = float('nan')
            result[key] = avg

        else:
            arr = np.array([float(v) for v in values], dtype=np.float64)
            result[key] = float(np.nanmean(arr))

    return result

def average_localization_dict_serialized(localization_dict_list):
    """
    Average a list of already-serialized localization dicts (loaded from JSON).
    Values are Python floats or [x, y] lists — no tensors.
    Used by plotting functions only.
    """
    if not localization_dict_list:
        return {}
    result = {}
    for key in localization_dict_list[0]:
        vals = [d[key] for d in localization_dict_list if key in d and d[key] is not None]
        if not vals:
            continue
        if isinstance(vals[0], (list, tuple)):
            result[key] = list(np.nanmean(np.array(vals, dtype=np.float64), axis=0))
        else:
            result[key] = float(np.nanmean(np.array([float(v) for v in vals], dtype=np.float64)))
    return result

def _true_per_frame(dicts_subset):
        """Pool count_error_sum/count_exact_sum/n_frames across all batches,
        then divide once — gives true per-frame mean rather than mean of batch means."""
        total_err_sum   = sum(d.get('count_error_sum', 0.0) for d in dicts_subset)
        total_exact_sum = sum(d.get('count_exact_sum', 0.0) for d in dicts_subset)
        total_n         = sum(d.get('n_frames',        0)   for d in dicts_subset)
        return (
            total_err_sum   / total_n if total_n > 0 else float('nan'),  # true count_error_mean
            total_exact_sum / total_n if total_n > 0 else float('nan'),  # true count_exact_acc
        )

def average_detection_dict_overbatches(detect_dict_list):
    """
    Aggregates detection metrics across batches.

    Precision / Recall / F1
        Micro-averaged: TP, FP, FN are SUMMED across batches then the
        metrics are RECOMPUTED from the totals. This is the statistically
        correct approach and is consistent with COCO / MMPose evaluation.

    tp / fp / fn
        Summed (they are raw integer counts).

    count_error_mean / count_error_std / count_exact_acc
        nanmean across batches — these are already per-frame statistics
        so averaging batch estimates is appropriate.

    detect_dict_list : list of detect_dicts returned by calculateError
    """
    if not detect_dict_list:
        return {}

    # ── Overall ──────────────────────────────────────────────────────────────
    overall_dicts = [d['overall'] for d in detect_dict_list if 'overall' in d]

    all_tp = sum(d.get('tp', 0) for d in overall_dicts)
    all_fp = sum(d.get('fp', 0) for d in overall_dicts)
    all_fn = sum(d.get('fn', 0) for d in overall_dicts)

    overall_prec = all_tp / (all_tp + all_fp) if (all_tp + all_fp) > 0 else float('nan')
    overall_rec  = all_tp / (all_tp + all_fn) if (all_tp + all_fn) > 0 else float('nan')
    overall_f1   = (2 * overall_prec * overall_rec / (overall_prec + overall_rec)
                    if not any(np.isnan([overall_prec, overall_rec]))
                       and (overall_prec + overall_rec) > 0
                    else float('nan'))

    # ↓ true per-frame values using pooled sums
    count_err_mean, count_exact_acc = _true_per_frame(overall_dicts)

    averaged = {
        'overall': {
            'precision':        overall_prec,
            'recall':           overall_rec,
            'f1':               overall_f1,
            'count_error_mean': count_err_mean,    # ← now true per-frame
            'count_exact_acc':  count_exact_acc,   # ← now true per-frame
            'tp': all_tp,
            'fp': all_fp,
            'fn': all_fn,
        },
        'per_category': {}
    }

    # ── Per-category ─────────────────────────────────────────────────────────
    all_cat_names = {
        cat
        for d in detect_dict_list
        for cat in d.get('per_category', {})
    }

    for cat in all_cat_names:
        cat_dicts = [
            d['per_category'][cat]
            for d in detect_dict_list
            if cat in d.get('per_category', {})
        ]
        tp_c = sum(d.get('tp', 0) for d in cat_dicts)
        fp_c = sum(d.get('fp', 0) for d in cat_dicts)
        fn_c = sum(d.get('fn', 0) for d in cat_dicts)

        prec = tp_c / (tp_c + fp_c) if (tp_c + fp_c) > 0 else float('nan')
        rec  = tp_c / (tp_c + fn_c) if (tp_c + fn_c) > 0 else float('nan')
        f1   = (2 * prec * rec / (prec + rec)
                if not any(np.isnan([prec, rec])) and (prec + rec) > 0
                else float('nan'))

        # ↓ true per-frame values using pooled sums
        count_err_mean_c, count_exact_acc_c = _true_per_frame(cat_dicts)

        averaged['per_category'][cat] = {
            'precision':        prec,
            'recall':           rec,
            'f1':               f1,
            'count_error_mean': count_err_mean_c,  
            'count_exact_acc':  count_exact_acc_c, 
            'tp': tp_c,
            'fp': fp_c,
            'fn': fn_c,
        }

    return averaged


#Helpers for computing gaverages of the detection dict:
def _recompute_prf(tp, fp, fn):
    """Recompute precision / recall / F1 from accumulated counts."""
    precision = tp / (tp + fp) if (tp + fp) > 0 else float('nan')
    recall    = tp / (tp + fn) if (tp + fn) > 0 else float('nan')
    f1 = (
        2 * precision * recall / (precision + recall)
        if not any(np.isnan([precision, recall]))
            and (precision + recall) > 0
        else float('nan')
    )
    return precision, recall, f1


def _aggregate_bucket(bucket_dicts):
    """
    Aggregate a list of same-bucket metric dicts from successive batches.

    bucket_dicts : list[dict], each dict has keys:
        precision, recall, f1,
        count_error_mean, count_error_std, count_exact_acc,
        tp, fp, fn
    """
    # ── Micro-average for detection metrics ────────────────────────────
    total_tp = sum(d['tp'] for d in bucket_dicts)
    total_fp = sum(d['fp'] for d in bucket_dicts)
    total_fn = sum(d['fn'] for d in bucket_dicts)
    precision, recall, f1 = _recompute_prf(total_tp, total_fp, total_fn) #We can't just take the average

    # ── Nanmean for per-frame count statistics ─────────────────────────
    # count_error_std is the within-batch std of per-frame count errors.
    # Nanmean gives the mean within-batch variability across the epoch.
    count_error_means = np.array(
        [d['count_error_mean'] for d in bucket_dicts], dtype=np.float64
    )
    count_error_stds = np.array(
        [d['count_error_std'] for d in bucket_dicts], dtype=np.float64
    )
    count_exact_accs = np.array(
        [d['count_exact_acc'] for d in bucket_dicts], dtype=np.float64
    )

    return {
        'precision':        precision,
        'recall':           recall,
        'f1':               f1,
        'count_error_mean': float(np.nanmean(count_error_means)),
        'count_error_std':  float(np.nanmean(count_error_stds)),
        'count_exact_acc':  float(np.nanmean(count_exact_accs)),
        'tp': total_tp,
        'fp': total_fp,
        'fn': total_fn,
    }



#######################Plotting and Statistic Computation Functions####################
#Summary of plotting and statistic computation functions:
"""
computeStats: Prints & saves mean±std for every metric; reports best epoch for train/valid
plotTrainingLoss: Left: raw per-batch loss trace with epoch markers; Right: per-epoch mean±std with best-epoch line
plotLocalizationMetrics: 2x5 grid — each column one metric (Euc mm, Euc px, X err, Y err, % err), top=train, bottom=valid, with ±1 std shading
plotDetectionMetrics: 2x5 grid — each column one metric (P, R, F1, count error, count exact acc), one line per category + bold overall
plotTestBoxplots: Box plots of per-batch localization and detection distributions on the test set
plotPerCategoryMetrics: Grouped P/R/F1 bar chart per category (micro-averaged over test set), values annotated
plotTPFPFN: Stacked TP/FP/FN bars per epoch — instantly reveals whether FP or FN is the dominant error
plotTestErrorHistogram: Density histograms of Euc distance (mm) and per-axis X/Y error with mean±std annotations
"""

######Helper Functions#########
def _loc_series(epoch_dicts, key):
    """Extract a scalar localization metric across epoch dicts → np.array."""
    out = []
    for d in epoch_dicts:
        v = d.get(key, float('nan'))
        try:
            out.append(float(v))
        except (TypeError, ValueError):
            out.append(float('nan'))
    return np.array(out)

def _loc_series_axis(epoch_dicts, key, axis):
    """Extract per-axis (0=X, 1=Y) localization metric across epoch dicts → np.array."""
    out = []
    for d in epoch_dicts:
        v = d.get(key)
        if v is None or not isinstance(v, (list, tuple)) or len(v) <= axis:
            out.append(float('nan'))
        else:
            out.append(float(v[axis]))
    return np.array(out)

def _det_series(epoch_dicts, key, category='overall'):
    """Extract a detection metric across epoch dicts → np.array."""
    out = []
    for d in epoch_dicts:
        if category == 'overall':
            v = d.get('overall', {}).get(key, float('nan'))
        else:
            v = d.get('per_category', {}).get(category, {}).get(key, float('nan'))
        out.append(float(v) if v is not None else float('nan'))
    return np.array(out)

def _get_categories(det_logger):
    """Discover category names from the first valid epoch dict."""
    for d in det_logger:
        cats = list(d.get('per_category', {}).keys())
        if cats:
            return cats
    return []

def _nan_safe_json(obj):
    """Recursively convert float NaN → None for JSON serialisation."""
    if isinstance(obj, dict):
        return {k: _nan_safe_json(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_nan_safe_json(v) for v in obj]
    if isinstance(obj, float) and np.isnan(obj):
        return None
    return obj

def _safe_save(fig, save_file):
    if save_file:
        os.makedirs(os.path.dirname(save_file) or '.', exist_ok=True)
        fig.savefig(save_file, format='svg', dpi=600, bbox_inches='tight')


# ─────────────────────────────────────────────────────────────────────────────
# 1.  Statistics summary computation
# ─────────────────────────────────────────────────────────────────────────────

def computeStats(test_logger, train_logger=None, valid_logger=None, save_file=None):
    """
    Compute, print and optionally save summary statistics.

    Train / Valid loggers  → already per-epoch averages, read directly.
    Test logger            → flat per-batch, micro-averaged here.
    """
    results = {}
    # ── Train / Valid  — already per-epoch averages, read directly ───────────
    for split, logger in [('train', train_logger), ('valid', valid_logger)]:
        if logger is None or not logger.get('loss'):
            continue
        raw_losses = logger['loss']

        if raw_losses and isinstance(raw_losses[0], list):
            epoch_losses = np.array(
                [np.nanmean(ep) for ep in raw_losses], dtype=np.float64
            )
        else:
            # already per-epoch scalars
            epoch_losses = np.array(raw_losses, dtype=np.float64)

        best_epoch = int(np.nanargmin(epoch_losses))
        n_loc = len(logger.get('localization_dict', []))
        n_det = len(logger.get('detection_dict',    []))

        # Guard: localization_dict / detection_dict may have fewer entries
        # than loss if some epochs were skipped during metric computation
        safe_best_loc = min(best_epoch, n_loc - 1) if n_loc > 0 else -1
        safe_best_det = min(best_epoch, n_det - 1) if n_det > 0 else -1

        final_loc      = logger['localization_dict'][-1] if n_loc > 0 else {}
        final_det      = logger['detection_dict'][-1]    if n_det > 0 else {}
        best_epoch_loc = logger['localization_dict'][safe_best_loc] if safe_best_loc >= 0 else {}
        best_epoch_det = logger['detection_dict'][safe_best_det]    if safe_best_det >= 0 else {}

        results[split] = {
            'num_epochs':            len(epoch_losses),
            'best_epoch':            best_epoch,
            'best_epoch_loss':       float(epoch_losses[best_epoch]),
            'best_epoch_euc_dist_mm': best_epoch_loc.get('euc_dist_mm_avg', float('nan')),
            'best_epoch_precision':  best_epoch_det.get('overall', {}).get('precision', float('nan')),
            'best_epoch_recall':     best_epoch_det.get('overall', {}).get('recall',    float('nan')),
            'best_epoch_f1':         best_epoch_det.get('overall', {}).get('f1',        float('nan')),
            'final_loss':            float(epoch_losses[-1]),
            'final_euc_dist_mm':     final_loc.get('euc_dist_mm_avg', float('nan')),
            'final_precision':       final_det.get('overall', {}).get('precision', float('nan')),
            'final_recall':          final_det.get('overall', {}).get('recall',    float('nan')),
            'final_f1':              final_det.get('overall', {}).get('f1',        float('nan')),
        }
        r = results[split]
        print("="*60)
        print(f"  {split.upper()} RESULTS  (num_epochs: {r['num_epochs']}):")
        print(f"  Best Epoch                  : {best_epoch} ")
        print(f"  Best Epoch Loss             : {r['best_epoch_loss']:.4f} ")
        print(f"  Best Euclidean Distance (mm): {r['best_epoch_euc_dist_mm']:.4f} ")
        print(f"  Best Precision; Recall; F1  : "
              f"{r['best_epoch_precision']:.4f}; {r['best_epoch_recall']:.4f}; {r['best_epoch_f1']:.4f}")
        print(f"  Final Loss                  : {r['final_loss']:.4f}")
        print(f"  Final Euclidean Dist (mm)   : {r['final_euc_dist_mm']:.4f}")
        print(f"  Final Precision; Recall; F1 : "
              f"{r['final_precision']:.4f}; {r['final_recall']:.4f}; {r['final_f1']:.4f}")


    # ── Test ─────────────────────────────────────────────────────────────────
    if test_logger is not None:
        loc_avg = test_logger['avg_localization']
        det_avg = test_logger['avg_detection']
        avg_loss=test_logger['avg_loss']

        #STD still needs the raw per-batch losses
        raw_losses=test_logger['raw_logger']['loss']
        loss_std=float(np.nanstd(raw_losses)) if raw_losses else float('nan')

        #Per axis error
        pa = loc_avg.get('peraxis_mm_err_avg', [float('nan'), float('nan')])

        results['test'] = {
            'loss_mean':               avg_loss,
            'loss_std':                loss_std,
            'euc_dist_mm_mean':        loc_avg.get('euc_dist_mm_avg',  float('nan')),
            'euc_dist_mm_std':         loc_avg.get('euc_dist_mm_std',  float('nan')),
            'euc_dist_px_mean':        loc_avg.get('euc_dist_px_avg',  float('nan')),
            'peraxis_mm_x_mean':       float(pa[0]) if isinstance(pa, (list,tuple)) else float('nan'),
            'peraxis_mm_y_mean':       float(pa[1]) if isinstance(pa, (list,tuple)) and len(pa)>1 else float('nan'),
            'perc_err_mean':           loc_avg.get('perc_err', float('nan')),
            'overall_precision':       det_avg.get('overall',{}).get('precision',        float('nan')),
            'overall_recall':          det_avg.get('overall',{}).get('recall',           float('nan')),
            'overall_f1':              det_avg.get('overall',{}).get('f1',               float('nan')),
            'overall_count_err_mean':  det_avg.get('overall',{}).get('count_error_mean', float('nan')),
            'overall_count_exact_acc': det_avg.get('overall',{}).get('count_exact_acc',  float('nan')),
        }
        for cat, cd in det_avg.get('per_category', {}).items():
            results['test'][f'{cat}_precision'] = cd.get('precision', float('nan'))
            results['test'][f'{cat}_recall']    = cd.get('recall',    float('nan'))
            results['test'][f'{cat}_f1']        = cd.get('f1',        float('nan'))
            results['test'][f'{cat}_count_err'] = cd.get('count_error_mean', float('nan'))

        r = results['test']
        print("="*60)
        print("TEST RESULTS:")
        print(f"  Loss                        : {r['loss_mean']:.4f} ± {r['loss_std']:.4f}")
        print(f"  Euclidean Distance (mm)     : {r['euc_dist_mm_mean']:.4f} ± {r['euc_dist_mm_std']:.4f}")
        print(f"  Euclidean Distance (px)     : {r['euc_dist_px_mean']:.4f}")
        print(f"  Per-axis Error:       X (mm): {r['peraxis_mm_x_mean']:.4f}, Y (mm): {r['peraxis_mm_y_mean']:.4f}")
        print(f"  Percentage Error            : {r['perc_err_mean']:.2f} %")
        print(f"  Overall Precision           : {r['overall_precision']:.4f}")
        print(f"  Overall Recall              : {r['overall_recall']:.4f}")
        print(f"  Overall F1                  : {r['overall_f1']:.4f}")
        print(f"  Count Error Mean            : {r['overall_count_err_mean']:.4f}")
        print(f"  Count Exact Accuracy        : {r['overall_count_exact_acc']:.4f}")
        for cat in det_avg.get('per_category', {}):
            print(f"  [{cat:<10s}]: "
                  f"P={r[f'{cat}_precision']:.4f}, "
                  f"R={r[f'{cat}_recall']:.4f}, "
                  f"F1={r[f'{cat}_f1']:.4f}, "
                  f"CountErr={r[f'{cat}_count_err']:.3f}")
            

    if save_file:
        os.makedirs(os.path.dirname(save_file) or '.', exist_ok=True)
        with open(save_file, 'w') as f:
            json.dump(_nan_safe_json(results), f, indent=2)
        print(f"\n  Stats saved → {save_file}")

    return results


# ─────────────────────────────────────────────────────────────────────────────
# 2.  Training Loss Plotter
# ─────────────────────────────────────────────────────────────────────────────
#Averaging helper:
def _rolling(arr, w=5):
    """Rolling mean from epoch 0 using expanding window for first w-1 epochs."""
    out = np.zeros(len(arr))
    for i in range(len(arr)):
        out[i] = np.nanmean(arr[max(0, i - w + 1): i + 1])
    return out

def plotTrainingLoss(train_logger, valid_logger=None, save_file=None):
    """
    Two-panel figure.
    Left  : per-epoch mean loss for train (+ valid if supplied),
            ±1 std shading using stored 'loss_std'.
    Right : Smoothed version (5-epoch rolling mean) to show the trend
            more clearly if training is noisy.
    """
    raw_train = train_logger.get('loss', [])
    if not raw_train:
        return
    #Gets the per-epoch mean and std
    if isinstance(raw_train[0], list):
        train_means = np.array([np.nanmean(ep) for ep in raw_train], dtype=np.float64)
    else:
        train_means = np.array(raw_train, dtype=np.float64)
    train_stds = np.array(train_logger.get('loss_std', np.zeros(len(train_means))), dtype=np.float64)
    if len(train_means) == 0:
        return

    #Number of epochs
    epochs = np.arange(len(train_means))
    best_valid_epoch = None

    valid_means = valid_stds = v_epochs = None
    if valid_logger and valid_logger.get('loss'):
        raw_valid = valid_logger.get('loss', [])
        if isinstance(raw_valid[0], list):
            valid_means = np.array([np.nanmean(ep) for ep in raw_valid], dtype=np.float64)
        else:
            valid_means = np.array(raw_valid, dtype=np.float64)
        valid_stds       = np.array(valid_logger.get('loss_std', np.zeros(len(valid_means))), dtype=np.float64)
        v_epochs         = np.arange(len(valid_means))
        best_valid_epoch = int(np.nanargmin(valid_means))

    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle('Training Loss', fontsize=14, fontweight='bold')

    # ── Left: mean ± std per epoch ───────────────────────────────────────────
    axes[0].plot(epochs, train_means, color='steelblue', linewidth=2.0, label='Train')
    axes[0].fill_between(epochs,
                          np.maximum(0, train_means - train_stds),
                          train_means + train_stds,
                          color='steelblue', alpha=0.2)
    if valid_means is not None:
        axes[0].plot(v_epochs, valid_means, color='orangered', linewidth=2.0, label='Valid')
        axes[0].fill_between(v_epochs,
                              np.maximum(0, valid_means - valid_stds),
                              valid_means + valid_stds,
                              color='orangered', alpha=0.2)
        axes[0].axvline(best_valid_epoch, color='orangered', linewidth=1.5,
                        linestyle=':', label=f'Best valid epoch ({best_valid_epoch})')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Per-epoch Loss ± Batch Std')
    axes[0].legend(fontsize=9)
    axes[0].grid(True, alpha=0.3)

    # ── Right: smoothed trend ────────────────────────────────────────────────
    w = min(5, len(train_means))
    sm_train = _rolling(train_means, w)
    axes[1].plot(epochs, train_means, color='steelblue', linewidth=0.8, alpha=0.4)
    axes[1].plot(epochs,  sm_train,    color='steelblue', linewidth=2.5, label=f'Train ({w}-ep smooth)')
    if valid_means is not None and len(valid_means) >= w:
        sm_valid = _rolling(valid_means, w)
        axes[1].plot(v_epochs, valid_means, color='orangered', linewidth=0.8, alpha=0.4)
        axes[1].plot(v_epochs, sm_valid, color='orangered', linewidth=2.5,
                     label=f'Valid ({w}-ep smooth)')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Loss')
    axes[1].set_title('Smoothed Loss Trend')
    axes[1].legend(fontsize=9)
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    _safe_save(fig, save_file)
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# 3.  Localization Metrics vs Epoch Plotting
# ─────────────────────────────────────────────────────────────────────────────

def plotLocalizationMetrics(train_logger, valid_logger, save_file=None):
    """
    2-row × 5-col figure.  Top = train, Bottom = valid.
    Logger localization_dict is already per-epoch averaged — used directly.
    """
    train_loc = train_logger.get('localization_dict', [])
    valid_loc = valid_logger.get('localization_dict', [])
    if not train_loc and not valid_loc:
        return

    t_ep = np.arange(len(train_loc))
    v_ep = np.arange(len(valid_loc))

    # (mean_key, std_key|None, axis_idx|None, y_label, title)
    metrics = [
        ('euc_dist_mm_avg',    'euc_dist_mm_std',    None, 'Distance (mm)', 'Euclidean Distance (mm)'),
        ('euc_dist_px_avg',    'euc_dist_px_std',    None, 'Distance (px)', 'Euclidean Distance (px)'),
        ('peraxis_mm_err_avg', 'peraxis_mm_err_std', 0,    'Error X (mm)',  'Per-axis Error X (mm)'),
        ('peraxis_mm_err_avg', 'peraxis_mm_err_std', 1,    'Error Y (mm)',  'Per-axis Error Y (mm)'),
        ('perc_err',            None,                None, 'Error (%)',     'Percentage Error (%)'),
    ]

    n_cols = len(metrics)
    fig, axes = plt.subplots(2, n_cols, figsize=(4*n_cols,8))
    fig.suptitle('Localization Metrics vs Epoch', fontsize=14, fontweight='bold')
    # Add shared row labels on the left side
    fig.text(0.01, 0.75, 'Train', va='center', rotation='vertical',
            fontsize=13, fontweight='bold', color='steelblue')
    fig.text(0.01, 0.25, 'Valid', va='center', rotation='vertical',
            fontsize=13, fontweight='bold', color='orangered')

    for col, (mean_key, std_key, axis_idx, ylabel, title) in enumerate(metrics):
        for row, (loc_dicts, ep, color, label) in enumerate([
            (train_loc, t_ep, 'steelblue', 'Train'),
            (valid_loc, v_ep, 'orangered', 'Valid'),
        ]):
            ax = axes[row, col]
            if axis_idx is not None:
                means = _loc_series_axis(loc_dicts, mean_key, axis_idx)
                stds  = _loc_series_axis(loc_dicts, std_key,  axis_idx) if std_key else None
            else:
                means = _loc_series(loc_dicts, mean_key)
                stds  = _loc_series(loc_dicts, std_key)  if std_key else None

            ax.plot(ep[:len(means)], means, color=color, linewidth=2.0, label='Mean')
            if stds is not None:
                ax.fill_between(ep[:len(means)],
                                np.maximum(0, means - stds),
                                means + stds,
                                color=color, alpha=0.2, label='±1 std')
            ax.set_xlabel('Epoch')
            ax.set_ylabel(ylabel)
            ax.set_title(f'{title}')
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)

    plt.tight_layout()
    _safe_save(fig, save_file)
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# 4.  Detection Metrics vs Epoch Plotting
# ─────────────────────────────────────────────────────────────────────────────

def plotDetectionMetrics(train_logger, valid_logger, save_file=None):
    """
    2-row × 5-col figure.  Top = train, Bottom = valid.
    Each subplot: bold black = overall, dashed coloured = per-category.
    Logger detection_dict is already per-epoch averaged — used directly.
    """
    train_det = train_logger.get('detection_dict', [])
    valid_det = valid_logger.get('detection_dict', [])
    if not train_det and not valid_det:
        return

    all_cats   = _get_categories(train_det or valid_det)
    cat_colors = cm.tab10(np.linspace(0.0, 0.7, max(len(all_cats), 1)))
    t_ep = np.arange(len(train_det))
    v_ep = np.arange(len(valid_det))

    det_metrics = [
        ('precision',        'Precision',         (0.0, 1.05)),
        ('recall',           'Recall',             (0.0, 1.05)),
        ('f1',               'F1 Score',           (0.0, 1.05)),
        ('count_error_mean', 'Count Error (mean)', None),
        ('count_exact_acc',  'Count Exact Acc',    (0.0, 1.05)),
    ]

    n_cols = len(det_metrics)
    fig, axes = plt.subplots(2, n_cols, figsize=(4*n_cols,8))
    fig.suptitle('Detection Metrics vs Epoch', fontsize=14, fontweight='bold')
    # Add shared row labels on the left side
    fig.text(0.01, 0.75, 'Train', va='center', rotation='vertical',
            fontsize=13, fontweight='bold', color='steelblue')
    fig.text(0.01, 0.25, 'Valid', va='center', rotation='vertical',
            fontsize=13, fontweight='bold', color='orangered')

    for col, (metric_key, ylabel, ylim) in enumerate(det_metrics):
        for row, (det_dicts, ep, split_label) in enumerate([
            (train_det, t_ep, 'Train'),
            (valid_det, v_ep, 'Valid'),
        ]):
            ax = axes[row, col]
            overall_vals = _det_series(det_dicts, metric_key, 'overall')
            ax.plot(ep[:len(overall_vals)], overall_vals,
                    color='black', linewidth=2.5, linestyle='-', label='Overall', zorder=3)
            for cat, color in zip(all_cats, cat_colors):
                cat_vals = _det_series(det_dicts, metric_key, cat)
                ax.plot(ep[:len(cat_vals)], cat_vals,
                        color=color, linewidth=1.8, linestyle='--', label=cat.capitalize())
            ax.set_xlabel('Epoch')
            ax.set_ylabel(ylabel)
            ax.set_title(f'{ylabel}')
            if ylim:
                ax.set_ylim(*ylim)
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)

    plt.tight_layout()
    _safe_save(fig, save_file)
    plt.close(fig)

# ─────────────────────────────────────────────────────────────────────────────
# 5.  Test-set box plots  (per-batch distributions)
# ─────────────────────────────────────────────────────────────────────────────

def plotTestBoxplots(test_logger, save_file=None):
    """
    Box plots of per-batch test metrics. Each box captures the distribution of
    batch-level means across the test set.

    Left panel  : localization metrics
    Right panel : detection metrics — overall then per-category

    Parameters
    ----------
    test_logger : dict  Flat per-batch logger from modelTester.
    save_file   : str   (optional) SVG save path.
    """
    loc_dicts = test_logger.get('localization_dict', [])
    det_dicts = test_logger.get('detection_dict',    [])
    if not loc_dicts or not det_dicts:
        return

    def _pa(d, i):
        v = d.get('peraxis_mm_err_avg')
        return float(v[i]) if isinstance(v, (list, tuple)) and len(v) > i else np.nan

    def _clean(lst):
        return [v for v in lst if not np.isnan(float(v))]

    print("Passed checks in plotTestBoxplots")
    # ── Localization data ────────────────────────────────────────────────────
    euc_mm  = [d.get('euc_dist_mm_avg', np.nan) for d in loc_dicts]
    axis_x  = [_pa(d, 0) for d in loc_dicts]
    axis_y  = [_pa(d, 1) for d in loc_dicts]
    euc_px  = [d.get('euc_dist_px_avg', np.nan) for d in loc_dicts]
    perc    = [d.get('perc_err',        np.nan) for d in loc_dicts]

    loc_data   = [euc_mm, axis_x, axis_y, euc_px, perc]
    loc_labels = ['Euc\n(mm)', 'X Err\n(mm)', 'Y Err\n(mm)', 'Euc\n(px)', '% Err']

    # ── Detection data ───────────────────────────────────────────────────────
    all_cats = list(det_dicts[0].get('per_category', {}).keys()) if det_dicts else []

    def _dget(key, cat='overall'):
        if cat == 'overall':
            return [d.get('overall', {}).get(key, np.nan) for d in det_dicts]
        return [d.get('per_category', {}).get(cat, {}).get(key, np.nan) for d in det_dicts]

    det_data   = [_dget('precision'), _dget('recall'), _dget('f1')]
    det_labels = ['Overall\nP', 'Overall\nR', 'Overall\nF1']
    det_colors = ['steelblue'] * 3

    cat_colors = cm.tab10(np.linspace(0.0, 0.7, max(len(all_cats), 1)))
    for cat, c in zip(all_cats, cat_colors):
        det_data   += [_dget('precision', cat), _dget('recall', cat), _dget('f1', cat)]
        det_labels += [f'{cat.capitalize()}\nP', f'{cat.capitalize()}\nR', f'{cat.capitalize()}\nF1']
        det_colors += [c, c, c]

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    fig.suptitle('Test Set — Per-batch Metric Distributions', fontsize=13, fontweight='bold')

    # Left: localization
    bp_loc = axes[0].boxplot([_clean(d) for d in loc_data],
                              patch_artist=True)
    axes[0].set_xticklabels(loc_labels,fontsize=8)
    loc_box_colors = cm.Blues(np.linspace(0.35, 0.75, len(loc_data)))
    for patch, fc in zip(bp_loc['boxes'], loc_box_colors):
        patch.set_facecolor(fc)
    axes[0].set_title('Localization Errors')
    axes[0].set_ylabel('Error')
    axes[0].grid(True, alpha=0.3, axis='y')

    # Right: detection
    bp_det = axes[1].boxplot([_clean(d) for d in det_data],
                              patch_artist=True)
    axes[1].set_xticklabels(det_labels,fontsize=8)
    for patch, fc in zip(bp_det['boxes'], det_colors):
        patch.set_facecolor(fc)
        patch.set_alpha(0.75)
    axes[1].set_title('Detection Metrics')
    axes[1].set_ylabel('Score')
    axes[1].set_ylim(-0.05, 1.1)
    axes[1].grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    _safe_save(fig, save_file)
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# 6.  Per-category grouped bar chart  (test set)
# ─────────────────────────────────────────────────────────────────────────────

def plotPerCategoryMetrics_Test(test_logger, save_file=None):
    """
    Grouped bar chart comparing Precision / Recall / F1 for Overall and each
    category, computed via micro-averaging over the full test set.

    Parameters
    ----------
    test_logger : dict  Flat per-batch logger from modelTester.
    save_file   : str   (optional) SVG save path.
    """
    det_avg = average_detection_dict_overbatches(test_logger.get('detection_dict', []))
    if not det_avg:
        return

    overall  = det_avg.get('overall', {})
    per_cat  = det_avg.get('per_category', {})

    groups     = ['Overall'] + [c.capitalize() for c in per_cat]
    precisions = [overall.get('precision', np.nan)] + [per_cat[c].get('precision', np.nan) for c in per_cat]
    recalls    = [overall.get('recall',    np.nan)] + [per_cat[c].get('recall',    np.nan) for c in per_cat]
    f1s        = [overall.get('f1',        np.nan)] + [per_cat[c].get('f1',        np.nan) for c in per_cat]

    x     = np.arange(len(groups))
    width = 0.25

    fig, ax = plt.subplots(figsize=(max(8, 3 * len(groups)), 5))
    bars_p = ax.bar(x - width, precisions, width, label='Precision', color='steelblue', alpha=0.85)
    bars_r = ax.bar(x,         recalls,    width, label='Recall',    color='orangered', alpha=0.85)
    bars_f = ax.bar(x + width, f1s,        width, label='F1',        color='seagreen',  alpha=0.85)

    for bars in (bars_p, bars_r, bars_f):
        ax.bar_label(bars, fmt='%.3f', fontsize=8, padding=2)

    ax.set_xticks(x)
    ax.set_xticklabels(groups, fontsize=11)
    ax.set_ylabel('Score')
    ax.set_title('Test Set: Detection Metrics by Category')
    ax.set_ylim(0, 1.18)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    _safe_save(fig, save_file)
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# 7.  TP / FP / FN stacked bars vs epoch for training and validation sets
# ─────────────────────────────────────────────────────────────────────────────

def plotTPFPFN_TrainValid(train_logger, valid_logger=None, save_file=None):
    """
    Stacked bar chart of cumulative TP / FP / FN per epoch (overall).

    Since TP/FP/FN are summed across batches within each epoch by the
    aggregation helpers, the bar heights reflect total detection counts
    for that epoch. Growing TP with shrinking FP/FN indicates the model
    is learning. Helps distinguish precision vs recall as the primary bottleneck.

    Parameters
    ----------
    train_logger : dict  Epoch-nested logger from ModelTrainer.
    valid_logger : dict  (optional) Same structure.
    save_file    : str   (optional) SVG save path.
    """
    def _extract_counts(logger):
        epoch_dicts = logger.get('detection_dict', [])   # ← remove the averaging call
        tp = np.array([d.get('overall', {}).get('tp', 0) for d in epoch_dicts])
        fp = np.array([d.get('overall', {}).get('fp', 0) for d in epoch_dicts])
        fn = np.array([d.get('overall', {}).get('fn', 0) for d in epoch_dicts])
        return tp, fp, fn

    has_valid = valid_logger is not None and valid_logger.get('detection_dict')
    n_cols    = 2 if has_valid else 1

    fig, axes = plt.subplots(1, n_cols, figsize=(8 * n_cols, 5))
    if n_cols == 1:
        axes = [axes]
    fig.suptitle('TP / FP / FN per Epoch (Overall)', fontsize=13, fontweight='bold')

    pairs = [(train_logger, 'Training')]
    if has_valid:
        pairs.append((valid_logger, 'Validation'))

    for ax, (logger, title) in zip(axes, pairs):
        tp, fp, fn = _extract_counts(logger)
        ep = np.arange(len(tp))
        ax.bar(ep, tp,          label='TP', color='seagreen',  alpha=0.85)
        ax.bar(ep, fp, bottom=tp,          label='FP', color='steelblue', alpha=0.85)
        ax.bar(ep, fn, bottom=tp + fp,     label='FN', color='tomato',    alpha=0.85)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Count (summed across batches)')
        ax.set_title(title)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    _safe_save(fig, save_file)
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# 8.  Test-set error histograms
# ─────────────────────────────────────────────────────────────────────────────

def plotTestErrorHistogram(test_logger, save_file=None):
    """
    Histograms of per-batch test localization errors with mean / ±1 std annotations.
    Particularly useful for ultrasound where understanding the mm error distribution
    is clinically relevant.

    Left  : Euclidean distance distribution (mm)
    Right : Per-axis X vs Y error distribution (mm) — overlaid histograms

    Parameters
    ----------
    test_logger : dict  Flat per-batch logger from modelTester.
    save_file   : str   (optional) SVG save path.
    """
    loc_dicts = test_logger.get('localization_dict', [])
    if not loc_dicts:
        return

    def _pa(d, i):
        v = d.get('peraxis_mm_err_avg')
        return float(v[i]) if isinstance(v, (list, tuple)) and len(v) > i else np.nan

    euc_mm  = np.array([d.get('euc_dist_mm_avg', np.nan) for d in loc_dicts], dtype=np.float64)
    axis_x  = np.array([_pa(d, 0) for d in loc_dicts], dtype=np.float64)
    axis_y  = np.array([_pa(d, 1) for d in loc_dicts], dtype=np.float64)

    euc_mm = euc_mm[~np.isnan(euc_mm)]
    axis_x = axis_x[~np.isnan(axis_x)]
    axis_y = axis_y[~np.isnan(axis_y)]

    if len(euc_mm) == 0:
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle('Test Set — Error Distributions (per-batch means)', fontsize=13, fontweight='bold')

    # ── Left: Euclidean distance ─────────────────────────────────────────────
    axes[0].hist(euc_mm, bins=30, color='steelblue', alpha=0.7,
                 edgecolor='white', density=True)
    mean_val, std_val = euc_mm.mean(), euc_mm.std()
    axes[0].axvline(mean_val,           color='black', linewidth=2.0, linestyle='-',
                    label=f'Mean = {mean_val:.2f} mm')
    axes[0].axvline(mean_val - std_val, color='gray',  linewidth=1.5, linestyle='--',
                    label=f'±1 std ({std_val:.2f} mm)')
    axes[0].axvline(mean_val + std_val, color='gray',  linewidth=1.5, linestyle='--')
    axes[0].set_xlabel('Euclidean Distance (mm)')
    axes[0].set_ylabel('Density')
    axes[0].set_title('Euclidean Distance Error (mm)')
    axes[0].legend(fontsize=9)
    axes[0].grid(True, alpha=0.3)

    # ── Right: Per-axis X vs Y ───────────────────────────────────────────────
    all_ax = np.concatenate([axis_x, axis_y])
    bins = np.histogram_bin_edges(all_ax[~np.isnan(all_ax)], bins=30)
    axes[1].hist(axis_x, bins=bins, color='steelblue', alpha=0.65, edgecolor='white',
                 density=True, label=f'X  mean={axis_x.mean():.2f} mm')
    axes[1].hist(axis_y, bins=bins, color='orangered', alpha=0.65, edgecolor='white',
                 density=True, label=f'Y  mean={axis_y.mean():.2f} mm')
    axes[1].axvline(axis_x.mean(), color='steelblue', linewidth=1.8, linestyle='--')
    axes[1].axvline(axis_y.mean(), color='orangered', linewidth=1.8, linestyle='--')
    axes[1].set_xlabel('Per-axis Error (mm)')
    axes[1].set_ylabel('Density')
    axes[1].set_title('Per-axis Error Distribution (mm)')
    axes[1].legend(fontsize=9)
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    _safe_save(fig, save_file)
    plt.close(fig)