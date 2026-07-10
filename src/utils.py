import json
import re
import os
import glob
from typing import Any, Dict, List, Optional, Tuple

import torch
import numpy as np
import pydicom
from scipy.ndimage import map_coordinates
from PIL import Image

#Class ID numbers
CLASS_ID_PLEURAL_LINE=1
CLASS_ID_B_LINE=2

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
            frame_to_pleura_sector[int(fn)] = pts
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
                        "bbox":bboxes,
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
                        "bbox":bboxes_bline,
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
                categories[b,t,:k_t]=item['flat_categories'][t] #Categories without keypoints is -1
                continue #Empty frame, skip
            keypoints[b,t,:k_t]=item['flat_keypoints'][t]
            visibility[b,t,:k_t]=True
            categories[b,t,:k_t]=item['flat_categories'][t]
    
    return {
        'images': images,
        'padding_mask': padding_mask,
        'keypoints': keypoints,
        'visibility': visibility,
        'categories': categories,
        'frame_nums': [item['frame_nums'] for item in clip_batch],
        'px_mul_x': [item['px_mul_x']   for item in clip_batch],
        'px_mul_y': [item['px_mul_y']   for item in clip_batch],
        'clip_len': [item['clip_len']    for item in clip_batch],
        'clip_id': [item['clip_id']    for item in clip_batch],
        'metadata': [item['metadata']    for item in clip_batch],
    }


   




#############################Dataclasses######################