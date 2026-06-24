import json
import re
import os
import glob
from copy import deepcopy
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pydicom
from scipy.ndimage import map_coordinates
from dataclasses import dataclass
import cv2
from PIL import Image
from preprocessing import (CLASS_ID_B_LINE, CLASS_ID_PLEURAL_LINE)

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


########################File Handling Functions#########################

def find_json_for_dicom(dicom_path: str, json_dir: str) -> Optional[str]:
    """Match DICOM to JSON by SOPInstanceUID when possible, else by filename stem."""
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

    stem = os.path.splitext(os.path.basename(dicom_path))[0]  # Filename without extension
    candidates = sorted(glob.glob(os.path.join(json_dir, f"{stem}*.json")))
    if candidates:
        return candidates[0]

    for p in sorted(glob.glob(os.path.join(json_dir, "*.json"))):
        p_name = os.path.basename(p)
        p_stem = os.path.splitext(p_name)[0]
        if stem in p_name or re.match(re.escape(stem), p_stem):
            return p
    return None

def os_make_dir(folder_path):
    if not os.path.exists(folder_path):
        os.makedirs(folder_path, exist_ok=True)


def export_clip_to_png_and_json(input_dicom_path,input_json_path,output_annotation_dir,output_image_dir,
                       filename_prefix,coordinate_space='scanline',num_lines=128,num_samples_per_line=128):
    """ Converts a dicom clip (scan) and json annotation to individual images and annotations in the
      desired output format (only .png and .json supported now). 
      filename_prefix is a string with: <annotator>_<site>_<patient_id>_<time>_<scan_id>"""
    
    #########Loading data and initializing parameters############
    json_data=load_annotation_json(input_json_path) #Loading json annotations
    scan_params = scanconversion_params_from_annotation(json_data) #Checks for any missing keys in the annotations, and returns annotations that have complete keys

    #Init dictionaries that will contain bline and pleura line annotations in mm
    frame_to_pleura_sector: Dict[int, np.ndarray] = {}
    frame_to_blines_sector: Dict[int, List[np.ndarray]] = {}

    #Fill the annotation dictionaries (in original sector space)
    frame_nos, pleura_list, blists, pleura_ok, _b_ok = parse_all_lines_from_annotation(json_data)
    for fn, pts, ok in zip(frame_nos, pleura_list, pleura_ok):
        if ok and pts.size > 0:
            frame_to_pleura_sector[int(fn)] = pts
    for fn, b_list in zip(frame_nos, blists):
        good = [b for b in b_list if b.size > 0]
        if good:
            frame_to_blines_sector[int(fn)] = good


    ################Loading in the ultrasound images###############
    frames_sector, psx, psy, _ds = load_ultrasound_frames_from_dicom(input_dicom_path)
    
    ###############Converting images (originally in sector) to scanlines (rectangular)#######
    if coordinate_space=='scanline':
        scan_tensor, scan_config = convert_to_scanlines(frames_sector, scan_params, num_lines=num_lines, num_samples_per_line=num_samples_per_line)
        if scan_tensor is None:
            raise ValueError("Empty frame array from DICOM.")

        #Convert scan tensor to np frames
        scan_hwc_scanline = np.moveaxis(scan_tensor, 1, 3)
        num_frames = scan_hwc_scanline.shape[0]

        #Init the dictionary to hold the annotations in the scanline space
        frame_to_pleura_scanline: Dict[int, np.ndarray] = {}

        #Read in annotations in scanline space
        for fn, mm in frame_to_pleura_sector.items():
            if fn < 0 or fn >= num_frames:
                continue
            frame_to_pleura_scanline[fn] = pleura_points_curvilinear_mm_to_scanlines(mm, psx, psy, scan_config)
        
        frame_to_bline_scanline: Dict[int, List[np.ndarray]] = {}
        for fn, mm_list in frame_to_blines_sector.items():
            if fn < 0 or fn >= num_frames:
                continue
            converted: List[np.ndarray] = []
            for mm in mm_list:
                if mm.size == 0:
                    continue
                converted.append(pleura_points_curvilinear_mm_to_scanlines(mm, psx, psy, scan_config))
            if converted:
                frame_to_bline_scanline[fn] = converted
        

    #Looping through all the image frames in this clip and saves them to separate files
    lbl_filename=os.path.join(output_annotation_dir,f'{filename_prefix}.json')
    output_json={
        "images": [],
        "annotations": [],
    }
    for f in range(num_frames):
        img_filename=os.path.join(output_image_dir,f'{filename_prefix}_{f}.png')
        im=np.asarray(scan_hwc_scanline[f]) if coordinate_space=='scanline' else np.asarray(frames_sector[f])

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
        
        #Saves image:
        Image.fromarray(im_u8).save(img_filename)

        #Update the image in the json:
        output_json["images"].append({
                    "id": f,
                    "file_name": str(img_filename),
                    "height": im.shape[0],
                    "width": im.shape[1],
                })

        #Handling the annotations and image labels in the output_json file:
        #Handling pleural keypoint annotations
        pleural_pts_s=frame_to_pleura_scanline.get(f) if coordinate_space=='scanline' else frame_to_pleura_sector.get(f)
        for pts_pleural in pleural_pts_s:
            if pts_pleural is not None and pts_pleural.size>0:
                output_json["annotations"].append({
                    "id": f, #Just the frame number
                    "category_id": CLASS_ID_PLEURAL_LINE,
                    "keypoints": [[pts_pleural[:,0],pts_pleural[:,1]]],
                   })                

        #Handling bline keypoint annotations
        bline_pts=frame_to_bline_scanline.get(f,[]) if coordinate_space=='scanline' else frame_to_blines_sector.get(f,[])
        for pts_b in bline_pts:
            if pts_b is not None and pts_b.size>0:
                output_json["annotations"].append({
                    "id": f, #Just the frame number
                    "category_id": CLASS_ID_B_LINE,
                    "keypoints": [[pts_b[:,0],pts_b[:,1]]],
                   })
        
    #Save the json to its annotation folder
    with open(lbl_filename,"w",encoding="utf-8") as file:
        json.dump(output_json,file,indent=4)
    
    

        
        


########################Helper Functions (for internal utils use)########################

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
) -> Tuple[List[int], List[np.ndarray], List[List[np.ndarray]], List[bool], List[bool]]:
    """
    Pleura (first line only) plus **all** B-line polylines per frame.

    Returns:
        frame_numbers
        pleura_points_mm: one (N,2) array per frame (empty if none)
        b_lines_points_mm: list per frame of zero or more (N,2) arrays
        pleura_valid: first pleura line usable
        b_lines_valid: at least one usable B-line on that frame
    """
    frames = data.get("frame_annotations")
    if not isinstance(frames, list):
        raise ValueError("Annotation JSON has no list 'frame_annotations'.")

    frame_numbers: List[int] = []
    pleura_mm: List[np.ndarray] = []
    blines_mm: List[List[np.ndarray]] = []
    pleura_ok: List[bool] = []
    b_ok: List[bool] = []

    for entry in frames:
        if "frame_number" not in entry:
            raise ValueError("frame_annotations entry missing 'frame_number'.")
        fn = int(entry["frame_number"])
        frame_numbers.append(fn)
        if not isinstance(entry, dict):
            raise ValueError("frame_annotations entries must be objects.")

        p_arr, p_valid = _first_polyline_mm_from_entry(
            entry, "pleura_lines", require_non_empty_line=require_non_empty_line
        )
        pleura_mm.append(p_arr)
        pleura_ok.append(p_valid)

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

#############################Dataclasses######################