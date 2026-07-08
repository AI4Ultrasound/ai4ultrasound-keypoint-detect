import cv2
import numpy as np
import random

#Custom imports
from AIUSDataset import AIUSDataset 
import datasplitter
from utils import CLASS_ID_B_LINE, CLASS_ID_PLEURAL_LINE

########################Init Params########################
##Config datasplitter
dataset_root='../../../Data/Keypoint_Detect_Data'
outputdata_format='COCO_like'
train_split=0.7
val_split=0.15
test_split=0.15
k_folds=None
equal_prop_tags=['site','annotator']
metadata_tags={
    'site': ['All'],
    'annotator': ['All'],
    'zone_label': ['All'],
    'patient_id': ['All'],
    'time': ['All'],
    'transducer_type': ['All'],
    'manufacturer_name': ['All'],
    'coordinate_space': 'scanline',
}
random_state=42
verbose=True

##Config AIUSDataset
us_framesize=(128,128)
line_type='both'
normalize_method=None
resampling_freq=None
convert_to_gray=True
img_augmentation=None
aug_prob=0.1
return_mode='frame'

##Config Display
NUM_DISPLAY_SAMPLES=10 #Number of frames to display
DISPLAY_WAIT_MS=0 #0=wait for keypress; >0 auto-advance in ms
WINDOW_SCALE=5 #Scale-up factor for small frame sizes
WINDOW_NAME='AIUSDataset Annotation Viewer'
KP_RADIUS=5 #Keypoint display radius (pixels)

#Colours and names for annotation categories
CATEGORY_COLOURS={
    CLASS_ID_PLEURAL_LINE: (0,255,0), #Pleural line = green
    CLASS_ID_B_LINE: (0,255,255), #B-line = yellow
}

CATEGORY_NAMES = {
    CLASS_ID_PLEURAL_LINE: 'Pleural line',
    CLASS_ID_B_LINE:       'B-line',    
}


######################Helper Functions###############
def draw_keypoints(display_img,keypoints,categories,window_scale): #May not need the px_mul_x
    """
    Daw all annotation keypoints onto a BGR image.
    """
    #Loop for all keypoints and categories
    for kp_tensor, cat_id in zip(keypoints, categories):
        cat_id_int = int(cat_id.item())
        color = CATEGORY_COLOURS.get(cat_id_int, (255, 255, 255))  # white fallback if category id doesn't match global constant

        kp_np = kp_tensor.numpy()  # (N_i, 2)
        if kp_np.ndim == 1:        # Single keypoint stored as (2,)
            kp_np = kp_np[np.newaxis, :]

        for kp in kp_np:
            px = int(round(float(kp[0])* window_scale)) 
            py = int(round(float(kp[1])* window_scale))
            cv2.circle(display_img, (px, py), KP_RADIUS, color, -1)          # filled dot
            cv2.circle(display_img, (px, py), KP_RADIUS + 1, (0, 0, 0), 1)  # thin black border

    return display_img

def draw_legend(display_img):
    """
    Draw a colour-coded category (pleural line vs. B-line) legend in top-left corner
    """
    x0, y0 = 8, 18
    for cat_id, name in CATEGORY_NAMES.items(): #Loops for all categories
        color = CATEGORY_COLOURS.get(cat_id, (255, 255, 255))
        cv2.circle(display_img, (x0 + 6, y0 - 5), 5, color, -1)
        cv2.circle(display_img, (x0 + 6, y0 - 5), 6, (0, 0, 0), 1)
        cv2.putText(display_img, name, (x0 + 18, y0),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
        y0 += 22
    return display_img

def draw_info_bar(display_img,frame_num,clip_id,n_anns,sample_idx,total):
    """
    Draws an info bar along the bottom of the image about this particular frame's metadata.
    """
    h = display_img.shape[0] #Gets image height
    text = (f'sample: {sample_idx + 1}/{total}  clip: {clip_id}  '
            f'frame #: {frame_num}  # annotations: {n_anns}')
    #Dark background strip with text superimposed on it
    cv2.rectangle(display_img,
                  (0, h - 20), (display_img.shape[1], h),
                  (30, 30, 30), -1)
    cv2.putText(display_img, text, (5, h - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, (210, 210, 210), 1, cv2.LINE_AA)
    return display_img

def tensor_to_display_bgr(img_unnorm):
    """
    Converts unnormalized image back to a 3-channel BGR image for OpenCV display.
    """
    if img_unnorm.ndim == 2:                            # Grayscale (H, W)
        return cv2.cvtColor(img_unnorm, cv2.COLOR_GRAY2BGR)
    elif img_unnorm.ndim == 3 and img_unnorm.shape[2] == 3:  # RGB (H, W, 3)
        return cv2.cvtColor(img_unnorm, cv2.COLOR_RGB2BGR)
    else:
        # Unexpected channel count — return as-is and let OpenCV handle it
        return img_unnorm
    

######################Main################

if __name__=='__main__':


    #Run datasplitter
    print('=' * 60)
    print('Step 1: Running datasplitter...')
    print('=' * 60)
    train_paths,val_paths,test_paths=datasplitter.datasplitter(dataset_root=dataset_root,
                          outdata_format=outputdata_format,
                          train_split=train_split,
                          val_split=val_split,
                          test_split=test_split,
                          k_folds=k_folds,
                          equal_prop_tags=equal_prop_tags,
                          metadata_tags=metadata_tags,
                          random_state=random_state,
                          verbose=verbose)
    
    #Run AIUSDataset init
    print()
    print('=' * 60)
    print('Step 2: Building AIUSDataset...')
    print('=' * 60)

    train_data=AIUSDataset(
                json_paths=train_paths,
                us_framesize=us_framesize,
                line_type=line_type,
                normalize_method=normalize_method,
                resampling_freq=resampling_freq,
                convert_to_gray=convert_to_gray,
                img_augmentations=img_augmentation,
                aug_prob=aug_prob,
                return_mode=return_mode
            )
    #Select sample indices to display
    n_display = min(NUM_DISPLAY_SAMPLES, len(train_data))
    if n_display == 0:
        print('No samples found in dataset — check paths and line_type config.')
        exit
    
    #Get random subset of images to display
    display_indices = random.sample(range(len(train_data)), n_display)

    #Display loop (with OpenCV)
    print()
    print('=' * 60)
    print(f'Step 3: Displaying {n_display} frames.')
    print('  Press any key to advance, ESC to quit.')
    print('=' * 60)

    cv2.namedWindow(WINDOW_NAME,cv2.WINDOW_AUTOSIZE)

    for loop_idx, sample_idx in enumerate(display_indices):

        # ---- Load sample from dataset ----
        sample     = train_data[sample_idx]
        img_tensor = sample['image']        # FloatTensor (C, H, W)
        keypoints  = sample['keypoints']    # list[FloatTensor(N_i, 2)]
        categories = sample['categories']   # LongTensor (K,)
        frame_num  = sample['frame_num']    # int
        #px_mul_x   = sample['px_mul_x']    # float: annotation x → pixel x
        #px_mul_y   = sample['px_mul_y']    # float: annotation y → pixel y
        clip_id    = sample['clip_id']      # str

        # ---- Console summary ----
        print(f'  [{loop_idx + 1:>{len(str(n_display))}}/{n_display}]  '
              f'clip: {clip_id:30s}  '
              f'frame: {frame_num:5d}  '
              f'annotations: {len(keypoints):3d}  '
              f'image shape: {tuple(img_tensor.shape)}')

        # ---- Unnormalize image tensor → uint8 numpy array ----
        img_unnorm = train_data.unnormalize_image(img_tensor, is_gpu=False)
        # Shape: (H, W) for grayscale  |  (H, W, 3) for colour

        # ---- Convert to 3-channel BGR for OpenCV annotation drawing ----
        img_bgr = tensor_to_display_bgr(img_unnorm)

        # ---- Scale up small images so they're easier to inspect ----
        if WINDOW_SCALE > 1:
            h, w = img_bgr.shape[:2]
            img_bgr = cv2.resize(img_bgr,
                                 (w * WINDOW_SCALE, h * WINDOW_SCALE),
                                 interpolation=cv2.INTER_NEAREST)

        # ---- Draw annotations ----
        img_bgr = draw_keypoints(img_bgr, keypoints, categories,WINDOW_SCALE)

        # ---- Draw legend and info bar ----
        img_bgr = draw_legend(img_bgr)
        img_bgr = draw_info_bar(img_bgr, frame_num, clip_id,
                                len(keypoints), loop_idx, n_display)

        # ---- Show frame ----
        cv2.imshow(WINDOW_NAME, img_bgr)
        key = cv2.waitKey(DISPLAY_WAIT_MS) & 0xFF
        if key == 27:   # ESC → exit early
            print('\n  ESC pressed — stopping display early.')
            break

    cv2.destroyAllWindows()
    print()
    print('Done.')
    