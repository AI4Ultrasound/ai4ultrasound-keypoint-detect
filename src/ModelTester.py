import torch
import torch.nn as nn
from tqdm import trange, tqdm
import os
import numpy as np
import json

import utils

def modelTester(model,test_loader,return_mode,matching_strategy,loss_fun,device,us_framesize,model_name_save,match_thresh_percentage=0.1):
    H_in=us_framesize[0]
    W_in=us_framesize[1]
    max_diagnoal=float(np.sqrt(H_in**2+W_in**2)) #Diagonal length
    match_thresh_pixel=max_diagnoal*match_thresh_percentage

    model.eval() #Set model to evaluation mode for this epoch
    test_logger={
        'loss':[],
        'localization_dict':[],
        'detection_dict':[],
    }
    with torch.no_grad():
        #Loop through the validation loader
        pbar=tqdm(test_loader, desc="Testing", leave=False,mininterval=1.0)
        for batch_idx,data in enumerate(pbar):
            
            if return_mode=='frame': #Using frame mode
                #Load the data
                us_frames=data['images'].to(device,non_blocking=True) #(B, C, H, W)
                target_keypoints=data['keypoints'].to(device,non_blocking=True) #(B,K,2) where K=number of keypoints
                target_areas=data['areas'].to(device,non_blocking=True) #(B,K)
                target_visibility=data['visibility'].to(device,non_blocking=True)
                target_categories=data['categories'].to(device,non_blocking=True)  #(B,K)
                px_mul_x=data['px_mul_x'] #Multiplier for pixel to mm (mm=pixel*px_mul), size (B)
                px_mul_y=data['px_mul_y']

                #Gets prediction and runs the loss
                if matching_strategy in ('fixed','hungarian'):
                    pred_keypoints,pred_categories=model(us_frames)
                    loss=loss_fun(pred_keypoints,target_keypoints,target_visibility,target_areas,target_categories,pred_categories=pred_categories)
                elif matching_strategy=='heatmap':
                    pred_heatmaps=model(us_frames) #shape (B,2,H',W')
                    loss = loss_fun(
                    pred_heatmaps, target_keypoints,
                    target_visibility, target_areas, target_categories,
                    image_shape = (H_in,W_in),  
                )

            elif return_mode=='clip': #Using the clip_collate_fn return where we pad the end of the time dimension with zeros
                us_frames=data['images'].to(device,non_blocking=True) #(B,T, C, H, W)
                padding_mask=data['padding_mask'].to(device,non_blocking=True) #(B,T)
                #Skip this batch if less than two batches are valid
                if padding_mask.sum()<2:
                    del us_frames,padding_mask
                    continue
                px_mul_x=data['px_mul_x'] #Multiplier for pixel to mm (mm=pixel*px_mul), (B,T)
                px_mul_y=data['px_mul_y']

                #Get target keypoints, visibiity and categories
                target_keypoints=data['keypoints'].to(device,non_blocking=True) #(B,T,K_max,2)  
                target_areas=data['areas'].to(device,non_blocking=True)  #(B,T,K)
                target_visibility=data['visibility'].to(device,non_blocking=True) #(B,T,K_max)
                target_categories=data['categories'].to(device,non_blocking=True) #(B,T,K_max)

                target_visibility=target_visibility & padding_mask.unsqueeze(-1) #mask out any visibility for frames which have been padded

                if not target_visibility.any():
                    del us_frames,padding_mask, target_keypoints,target_visibility,target_categories,target_areas
                    continue

                #Gets prediction and runs the loss
                if matching_strategy in ('fixed','hungarian'):
                    pred_keypoints,pred_categories=model(us_frames)
                    loss=loss_fun(pred_keypoints,target_keypoints,target_visibility,target_areas,target_categories,pred_categories=pred_categories)
                elif matching_strategy=='heatmap':
                    pred_heatmaps=model(us_frames) #shape (B,2,H',W')
                    loss = loss_fun(
                    pred_heatmaps, target_keypoints,
                    target_visibility, target_areas, target_categories,
                    image_shape = (H_in,W_in),  
                )
            
            else:
                raise ValueError(f"Unknown return_mode: '{return_mode}'")
            
            #Check that loss is not nan
            if torch.isnan(loss):
                continue
            ###########Compute Validation Error and Log Loss+Error###########
            unique_categories = torch.unique(target_categories)
            unique_categories = unique_categories[unique_categories >0]
            num_categories = unique_categories.numel()

            if matching_strategy=='heatmap':
                #Compute error on predicted heatmap using the utils compute error function
                localization_dict,detection_dict=utils.calculateError(pred=pred_heatmaps,pred_categories=None,target_keypoints=target_keypoints,visibility=target_visibility,areas=target_areas,categories=target_categories,
                                                                    return_mode=return_mode,matching_strategy=matching_strategy,num_categories=num_categories,image_shape=(H_in,W_in),
                                                                    px_mul_x=px_mul_x,px_mul_y=px_mul_y,match_threshold=match_thresh_pixel,max_diagnoal=max_diagnoal)
            else:
                localization_dict,detection_dict=utils.calculateError(pred=pred_keypoints,pred_categories=pred_categories,target_keypoints=target_keypoints,visibility=target_visibility,areas=target_areas,categories=target_categories,
                                                                    return_mode=return_mode,matching_strategy=matching_strategy,num_categories=num_categories,image_shape=(H_in,W_in),
                                                                    px_mul_x=px_mul_x,px_mul_y=px_mul_y,match_threshold=match_thresh_pixel,max_diagnoal=max_diagnoal)            
            #Update the epoch train metrics accumulator
            test_logger['loss'].append(loss.item())
            test_logger['localization_dict'].append(utils._localization_dict_to_serializable(localization_dict))
            test_logger['detection_dict'].append(detection_dict)
            
            #Delete data to save memory
            if matching_strategy=='heatmap':
                del us_frames, pred_heatmaps,target_keypoints,target_areas,target_visibility,target_categories,loss
            else:
                del us_frames, pred_keypoints,pred_categories,target_keypoints,target_areas,target_visibility,target_categories,loss
    
            #Update the progress bar:
            pbar.set_postfix_str('Test Err: EucDist= %.4f (mm), P= %.4f, R= %.4f, F1= %.4f'
                                    % (test_logger['localization_dict'][-1]['euc_dist_mm_avg'],test_logger['detection_dict'][-1]['overall']['precision'],test_logger['detection_dict'][-1]['overall']['recall'],test_logger['detection_dict'][-1]['overall']['f1']))

    #Computing average results
    avg_localization = utils.average_localization_dict_serialized(test_logger['localization_dict'])
    avg_detection    = utils.average_detection_dict_overbatches(test_logger['detection_dict'])
    avg_loss         = float(np.mean(test_logger['loss'])) if test_logger['loss'] else float('nan')
    
    summary_results = {
    'avg_loss':         avg_loss,
    'avg_localization': avg_localization,
    'avg_detection':    avg_detection,
    'raw_logger':       test_logger,
    }

    #Saving the testing results
    save_dir=os.path.join('../stats','test_results')
    os.makedirs(save_dir,exist_ok=True)
    save_path=os.path.join(save_dir,model_name_save+'.pt')
    #Make sure the path doesn't exist
    num_val=0
    while os.path.exists(save_path):
        save_path=os.path.join(save_dir,model_name_save+'_'+str(num_val)+'.pt')
        num_val+=1
    torch.save(summary_results, save_path)

    
    return summary_results