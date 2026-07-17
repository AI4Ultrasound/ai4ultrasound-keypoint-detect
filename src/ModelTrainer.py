import torch
import torch.nn as nn
from tqdm import trange, tqdm
import os
import numpy as np
import json

import utils

METRIC_EVERY_N_BATCHES=5 #Don't compute the training detection and localization errors on every batch, do it on every N'th batch (to save loop time)


class ModelTrainer(nn.Module):
    def __init__(self,model,loss_fun,optimizer,train_loader,valid_loader,
                 num_epochs,device,loss_type,us_framesize,LR_scheduler=None,return_mode='frame',
                 checkpoint_savedir='models',model_name_save='model',model_name_load='model',start_from_checkpoint=False,
                 matching_strategy='heatmap',match_thresh_percentage=0.1,verbose=False):
        
        super(ModelTrainer,self).__init__()
        ##########Init Class Params#########
        self.model=model
        self.loss_fun=loss_fun
        self.optimizer=optimizer
        self.train_loader=train_loader
        self.valid_loader=valid_loader
        self.num_epochs=num_epochs
        self.device=device
        self.loss_type=loss_type
        self.LR_scheduler=LR_scheduler
        self.return_mode=return_mode
        self.matching_strategy=matching_strategy
        self.H_in=us_framesize[0]
        self.W_in=us_framesize[1]
        self.us_framesize=us_framesize
        self.max_diagnoal=float(np.sqrt(self.H_in**2+self.W_in**2)) #Diagonal length
        #Match percentage is the percentage of the image size that below which we consider a predicted keypoint to be the same keypoint as a ground truth
        self.match_thresh_pixel=self.max_diagnoal*match_thresh_percentage
        self.verbose=verbose

        #Initalize the training and validation loggers:
        self.training_logger={'loss':[],'localization_dict':[],'detection_dict':[]} #We log each batch's within each epochs loss, detection and localization dicts
        self.valid_logger={'loss':[],'localization_dict':[],'detection_dict':[]} 

        #Checkpointing params
        self.checkpoint_savedir=checkpoint_savedir
        self.model_name_save=model_name_save
        self.model_name_load=model_name_load
        #Create the checkpoint savepath
        self.checkpoint_savepath=os.path.join(checkpoint_savedir,model_name_save+ '.pt') if checkpoint_savedir else None

        #Create a different savepath if this exists
        num_val=0
        while os.path.exists(self.checkpoint_savepath):
            self.checkpoint_savepath=os.path.join(checkpoint_savedir,model_name_save+'_'+str(num_val)+'.pt') if checkpoint_savedir else None
            num_val+=1
            
        self.checkpoint_loadpath=os.path.join(checkpoint_savedir,model_name_load+'.pt') if checkpoint_savedir else None

        self.start_epoch=0
        self.best_valid_loss=float(10000.0) #Keep model with best validation loss

        #Logger save path (saved at end of training)
        logger_save_dir=os.path.join('logs',model_name_save)
        if not os.path.isdir(logger_save_dir):
            os.makedirs(logger_save_dir,exist_ok=True)
        
        self.train_logger_save_path=os.path.join(logger_save_dir,'train_logger.json')
        self.valid_logger_save_path=os.path.join(logger_save_dir,'valid_logger.json')
        
        #Load from checkpoint
        if start_from_checkpoint and os.path.isfile(self.checkpoint_loadpath):
            self.loadCheckpoint()
        
    #######Checkpointing Methods#######
    def saveCheckpoint(self,epoch,valid_loss,localization_dict_avg,detection_dict_avg): #Save model to file for checkpointing
        if self.checkpoint_savepath is None:
            return #No file name defined
        if not os.path.exists(self.checkpoint_savedir):
            os.makedirs(self.checkpoint_savedir)
        self.best_valid_loss=valid_loss
        torch.save({
            'epoch':epoch+1,
            'model_state_dict':self.model.state_dict(),
            'optimizer_state_dict':self.optimizer.state_dict(),
            'best_valid_loss':self.best_valid_loss,
            #'training_logger':self.training_logger,
            #'valid_logger':self.valid_logger,
            'localization_dict_avg':localization_dict_avg,
            'detection_dict_avg':detection_dict_avg
        },self.checkpoint_savepath)

    def loadCheckpoint(self):
        #Loads from checkpoint if we are doing that
        checkpoint=torch.load(self.checkpoint_loadpath,, map_location=self.device, weights_only=False)
        self.model.load_state_dict(checkpoint['model_state_dict'])  
        self.model=self.model.to(self.device)
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict']) 
        self.start_epoch=checkpoint['epoch']
        self.best_valid_loss=checkpoint['best_valid_loss']     
        #self.training_logger=checkpoint['training_logger']   
        #self.valid_logger=checkpoint['valid_logger']

        if self.verbose:
            print(f"Loaded checkpoint from epoch {self.start_epoch}, best validation loss: {self.best_valid_loss:.4f}")
    
    def train(self):
        #######Main Trainer Function where we conduct both training and validation
        pbar=trange(self.start_epoch,self.num_epochs,leave=False,desc="Epoch")

        for epoch in pbar: #Loops for num_epochs
            
            ###################Training################
            self.model.train() #Sets model to training mode

            #Init epoch metric accumulator
            epoch_train={
                'loss':[],
                'localization_dict':[],
                'detection_dict':[],
            }

            #Loops through each batch in the training loader
            for batch_idx,data in enumerate(tqdm(self.train_loader, desc="Training", leave=False,mininterval=1.0)):

                #Load the data
                if self.return_mode=='frame':
                    us_frames=data['images'].to(self.device,non_blocking=True) #(B, C, H, W)
                    target_keypoints=data['keypoints'].to(self.device,non_blocking=True) #(B,K,2) where K=number of keypoints
                    target_areas=data['areas'].to(self.device,non_blocking=True) #(B,K)
                    target_visibility=data['visibility'].to(self.device,non_blocking=True)
                    target_categories=data['categories'].to(self.device,non_blocking=True)  #(B,K)
                    px_mul_x=data['px_mul_x'] #Multiplier for pixel to mm (mm=pixel*px_mul), size (B)
                    px_mul_y=data['px_mul_y']

                    #Gets prediction and runs the loss
                    if self.matching_strategy in ('fixed','hungarian'):
                        pred_keypoints,pred_categories=self.model(us_frames)
                        loss=self.loss_fun(pred_keypoints,target_keypoints,target_visibility,target_areas,target_categories,pred_categories=pred_categories)
                    elif self.matching_strategy=='heatmap':
                        pred_heatmaps=self.model(us_frames) #shape (B,2,H',W')
                        loss = self.loss_fun(
                        pred_heatmaps, target_keypoints,
                        target_visibility, target_areas, target_categories,
                        image_shape = (self.H_in,self.W_in),  
                    )

                if self.return_mode=='clip': #Using the clip_collate_fn return where we pad the end of the time dimension with zeros
                    us_frames=data['images'].to(self.device,non_blocking=True) #(B,T, C, H, W)
                    padding_mask=data['padding_mask'].to(self.device,non_blocking=True) #(B,T)
                    #Skip this batch if less than two batches are valid
                    if padding_mask.sum()<2:
                        del us_frames,padding_mask
                        continue
                    px_mul_x=data['px_mul_x'] #Multiplier for pixel to mm (mm=pixel*px_mul), (B,T)
                    px_mul_y=data['px_mul_y']

                    #Get target keypoints, visibiity and categories
                    target_keypoints=data['keypoints'].to(self.device,non_blocking=True) #(B,T,K_max,2)  
                    target_areas=data['areas'].to(self.device,non_blocking=True)  #(B,T,K)
                    target_visibility=data['visibility'].to(self.device,non_blocking=True) #(B,T,K_max)
                    target_categories=data['categories'].to(self.device,non_blocking=True) #(B,T,K_max)

                    target_visibility=target_visibility & padding_mask.unsqueeze(-1) #mask out any visibility for frames which have been padded

                    if not target_visibility.any():
                        del us_frames,padding_mask, target_keypoints,target_visibility,target_categories,target_areas
                        continue

                    #Gets prediction and runs the loss
                    if self.matching_strategy in ('fixed','hungarian'):
                        pred_keypoints,pred_categories=self.model(us_frames)
                        loss=self.loss_fun(pred_keypoints,target_keypoints,target_visibility,target_areas,target_categories,pred_categories=pred_categories)
                    elif self.matching_strategy=='heatmap':
                        pred_heatmaps=self.model(us_frames) #shape (B,2,H',W')
                        loss = self.loss_fun(
                        pred_heatmaps, target_keypoints,
                        target_visibility, target_areas, target_categories,
                        image_shape = (self.H_in,self.W_in),  
                    )
                        
                

                if torch.isnan(loss):
                    self.optimizer.zero_grad()
                    if self.matching_strategy=='heatmap':
                        del us_frames, pred_heatmaps,target_keypoints,target_areas,target_visibility,target_categories,loss
                    else:
                        del us_frames, pred_keypoints,pred_categories,target_keypoints,target_areas,target_visibility,target_categories,loss
                    continue
                else:
                    #Backpropagation
                    self.optimizer.zero_grad()
                    loss.backward()
                    #Step the optimizer
                    self.optimizer.step()
                    #If we are using OneCycleLR scheduler
                    if self.LR_scheduler is not None:
                        if isinstance(self.LR_scheduler,torch.optim.lr_scheduler.OneCycleLR):
                            self.LR_scheduler.step() #Step every batch for one cycle lr

                    #Detach from computational graph
                    if self.matching_strategy=='heatmap':
                        pred_heatmaps=pred_heatmaps.detach()
                    else:
                        pred_categories=pred_categories.detach()
                        pred_keypoints=pred_keypoints.detach()
                    
                    target_keypoints=target_keypoints.detach()
                    target_visibility=target_visibility.detach()
                    target_categories=target_categories.detach()

                

                ###########Compute Training Error and Log Loss+Error###########
                if batch_idx % METRIC_EVERY_N_BATCHES==0: #Compute error for this batch
                    unique_categories = torch.unique(target_categories)
                    unique_categories = unique_categories[unique_categories >0]
                    num_categories = unique_categories.numel()

                    if self.matching_strategy=='heatmap':
                        #Compute error on predicted heatmap using the utils compute error function
                        localization_dict,detection_dict=utils.calculateError(pred=pred_heatmaps,pred_categories=None,target_keypoints=target_keypoints,visibility=target_visibility,areas=target_areas,categories=target_categories,
                                                                            return_mode=self.return_mode,matching_strategy=self.matching_strategy,num_categories=num_categories,image_shape=(self.H_in,self.W_in),
                                                                            px_mul_x=px_mul_x,px_mul_y=px_mul_y,match_threshold=self.match_thresh_pixel,max_diagnoal=self.max_diagnoal)
                    else:
                        localization_dict,detection_dict=utils.calculateError(pred=pred_keypoints,pred_categories=pred_categories,target_keypoints=target_keypoints,visibility=target_visibility,areas=target_areas,categories=target_categories,
                                                                            return_mode=self.return_mode,matching_strategy=self.matching_strategy,num_categories=num_categories,image_shape=(self.H_in,self.W_in),
                                                                            px_mul_x=px_mul_x,px_mul_y=px_mul_y,match_threshold=self.match_thresh_pixel,max_diagnoal=self.max_diagnoal)
                    #Update the epoch train metrics accumulator
                    epoch_train['localization_dict'].append(self._localization_dict_to_serializable(localization_dict))
                    epoch_train['detection_dict'].append(detection_dict)
                
                #Update the epoch train metrics accumulator with the loss
                epoch_train['loss'].append(loss.item())

                #Delete data to save memory
                if self.matching_strategy=='heatmap':
                    del us_frames, pred_heatmaps,target_keypoints,target_areas,target_visibility,target_categories,loss
                else:
                    del us_frames, pred_keypoints,pred_categories,target_keypoints,target_areas,target_visibility,target_categories,loss
                
            ##################Validation################
            self.model.eval() #Set model to evaluation mode for this epoch
            epoch_valid={
                'loss':[],
                'localization_dict':[],
                'detection_dict':[],
            }
            with torch.no_grad():
                #Loop through the validation loader
                for data in tqdm(self.valid_loader, desc="Validation", leave=False,mininterval=1.0):
                    
                    if self.return_mode=='frame': #Using frame mode
                        #Load the data
                        us_frames=data['images'].to(self.device,non_blocking=True) #(B, C, H, W)
                        target_keypoints=data['keypoints'].to(self.device,non_blocking=True) #(B,K,2) where K=number of keypoints
                        target_areas=data['areas'].to(self.device,non_blocking=True) #(B,K)
                        target_visibility=data['visibility'].to(self.device,non_blocking=True)
                        target_categories=data['categories'].to(self.device,non_blocking=True)  #(B,K)
                        px_mul_x=data['px_mul_x'] #Multiplier for pixel to mm (mm=pixel*px_mul), size (B)
                        px_mul_y=data['px_mul_y']

                        #Gets prediction and runs the loss
                        if self.matching_strategy in ('fixed','hungarian'):
                            pred_keypoints,pred_categories=self.model(us_frames)
                            loss=self.loss_fun(pred_keypoints,target_keypoints,target_visibility,target_areas,target_categories,pred_categories=pred_categories)
                        elif self.matching_strategy=='heatmap':
                            pred_heatmaps=self.model(us_frames) #shape (B,2,H',W')
                            loss = self.loss_fun(
                            pred_heatmaps, target_keypoints,
                            target_visibility, target_areas, target_categories,
                            image_shape = (self.H_in,self.W_in),  
                        )

                    if self.return_mode=='clip': #Using the clip_collate_fn return where we pad the end of the time dimension with zeros
                        us_frames=data['images'].to(self.device,non_blocking=True) #(B,T, C, H, W)
                        padding_mask=data['padding_mask'].to(self.device,non_blocking=True) #(B,T)
                        #Skip this batch if less than two batches are valid
                        if padding_mask.sum()<2:
                            del us_frames,padding_mask
                            continue
                        px_mul_x=data['px_mul_x'] #Multiplier for pixel to mm (mm=pixel*px_mul), (B,T)
                        px_mul_y=data['px_mul_y']

                        #Get target keypoints, visibiity and categories
                        target_keypoints=data['keypoints'].to(self.device,non_blocking=True) #(B,T,K_max,2)  
                        target_areas=data['areas'].to(self.device,non_blocking=True)  #(B,T,K)
                        target_visibility=data['visbility'].to(self.device,non_blocking=True) #(B,T,K_max)
                        target_categories=data['categories'].to(self.device,non_blocking=True) #(B,T,K_max)

                        target_visibility=target_visibility & padding_mask.unsqueeze(-1) #mask out any visibility for frames which have been padded

                        if not target_visibility.any():
                            del us_frames,padding_mask, target_keypoints,target_visibility,target_categories,target_areas
                            continue

                        #Gets prediction and runs the loss
                        if self.matching_strategy in ('fixed','hungarian'):
                            pred_keypoints,pred_categories=self.model(us_frames)
                            loss=self.loss_fun(pred_keypoints,target_keypoints,target_visibility,target_areas,target_categories,pred_categories=pred_categories)
                        elif self.matching_strategy=='heatmap':
                            pred_heatmaps=self.model(us_frames) #shape (B,2,H',W')
                            loss = self.loss_fun(
                            pred_heatmaps, target_keypoints,
                            target_visibility, target_areas, target_categories,
                            image_shape = (self.H_in,self.W_in),  
                        )
                    
                    #Check that loss is not nan
                    if torch.isnan(loss):
                        continue
                    ###########Compute Validation Error and Log Loss+Error###########
                    if batch_idx % METRIC_EVERY_N_BATCHES==0: #Compute error for this batch
                        unique_categories = torch.unique(target_categories)
                        unique_categories = unique_categories[unique_categories >0]
                        num_categories = unique_categories.numel()

                        if self.matching_strategy=='heatmap':
                            #Compute error on predicted heatmap using the utils compute error function
                            localization_dict,detection_dict=utils.calculateError(pred=pred_heatmaps,pred_categories=None,target_keypoints=target_keypoints,visibility=target_visibility,areas=target_areas,categories=target_categories,
                                                                                return_mode=self.return_mode,matching_strategy=self.matching_strategy,num_categories=num_categories,image_shape=(self.H_in,self.W_in),
                                                                                px_mul_x=px_mul_x,px_mul_y=px_mul_y,match_threshold=self.match_thresh_pixel,max_diagnoal=self.max_diagnoal)
                        else:
                            localization_dict,detection_dict=utils.calculateError(pred=pred_keypoints,pred_categories=pred_categories,target_keypoints=target_keypoints,visibility=target_visibility,areas=target_areas,categories=target_categories,
                                                                                return_mode=self.return_mode,matching_strategy=self.matching_strategy,num_categories=num_categories,image_shape=(self.H_in,self.W_in),
                                                                                px_mul_x=px_mul_x,px_mul_y=px_mul_y,match_threshold=self.match_thresh_pixel,max_diagnoal=self.max_diagnoal)
                        epoch_valid['localization_dict'].append(self._localization_dict_to_serializable(localization_dict))
                        epoch_valid['detection_dict'].append(detection_dict)
                    
                    #Update the epoch train metrics accumulator
                    epoch_valid['loss'].append(loss.item())
                    
                    #Delete data to save memory
                    if self.matching_strategy=='heatmap':
                        del us_frames, pred_heatmaps,target_keypoints,target_areas,target_visibility,target_categories,loss
                    else:
                        del us_frames, pred_keypoints,pred_categories,target_keypoints,target_areas,target_visibility,target_categories,loss

            
            #First check that we have validation loss
            if len(epoch_valid['loss'])==0:
                pbar.set_postfix_str(f'Epoch {epoch}: no valid metrics (all NaN)')
                continue
            
            #Compute average of the localization and detection dicts for validation
            valid_localization_dict_avg=self.average_localization_dict_overbatches(epoch_valid['localization_dict'])
            valid_detection_dict_avg=self.average_detection_dict_overbatches(epoch_valid['detection_dict'])

            #Compute validation loss average for this epoch, and see if it is better than best validation loss, if so save the checkpoint
            valid_loss_avg=np.mean(epoch_valid['loss'])
            if valid_loss_avg<self.best_valid_loss:
                self.saveCheckpoint(epoch,valid_loss=valid_loss_avg,localization_dict_avg=valid_localization_dict_avg,detection_dict_avg=valid_detection_dict_avg)

            #Compute the average of training localization and detection dicts
            train_localization_dict_avg=self.average_localization_dict_overbatches(epoch_train['localization_dict'])
            train_detection_dict_avg=self.average_detection_dict_overbatches(epoch_train['detection_dict'])

            #Update the progress bar:
            pbar.set_postfix_str('Train Err: EucDist= %.4f (mm), P= %.4f, R= %.4f, F1= %.4f; Valid Err: EucDist= %.4f (mm), P= %.4f, R= %.4f, F1= %.4f'
                                  % (train_localization_dict_avg['euc_dist_mm_avg'],train_detection_dict_avg['overall']['precision'],train_detection_dict_avg['overall']['recall'],train_detection_dict_avg['overall']['f1'],
                                     valid_localization_dict_avg['euc_dist_mm_avg'],valid_detection_dict_avg['overall']['precision'],valid_detection_dict_avg['overall']['recall'],valid_detection_dict_avg['overall']['f1']))

            #Update scheduler if we are using Plateau/Cosine:
            if self.LR_scheduler is not None:
                if isinstance(self.LR_scheduler,torch.optim.lr_scheduler.ReduceLROnPlateau):
                    self.LR_scheduler.step(valid_loss_avg) #Step every epoch for ReduceLROnPlateau
                elif isinstance(self.LR_scheduler,torch.optim.lr_scheduler.CosineAnnealingLR):
                    self.LR_scheduler.step() #Step every epoch for CosineAnnealingLR
            
            #Logging the final results
            for key,value in epoch_train.items():
                self.training_logger[key].append(value) #Separate list of batch results for each epoch
            for key,value in epoch_valid.items():
                self.valid_logger[key].append(value)

            #Saves the loggers to the logs folders
            with open(self.train_logger_save_path,"w") as f:
                json.dump(self.training_logger,f)

            with open(self.valid_logger_save_path,"w") as f:
                json.dump(self.valid_logger,f)

    def average_localization_dict_overbatches(self,localization_dict_list):
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
                # FIX 2: nanmean excludes NaN stds rather than propagating them
                arr = np.array([float(v) for v in values], dtype=np.float64)
                result[key] = float(np.nanmean(arr))

        return result
    
    def average_detection_dict_overbatches(self,detect_dict_list):
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

        # Discover which categories exist from the first dict
        cat_names = list(detect_dict_list[0]['per_category'].keys())
         # ── Per-category aggregation ───────────────────────────────────────────
        per_category_avg = {}
        for cat_name in cat_names:
            bucket_dicts = [d['per_category'][cat_name] for d in detect_dict_list] #Get list of dicts for per-category results for this category
            per_category_avg[cat_name] = self._aggregate_bucket(bucket_dicts)

        # ── Overall aggregation ────────────────────────────────────────────────
        overall_avg = self._aggregate_bucket([d['overall'] for d in detect_dict_list])

        return {
            'per_category': per_category_avg,
            'overall':      overall_avg,
        }


    #Helpers for computing gaverages of the detection dict:
    def _recompute_prf(self,tp, fp, fn):
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

    
    def _aggregate_bucket(self,bucket_dicts):
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
        precision, recall, f1 = self._recompute_prf(total_tp, total_fp, total_fn) #We can't just take the average

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
    
    def _localization_dict_to_serializable(self, d):
        """Convert any tensors in a localization_dict to plain Python types."""
        out = {}
        for k, v in d.items():
            if isinstance(v, torch.Tensor):
                out[k] = v.detach().cpu().tolist()   # scalar → float, (2,) → [x, y]
            else:
                out[k] = v
        return out

