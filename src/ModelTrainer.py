import torch
import torch.nn as nn
from tqdm import trange, tqdm
import os
import numpy as np
import json

import utils




class ModelTrainer(nn.Module):
    def __init__(self,model,loss_fun,optimizer,train_loader,valid_loader,
                 num_epochs,device,loss_type,us_framesize,LR_scheduler=None,return_mode='frame',
                 checkpoint_savedir='models',model_name_save='model',model_name_load='model',start_from_checkpoint=False,
                 matching_strategy='heatmap',match_thresh_percentage=0.1,verbose=False,metric_every_n_batches=5):
        
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
        self.metric_every_n_batches=metric_every_n_batches #Don't compute the training detection and localization errors on every batch, do it on every N'th batch (to save loop time)

        #Initalize the training and validation loggers:
        self.training_logger={'loss':[],'loss_std':[],'localization_dict':[],'detection_dict':[]} #We log each batch's within each epochs loss, detection and localization dicts
        self.valid_logger={'loss':[],'loss_std':[],'localization_dict':[],'detection_dict':[]} 

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
        logger_save_dir=os.path.join('../logs',model_name_save)
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
        checkpoint=torch.load(self.checkpoint_loadpath, map_location=self.device, weights_only=False)
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

                elif self.return_mode=='clip': #Using the clip_collate_fn return where we pad the end of the time dimension with zeros
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
                
                else:
                    raise ValueError(f"Unknown return_mode: '{self.return_mode}'")
                        
                

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
                if batch_idx % self.metric_every_n_batches==0: #Compute error for this batch
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
                    epoch_train['localization_dict'].append(utils._localization_dict_to_serializable(localization_dict))
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

                    elif self.return_mode=='clip': #Using the clip_collate_fn return where we pad the end of the time dimension with zeros
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
                    else:
                        raise ValueError(f"Unknown return_mode: '{self.return_mode}'")
                    
                    #Check that loss is not nan
                    if torch.isnan(loss):
                        continue
                    ###########Compute Validation Error and Log Loss+Error###########
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
                    epoch_valid['loss'].append(loss.item())
                    epoch_valid['localization_dict'].append(utils._localization_dict_to_serializable(localization_dict))
                    epoch_valid['detection_dict'].append(detection_dict)
                    
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
            valid_localization_dict_avg=utils.average_localization_dict_serialized(epoch_valid['localization_dict'])
            valid_detection_dict_avg=utils.average_detection_dict_overbatches(epoch_valid['detection_dict'])

            #Compute validation loss average for this epoch, and see if it is better than best validation loss, if so save the checkpoint
            valid_loss_avg=np.mean(epoch_valid['loss'])
            if valid_loss_avg<self.best_valid_loss:
                self.saveCheckpoint(epoch,valid_loss=valid_loss_avg,localization_dict_avg=valid_localization_dict_avg,detection_dict_avg=valid_detection_dict_avg)

            #Compute the average of training localization and detection dicts
            train_localization_dict_avg=utils.average_localization_dict_serialized(epoch_train['localization_dict'])
            train_detection_dict_avg=utils.average_detection_dict_overbatches(epoch_train['detection_dict'])

            #Update the progress bar:
            pbar.set_postfix_str('Tr Er: EuDis=%.4f (mm), P=%.4f, R=%.4f, F1=%.4f; Val Er: EuDist=%.4f (mm), P=%.4f, R=%.4f, F1=%.4f'
                                  % (train_localization_dict_avg['euc_dist_mm_avg'],train_detection_dict_avg['overall']['precision'],train_detection_dict_avg['overall']['recall'],train_detection_dict_avg['overall']['f1'],
                                     valid_localization_dict_avg['euc_dist_mm_avg'],valid_detection_dict_avg['overall']['precision'],valid_detection_dict_avg['overall']['recall'],valid_detection_dict_avg['overall']['f1']))

            #Update scheduler if we are using Plateau/Cosine:
            if self.LR_scheduler is not None:
                if isinstance(self.LR_scheduler,torch.optim.lr_scheduler.ReduceLROnPlateau):
                    self.LR_scheduler.step(valid_loss_avg) #Step every epoch for ReduceLROnPlateau
                elif isinstance(self.LR_scheduler,torch.optim.lr_scheduler.CosineAnnealingLR):
                    self.LR_scheduler.step() #Step every epoch for CosineAnnealingLR
            
            #######Logging the final results########
            #Store the loss, keep the raw batches nested within each epoch
            self.training_logger['loss'].append(epoch_train['loss'])
            self.training_logger['loss_std'].append(float(np.std(epoch_train['loss']))) #Compute std across batch
            self.valid_logger['loss'].append(epoch_valid['loss'])
            self.valid_logger['loss_std'].append(float(np.std(epoch_valid['loss']))) #Compute std across batch

            #Localization and detection are stored as averages across the batch
            self.training_logger['localization_dict'].append(utils._localization_dict_to_serializable(train_localization_dict_avg))
            self.training_logger['detection_dict'].append(train_detection_dict_avg)
            self.valid_logger['localization_dict'].append(utils._localization_dict_to_serializable(valid_localization_dict_avg))
            self.valid_logger['detection_dict'].append(valid_detection_dict_avg)

            #Saves the loggers to the logs folders
            with open(self.train_logger_save_path,"w") as f:
                json.dump(self.training_logger,f)

            with open(self.valid_logger_save_path,"w") as f:
                json.dump(self.valid_logger,f) 


