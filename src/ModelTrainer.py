import torch
import torch.nn as nn
from tqdm import trange, tqdm
import os


class ModelTrainer(nn.Module):
    def __init__(self,model,loss_fun,optimizer,train_loader,valid_loader,
                 num_epocs,device,loss_type,us_framesize,LR_scheduler=None,return_mode='frame',
                 checkpoint_savedir='models',model_name_save='model',model_name_load='model',start_from_checkpoint=False,
                 matching_strategy='heatmap'):
        
        super(ModelTrainer,self).__init__()
        ##########Init Class Params#########
        self.model=model
        self.loss_fun=loss_fun
        self.optimizer=optimizer
        self.train_loader=train_loader
        self.valid_loader=valid_loader
        self.num_epochs=num_epocs
        self.device=device
        self.loss_type=loss_type
        self.LR_scheduler=LR_scheduler
        self.return_mode=return_mode
        self.matching_strategy=matching_strategy
        self.H_in=us_framesize[0]
        self.W_in=us_framesize[1]


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
    def saveCheckpoint(self,epoch,valid_loss,euc_mm_err_avg,euc_mm_err_std,peraxis_mm_err_avg,peraxis_mm_err_std): #Save model to file for checkpointing
        if self.checkpoint_savepath is None:
            return #No file name defined
        if not os.path.exists(self.checkpoint_savedir):
            os.makedirs(self.checkpoint_savedir)
        self.best_valid_loss=valid_loss
        torch.save({
            'epoch':epoch,
            'model_state_dict':self.model.state_dict(),
            'optimizer_state_dict':self.optimizer.state_dict(),
            'best_valid_loss':self.best_valid_loss,
            'training_logger':self.training_logger,
            'valid_logger':self.valid_logger,
            'euc_mm_err_avg':euc_mm_err_avg,
            'euc_mm_err_std':euc_mm_err_std,
            'peraxis_mm_err_avg':peraxis_mm_err_avg,
            'peraxis_mm_err_std':peraxis_mm_err_std
        },self.checkpoint_savepath)

    def loadCheckpoint(self):
        #Loads from checkpoint if we are doing that
        checkpoint=torch.load(self.checkpoint_loadpath)
        self.model.load_state_dict(checkpoint['model_state_dict'])  
        self.model=self.model.to(self.device)
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict']) 
        self.start_epoch=checkpoint['epoch']
        self.best_valid_loss=checkpoint['best_valid_loss']     
        self.training_logger=checkpoint['training_logger']   
        self.valid_logger=checkpoint['valid_logger']

        if self.verbose:
            print(f"Loaded checkpoint from epoch {self.start_epoch}, best validation loss: {self.best_valid_loss:.4f}")
    
    def train(self):
        #######Main Trainer Function where we conduct both training and validation
        pbar=trange(self.start_epoch,self.num_epochs,leave=False,desc="Epoch")

        for epoch in pbar: #Loops for num_epochs
            
            ###################Training################
            self.model.train() #Sets model to training mode

            #Loops through the training loader
            for data in tqdm(self.train_loader, desc="Training", leave=False,mininterval=1.0):

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
                    target_categories=target_categories.detach()

                    #Delete data to save memory
                    if self.matching_strategy=='heatmap':
                        del us_frames, pred_heatmaps,target_keypoints,target_areas,target_visibility,target_categories,loss
                    else:
                        del us_frames, pred_keypoints,pred_categories,target_keypoints,target_areas,target_visibility,target_categories,loss
                

                ###########Log Training Error###########






