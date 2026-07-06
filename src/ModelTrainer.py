import torch
import torch.nn as nn
from tqdm import trange, tqdm


class ModelTrainer(nn.Module):
    def __init__(self,model,loss_fun,optimizer,train_loader,valid_loader,
                 num_epocs,device,loss_type,LR_scheduler=None,return_mode='frame',
                 checkpoint_savedir='models',model_name_save='model',model_name_load='model',start_from_checkpoint=False,):
        
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

        #Checkpointing params
        self.checkpoint_savedir=checkpoint_savedir
        self.model_name_save=model_name_save
        self.model_name_load=model_name_load


        self.start_epoch=0
        self.best_valid_loss=float(10000.0) #Keep model with best validation loss
    
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
                    target_keypoints=data['keypoints'].to(self.device,non_blocking=True) #(B,list[N_t,K_i,2]) where N_t=# of annotations/frame and K_i=number of keypoints per annotation
                    categories=data['categories'].to(self.device,non_blocking=True)  #(B,N_t)
                    px_mul_x=data['px_mul_x'] #Multiplier for pixel to mm (mm=pixel*px_mul)
                    px_mul_y=data['px_mul_y']

                if self.return_mode=='clip': #Using the clip_collate_fn return where we pad the end of the time dimension with zeros
                    us_frames=data['images'].to(self.device,non_blocking=True) #(B,T, C, H, W)
                    padding_mask=data['padding_mask'].to(self.device,non_blocking=True) #(B,T,)
                    target_keypoints=data['keypoints'].to(self.device,non_blocking=True) #(B,T,list[N_t,K_i,2])                    
                    categories=data['categories'].to(self.device,non_blocking=True) #(B,T,N_t)
                    px_mul_x=data['px_mul_x'] #Multiplier for pixel to mm (mm=pixel*px_mul)
                    px_mul_y=data['px_mul_y']
                #Gets the prediction:
                pred_keypoints=self.model(us_frames)
                #Runs the loss function

                loss=self.loss_fun(pred_keypoints,target_keypoints)

                if torch.isnan(loss):
                    self.optimizer.zero_grad()
                    del us_frames, target_keypoints,categories,loss
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

                    #Detach pred_keypoints and target_keypoints from computational graph
                    pred_keypoints=pred_keypoints.detach()
                    target_keypoints=target_keypoints.detach()

                    #Delete data to save memory
                    del us_frames, target_keypoints,categories,loss
                






