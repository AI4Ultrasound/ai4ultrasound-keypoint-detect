import torch
import torch.nn as nn


class KeypointLoss(nn.Module):
    #############Init
    def __init__(self,loss_type,weights,device,sigmas=None,return_mode='frame'):
        '''
        Inputs:
            - loss_type: supported loss functions:
                - 'L1' = MAE loss = weights[0]*sum(|y_targ,i-y_pred,i|)/N
                - 'L2' = MSE loss = weights[0]*sum((y_targ,i-y_pred,i)^2)/N
                - 'OKS' = weights[0]*(1-OKS score) = 1-sum(KSi)/N 
                - 'OKS+L1' = weights[0]*OKS + weights[1]*L1
                - 'OKS+L2' = weights[0]*OKS + weights[1]*L2
            - weights: list for each term in the selected loss function
            - sigmas: per-keypoint standard deviation. List of 2 values. pleural-line std and b-line std.
            Only used with OKS. Entry 0 is pleural line sigma, entry 1 is b-line sigma
            - return_mode: if we are estimating keypoints frame-by-frame (keypoints are (B,K_t), or at the clip level (keypoints are (B,T, K_t))
                - 'frame' loss computed over B samples
                - 'clip' loss computed over B*T samples

        Sources:
            - L1, L2 and OKS loss comes from: https://ieeexplore.ieee.org/document/10586698
        
        '''
        super().__init__()
        #Init supported loss types:
        supported=['L1','L2','OKS','OKS+L1','OKS+L2']

        #Checks that the loss_type is supported
        if loss_type not in supported:
            raise ValueError(f"Unsupported loss_type '{loss_type}'. Choose from {supported}")
        if loss_type in ('OKS','OKS+L1','OKS+L2') and sigmas is None:
            raise ValueError("sigmas must be provided when loss_type='OKS', 'OKS+L1' or 'OKS+L2'")
        if return_mode not in ('frame','clip'):
            raise ValueError(f"Unsupported return_mode '{return_mode}'. Choose from 'frame' or 'clip'")
        #Init the loss type and weights class vars
        self.loss_type=loss_type
        self.weights=weights.to(device)
        self.return_mode=return_mode

        if self.loss_type=='L1':
            self.loss_fun=self.L1Loss()
        
        elif self.loss_type=='L2':
            self.loss_fun=self.L2Loss()
        
        elif self.loss_type=='OKS':
            #Register sigmas as a buffer so sigmas move with the model
            self.sigmas=sigmas.to(device)
            self.register_buffer('sigmas',self.sigmas)
            self.loss_fun=self.OKSLoss

        elif self.loss_type=='OKS+L1':
            self.sigmas=sigmas.to(device)
            self.register_buffer('sigmas',self.sigmas)
            self.loss_fun_oks=self.OKSLoss
            self.loss_fun_l1=self.L1Loss()
        
        elif self.loss_type=='OKS+L2':
            self.sigmas=sigmas.to(device)
            self.register_buffer('sigmas',self.sigmas)
            self.loss_fun_oks=self.OKSLoss
            self.loss_fun_l2=self.L2Loss()
        
        else:
            raise ValueError(f"Unsupported loss_type: {self.loss_type}")
    ##############Loss Function Helpers#############
    def L1Loss(self):
        l1_loss=nn.L1Loss(reduction='mean') #Returns mean of MAE between points
        return l1_loss
    def L2Loss(self):
        l2_loss=nn.MSELoss(reduction='mean') #Returns mean of MSE between points
        return l2_loss
    
    def OKSLoss(self,pred,target,visibility,areas,categories):
        '''
        Computes the OKS loss: 1-OKS, handling variable numbers of keypoints.
        Inputs:
            pred=predicted keypoints (B,K,2) - padded to max K
            target=target keypoints (B,K,2) - padded to max K
            visibility=keypoint flags (B,K) - 1=present, 0=absent/padded (this is dirac delta)
            areas=area of bounding box (B,K) - each keypoint's object area
            categories: 1 = pleural line, 2=b-line, 0=padded (because we pad to max K). Used to assign correct sigma to each keypoint
        '''
        #Get the squared euclidean distance between points d^2 = (x1-x2)^2 + (y1-y2)^2
        dist_sq=torch.sum((pred-target)**2,dim=-1) # Shape: (B, K)

        #Get the normalization factor of the kepoint similarity metric (denominator):
        s_sq=(0.53*areas)**2 #Areas of keypoint bounding boxes multiplied by huristic 0.53 squared(B,K) 

        #Get the per-keypoint sigma
        #self.sigmas = [sigma_pleural,sigma_bline], shape (2,)
        sigma_idx = (categories - 1).clamp(min=0)            # (B, K), values in {0, 1}, pleural = 0, b-line = 1, padded = 0. Padded = 0 okay because we mask anyways
        sigma_map=self.sigmas[sigma_idx]

        #Per-keypoint sigma term
        two_sigma_sq=(2.0*sigma_map)**2 #(B,K)
        #KS denominator
        denominator=2*s_sq*two_sigma_sq +1e-8 #(B,K)

        #Per-keypoint similarity score
        #KS_i = exp(-d_i^2 / (2 * s^2 * (2*sigma_i)^2))
        ks=torch.exp(-dist_sq/denominator)

        #Extracts the visibility mask
        vis_mask=(visibility>0).float()     #(B,K)
        num_visible=vis_mask.sum(dim=-1) #(B,)

        #OKS per image, shape (B,)
        oks_per_image=(ks*vis_mask).sum(dim=-1)/(num_visible+1e-8) #averages KS values for this image

        #Only average images in batch that have keypoints
        valid_images=(num_visible>0).float() #(B,)
        n_valid=valid_images.sum().clamp(min=1.0)

        #OKS loss = mean(1-OKS) over batch
        loss=((1.0-oks_per_image)*valid_images).sum()/n_valid

        return loss


    

    ##############Forward Method###########

    def forward(self,pred,target,visibility=None,areas=None,categories=None):
        '''
        pred=predicted keypoints (B,K,2) - padded to max K when loss_type=OKS
        target=target keypoints (B,K,2) - padded to max K when loss_type=OKS
        visibility=keypoint flags (B,K) - 1=present, 0=absent/padded (this is dirac delta) - used when loss_type=OKS
        areas=area of bounding box (B,K) - each keypoint's object area - used when when loss_type=OKS
        categories: 1 = pleural line, 2=b-line, 0=padded (because we pad to max K). Used to assign correct sigma to each keypoint - used when loss_type=OKS
        '''

        ######If we are doing 'clip' mode, we convert from (B,T,K,2) to (B*T,K,2)
        if pred.dim()==4 and self.return_mode=='clip':
            B,T,K,C=pred.shape
            pred=pred.view(B*T,K,C) #Reshape to (B*T,K,2)
            target=target.view(B*T,K,C)
            if visibility  is not None: visibility  = visibility.view(B * T, K)
            if areas       is not None: areas       = areas.view(B * T, K)
            if categories  is not None: categories  = categories.view(B * T, K)
        elif pred.dim()==3 and self.return_mode=='frame':
            pass
        else:
            raise ValueError(f"Error with return_mode, got pred dim: {pred.dim()}D and return_mode: {self.return_mode}")
        
        # --- Compute loss on (B_eff, K, 2) tensors ---
        # B_eff = B in frame mode, B*T in clip mode

        if self.loss_type=='L1':
            loss=self.weights[0]*self.loss_fun(pred,target)
        elif self.loss_type=='L2':
            loss=self.weights[0]*self.loss_fun(pred,target)
        elif self.loss_type=='OKS':
            loss=self.weights[0]*self.loss_fun(pred,target,visibility,areas,categories)
        elif self.loss_type=='OKS+L1':
            vis_mask = (visibility > 0).unsqueeze(-1).float()  # (B, K, 1)
            pred_masked   = pred * vis_mask #Zero out the predictions for padded iamges
            target_masked = target * vis_mask 
            loss=(self.weights[0]*self.loss_fun_oks(pred,target,visibility,areas,categories)
            + self.weights[1]*self.loss_fun_l1(pred_masked,target_masked))
        elif self.loss_type=='OKS+L2':
            vis_mask = (visibility > 0).unsqueeze(-1).float()  # (B, K, 1)
            pred_masked   = pred * vis_mask #Zero out the predictions for padded iamges
            target_masked = target * vis_mask 
            loss=(self.weights[0]*self.loss_fun_oks(pred,target,visibility,areas,categories)
                  + self.weights[1]*self.loss_fun_l2(pred_masked,target_masked))
        else:
            raise ValueError(f"Unsupported loss_type: {self.loss_type}")

    

        #Returns the loss value:
        return loss
        