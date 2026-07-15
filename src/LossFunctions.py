import torch
import torch.nn as nn
import utils



class KeypointLoss(nn.Module):
    SUPPORTED_LOSS=('L1','L2','OKS','OKS+L1','OKS+L2')
    SUPPORTED_MATCHING=('fixed', 'hungarian', 'heatmap')
    #############Init
    def __init__(self,loss_type,weights,device,
                 sigmas=None,return_mode='frame',num_categories=2,heatmap_sigma=2.0,matching_strategy='heatmap'):
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
            - num_categories: Number of keypoint categories.  Used only for heatmap mode (default 2).
            - heatmap_sigma: Gaussian σ in heatmap pixels for target generation. Used only for heatmap mode (default 2.0).
            - matching_strategy:
                - 'fixed': pred[b,k] is already aligned to target[b,k] by index.
                - 'hungarian': per-image optimal assignment solved before the loss.
                - 'heatmap': model outputs spatial heatmaps; loss is pixel-wise.

        Sources:
            - L1, L2 and OKS loss comes from: https://ieeexplore.ieee.org/document/10586698
        
        '''
        super().__init__()
        #Init supported loss types:
        

        #Checks that the loss_type is supported
        if loss_type not in self.SUPPORTED_LOSS:
            raise ValueError(f"Unsupported loss_type '{loss_type}'. Choose from {self.SUPPORTED_LOSS}")
        
        if matching_strategy not in self.SUPPORTED_MATCHING:
            raise ValueError(f"Unsupported matching_strategy '{matching_strategy}'. Choose from {self.SUPPORTED_MATCHING}")

        if loss_type in ('OKS','OKS+L1','OKS+L2') and sigmas is None:
            raise ValueError("sigmas must be provided when loss_type='OKS', 'OKS+L1' or 'OKS+L2'")
        if return_mode not in ('frame','clip'):
            raise ValueError(f"Unsupported return_mode '{return_mode}'. Choose from 'frame' or 'clip'")
        
        if matching_strategy == 'heatmap' and loss_type in ('OKS', 'OKS+L1', 'OKS+L2'):
            raise ValueError("OKS requires coordinate predictions and is incompatible with heatmap matching. Use 'L1' or 'L2' for heatmap mode.")
        
        #Init the loss type and weights class vars
        self.loss_type=loss_type
        self.return_mode=return_mode
        self.matching_strategy=matching_strategy
        self.num_categories=num_categories
        self.heatmap_sigma=heatmap_sigma


        #Register weights as a buffer so model.to(device) moves it automatically
        if not isinstance(weights, torch.Tensor):
            weights = torch.tensor(weights, dtype=torch.float32)
        self.register_buffer('weights', weights.to(device))

        #Cast sigmas (keypoint standard deviation)
        if loss_type in ('OKS', 'OKS+L1', 'OKS+L2'):
                if not isinstance(sigmas, torch.Tensor):
                    sigmas = torch.tensor(sigmas, dtype=torch.float32)
                self.register_buffer('sigmas', sigmas.to(device))

    ##############Loss Function Helpers#############    
    def OKSLoss(self,pred,target,visibility,areas,categories):
        '''
        Computes the OKS loss: 1-OKS, handling variable numbers of keypoints.
        Inputs:
            pred=predicted keypoints (B,K,2) - padded to max K
            target=target keypoints (B,K,2) - padded to max K
            visibility=keypoint flags (B,K) - 1=present, 0=absent/padded (this is dirac delta)
            areas=area of bounding box (B,K) - each keypoint's object area
            categories: 1 = pleural line, 2=b-line, -1=padded/empty (because we pad to max K). Used to assign correct sigma to each keypoint
        '''
        #Get the squared euclidean distance between points d^2 = (x1-x2)^2 + (y1-y2)^2
        dist_sq=torch.sum((pred-target)**2,dim=-1) # Shape: (B, K)

        #Get the normalization factor of the kepoint similarity metric (denominator):
        s_sq=areas #Areas of keypoint bounding boxes multiplied by huristic 0.53 squared(B,K) 

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
    def _compute_keypoint_loss(self, pred, target, visibility, areas, categories):
        #Create a vis_mask that invisible/padded keypoints are zeroid out
        vis_mask = (visibility > 0).unsqueeze(-1).float()
        n_vis= vis_mask.sum().clamp(min=1.0)           # total visible keypoints

        # --- Compute loss on (B_eff, K, 2) tensors ---
        # B_eff = B in frame mode, B*T in clip mode

        if self.loss_type=='L1':
            loss = self.weights[0] * ((torch.abs(pred - target) * vis_mask).sum() / n_vis)
        elif self.loss_type=='L2':
            loss = self.weights[0] * ((((pred - target) ** 2) * vis_mask).sum() / n_vis)
        elif self.loss_type=='OKS':
            loss = self.weights[0] * self.OKSLoss(pred, target, visibility, areas, categories)
        elif self.loss_type=='OKS+L1':
            l1=(torch.abs(pred - target) * vis_mask).sum() / n_vis
            #Combined OKS and L1 loss
            loss=self.weights[0] * self.OKSLoss(pred, target, visibility, areas, categories)+ self.weights[1] * l1
        elif self.loss_type=='OKS+L2':
            l2=(((pred - target) ** 2) * vis_mask).sum() / n_vis
            #Combined OKS and L2 loss
            loss=self.weights[0] * self.OKSLoss(pred, target, visibility, areas, categories)+ self.weights[1] * l2
        else:
            raise ValueError(f"Unsupported loss_type: {self.loss_type}")
        
        return loss
    
    def _compute_heatmap_loss(self, pred, target_heatmaps, target_weight):
        """
        Pixel-wise L1 or L2 between pred and target heatmaps.
        Channels with no ground-truth keypoints are zeroed via target_weight,
        matching the MMPose KeypointMSELoss(use_target_weight=True) convention.

        pred             : (B, C, H', W')
        target_heatmaps  : (B, C, H', W')
        target_weight    : (B, C)  float   1=channel active  0=no keypoints
        """
        weight = target_weight[:, :, None, None]   # (B, C, 1, 1) — broadcast

        if self.loss_type == 'L1':
            loss= self.weights[0] * (
                torch.abs(pred - target_heatmaps) * weight
            ).mean()
        elif self.loss_type == 'L2':
            loss= self.weights[0] * (
                ((pred - target_heatmaps) ** 2) * weight
            ).mean()
        else:
            raise ValueError(
                f"loss_type '{self.loss_type}' is not supported for heatmap mode. "
                f"Use 'L1' or 'L2'."
            )
        return loss

    ##############Forward Methods###########

    def forward(self,pred,target,visibility=None,areas=None,categories=None,pred_categories=None,image_shape=None):
        '''
        Depends on the matching_strategy:
        'fixed' / 'hungarian':
            pred            : (B, K_pred, 2)
            pred_categories : (B, K_pred, num_classes)  optional — used in
                              the Hungarian cost matrix to help separate
                              pleural vs b-line predictions during assignment.

        'heatmap':
            pred            : (B, num_cat, H', W')
            image_shape     : (H_in, W_in)             required

        All modes:
            target     : (B, K, 2)    ground-truth keypoint pixel coordinates
            visibility : (B, K)  bool
            areas      : (B, K)  float   bounding-box area [px²]  (OKS only)
            categories : (B, K)  long    1=pleural  2=bline  -1=padded

        ***Note: in the case when return_mode=='clip' it is (B,T..) dimensions above
        '''

        ######If we are doing 'clip' mode
        if self.return_mode=='clip':
            if self.matching_strategy in ('fixed','hungarian'):
                if pred.dim() != 4:
                    raise ValueError(
                        f"clip + {self.matching_strategy}: expected 4-D pred "
                        f"(B,T,K,2), got {pred.dim()}-D"
                    )
                #convert from (B,T,K,2) to (B*T,K,2)
                B,T,K_pred,_=pred.shape
                pred   = pred.reshape(B * T, K_pred, 2)
                target = target.reshape(B * T, target.shape[2], 2)
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
                target = target.reshape(B * T, target.shape[2], 2)
                if visibility  is not None: visibility  = visibility.reshape(B * T, -1)
                if areas       is not None: areas       = areas.reshape(B * T, -1)
                if categories  is not None: categories  = categories.reshape(B * T, -1)
                
        elif self.return_mode=='frame':
            expected = 3 if self.matching_strategy in ('fixed', 'hungarian') else 4
            if pred.dim() != expected:
                raise ValueError(
                    f"frame + {self.matching_strategy}: expected {expected}-D pred, "
                    f"got {pred.dim()}-D"
                )
        else:
            raise ValueError(f"Error with return_mode, got pred dim: {pred.dim()}D and return_mode: {self.return_mode}")
        
        # Default: treat all positions as visible (no padding present)
        if visibility is None:
            visibility = torch.ones(pred.shape[:2], dtype=torch.bool, device=pred.device)
        
        

        ###############Do matching -> Then Loss
        if self.matching_strategy=='fixed':
            loss=self._compute_keypoint_loss(pred, target, visibility, areas, categories)
        elif self.matching_strategy=='hungarian':
            pred_matched, target_matched, vis_matched, areas_matched, cats_matched = utils.apply_hungarian_matching(pred, pred_categories, target, visibility, areas, categories)
            loss=self._compute_keypoint_loss(pred_matched, target_matched, vis_matched, areas_matched, cats_matched)
        elif self.matching_strategy=='heatmap':
            if image_shape is None:
                raise ValueError(
                    "image_shape=(H_in, W_in) must be supplied for heatmap matching."
                )
            H_in, W_in   = image_shape
            H_out, W_out = pred.shape[2], pred.shape[3]
            if H_out is None or W_out is None:
                raise ValueError(
                    "(H_in, W_in) must be supplied for heatmap matching."
                )
            heatmaps, weight = utils.make_target_heatmaps(target, visibility, categories, H_out, W_out, H_in, W_in,self.num_categories,self.heatmap_sigma)
            loss=self._compute_heatmap_loss(pred,heatmaps,weight)
            
            
        #Returns the loss value:
        return loss
        