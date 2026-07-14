import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment


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
    
    def _apply_hungarian_matching(self, pred, pred_cats_logits, target, visibility, areas, categories):
        """
        Solve the optimal 1-to-1 assignment per image, then repack the
        matched pairs into padded tensors so _compute_keypoint_loss can be
        applied identically to the fixed-matching path.

        Inputs
        ------
        pred             : (B, K_pred, 2)
        pred_cats_logits : (B, K_pred, num_classes)  or  None
        target           : (B, K_tgt,  2)
        visibility       : (B, K_tgt)   bool
        areas            : (B, K_tgt)   float  or  None
        categories       : (B, K_tgt)   long

        Returns  (padded to K_max = largest matched-pair count in the batch)
        -------
        matched_pred  : (B, K_max, 2)
        matched_tgt   : (B, K_max, 2)
        matched_vis   : (B, K_max)    bool   True=real pair, False=padding
        matched_areas : (B, K_max)    float  or  None
        matched_cats  : (B, K_max)    long   category of the matched target
        """

        B=pred.shape[0]
        device=pred.device

        # ---- solve per image -------------------------------------------
        all_pred_idx, all_tgt_idx = [], []
        #Loop through the batch
        for b in range(B):
            pi, tj = self._hungarian_match_single(
                pred_kps         = pred[b],
                pred_cats_logits = (pred_cats_logits[b]
                                    if pred_cats_logits is not None else None),
                target_kps  = target[b],
                target_cats = categories[b],
                vis_mask    = visibility[b],
            )

            #Append to list containing all the matching indices
            all_pred_idx.append(pi)
            all_tgt_idx.append(tj)

        # ---- pad to largest matched count ------------------------------
        counts = [idx.shape[0] for idx in all_pred_idx]
        K_max  = max(counts) if any(c > 0 for c in counts) else 1

        #Pre-allocate tensors
        matched_pred=torch.zeros(B, K_max, 2, dtype=pred.dtype,   device=device)
        matched_tgt=torch.zeros(B, K_max, 2, dtype=target.dtype, device=device)
        matched_vis=torch.zeros(B, K_max,    dtype=torch.bool,   device=device)
        matched_areas=(torch.zeros(B, K_max, dtype=areas.dtype, device=device)
                         if areas is not None else None)
        matched_cats=torch.full((B, K_max), -1, dtype=torch.long, device=device)

        for b in range(B):
            pred_i=all_pred_idx[b].to(device)
            tgt_j=all_tgt_idx[b].to(device)
            N=pred_i.shape[0]
            if N == 0:
                continue
            matched_pred[b, :N]=pred[b][pred_i] #Gets the matching pred and target up to max size of pred
            matched_tgt[b,  :N]=target[b][tgt_j]
            matched_vis[b,  :N]=True
            if areas is not None:
                matched_areas[b, :N]=areas[b][tgt_j]
            matched_cats[b, :N]=categories[b][tgt_j]

        return matched_pred, matched_tgt, matched_vis, matched_areas, matched_cats

    def _hungarian_match_single(
        self, pred_kps, pred_cats_logits, target_kps, target_cats, vis_mask
    ):
        """
        Solve the assignment problem for one image (finds closest matching keypoints).

        pred_kps         : (K_pred, 2)
        pred_cats_logits : (K_pred, num_classes)  or  None
        target_kps       : (K_tgt,  2)
        target_cats      : (K_tgt,) long   1=pleural  2=bline  -1=padded
        vis_mask         : (K_tgt,) bool

        Returns
        -------
        pred_idx   : (N_matched,) long — matched row indices in pred
        target_idx : (N_matched,) long — matched row indices in target
        """
        real_tgt_idx = vis_mask.nonzero(as_tuple=False).squeeze(1)   # (N_real,)
        N_real = real_tgt_idx.shape[0]
        if N_real == 0:
            return (torch.empty(0, dtype=torch.long),
                    torch.empty(0, dtype=torch.long)) #Returns empty indexes if there are no visible keypoints

        K_pred = pred_kps.shape[0]

        # Spatial cost: squared L2  (K_pred × N_real)
        p    = pred_kps.unsqueeze(1)                       # (K_pred, 1,      2)
        t    = target_kps[real_tgt_idx].unsqueeze(0)       # (1,      N_real, 2)
        cost = torch.sum((p - t) ** 2, dim=-1)             # (K_pred, N_real)

        # Optional category cost: CE for every pred–target combination
        # This gives the category head gradient signal through the cost matrix
        # NOTE: linear_sum_assignment uses .detach() so gradients do NOT flow
        # through the matching decision itself — only through the loss on matched pairs.
        # Add an explicit category CE loss term in your training loop if you want
        # stronger gradient signal to the category head.
        if pred_cats_logits is not None:
            cat_cost = torch.zeros(K_pred, N_real, device=pred_kps.device)
            for j, tgt_j in enumerate(real_tgt_idx):
                # Category labels are 1-indexed; CE expects 0-indexed
                label          = (target_cats[tgt_j] - 1).clamp(min=0).expand(K_pred)
                cat_cost[:, j] = F.cross_entropy(
                    pred_cats_logits, label, reduction='none'
                )
            cost = cost + cat_cost                          # (K_pred, N_real)

        # Solve on CPU (scipy requirement)
        pred_i, tgt_j = linear_sum_assignment(cost.detach().cpu().numpy())

        pred_i = torch.tensor(pred_i, dtype=torch.long)
        tgt_j  = real_tgt_idx[torch.tensor(tgt_j, dtype=torch.long)]
        return pred_i, tgt_j
    
    def _make_target_heatmaps(self, keypoints, visibility, categories, H_out, W_out, H_in, W_in):
        """
        Creates Gaussian heatmaps from target keypoint coordinates.
        keypoints  : (B, K, 2)  pixel coords in input image space (x, y)
        visibility : (B, K)   bool
        categories : (B, K)   long  1=pleural  2=bline  -1=padded
        H_out/W_out : heatmap spatial dimensions
        H_in/W_in   : input image spatial dimensions  (for coordinate scaling)

        Returns
        -------
        heatmaps      : (B, num_categories, H_out, W_out)  float32
        target_weight : (B, num_categories)                 float32
                        1.0 if channel has >= 1 visible keypoint, else 0.0
        """

        B, K, _ = keypoints.shape
        device  = keypoints.device
        scale_x = W_out / W_in
        scale_y = H_out / H_in

        yy = torch.arange(H_out, dtype=torch.float32, device=device)
        xx = torch.arange(W_out, dtype=torch.float32, device=device)
        grid_y, grid_x = torch.meshgrid(yy, xx, indexing='ij')   # (H_out, W_out)

        heatmaps      = torch.zeros(B, self.num_categories, H_out, W_out,
                                    dtype=torch.float32, device=device)
        target_weight = torch.zeros(B, self.num_categories,
                                    dtype=torch.float32, device=device)

        for b in range(B):
            for k in range(K):
                if not visibility[b, k]:
                    continue
                cat_idx = int(categories[b, k].item()) - 1    # 1→0, 2→1
                if cat_idx < 0:
                    continue   # padded slot

                cx = keypoints[b, k, 0].item() * scale_x
                cy = keypoints[b, k, 1].item() * scale_y

                gauss = torch.exp(
                    -((grid_x - cx) ** 2 + (grid_y - cy) ** 2)
                    / (2.0 * self.heatmap_sigma ** 2)
                )   # (H_out, W_out)

                # max-blend: N keypoints of the same category → N distinct peaks
                heatmaps[b, cat_idx]      = torch.max(heatmaps[b, cat_idx], gauss)
                target_weight[b, cat_idx] = 1.0

        return heatmaps, target_weight

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
            pred_matched, target_matched, vis_matched, areas_matched, cats_matched = self._apply_hungarian_matching(pred, pred_categories, target, visibility, areas, categories)
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
            heatmaps, weight = self._make_target_heatmaps(target, visibility, categories, H_out, W_out, H_in, W_in)
            loss=self._compute_heatmap_loss(pred,heatmaps,weight)
            
            
        #Returns the loss value:
        return loss
        