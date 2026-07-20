import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.rnn as rnn_utils
from mmengine.config import Config
from mmpose.models.backbones import HRNet
from mmpose.models.heads import HeatmapHead
from mmengine.runner import load_checkpoint
from mmpose.apis import init_model
from mim import download as mim_download

import os

########HRNet-W48 architecture config

_MMPOSE_CONFIG_ALIAS = {
    'pose_hrnet_w48_udp': 'td-hm_hrnet-w48_udp-8xb32-210e_coco-256x192',
}

_HRNET_W48_OUT_CH = 48



class HRNet_UDP(nn.Module):
    def __init__(self,model_type,device,return_mode='frame',num_categories=1,in_channels=1,mmpose_config=None,pretrained_backbone=None,mmpose_cache_dir='mmpose_model_cache'):
        """
        Wraps MMPose's HRNet-W48 UDP with a lightweight custom heatmap head. 

        Input:
        - model_type: 'pose_hrnet_w48_udp'
        - device: torch.device
        - return_mode: 'frame' or 'clip'
        - num_categories: Number of heatmap output channels (1 for pleuraline or bline only, 2 for both)
        - in_channels: 1 for grayscale, 3 for RGB
        - pretrained_backbone: Path to MMPose .pth checkpoint 
        """
        super(HRNet_UDP,self).__init__()
        self.model_type=model_type
        self.return_mode=return_mode
        self.device=device
        self.num_categories=num_categories
        self.in_channels=in_channels


        if self.model_type=='pose_hrnet_w48_udp':
            self._build_hrnet_w48_udp(mmpose_config=mmpose_config,pretrained_backbone=pretrained_backbone,cache_dir=mmpose_cache_dir)
            #config_url='configs/body_2d_keypoint/topdown_heatmap/coco/td-hm_hrnet-w48_udp-8xb32-210e_coco-256x192.py'
        else:
            raise ValueError(f"Unknown model_type='{model_type}'. Currently supported: 'pose_hrnet_w48_udp'.")
      

    def _resolve_config_and_checkpoint(self,mmpose_config,pretrained_backbone,cache_dir):
        """
        Go through following steps for the config:
        1. mmpose_config is an existing local .py file => use it directly.
           (pretrained_backbone is also used as-is in this case)
        2. mmpose_config is None => resolve alias from _MMPOSE_CONFIG_ALIAS and
           download via mim into cache_dir.
        3. mmpose_config is a string but NOT a file => treat as a mim config
           alias and download.
        """

        #Local .py supplied directly
        if mmpose_config is not None and os.path.isfile(mmpose_config):
            return mmpose_config, pretrained_backbone
        
        #If the model is not stored localy, we need a mim download
        config_alias = (
            _MMPOSE_CONFIG_ALIAS[self.model_type]   # Case 2: use built-in alias
            if mmpose_config is None
            else mmpose_config                       # Case 3: user-supplied alias
        )

        os.makedirs(cache_dir, exist_ok=True)
        expected_config = os.path.join(cache_dir, config_alias + '.py')

        if not os.path.isfile(expected_config):
            mim_download('mmpose',configs=[config_alias],dest_root=cache_dir)
        
        #Locate the downloaded .py (filename may have extra metadata)
        if not os.path.isfile(expected_config):
            candidates = [
                f for f in os.listdir(cache_dir)
                if f.endswith('.py') and config_alias in f
            ]
            if not candidates:
                raise FileNotFoundError(
                    f"No .py config found for alias '{config_alias}' in "
                    f"'{cache_dir}'.\n"
                    f"Try:  mim download mmpose --config {config_alias} "
                    f"--dest {cache_dir}"
                )
            expected_config = os.path.join(cache_dir, sorted(candidates)[0])
        #Auto detect the checkpoint ig none is provided
        resolved_ckpt = pretrained_backbone
        if resolved_ckpt is None:
            pth_files = sorted(
                [f for f in os.listdir(cache_dir) if f.endswith('.pth')],
                key=lambda f: os.path.getmtime(os.path.join(cache_dir, f)),
            )
            if pth_files:
                resolved_ckpt = os.path.join(cache_dir, pth_files[-1])
                print(
                    f"[HeatmapKeypointModels] Auto-detected checkpoint: "
                    f"{resolved_ckpt}"
                )

        return expected_config, resolved_ckpt

    def _build_hrnet_w48_udp(self,mmpose_config,pretrained_backbone,cache_dir):
        """
        Assembles three sub-modules:
          1. input_adapter  — maps in_channels => 3 (no-op when in_channels==3)
          2. backbone       — HRNet-W48 from MMPose
          3. heatmap_head   — lightweight conv block => num_categories channels
        """

        ############Define the input channel adapter##############
        #HRNet's first conv layer expects 3 input channels
        #If images are grayscale, we must project to 3 channels
        if self.in_channels != 3:
            self.input_adapter = nn.Sequential(
                nn.Conv2d(self.in_channels, 3, kernel_size=1, bias=False),
                nn.BatchNorm2d(3),
                nn.ReLU(inplace=True),
            )
            # Initialise so that the adapter starts close to "repeat the channel"
            nn.init.constant_(self.input_adapter[0].weight,
                               1.0 / self.in_channels)
            nn.init.constant_(self.input_adapter[1].weight, 1.0)
            nn.init.constant_(self.input_adapter[1].bias,   0.0)
        else:
            self.input_adapter = nn.Identity()

        
        ############Setup the HRNet-W48 with UDP backbone#############
        config_path, ckpt_path = self._resolve_config_and_checkpoint(mmpose_config, pretrained_backbone, cache_dir)
        full_mmpose_model=init_model(config=config_path,checkpoint=ckpt_path,device=str(self.device))

        #Extract the backbone
        self.backbone=full_mmpose_model.backbone
        for param in self.backbone.parameters():
            param.requires_grad_(True)

        #Free the full model
        del full_mmpose_model


        ############Define the heatmap head###########
        self.heatmap_head = nn.Sequential(
            nn.Conv2d(_HRNET_W48_OUT_CH, _HRNET_W48_OUT_CH,
                      kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(_HRNET_W48_OUT_CH),
            nn.ReLU(inplace=True),
            nn.Conv2d(_HRNET_W48_OUT_CH, self.num_categories, kernel_size=1),
        )

        #Intialize the head:
        for m in self.heatmap_head.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias,   0.0)

    def forward(self,x):
        """
        Frame mode:
        - x: (B,C,H,W)
        - return: (B,num_categories,H/4,W/4)

        Clip modeL
        - x: (B,T,C,H,W)
        -Internally flattened to (B*T,C,H,W) then unflattened to (B,T,num_categories,H/4,W/4)
        """
        if self.return_mode=='clip':
            B, T, C, H, W=x.shape
            x_flat=x.view(B * T, C, H, W)
            hm_flat=self._forward_single(x_flat)  # (B*T, num_cat, H', W')
            _, C_out, Hp, Wp=hm_flat.shape
            return hm_flat.view(B, T, C_out, Hp, Wp)        # (B, T, num_cat, H', W')
        else:
            return self._forward_single(x)
        
    def _forward_single(self,x):
        x=self.input_adapter(x) #(B,3,H,W)
        feature_maps=self.backbone(x) #list/tuple of 4 tensors
        x=feature_maps[0] # (B, 48, H/4, W/4)
        x=self.heatmap_head(x) #converts to (B,num_categories,H/4,W/4)
        return torch.sigmoid(x) #Pass through a sigmoid layer

        
        



        

