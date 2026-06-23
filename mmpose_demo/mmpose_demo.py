import torch
original_load = torch.load
torch.load = lambda *args, **kwargs: original_load(
    *args, **{**kwargs, 'weights_only': False}
)

from mmpose.apis import inference_topdown, init_model
from mmpose.utils import register_all_modules
from mmpose.visualization import PoseLocalVisualizer
import mmcv

register_all_modules()

config_file = 'td-hm_hrnet-w48_8xb32-210e_coco-256x192.py'
checkpoint_file = 'td-hm_hrnet-w48_8xb32-210e_coco-256x192-0e67c616_20220913.pth'
model = init_model(config_file, checkpoint_file, device='cuda:0')  # or device='cuda:0'

#This runs the inference
results = inference_topdown(model, 'demo.jpg')

#Visualize the results
img=mmcv.imread('demo.jpg',channel_order='rgb')

visualizer = PoseLocalVisualizer()
visualizer.set_dataset_meta(model.dataset_meta)
visualizer.add_datasample('result',img,data_sample=results[0],draw_gt=False,
                          draw_heatmap=True,draw_bbox=True,show=False,out_file='demo_result.jpg')
print('Done. Check demo_result.jpg for the result.')

keypoints = results[0].pred_instances.keypoints
print("Keypoints shape:", keypoints.shape)
print("Keypoints:", keypoints)