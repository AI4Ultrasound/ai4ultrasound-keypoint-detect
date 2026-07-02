#This script is just to debug and test the performance of the datasplitter

import datasplitter
import time #Used to calculate run-time
from AIUSDataset import AIUSDataset

#Setup the datasplitter:

##############Config datasplitter
dataset_root='../../../Data/Keypoint_Detect_Data'
outputdata_format='COCO_like'
train_split=0.7
val_split=0.15
test_split=0.15
k_folds=None
equal_prop_tags=['site','annotator']
metadata_tags={
    'site': ['All'],
    'annotator': ['All'],
    'zone_label': ['All'],
    'patient_id': ['All'],
    'time': ['All'],
    'transducer_type': ['All'],
    'manufacturer_name': ['All'],
    'coordinate_space': 'scanline',
}
random_state=42
verbose=True

start_time=time.perf_counter() #Start timer

train_paths,val_paths,test_paths=datasplitter.datasplitter(dataset_root=dataset_root,
                          outdata_format=outputdata_format,
                          train_split=train_split,
                          val_split=val_split,
                          test_split=test_split,
                          k_folds=k_folds,
                          equal_prop_tags=equal_prop_tags,
                          metadata_tags=metadata_tags,
                          random_state=random_state,
                          verbose=verbose)



###############Config 1 dataloader
# us_framesize=(128,128)
# line_type='pleuraline'
# normalize_method='basic'
# resampling_freq=None
# convert_to_gray=True
# img_augmentation=None
# aug_prob=0.1
# return_mode='frame'


# start_time=time.perf_counter() #Start timer

# train_data=AIUSDataset(
#     json_paths=train_paths,
#     us_framesize=us_framesize,
#     line_type=line_type,
#     normalize_method=normalize_method,
#     resampling_freq=resampling_freq,
#     convert_to_gray=convert_to_gray,
#     img_augmentations=img_augmentation,
#     aug_prob=aug_prob,
#     return_mode=return_mode
# )

# print('===================')
# end_time=time.perf_counter() #End timer
# execution_time=end_time-start_time
# print(f"Dataloader _init_ runtime: {execution_time:.6f} seconds")
# print('===================')

# start_time=time.perf_counter() #Start timer

# train_data.__getitem__(1) #Tests the getitem

# print('===================')
# end_time=time.perf_counter() #End timer
# execution_time=end_time-start_time
# print(f"Dataloader __getitem__ with return_mode: {return_mode} runtime: {execution_time:.6f} seconds")
# print('===================')


# ###############Config 2 dataloader
# us_framesize=(128,128)
# line_type='pleuraline'
# normalize_method='basic'
# resampling_freq=None
# convert_to_gray=True
# img_augmentation=None
# aug_prob=0.1
# return_mode='clip'


# start_time=time.perf_counter() #Start timer

# train_data=AIUSDataset(
#     json_paths=train_paths,
#     us_framesize=us_framesize,
#     line_type=line_type,
#     normalize_method=normalize_method,
#     resampling_freq=resampling_freq,
#     convert_to_gray=convert_to_gray,
#     img_augmentations=img_augmentation,
#     aug_prob=aug_prob,
#     return_mode=return_mode
# )

# print('===================')
# end_time=time.perf_counter() #End timer
# execution_time=end_time-start_time
# print(f"Dataloader _init_ runtime: {execution_time:.6f} seconds")
# print('===================')

# start_time=time.perf_counter() #Start timer

# train_data.__getitem__(1) #Tests the getitem

# print('===================')
# end_time=time.perf_counter() #End timer
# execution_time=end_time-start_time
# print(f"Dataloader __getitem__ with return_mode: {return_mode} runtime: {execution_time:.6f} seconds")
# print('===================')

# ###############Config 3 dataloader
# us_framesize=(128,128)
# line_type='pleuraline'
# normalize_method='basic'
# resampling_freq=None
# convert_to_gray=True
# img_augmentation=None
# aug_prob=0.1
# return_mode='frame'


# start_time=time.perf_counter() #Start timer

# train_data=AIUSDataset(
#     json_paths=train_paths,
#     us_framesize=us_framesize,
#     line_type=line_type,
#     normalize_method=normalize_method,
#     resampling_freq=resampling_freq,
#     convert_to_gray=convert_to_gray,
#     img_augmentations=img_augmentation,
#     aug_prob=aug_prob,
#     return_mode=return_mode
# )

# print('===================')
# end_time=time.perf_counter() #End timer
# execution_time=end_time-start_time
# print(f"Dataloader _init_ runtime: {execution_time:.6f} seconds")
# print('===================')

# start_time=time.perf_counter() #Start timer

# train_data.__getitem__(1) #Tests the getitem

# print('===================')
# end_time=time.perf_counter() #End timer
# execution_time=end_time-start_time
# print(f"Dataloader __getitem__ with return_mode: {return_mode} runtime: {execution_time:.6f} seconds")
# print('===================')

# ###############Config 4 dataloader
# us_framesize=(128,128)
# line_type='pleuraline'
# normalize_method='stand'
# resampling_freq=None
# convert_to_gray=True
# img_augmentation=None
# aug_prob=0.1
# return_mode='frame'


# start_time=time.perf_counter() #Start timer

# train_data=AIUSDataset(
#     json_paths=train_paths,
#     us_framesize=us_framesize,
#     line_type=line_type,
#     normalize_method=normalize_method,
#     resampling_freq=resampling_freq,
#     convert_to_gray=convert_to_gray,
#     img_augmentations=img_augmentation,
#     aug_prob=aug_prob,
#     return_mode=return_mode
# )

# print('===================')
# end_time=time.perf_counter() #End timer
# execution_time=end_time-start_time
# print(f"Dataloader _init_ runtime: {execution_time:.6f} seconds")
# print('===================')

# start_time=time.perf_counter() #Start timer

# train_data.__getitem__(1) #Tests the getitem

# print('===================')
# end_time=time.perf_counter() #End timer
# execution_time=end_time-start_time
# print(f"Dataloader __getitem__ with return_mode: {return_mode} runtime: {execution_time:.6f} seconds")
# print('===================')

###############Config 5 dataloader
us_framesize=(128,128)
line_type='pleuraline'
normalize_method='stand'
resampling_freq=None
convert_to_gray=True
img_augmentation=['gaussnoise']
aug_prob=0.1
return_mode='frame'


start_time=time.perf_counter() #Start timer

train_data=AIUSDataset(
    json_paths=train_paths,
    us_framesize=us_framesize,
    line_type=line_type,
    normalize_method=normalize_method,
    resampling_freq=resampling_freq,
    convert_to_gray=convert_to_gray,
    img_augmentations=img_augmentation,
    aug_prob=aug_prob,
    return_mode=return_mode
)

print('===================')
end_time=time.perf_counter() #End timer
execution_time=end_time-start_time
print(f"Dataloader _init_ runtime: {execution_time:.6f} seconds")
print('===================')

start_time=time.perf_counter() #Start timer

train_data.__getitem__(1) #Tests the getitem

print('===================')
end_time=time.perf_counter() #End timer
execution_time=end_time-start_time
print(f"Dataloader __getitem__ with return_mode: {return_mode} runtime: {execution_time:.6f} seconds")
print('===================')