#This script is just to debug and test the performance of the datasplitter

import datasplitter
import time #Used to calculate run-time


###############Config 1 datasplitter
# dataset_root='../../../Data/Keypoint_Detect_Data'
# outputdata_format='COCO_like'
# train_split=0.6
# val_split=0.2
# test_split=0.2
# k_folds=None
# equal_prop_tags=None
# metadata_tags={
#     'site': ['All'],
#     'annotator': ['All'],
#     'zone_label': ['All'],
#     'patient_id': ['All'],
#     'time': ['All'],
#     'transducer_type': ['All'],
#     'manufacturer_name': ['All'],
#     'coordinate_space': 'scanline',
# }
# random_state=42
# verbose=True

# start_time=time.perf_counter() #Start timer

# train_paths,val_paths,test_paths=datasplitter.datasplitter(dataset_root=dataset_root,
#                           outdata_format=outputdata_format,
#                           train_split=train_split,
#                           val_split=val_split,
#                           test_split=test_split,
#                           k_folds=k_folds,
#                           equal_prop_tags=equal_prop_tags,
#                           metadata_tags=metadata_tags,
#                           random_state=random_state,
#                           verbose=verbose)

# print('===================')
# end_time=time.perf_counter() #End timer
# execution_time=end_time-start_time
# print(f"datasplitter.py runtime: {execution_time:.6f} seconds")
# print('===================')
# print('Train Paths: ')
# for a in train_paths:
#     print(a)
# print('Val Paths: ')
# for a in val_paths:
#     print(a)
# print('Test Paths: ')
# for a in test_paths:
#     print(a)


###############Config 2 (with k-folds) datasplitter
# dataset_root='../../../Data/Keypoint_Detect_Data'
# outputdata_format='COCO_like'
# train_split=0.6
# val_split=0.2
# test_split=0.2
# k_folds=5
# equal_prop_tags=None
# metadata_tags={
#     'site': ['All'],
#     'annotator': ['All'],
#     'zone_label': ['All'],
#     'patient_id': ['All'],
#     'time': ['All'],
#     'transducer_type': ['All'],
#     'manufacturer_name': ['All'],
#     'coordinate_space': 'scanline',
# }
# random_state=42
# verbose=True

# start_time=time.perf_counter() #Start timer

# folds=datasplitter.datasplitter(dataset_root=dataset_root,
#                           outdata_format=outputdata_format,
#                           train_split=train_split,
#                           val_split=val_split,
#                           test_split=test_split,
#                           k_folds=k_folds,
#                           equal_prop_tags=equal_prop_tags,
#                           metadata_tags=metadata_tags,
#                           random_state=random_state,
#                           verbose=verbose)

# print('===================')
# end_time=time.perf_counter() #End timer
# execution_time=end_time-start_time
# print(f"datasplitter.py runtime: {execution_time:.6f} seconds")
# print('===================')

###############Config 3 (with k-folds and 1 equal props) datasplitter
# dataset_root='../../../Data/Keypoint_Detect_Data'
# outputdata_format='COCO_like'
# train_split=0.6
# val_split=0.2
# test_split=0.2
# k_folds=5
# equal_prop_tags=['site']
# metadata_tags={
#     'site': ['All'],
#     'annotator': ['All'],
#     'zone_label': ['All'],
#     'patient_id': ['All'],
#     'time': ['All'],
#     'transducer_type': ['All'],
#     'manufacturer_name': ['All'],
#     'coordinate_space': 'scanline',
# }
# random_state=42
# verbose=True

# start_time=time.perf_counter() #Start timer

# folds=datasplitter.datasplitter(dataset_root=dataset_root,
#                           outdata_format=outputdata_format,
#                           train_split=train_split,
#                           val_split=val_split,
#                           test_split=test_split,
#                           k_folds=k_folds,
#                           equal_prop_tags=equal_prop_tags,
#                           metadata_tags=metadata_tags,
#                           random_state=random_state,
#                           verbose=verbose)

# print('===================')
# end_time=time.perf_counter() #End timer
# execution_time=end_time-start_time
# print(f"datasplitter.py runtime: {execution_time:.6f} seconds")
# print('===================')

###############Config 4 (with k-folds and 1 equal props) datasplitter
dataset_root='../../../Data/Keypoint_Detect_Data'
outputdata_format='COCO_like'
train_split=0.6
val_split=0.2
test_split=0.2
k_folds=5
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

folds=datasplitter.datasplitter(dataset_root=dataset_root,
                          outdata_format=outputdata_format,
                          train_split=train_split,
                          val_split=val_split,
                          test_split=test_split,
                          k_folds=k_folds,
                          equal_prop_tags=equal_prop_tags,
                          metadata_tags=metadata_tags,
                          random_state=random_state,
                          verbose=verbose)

print('===================')
end_time=time.perf_counter() #End timer
execution_time=end_time-start_time
print(f"datasplitter.py runtime: {execution_time:.6f} seconds")
print('===================')