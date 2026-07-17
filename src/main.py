########Imports########
#Custom classes and imports
import datasplitter
from AIUSDataset import AIUSDataset
from ModelTrainer import ModelTrainer
import ModelTester
import utils
import LossFunctions

if __name__=='__main__':
    hyperparameters={
        #Datasplitter params
        'dataset_root': '../../../Data/Keypoint_Detect_Data',
        'train_split': 0.7,
        'val_split': 0.15,
        'test_split': 0.15,
        'k_folds': None,
        'equal_prop_tags': ['site','annotator'],
        'metadata_tags': 
        {
            'site': ['All'],
            'annotator': ['All'],
            'zone_label': ['All'],
            'patient_id': ['All'],
            'time': ['All'],
            'transducer_type': ['All'],
            'manufacturer_name': ['All'],
            'coordinate_space': 'scanline',
        },
        'random_state': 42,
        'datasplitter_verbose': True,
        'outputdata_format': 'COCO_like',

        #Dataset params
        'us_framesize': (128,128),
        'line_type': 'pleuraline',
        'normalize_method': 'stand',
        'resampling_freq': None,
        'convert_to_gray': True,
        'img_augmentation': ['gaussnoise'],
        'aug_prob': 0.1,
        'return_mode': 'frame',
    }


    ######################Setup Datasets#################
    #Split dataset (get train,val,test strings)
    train_paths,val_paths,test_paths=datasplitter.datasplitter(dataset_root=dataset_root,
                          outdata_format=hyperparameters['outputdata_format'],
                          train_split=hyperparameters['train_split'],
                          val_split=hyperparameters['val_split'],
                          test_split=hyperparameters['test_split'],
                          k_folds=hyperparameters['k_folds'],
                          equal_prop_tags=hyperparameters['equal_prop_tags'],
                          metadata_tags=hyperparameters['metadata_tags'],
                          random_state=hyperparameters['random_state'],
                          verbose=hyperparameters['datasplitter_verbose'])

