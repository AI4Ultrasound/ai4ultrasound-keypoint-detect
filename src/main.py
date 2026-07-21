########Imports########
#Python imports
import torch
from torch import optim
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset
import os
import json
import random
import numpy as np


#Custom classes and imports
import datasplitter
from AIUSDataset import AIUSDataset
from LossFunctions import KeypointLoss
from ModelTrainer import ModelTrainer
import ModelTester
import utils
from Models import HRNet_UDP

if __name__=='__main__':
    hyperparameters={
        #Name of saved/load model (change each of these on every re-run)
        'base_model_name_save': 'model_July20',
        'base_model_name_load': 'model_July20',
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
        'normalize_method': 'basic',
        'resampling_freq': None,
        'convert_to_gray': True,
        'img_augmentation': None,
        'aug_prob': 0.1,
        'return_mode': 'frame',

        #Batch size and num epochs
        'batch_size': 32,
        'num_epochs': 100,

        #Learning rate and scheduler
        'learning_rate': 0.01, #Used as maximum learning rate in OneCycleLR case
        'lr_scheduler_name': 'OneCycleLR',

        #Model Parameters
        'return_mode': 'frame',
        'heatmap_head_dropout': 0.2,
        'model_type': 'pose_hrnet_w48_udp',
        'mmpose_config': '../mmpose_model_cache/td-hm_hrnet-w48_udp-8xb32-210e_coco-256x192.py', #Set to None on first run: mim downloads the model config into the mmpose_cache_dir automatically
        'pretrained_backbone': '../mmpose_model_cache/td-hm_hrnet-w48_udp-8xb32-210e_coco-256x192-3feaef8f_20220913.pth', #Set to None on first run
        'mmpose_cache_dir': '../mmpose_model_cache',

        #Loss Function Parameters
        'loss_type': 'L1',
        'weights': [1.0],
        'sigmas': None,
        'heatmap_sigma': 2,
        'matching_strategy': 'heatmap',

        #Training Class Parameters
        'match_thresh_percentage': 0.15, #A predicted keypoint within this percentage of the screen diagonal of the GT keypoint is registered as a correct prediction
        'metric_every_n_batches': 5, 

        #Model Testing Params
        'batch_size_test': 1,
    }

    #Setup name of model to save
    model_name_save=f"{hyperparameters['base_model_name_save']}"
    model_name_load=f"{hyperparameters['base_model_name_save']}"

    ##############Random Seed##############
    torch.manual_seed(hyperparameters['random_state'])
    torch.cuda.manual_seed_all(hyperparameters['random_state'])
    random.seed(hyperparameters['random_state'])
    np.random.seed(hyperparameters['random_state'])


    ######################Setup Datasets#################
    ####Split dataset (get train,val,test strings)
    train_paths,val_paths,test_paths=datasplitter.datasplitter(dataset_root=hyperparameters['dataset_root'],
                          outdata_format=hyperparameters['outputdata_format'],
                          train_split=hyperparameters['train_split'],
                          val_split=hyperparameters['val_split'],
                          test_split=hyperparameters['test_split'],
                          k_folds=hyperparameters['k_folds'],
                          equal_prop_tags=hyperparameters['equal_prop_tags'],
                          metadata_tags=hyperparameters['metadata_tags'],
                          random_state=hyperparameters['random_state'],
                          verbose=hyperparameters['datasplitter_verbose'])


    #####Create dataset classes
    train_data=AIUSDataset(
                        json_paths=train_paths,
                        us_framesize=hyperparameters['us_framesize'],
                        line_type=hyperparameters['line_type'],
                        normalize_method=hyperparameters['normalize_method'],
                        resampling_freq=hyperparameters['resampling_freq'],
                        convert_to_gray=hyperparameters['convert_to_gray'],
                        img_augmentations=hyperparameters['img_augmentation'],
                        aug_prob=hyperparameters['aug_prob'],
                        return_mode=hyperparameters['return_mode'])
    
    valid_data=AIUSDataset(
                        json_paths=val_paths,
                        us_framesize=hyperparameters['us_framesize'],
                        line_type=hyperparameters['line_type'],
                        normalize_method=hyperparameters['normalize_method'],
                        resampling_freq=hyperparameters['resampling_freq'],
                        convert_to_gray=hyperparameters['convert_to_gray'],
                        img_augmentations=hyperparameters['img_augmentation'],
                        aug_prob=hyperparameters['aug_prob'],
                        return_mode=hyperparameters['return_mode'])
    
    test_data=AIUSDataset(
                        json_paths=test_paths,
                        us_framesize=hyperparameters['us_framesize'],
                        line_type=hyperparameters['line_type'],
                        normalize_method=hyperparameters['normalize_method'],
                        resampling_freq=hyperparameters['resampling_freq'],
                        convert_to_gray=hyperparameters['convert_to_gray'],
                        img_augmentations=hyperparameters['img_augmentation'],
                        aug_prob=hyperparameters['aug_prob'],
                        return_mode=hyperparameters['return_mode'])
    
    ########################Setup Device####################
    device = torch.device(0 if torch.cuda.is_available() else 'cpu')
    print('using device: '+str(device))

    # #Cast the stats of the dataset objects to tensors and GPU for unnormalization
    # train_data.cast_stat_dicts_to_tensor_and_device(device)
    # valid_data.cast_stat_dicts_to_tensor_and_device(device)

    # ######################Setup Dataloader##################
    # train_loader=DataLoader(dataset=train_data,batch_size=hyperparameters['batch_size'],shuffle=True,num_workers=0,pin_memory=True,collate_fn=utils.frame_collate_fn) #num_workers=0 works fastest
    # valid_loader=DataLoader(dataset=valid_data,batch_size=hyperparameters['batch_size'],shuffle=False,num_workers=0,pin_memory=True,collate_fn=utils.frame_collate_fn)

    ##################Initialize model, optimizer and lr scheduler########
    #####Create model
    if hyperparameters['line_type'] in ('pleuraline','bline'):
        num_categories=1
    else:
        num_categories=2

    in_channels=1 if hyperparameters['convert_to_gray'] else 3

    model=HRNet_UDP(
        model_type=hyperparameters['model_type'],
        device=device,
        return_mode=hyperparameters['return_mode'],
        num_categories=num_categories,
        in_channels=in_channels,
        mmpose_config=hyperparameters['mmpose_config'],
        pretrained_backbone=hyperparameters['pretrained_backbone'],
        mmpose_cache_dir=hyperparameters['mmpose_cache_dir'],
        heatmap_head_dropout=hyperparameters['heatmap_head_dropout']
    ).to(device=device)


    # #####Initializes the optimizer with the specified learning rate
    # optimizer = optim.Adam(model.parameters(), lr=hyperparameters['learning_rate'])

    # #####See how many parameters model has:
    # num_model_params = 0
    # for param in model.parameters():
    #     num_model_params += param.flatten().shape[0]

    # print("-This Model Has %d (Approximately %d Million) Parameters!" % (num_model_params, num_model_params//1e6))
    # print("-------------------")


    # ####Setup learning rate scheduler if we are using one
    # if hyperparameters['lr_scheduler_name']=='OneCycleLR':            
    #     LR_scheduler=torch.optim.lr_scheduler.OneCycleLR(
    #                 optimizer,
    #                 max_lr=hyperparameters['learning_rate'],
    #                 steps_per_epoch=len(train_loader),
    #                 epochs=hyperparameters['num_epochs'],
    #                 pct_start=0.3,
    #                 div_factor=25,
    #             )
    # elif hyperparameters['lr_scheduler_name']=='ReduceLROnPlateau':            
    #     LR_scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau(
    #                 optimizer, mode='min', factor=0.5, patience=10, min_lr=1e-6,
    #             )
    # elif hyperparameters['lr_scheduler_name']=='CosineAnnealingLR':
    #     LR_scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(
    #                 optimizer, T_max=hyperparameters['num_epochs'], eta_min=1e-6)
    # else:
    #     LR_scheduler=None

    
    ###################Loss Function##################
    loss_fun=KeypointLoss(
        loss_type=hyperparameters['loss_type'],
        weights=hyperparameters['weights'],
        device=device,
        sigmas=hyperparameters['sigmas'],
        return_mode=hyperparameters['return_mode'],
        num_categories=num_categories,
        heatmap_sigma=hyperparameters['heatmap_sigma'],
        matching_strategy=hyperparameters['matching_strategy'])
    

    ##################Model Training##################
    # train_obj=ModelTrainer(
    #     model=model,
    #     loss_fun=loss_fun,
    #     optimizer=optimizer,
    #     train_loader=train_loader,
    #     valid_loader=valid_loader,
    #     num_epochs=hyperparameters['num_epochs'],
    #     device=device,
    #     loss_type=hyperparameters['loss_type'],
    #     us_framesize=hyperparameters['us_framesize'],
    #     LR_scheduler=LR_scheduler,
    #     return_mode=hyperparameters['return_mode'],
    #     checkpoint_savedir='../models',
    #     model_name_save=model_name_save,
    #     model_name_load=model_name_load,
    #     start_from_checkpoint=False,
    #     matching_strategy=hyperparameters['matching_strategy'],
    #     match_thresh_percentage=hyperparameters['match_thresh_percentage'],
    #     verbose=False,
    #     metric_every_n_batches=hyperparameters['metric_every_n_batches'])
    
    # #Run training
    # train_obj.train()

    ##################Model Testing###################
    #Load best model and run tester
    checkpoint_path=os.path.join('../models',model_name_load+'.pt')
    if os.path.exists(checkpoint_path):
        checkpoint_model=torch.load(checkpoint_path,map_location=device, weights_only=False) #Use best model on validation set for testing
        model.load_state_dict(checkpoint_model['model_state_dict'])
        model=model.to(device)
        test_loader=DataLoader(dataset=test_data,batch_size=hyperparameters['batch_size_test'],shuffle=False,num_workers=0,collate_fn=utils.frame_collate_fn,pin_memory=True)
        test_results=ModelTester.modelTester(
            model=model,
            test_loader=test_loader,
            return_mode=hyperparameters['return_mode'],
            matching_strategy=hyperparameters['matching_strategy'],
            loss_fun=loss_fun,
            device=device,
            us_framesize=hyperparameters['us_framesize'],
            model_name_save=model_name_save,
            match_thresh_percentage=hyperparameters['match_thresh_percentage'])

    else:
        print('No checkpoint found at: '+checkpoint_path+ ' skipping testing...')

    
    ###############Plotting Results############
    with open(os.path.join('../logs', str(model_name_save), 'train_logger.json')) as f:
        train_logger=json.load(f)

    with open(os.path.join('../logs', str(model_name_save), 'valid_logger.json')) as f:
        valid_logger=json.load(f)

    #Compute stats
    stats_dir_name='../stats'
    os.makedirs(stats_dir_name, exist_ok=True)
    stats_save_file=os.path.join(stats_dir_name, model_name_save + '_stats.json')
    utils.computeStats(test_logger=test_results,train_logger=train_logger,valid_logger=valid_logger,save_file=stats_save_file)

    #Plotting results
    plot_dir_name=os.path.join('../figs', model_name_save)
    os.makedirs(plot_dir_name, exist_ok=True)

    utils.plotTrainingLoss(train_logger=train_logger,valid_logger=valid_logger,save_file=os.path.join(plot_dir_name,'loss_train.svg'))
    utils.plotLocalizationMetrics(train_logger=train_logger,valid_logger=valid_logger,save_file=os.path.join(plot_dir_name,'localization_trainvalid.svg'))
    utils.plotDetectionMetrics(train_logger=train_logger,valid_logger=valid_logger,save_file=os.path.join(plot_dir_name,'detection_trainvalid.svg'))
    utils.plotTPFPFN_TrainValid(train_logger=train_logger,valid_logger=valid_logger,save_file=os.path.join(plot_dir_name,'detection_barchart_trainvalid.svg'))
    
    
    
    if test_results is not None:
        utils.plotTestBoxplots(test_logger=test_results,save_file=os.path.join(plot_dir_name,'boxplot_test.svg'))
        utils.plotPerCategoryMetrics_Test(test_logger=test_results,save_file=os.path.join(plot_dir_name,'percategory_test.svg'))
        utils.plotTestErrorHistogram(test_logger=test_results,save_file=os.path.join(plot_dir_name,'localizationhist_test.svg'))
    else:
        print('test_results unavailable — skipping test plots.')
    
    #Cleanup memory
    del train_data,valid_data,test_data
    # del train_loader,valid_loader
    # del optimizer,LR_scheduler,loss_fun,train_obj
    del model
    torch.cuda.empty_cache()