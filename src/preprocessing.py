'''
Description: Sets up desired directory structure and file format (e.g. COCO)like) for model training.  Converts raw DCM and JSON files
(organized per annotater per site per scan) to desired repo structure and file format. 
We also save annotations and images in both cartesion and polar coordinates to allow for training models in both configurations.
'''

import argparse
import sys
import os
import glob
import utils

INPUT_RAW_DIR_PATH_DEFAULT='../../../Data/Group-001'
OUTPUT_DIR_PATH_DEFAULT='../../../Data/Keypoint_Detect_Data'
SITE_STR_DEFAULT=['CARVD','Lahey']
if __name__ == '__main__':
    #Sets up command line args
    p=argparse.ArgumentParser(description='Convert Raw DCM and JSON files (organized per annotator per site) to desired repo format')
    p.add_argument('--input_dir',type=str,default=INPUT_RAW_DIR_PATH_DEFAULT,help='Path to raw data base directory (directory containing separate annotators)')
    p.add_argument('--outdata_format',type=str,default='COCO_like',help='Format of output data for model training (e.g. "COCO_like" has an annotation and image subfolder in a COCO_Data root folder)') #Only supports COCO for now
    p.add_argument('--output_dir',type=str,default=OUTPUT_DIR_PATH_DEFAULT,help='Path to base output directory') 
    p.add_argument('--site_str',type=list,default=SITE_STR_DEFAULT,help='List of US hospital sites to process')
    p.add_argument('--coordinate_space',type=str,default='both',help='Coordinate space to save images and annotations in. Options: both, scanline (rectangular), sector (original fan). both saves doubles of images and annotations for each coordinate')
    #Parse command line args
    args=p.parse_args()
    input_dir=args.input_dir
    output_dir=args.output_dir
    outdata_format=args.outdata_format
    site_str=args.site_str
    coordinate_space=args.coordinate_space

    #Check that input directory exists
    if not os.path.exists(input_dir):
        print('Input directory: ' + input_dir + ' does not exist.')
        sys.exit(1)
    
    #Create the output directory if it does not exist
    utils.os_make_dir(output_dir)
    
    #Loops through the annotators in the input directory and converts their data to the desired output format
    for annotator in os.listdir(input_dir):
        annotator_path=os.path.join(input_dir,annotator)
        #Check that annotator path is a directory
        if not os.path.isdir(annotator_path):
            print('Skipping path: ' + annotator_path+' since it is not a directory.')
            continue

        #Loops through the sites for this annotator 
        for site in os.listdir(annotator_path):
            if any(s in site for s in site_str):
                site_path=os.path.join(annotator_path,site)
                #Check that the site path is a directory
                if not os.path.isdir(site_path):
                    print('Skipping path: ' + site_path+' since it is not a directory.')
                    continue

                #Finds the dcm files in this site/annotator directory
                dcm_files = sorted(glob.glob(os.path.join(site_path, '*.dcm'))) + \
                            sorted(glob.glob(os.path.join(site_path, '*.DCM')))
                if not dcm_files:
                    print(f"No DICOM files found in {site_path} skipping this site")
                    continue
               
                #Looping for all the dcm files (clips) in this site/annotator directory
                for dcm in dcm_files:
                    # Process each DICOM file
                    json_file=utils.find_json_for_dicom(dcm, site_path)

                    if json_file is None:
                        print(f"Skipping {os.path.basename(dcm)}: no matching JSON")
                        continue
                    print(f"Found matching JSON: {os.path.basename(json_file)} for DICOM: {os.path.basename(dcm)}")
                    
                    ######Converting JSON (with annotations) and DICOM to desired output format (e.g. COCO)######
                    #Select the output directory format
                    if outdata_format=='COCO_like':
                        output_dir_root=os.path.join(output_dir,'COCO_Data')
                        utils.os_make_dir(output_dir_root)

                        #Annotations are stored in "annotations" subfolder as .json's and images in "images" subfolder as .png's
                        #Each file has name based on: <annotator>_<site>_<patient_id>_<time>_<scan_id>_<frame_num>.<ext>
                        output_dir_annotations=os.path.join(output_dir_root,'annotations')
                        output_dir_images=os.path.join(output_dir_root,'images')
                        utils.os_make_dir(output_dir_annotations)
                        utils.os_make_dir(output_dir_images)

                        #Create separate folders for sector (fan) or scanline (rectangular) under both annotations and images subfolders
                        if coordinate_space=='both' or coordinate_space=='sector':
                            utils.os_make_dir(os.path.join(output_dir_annotations,'sector'))
                            utils.os_make_dir(os.path.join(output_dir_images,'sector'))
                        if coordinate_space=='both' or coordinate_space=='scanline':
                            utils.os_make_dir(os.path.join(output_dir_annotations,'scanline'))
                            utils.os_make_dir(os.path.join(output_dir_images,'scanline'))

                        #Call export_clip_to_png which saves png's in output_dir_images and json annotations in output_dir_images
                        #Do it for each the polar and cartesion space files
                         

                        


                        
                        

                        



                print(f"Finished processing site: {site} for annotator: {annotator}")
                

    