import os,shutil,argparse
import json

import numpy             as np
import matplotlib.pyplot as plt
import tensorflow        as tf

from math                   import ceil,floor
from tensorflow.keras.utils import load_img,save_img,img_to_array


os.environ['KMP_DUPLICATE_LIB_OK']='True'

def identify_source_type(label_data):
    # should identify source type based on one example and then
    # check output if all examples have the same structure
    if not 'data' in label_data.keys():
        return -1
    if not isinstance(label_data['data'], list):
        return -1
    
    first_label = label_data['data'][0]
    if 'folder' in first_label.keys():
        if not isinstance(first_label['folder'], str):
            return -1
        if not first_label['folder'].endswith('/'):
            return -1
        
        if 'filename' not in first_label.keys():
            return -1
        if first_label['filename'].endswith('.jpg'):
            return 1
        else:
            return 3
    elif 'path' in first_label.keys():
        if not isinstance(first_label['path'], str):
            return -1
        if not first_label['path'].endswith('.png'):
            return -1
        else:
            return 2
    else:
        return -1

def process_image_labels(config_data, label_data, print_debug_info):
    
    source_type = identify_source_type(label_data)
    if source_type==-1:
        return -1,'source type could not be identified',{}

    data_list     = []
    new_filename_flag = True
    for idx,label in enumerate(label_data['data']):
        if not isinstance(label, dict):
            # log sth here
            continue
        
        if new_filename_flag:
            if source_type==1:
                curr_filename = label['filename']
            elif source_type==2:
                slash_idx     = label['path'].rindex('/')
                curr_filename = label['path'][slash_idx+1:-3] + 'jpg'
            elif source_type==3:
                curr_filename = label['filename'] + '.jpg'
            else:
                raise NotImplementedError
            
            data_dict     = {}
            image_labels_only_list = []
            if source_type==1:
                filename_path = os.path.join(label['folder'], label['filename'])
                data_dict['path'    ] = label['folder']
            elif source_type==2:
                filename_path = label['path']
                data_dict['path'    ] = label['path'][:slash_idx+1]
            elif source_type==3:
                filename_path = os.path.join(label['folder'], label['filename'] + '.jpg')
                data_dict['path'    ] = label['folder']
            else:
                raise NotImplementedError
            data_dict['filename'] = curr_filename
            
            if not os.path.isfile(filename_path):
                return -1,'file specified in metadata does not exist',{}
            image = load_img(
                 path       = filename_path,
                 color_mode = 'rgb',
                )
            image = img_to_array(img=image, dtype=np.int32)
            
            if source_type==1:
                if image.shape[0] != label['image_height']:
                    # log sth here
                    return -1,'image height does not match height in metadata',{}
                if image.shape[1] != label['image_width']:
                    # log sth here
                    return -1,'image width does not match width in metadata',{}
            elif source_type==2:
                if image.shape[0] != int(label['image_height']):
                    # log sth here
                    return -1,'image height does not match height in metadata',{}
                if image.shape[1] != int(label['image_width']):
                    # log sth here
                    return -1,'image width does not match width in metadata',{}

            data_dict['image_height'] = image.shape[0]
            data_dict['image_width' ] = image.shape[1]
            data_dict_labels_list     = []
        else:
            # another label for the same image 
            if source_type==3:
                if image.shape[0] < label['label_height']:
                    # log sth here
                    #return -1,'image height is smaller than label height in metadata',{}
                    continue
                if image.shape[1] < label['label_width']:
                    # log sth here
                    #return -1,'image width is smaller than label width in metadata',{}
                    continue

        image_labels_only_list.append(np.ones(shape=image.shape[0:2], dtype=bool))
        if source_type==1:
            offset_height_ = round(label['ymin']*image.shape[0])-1
            offset_width_  = round(label['xmin']*image.shape[1])-1
            
            image_cropped = tf.image.crop_to_bounding_box(
                                      image         = image,
                                      offset_height = offset_height_,
                                      offset_width  = offset_width_,
                                      target_height = round(label['ymax']*image.shape[0])-offset_height_,
                                      target_width  = round(label['xmax']*image.shape[1])-offset_width_,
                                     )
            
            image_labels_only_list[-1][:,:offset_width_                      ] = 0
            image_labels_only_list[-1][:,round(label['xmax']*image.shape[1]):] = 0
            
            image_labels_only_list[-1][:offset_height_,offset_width_:round(label['xmax']*image.shape[1])                     ] = 0
            image_labels_only_list[-1][round(label['ymax']*image.shape[0]):,offset_width_:round(label['xmax']*image.shape[1])] = 0
            
            label_dict = {
                          'xmin': label['xmin'],
                          'ymin': label['ymin'],
                          'xmax': label['xmax'],
                          'ymax': label['ymax'],
                          'label': 'Raccoon',
                          'width_to_height': (round(label['xmax']*image.shape[1])-offset_width_)/(round(label['ymax']*image.shape[0])-offset_height_),
            }
        elif source_type==2:            
            offset_height_ = label['ymin']-1
            offset_width_  = label['xmin']-1
            image_cropped = tf.image.crop_to_bounding_box(
                                      image         = image,
                                      offset_height = offset_height_,
                                      offset_width  = offset_width_,
                                      target_height = label['ymax']-offset_height_,
                                      target_width  = label['xmax']-offset_width_,
                                     )
            
            image_labels_only_list[-1][:,:offset_width_] = 0
            image_labels_only_list[-1][:,label['xmax']:] = 0
            
            image_labels_only_list[-1][:offset_height_,offset_width_:label['xmax']] = 0
            image_labels_only_list[-1][label['ymax']:,offset_width_:label['xmax']] = 0
            
            label_dict = {
                          'xmin' : label['xmin']/data_dict['image_width' ],
                          'ymin' : label['ymin']/data_dict['image_height'],
                          'xmax' : label['xmax']/data_dict['image_width' ],
                          'ymax' : label['ymax']/data_dict['image_height'],
                          'label': 'Raccoon',
                          'width_to_height': (label['xmax']-offset_width_)/(label['ymax']-offset_height_),
            }
        elif source_type==3:
            offset_height_ = floor(label['label_vert_center']-label['label_height']/2)-1
            offset_width_  = floor(label['label_horz_center']-label['label_width']/2)-1
            image_cropped = tf.image.crop_to_bounding_box(
                                      image         = image,
                                      offset_height = offset_height_,
                                      offset_width  = offset_width_,
                                      target_height = label['label_height'],
                                      target_width  = label['label_width'],
                                     )
            
            image_labels_only_list[-1][:,:offset_width_] = 0
            image_labels_only_list[-1][:,(floor(label['label_horz_center']+label['label_width']/2)-1):] = 0

            image_labels_only_list[-1][:offset_height_,offset_width_:(floor(label['label_horz_center']+label['label_width']/2)-1)] = 0
            image_labels_only_list[-1][(floor(label['label_vert_center']+label['label_height']/2)-1):,offset_width_:(floor(label['label_horz_center']+label['label_width']/2)-1)] = 0
            
            label_dict = {
                          'xmin': (offset_width_  + 1)/data_dict['image_width' ],
                          'ymin': (offset_height_ + 1)/data_dict['image_height'],
                          'xmax': (offset_width_  + 1 + label['label_width' ])/data_dict['image_width' ],
                          'ymax': (offset_width_  + 1 + label['label_height'])/data_dict['image_height'],
                          'label': 'Raccoon',
                          'width_to_height': label['label_width']/label['label_height']
            }
        else:
            raise NotImplementedError
        
        image_cropped_padded = tf.image.resize_with_pad(
                                                        image=image_cropped,
                                                        target_height=config_data['max_height'],
                                                        target_width=config_data['max_width'],
                                                       )
        
        save_filename = data_dict['filename']
        while True:
            if os.path.isfile(os.path.join(config_data['output_folder'], save_filename)):
                extension_idx = save_filename.rfind('.')
                save_filename = save_filename[0:extension_idx] + '_' + save_filename[extension_idx:]
            else:
                break
        
        save_img(
                 path = os.path.join(config_data['output_folder'], save_filename),
                 x    = image_cropped_padded,
                )
        if print_debug_info:
            print('Saved \'' + os.path.join(config_data['output_folder'], save_filename) + '\'')
        
        data_dict_labels_list.append(label_dict)
        
        
        idx_aux = idx+1
        while idx_aux<len(label_data['data']):
            if isinstance(label_data['data'][idx_aux], dict):
                break
        
        new_filename_flag = True
        if idx_aux<len(label_data['data']):
            next_label = label_data['data'][idx_aux]
            if source_type==1:
                next_filename = next_label['filename']
            elif source_type==2:
                slash_idx     = next_label['path'].rindex('/')
                next_filename = next_label['path'][slash_idx+1:-3] + 'jpg'
            elif source_type==3:
                next_filename = next_label['filename'] + '.jpg'
            else:
                raise NotImplementedError
            if next_filename==curr_filename:
               new_filename_flag = False

        if new_filename_flag: # meaning: end of label_data['data'] or new filename
            if len(image_labels_only_list)>1: # image with more than one label
                for idx_ in range(len(image_labels_only_list)):
                    image_labels_aux = np.zeros(shape=image.shape, dtype=image.dtype)
                    image_labels_aux[image_labels_only_list[idx_]] = image[image_labels_only_list[idx_]]
                    save_img(
                     path = curr_filename.replace('.jpg','_label_' + str(idx_+1) + '.jpg'),
                     x    = image_labels_aux,
                    )
                   
                for idx_ in range(len(image_labels_only_list)-1):
                    image_labels_only_list[-1] = np.bitwise_or(image_labels_only_list[-1],image_labels_only_list[idx_])
            
            image_labels_aux = np.zeros(shape=image.shape, dtype=image.dtype)
            image_labels_aux[image_labels_only_list[-1]] = image[image_labels_only_list[-1]]
            save_img(
                     path = curr_filename.replace('.jpg','_all_labels.jpg'),
                     x    = image_labels_aux,
                    )
            data_dict['label_area_perc'] = image_labels_only_list[-1].sum()/np.prod(image.shape[0:2])
            if data_dict['label_area_perc']>1.0:
                # log error for calculated area value here
                pass
            data_dict['labels'] = data_dict_labels_list
            data_list.append(data_dict)
    
    return 0,'',data_list

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-print_debug_info", action="store_true", help="print some debugging info")
    opt = parser.parse_args()    
    print_debug_info = opt.print_debug_info
    
    with open(os.path.join(os.getcwd(), 'config.json'), 'r', encoding='utf-8') as f:
        config_data = json.load(f)
    
    # clear output file, output folder and log file
    if os.path.isfile(config_data['output_file']):
        os.remove(config_data['output_file'])
    if os.path.isdir(config_data['output_folder']):        
        for root, dirs, files in os.walk(config_data['output_folder']):
            for f in files:
                os.remove(os.path.join(root, f))
            for d in dirs:
                shutil.rmtree(os.path.join(root, d))
    else:
        os.mkdir(config_data['output_folder'])
    
    # find folders with files to process
    image_folders = [name for name in os.listdir(os.getcwd()) if name.startswith('input_images_source_') \
                     and os.path.isdir(name)]
    
    data_output_dict = {}
    data_output_dict['data'] = []
    for image_folder in image_folders:
        # load label metadata if file 'labels.json' exists in the folder 
        if not os.path.isfile(os.path.join(os.getcwd(), image_folder, 'labels.json')):
            # log sth here
            continue
            
        with open(os.path.join(os.getcwd(), image_folder, 'labels.json'), 'r', encoding='utf-8') as f:
            label_data = json.load(f)
        
        return_code, return_msg, return_list = process_image_labels(config_data, label_data, print_debug_info)
        if return_code==0:
            data_output_dict['data'].extend(return_list)
        #break
        
        with open(config_data['output_file'], 'w', encoding='utf-8') as f:
            json.dump(data_output_dict, f, indent=4)
        

if __name__ == '__main__':
    main()
    
    
    
    # no final: rever python meetup (PEP8)
    # remove comments
