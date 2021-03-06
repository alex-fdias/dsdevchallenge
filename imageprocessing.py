import os,shutil,argparse,inspect
import json

import numpy             as np
import matplotlib.pyplot as plt
import tensorflow        as tf

from datetime               import datetime
from math                   import ceil,floor
from tensorflow.keras.utils import load_img,save_img,img_to_array


os.environ['KMP_DUPLICATE_LIB_OK']='True'

class ImageProcessing:
    def __init__(self, path=os.getcwd(), config_filename='config.json', log_filename='processing.log', save_debug_imgs=False):
        self.path              = path
        self.folder_debug_imgs = 'images_debug'
        self.config_filename   = config_filename
        self.log_filename      = log_filename
        self.save_debug_imgs   = save_debug_imgs
        
        self._run_time                  = None
        self._data_label_width_to_ratio = None
        

        self.print_init_msg()
        self.load_config()              
        self.clean_files()
        
    @property
    def run_time(self):
        return self._run_time

    @property
    def data_label_width_to_ratio(self):
        return self._data_label_width_to_ratio

    @staticmethod
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

    def print_save_log_file(self, output_str):
        print(output_str, end='')
        with open(os.path.join(self.path, self.log_filename), 'a', encoding='utf-8') as f:
            f.write(output_str)

    @staticmethod
    def image_valid_against_metadata(label, image, source_type):
        if image is None:
            return False
        else:
            if source_type==1:
                if label['image_height'] != image.shape[0]:
                    self.print_save_log_file(f"{datetime.now().strftime('[%Y-%m-%d %H:%M:%S]')} {self.__class__.__module__}.{self.__class__.__name__} {inspect.currentframe().f_code.co_name} _: 'Label metadata height does not match image height'\n")
                    return False
                if label['image_width'] != image.shape[1]:
                    self.print_save_log_file(f"{datetime.now().strftime('[%Y-%m-%d %H:%M:%S]')} {self.__class__.__module__}.{self.__class__.__name__} {inspect.currentframe().f_code.co_name} _: 'Label metadata width does not match image width'\n")
                    return False
            elif source_type==2:
                if int(label['image_height']) != image.shape[0]:
                    self.print_save_log_file(f"{datetime.now().strftime('[%Y-%m-%d %H:%M:%S]')} {self.__class__.__module__}.{self.__class__.__name__} {inspect.currentframe().f_code.co_name} _: 'Label metadata height does not match image height'\n")
                    return False
                if int(label['image_width']) != image.shape[1]:
                    self.print_save_log_file(f"{datetime.now().strftime('[%Y-%m-%d %H:%M:%S]')} {self.__class__.__module__}.{self.__class__.__name__} {inspect.currentframe().f_code.co_name} _: 'Label metadata width does not match image width'\n")
                    return False
            elif source_type==3:
                if floor(label['label_vert_center']-label['label_height']/2)<1:
                    self.print_save_log_file(f"{datetime.now().strftime('[%Y-%m-%d %H:%M:%S]')} {self.__class__.__module__}.{self.__class__.__name__} {inspect.currentframe().f_code.co_name} _: 'Label metadata for height is outside boundaries (less than 1)'\n")
                    return False
                if floor(label['label_vert_center']+label['label_height']/2)>image.shape[0]:
                    self.print_save_log_file(f"{datetime.now().strftime('[%Y-%m-%d %H:%M:%S]')} {self.__class__.__module__}.{self.__class__.__name__} {inspect.currentframe().f_code.co_name} _: 'Label metadata for height is outside boundaries (greater than image height)'\n")
                    return False
                if floor(label['label_horz_center']-label['label_width']/2)<1:
                    self.print_save_log_file(f"{datetime.now().strftime('[%Y-%m-%d %H:%M:%S]')} {self.__class__.__module__}.{self.__class__.__name__} {inspect.currentframe().f_code.co_name} _: 'Label metadata for width is outside boundaries (less than 1)'\n")
                    return False
                if floor(label['label_horz_center']+label['label_width']/2)>image.shape[1]:
                    self.print_save_log_file(f"{datetime.now().strftime('[%Y-%m-%d %H:%M:%S]')} {self.__class__.__module__}.{self.__class__.__name__} {inspect.currentframe().f_code.co_name} _: 'Label metadata for width is outside boundaries (greater than image width)'\n")
                    return False
            else:
                raise NotImplementedError
        
        return True

    def print_init_msg(self):
        self.print_save_log_file(f"\n{datetime.now().strftime('[%Y-%m-%d %H:%M:%S]')} {self.__class__.__module__}.{self.__class__.__name__} {inspect.currentframe().f_code.co_name} _: 'Image processing instance created'\n")
    
    def load_config(self):
        with open(os.path.join(self.path, self.config_filename), 'r', encoding='utf-8') as f:
            self.config_data = json.load(f)
        
        self.print_save_log_file(f"{datetime.now().strftime('[%Y-%m-%d %H:%M:%S]')} {self.__class__.__module__}.{self.__class__.__name__} {inspect.currentframe().f_code.co_name} _: 'Loaded run configuration data'\n")
    
    def clean_files(self):
        # clear output file, output folder
        if os.path.isfile(os.path.join(self.path, self.config_data['output_file'])):
            os.remove(os.path.join(self.path, self.config_data['output_file']))
        if os.path.isdir(os.path.join(self.path, self.config_data['output_folder'])):        
            for root, dirs, files in os.walk(os.path.join(self.path, self.config_data['output_folder'])):
                for f in files:
                    os.remove(os.path.join(root, f))
                for d in dirs:
                    shutil.rmtree(os.path.join(root, d))
        else:
            os.mkdir(os.path.join(self.path, self.config_data['output_folder']))

        # clear debug output folder
        if os.path.isdir(os.path.join(self.path, self.folder_debug_imgs)):
            if self.save_debug_imgs:
                for root, dirs, files in os.walk(os.path.join(self.path, self.folder_debug_imgs)):
                    for f in files:
                        os.remove(os.path.join(root, f))
                    for d in dirs:
                        shutil.rmtree(os.path.join(root, d))
            else:
                shutil.rmtree(os.path.join(self.path, self.folder_debug_imgs))
        elif self.save_debug_imgs:
            os.mkdir(os.path.join(self.path, self.folder_debug_imgs))        
        
        if os.path.isfile(os.path.join(self.path, 'plot_distribution.pdf')):
            os.remove(os.path.join(self.path, 'plot_distribution.pdf'))

        self.print_save_log_file(f"{datetime.now().strftime('[%Y-%m-%d %H:%M:%S]')} {self.__class__.__module__}.{self.__class__.__name__} {inspect.currentframe().f_code.co_name} _: 'Cleared files of previous run'\n")
    
    def run(self):
        self.print_save_log_file(f"{datetime.now().strftime('[%Y-%m-%d %H:%M:%S]')} {self.__class__.__module__}.{self.__class__.__name__} {inspect.currentframe().f_code.co_name} _: 'Started image processing run'\n")
    
        elapsed_time = datetime.now()
        
        # find folders with files to process
        image_folders = [name for name in os.listdir(self.path) if name.startswith('input_images_source_') and os.path.isdir(name)]
        
        data_output_dict          = {}
        data_output_dict['data']  = []
        self._data_label_width_to_ratio = []
        for image_folder in image_folders:
            # load label metadata if file 'labels.json' exists in the folder 
            if not os.path.isfile(os.path.join(self.path, image_folder, 'labels.json')):
                self.print_save_log_file(f"{datetime.now().strftime('[%Y-%m-%d %H:%M:%S]')} {self.__class__.__module__}.{self.__class__.__name__} {inspect.currentframe().f_code.co_name} _: 'Folder \"{os.path.join(self.path, image_folder)}\" has no \"labels.json\" file'\n")
                continue
            
            with open(os.path.join(self.path, image_folder, 'labels.json'), 'r', encoding='utf-8') as f:
                label_data = json.load(f)
            
            source_type = self.__class__.identify_source_type(label_data)
            if source_type==-1:
                self.print_save_log_file(f"{datetime.now().strftime('[%Y-%m-%d %H:%M:%S]')} {self.__class__.__module__}.{self.__class__.__name__} {inspect.currentframe().f_code.co_name} _: 'Source type could not be identified for folder \"{os.path.join(self.path, image_folder)}\"'\n")
                continue
            self.print_save_log_file(f"{datetime.now().strftime('[%Y-%m-%d %H:%M:%S]')} {self.__class__.__module__}.{self.__class__.__name__} {inspect.currentframe().f_code.co_name} _: 'Source type \"{source_type}\" identified for folder \"{os.path.join(self.path, image_folder)}\"'\n")               

            self.print_save_log_file(f"{datetime.now().strftime('[%Y-%m-%d %H:%M:%S]')} {self.__class__.__module__}.{self.__class__.__name__} {inspect.currentframe().f_code.co_name} _: 'Processing files in folder \"{os.path.join(self.path, image_folder)}\"'\n")            
            return_code, return_lists = self.process_image_labels(label_data, source_type)
            if return_code==0:
                data_output_dict['data'].extend(return_lists[0])
                self._data_label_width_to_ratio.extend(return_lists[1])
            
        with open(os.path.join(self.path, self.config_data['output_file']), 'w', encoding='utf-8') as f:
            json.dump(data_output_dict, f, indent=2)
            
        elapsed_time = datetime.now() - elapsed_time

        self.print_save_log_file(f"{datetime.now().strftime('[%Y-%m-%d %H:%M:%S]')} {self.__class__.__module__}.{self.__class__.__name__} {inspect.currentframe().f_code.co_name} _: 'Finished image processing, run time {elapsed_time}'\n")
        #print('Run time: ', elapsed_time)


    def validate_image_label_metadata(self, label, label_idx, source_type, last_filename_path):
        if not isinstance(label, dict):
            self.print_save_log_file(f"{datetime.now().strftime('[%Y-%m-%d %H:%M:%S]')} {self.__class__.__module__}.{self.__class__.__name__} {inspect.currentframe().f_code.co_name} _: 'Label metadata dictionary non-existent at position {label_idx+1}'\n")
            return {}, last_filename_path

        # validate existence of required fields, field types and field values
        if source_type==1:
            if 'folder' not in label.keys():
                return {}, last_filename_path
            elif not isinstance(label['folder'], str):
                return {}, last_filename_path
                
            if 'filename' not in label.keys():
                return {}, last_filename_path
            elif not isinstance(label['filename'], str):
                return {}, last_filename_path
                
            if 'xmin' not in label.keys():
                return {}, last_filename_path
            elif not isinstance(label['xmin'], float):
                return {}, last_filename_path
                
            if 'ymin' not in label.keys():
                return {}, last_filename_path
            elif not isinstance(label['ymin'], float):
                return {}, last_filename_path
                
            if 'ymax' not in label.keys():
                return {}, last_filename_path
            elif not isinstance(label['ymax'], float):
                return {}, last_filename_path

            if 'image_height' not in label.keys():
                return {}, last_filename_path
            elif not isinstance(label['image_height'], int):
                return {}, last_filename_path

            if 'image_width' not in label.keys():
                return {}, last_filename_path
            elif not isinstance(label['image_width'], int):
                return {}, last_filename_path
            
            if label['xmin'] < 1/label['image_width']:
                return {}, last_filename_path
            elif label['xmax'] > 1.0:
                return {}, last_filename_path
            elif label['xmax'] < label['xmin']:
                return {}, last_filename_path

            if label['ymin'] < 1/label['image_height']:
                return {}, last_filename_path
            elif label['ymax'] > 1.0:
                return {}, last_filename_path
            elif label['ymax'] < label['ymin']:
                return {}, last_filename_path
            
            if label['image_height'] < 1:
                return {}, last_filename_path
            if label['image_width']  < 1:
                return {}, last_filename_path
        elif source_type==2:
            pass
        elif source_type==3:
            pass
        else:
            raise NotImplementedError

        data_dict = {}
        if source_type==1:
            data_dict['path'    ] = label['folder']
            data_dict['filename'] = label['filename']
            curr_filename_path = os.path.join(label['folder'].replace('/','\\'), label['filename'])
        elif source_type==2:
            slash_idx     = label['path'].rindex('/')
            data_dict['path'    ] = label['path'][:slash_idx+1]
            data_dict['filename'] = label['path'][slash_idx+1:-3] + 'jpg'
            curr_filename_path = label['path'].replace('/','\\')
        elif source_type==3:
            data_dict['path'    ] = label['folder']
            data_dict['filename'] = label['filename'] + '.jpg'
            curr_filename_path = os.path.join(label['folder'].replace('/','\\'), data_dict['filename'])
        else:
            raise NotImplementedError

        if curr_filename_path != last_filename_path:
            last_filename_path = curr_filename_path

        return data_dict, last_filename_path

    def process_image_labels(self, label_data, source_type):
        data_list                 = []
        label_width_to_ratio_list = []
        last_filename_path = ''
        image = None
        for idx,label in enumerate(label_data['data']):
            data_dict, curr_filename_path = self.validate_image_label_metadata(label, idx, source_type, last_filename_path)

            image_label_masks_list = [] # should be loaded from data_output.json or kept (if same filename as before)

            if curr_filename_path != last_filename_path:
                last_filename_path = curr_filename_path
                image              = None
                
            if image is None and len(last_filename_path)>0: # if image hasn't been loaded for the current filename and path
                if not os.path.isfile(os.path.join(self.path, last_filename_path)):
                    self.print_save_log_file(f"{datetime.now().strftime('[%Y-%m-%d %H:%M:%S]')} {self.__class__.__module__}.{self.__class__.__name__} {inspect.currentframe().f_code.co_name} _: 'File \"{os.path.join(self.path, last_filename_path)}\" specified in label metadata not found'\n")
                else:
                    self.print_save_log_file(f"{datetime.now().strftime('[%Y-%m-%d %H:%M:%S]')} {self.__class__.__module__}.{self.__class__.__name__} {inspect.currentframe().f_code.co_name} _: 'Loading image \"{os.path.join(self.path, last_filename_path)}\"'\n")
                    
                    image = load_img(
                         path       = last_filename_path,
                         color_mode = 'rgb',
                        )
                    image = img_to_array(img=image, dtype=np.int32)
                    
            if self.__class__.image_valid_against_metadata(label, image, source_type):
                # load previous label data, if existent
                data_list_match=None
                for data_elem_idx, data_elem in enumerate(data_list):
                    if data_elem['path']==data_dict['path'] and data_elem['filename']==data_dict['filename']:
                        data_list_match=data_elem
                        break
                if data_list_match is None:
                    data_dict['image_height'] = image.shape[0]
                    data_dict['image_width' ] = image.shape[1]
                    data_dict_labels_list     = []
                else:
                    data_dict             = data_list_match
                    data_dict_labels_list = data_dict['labels']
                
                image_label_mask = np.ones(shape=image.shape[0:2], dtype=bool)
                
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
                    
                    image_label_mask[:,:offset_width_                      ] = 0
                    image_label_mask[:,round(label['xmax']*image.shape[1]):] = 0
                    
                    image_label_mask[:offset_height_,offset_width_:round(label['xmax']*image.shape[1])                     ] = 0
                    image_label_mask[round(label['ymax']*image.shape[0]):,offset_width_:round(label['xmax']*image.shape[1])] = 0
                    
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
                    
                    image_label_mask[:,:offset_width_] = 0
                    image_label_mask[:,label['xmax']:] = 0
                    
                    image_label_mask[:offset_height_,offset_width_:label['xmax']] = 0
                    image_label_mask[label['ymax']:,offset_width_:label['xmax']] = 0
                    
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
                    
                    image_label_mask[:,:offset_width_] = 0
                    image_label_mask[:,(floor(label['label_horz_center']+label['label_width']/2)-1):] = 0

                    image_label_mask[:offset_height_,offset_width_:(floor(label['label_horz_center']+label['label_width']/2)-1)] = 0
                    image_label_mask[(floor(label['label_vert_center']+label['label_height']/2)-1):,offset_width_:(floor(label['label_horz_center']+label['label_width']/2)-1)] = 0
                    
                    label_dict = {
                                  'xmin': (offset_width_  + 1)/data_dict['image_width' ],
                                  'ymin': (offset_height_ + 1)/data_dict['image_height'],
                                  'xmax': (offset_width_  + label['label_width' ])/data_dict['image_width' ],
                                  'ymax': (offset_height_ + label['label_height'])/data_dict['image_height'],
                                  'label': 'Raccoon',
                                  'width_to_height': label['label_width']/label['label_height']
                    }
                else:
                    raise NotImplementedError
                label_width_to_ratio_list.append(label_dict['width_to_height'])
                
                image_cropped_padded = tf.image.resize_with_pad(
                                                                image=image_cropped,
                                                                target_height=self.config_data['max_height'],
                                                                target_width=self.config_data['max_width'],
                                                               )
                
                save_filename = data_dict['filename']
                while True:
                    if os.path.isfile(os.path.join(self.path, self.config_data['output_folder'], save_filename)):
                        extension_idx = save_filename.rfind('.')
                        save_filename = save_filename[0:extension_idx] + '_' + save_filename[extension_idx:]
                    else:
                        break
                
                save_img(
                         path = os.path.join(self.path, self.config_data['output_folder'], save_filename),
                         x    = image_cropped_padded,
                        )
                self.print_save_log_file(f"{datetime.now().strftime('[%Y-%m-%d %H:%M:%S]')} {self.__class__.__module__}.{self.__class__.__name__} {inspect.currentframe().f_code.co_name} _: \'Saved output file \"{os.path.join(self.path, self.config_data['output_folder'], save_filename)}\"\'\n")
                
                data_dict_labels_list.append(label_dict)
                
                image_labels_aux = np.zeros(shape=image.shape, dtype=image.dtype)
                image_labels_aux[image_label_mask] = image[image_label_mask]
                if self.save_debug_imgs:
                    save_img(
                     path = os.path.join(self.path, self.folder_debug_imgs, data_dict['filename'].replace('.jpg','_label_' + str(len(data_dict_labels_list)) + '.jpg')),
                     x    = image_labels_aux,
                    )
                if len(data_dict_labels_list)>0:
                    # if image already had previous label(s), load previous label data in order to update label metadata and label area percentage in output file
                    for idx_,label_elem in enumerate(data_dict_labels_list):
                        image_label_mask_aux = np.zeros(shape=image.shape[0:2], dtype=bool)
                        
                        image_label_mask_aux[round(label_elem['ymin']*data_dict['image_height']-1):round(label_elem['ymax']*data_dict['image_height']), \
                                             round(label_elem['xmin']*data_dict['image_width' ]-1):round(label_elem['xmax']*data_dict['image_width' ])
                                             ] = True
                        
                        image_label_mask = np.bitwise_or(image_label_mask, image_label_mask_aux)
                        
                    image_labels_aux[image_label_mask] = image[image_label_mask]

                if self.save_debug_imgs:
                    save_img(
                             path = os.path.join(self.path, self.folder_debug_imgs, data_dict['filename'].replace('.jpg','_all_labels.jpg')),
                             x    = image_labels_aux,
                            )
                
                data_dict['label_area_perc'] = image_label_mask.sum()/np.prod(image.shape[0:2])
                data_dict['labels'] = data_dict_labels_list
                if data_list_match is None:
                    data_list.append(data_dict)
                else:
                    data_list[data_elem_idx] = data_dict
            
            #print(data_list)
        return 0, (data_list, label_width_to_ratio_list)
    
    def plot_distribution(self, show_window=False):
        if self._data_label_width_to_ratio is not None and len(self._data_label_width_to_ratio)>0:
            fig, axs = plt.subplots(1, 2, figsize=(9.0,4.8))
            axs[0].scatter(np.arange(1,len(self._data_label_width_to_ratio)+1, step=1), self._data_label_width_to_ratio)
            axs[0].set_xlabel('label no.')
            axs[0].set_ylabel('width-to-height ratio')
            axs[0].set_xlim(-.5+1, len(self._data_label_width_to_ratio)+.5)
            axs[0].set_xticks(np.arange(1,len(self._data_label_width_to_ratio)+1, step=1))
            n, _, _ = axs[1].hist(x=self._data_label_width_to_ratio)
            #axs[0].set_xlabel('label no.')
            axs[1].set_xlabel('width-to-height ratio')
            axs[1].set_ylabel('count')
            axs[1].set_yticks(np.arange(0,max(n)+1, step=1))
            
            plt.savefig(os.path.join(self.path,'plot_distribution.pdf'))
            
            self.print_save_log_file(f"{datetime.now().strftime('[%Y-%m-%d %H:%M:%S]')} {self.__class__.__module__}.{self.__class__.__name__} {inspect.currentframe().f_code.co_name} _: \'Saved output distribution plot file \"{os.path.join(self.path,'plot_distribution.pdf')}\"\'\n")
            if show_window:
                plt.show()
        else:
            self.print_save_log_file(f"{datetime.now().strftime('[%Y-%m-%d %H:%M:%S]')} {self.__class__.__module__}.{self.__class__.__name__} {inspect.currentframe().f_code.co_name} _: \'No output distribution data to plot'\n")