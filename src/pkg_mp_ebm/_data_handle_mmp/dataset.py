import os
import sys
import glob
from typing import Optional, Union

import numpy as np
import pandas as pd # type: ignore
# from skimage import io
from PIL import Image # type: ignore

import torch
import torchvision # type: ignore
from torch.utils.data import Dataset

from . import utils_np


class io(): 
    '''
    This class is used in case skimage not feasible. It does the same as "skimage.io".
    '''
    @staticmethod
    def imread(path) -> torch.Tensor:
        tr = torchvision.transforms.PILToTensor()
        return tr(Image.open(path)).permute(1,2,0)

class ImageStackDataset(Dataset):
    """This is a Pytorch-style Dataset generating a stack of images as the input.

    Notes:
        The method "__getitem__" returns a dictionary with ['input', 'target', 'index', 'traj', 'time'] where 
        'input'-np.ndarray/tensor, 'target'-np.ndarray/tensor, 'traj'-np.ndarray.
    """
    def __init__(self, csv_path: str, root_dir: str, transform:torchvision.transforms=None, 
                 pred_offset_range:Optional[tuple]=None, ref_image_name:Optional[str]=None, image_ext='png'):
        """
        Args:
            csv_path: Path to the CSV file with the info of the whole dataset.
            root_dir: Directory with all image folders (root_dir - video_folder - imgs/csvs).
            transform: Transform (such as rotation and flipping) the images.
            pred_offset_range: The prediction offset range, should be (offset_min, offset_max).
            ref_image_name: If the reference image has a name, specify here; otherwise regard the folder name as its name.
            image_ext: The format of the raw images, such as "png" and "jpg".
        """
        super().__init__()
        self.info_frame = pd.read_csv(csv_path)
        self.root_dir = root_dir
        self.tr = transform

        if image_ext not in ['png', 'jpg', 'jpeg']:
            raise ValueError(f'Unrecognized image type {image_ext}.')
        self.T_range = pred_offset_range
        self.ext = image_ext  # should not have '.'
        self.background = ref_image_name # if not None, use this as the background image

        csv_str = 'p' # in csv files, 'p' means position
        self.input_len = len([x for x in list(self.info_frame) if csv_str in x]) # length of input time step
        self.img_shape = self.__check_img_shape(self.background)

    def __check_img_shape(self, img_name=None):
        info = self.info_frame.iloc[0]
        video_folder = str(info['index'])
        if img_name is None:
            img_name = video_folder + '.' + self.ext
        img_path = os.path.join(self.root_dir, video_folder, img_name)
        image = self.togray(io.imread(img_path))
        return image.shape

    def __len__(self):
        return len(self.info_frame)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        input_img = np.empty(shape=[self.img_shape[0],self.img_shape[1],0])
        info = self.info_frame.iloc[idx]
        traj = []
        
        if self.background is not None:
            img_name = self.background
        else:
            img_name = f'{info["index"]}.{self.ext}'
            
        img_path = os.path.join(self.root_dir, str(info['index']), img_name)
        image = self.togray(io.imread(img_path))

        for i in range(self.input_len):
            position = info['p{}'.format(i)]
            this_x = float(position.split('_')[0])
            this_y = float(position.split('_')[1])
            traj.append([this_x,this_y])

            # obj_map = utils_np.np_gaudist_map((this_x, this_y), np.zeros_like(image), sigmas=[20,20]) # TODO: change the input map here
            obj_map = utils_np.np_dist_map((this_x, this_y), np.zeros_like(image))
            input_img = np.concatenate((input_img, obj_map[:,:,np.newaxis]), axis=2)
        input_img = np.concatenate((input_img, image[:,:,np.newaxis]), axis=2)
        
        if self.T_range is None:
            label_name_list = [x for x in list(self.info_frame) if 'T' in x]
        else:
            label_name_list = [f'T{x}' for x in range(self.T_range[0], self.T_range[1]+1)]
        label_list = list(info[label_name_list].values)
        label = [(float(x.split('_')[0]), float(x.split('_')[1])) for x in label_list]

        sample = {'input':input_img, 'target':np.array(label)} # label is TxDo
        if self.tr:
            sample = self.tr(sample)
        sample['index'] = info['index']
        sample['traj'] = np.array(traj)
        sample['time'] = info['t']

        return sample

    @DeprecationWarning # TODO: Check this function
    def rescale_label(self, label, original_scale): # x,y & HxW
        current_scale = self.__check_img_shape()
        rescale = (current_scale[0]/original_scale[0] , current_scale[1]/original_scale[1])
        return (label[0]*rescale[1], label[1]*rescale[0])

    @staticmethod
    def togray(image: Union[np.ndarray, torch.Tensor], normalize=True):
        if (len(image.shape)==2):
            return image
        elif (len(image.shape)==3) and (image.shape[2]==1):
            return image[:,:,0]
        else:
            image = image[:,:,:3] # ignore alpha
            img = image[:,:,0]/3 + image[:,:,1]/3 + image[:,:,2]/3
            if normalize:
                img /= 255
            return img


