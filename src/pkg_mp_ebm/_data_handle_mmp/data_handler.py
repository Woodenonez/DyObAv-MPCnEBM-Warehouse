from typing import TypedDict

import numpy as np

import torch
import torchvision # type: ignore
from torch.utils.data import Dataset
from torch.utils.data import DataLoader


class SampleInfo(TypedDict):
    input: np.ndarray
    target: np.ndarray

    index: int
    time: float
    traj: np.ndarray


class DataHandler():
    def __init__(self, dataset: Dataset, batch_size=64, shuffle=True, num_workers=0):
        self.dataset = dataset
        self.dataloader = DataLoader(self.dataset, batch_size, shuffle, num_workers=num_workers) # create the dataloader from the dataset
        self.__iter = iter(self.dataloader)

    def return_batch(self):
        try:
            sample_batch = next(self.__iter)
        except StopIteration:
            self.reset_iter()
            sample_batch = next(self.__iter)
        data:torch.Tensor = sample_batch['input'] # type: ignore
        labels:torch.Tensor = sample_batch['target'] # type: ignore
        return data, labels

    def reset_iter(self):
        self.__iter = iter(self.dataloader)

    def get_num_data(self):
        return len(self.dataset)

    def get_num_batch(self):
        return len(self.dataloader) # the number of batches, only for training dataset


class Rescale(object):
    def __init__(self, output_size: tuple, tolabel=False):
        '''
        Args:
            output_size: (height * width)
        '''
        super().__init__()
        self.output_size = output_size
        self.tolabel = tolabel

    def __call__(self, sample: SampleInfo):
        image, labels_ = sample['input'], sample['target']
        h, w = image.shape[:2]
        h_new, w_new = self.output_size
        if (h==h_new) & (w==w_new): # if no need to resize, skip
            return sample

        img = torchvision.transforms.Resize((h_new,w_new))(image)
        if self.tolabel:
            labels = [(x[0]*w_new/w, x[1]*h_new/h) for x in labels_]
        return {'input':img, 'target':labels}

class ToGray(object):
    # For RGB the weight could be (0.299R, 0.587G, 0.114B)
    def __init__(self, weight=None):
        super().__init__()
        self.weight = [1/3, 1/3, 1/3] # default weights
        if weight is not None:
            self.weight[0] = round(weight[0]/sum(weight),3)
            self.weight[1] = round(weight[1]/sum(weight),3)
            self.weight[2] = 1 - self.weight[0] - self.weight[1]

    def __call__(self, sample: SampleInfo):
        image, label = sample['input'], sample['target']
        if (len(image.shape)==2) or (image.shape[2] == 1):
            return sample
        else:
            image = image[:,:,:3] # ignore alpha
            img = self.weight[0]*image[:,:,0] + self.weight[1]*image[:,:,1] + self.weight[2]*image[:,:,2]
        return {'input':img[:,np.newaxis], 'target':label}

class DelAlpha(object):
    # From RGBA to RGB
    def __call__(self, sample: SampleInfo):
        image, label = sample['input'], sample['target']
        if (len(image.shape)==2) or (image.shape[2] == 1):
            return sample
        else:
            return {'input':image[:,:,:3], 'target':label}

class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""
    def __call__(self, sample: SampleInfo):
        image, label = sample['input'], sample['target']
        image = image.transpose((2, 0, 1)) # numpy: H x W x C -> torch: C X H X W
        return {'input':  torch.from_numpy(image),
                'target': torch.from_numpy(label)}

class MaxNormalize(object):
    def __init__(self, max_pixel=255, max_label=None):
        super().__init__()
        self.mp = max_pixel
        self.ml = max_label

    def __call__(self, sample: SampleInfo):
        image, labels_ = sample['input'], sample['target']
        if not isinstance(self.mp, (tuple,list)):
            self.mp = [self.mp]*image.shape[2]
        for i in range(image.shape[2]):
            image[:,:,:i] = image[:,:,:i]/self.mp[i]
        if self.ml is not None:
            if self.ml is tuple:
                labels = [(x[0]/self.ml[0], x[1]/self.ml[1]) for x in labels_]
            else:
                labels = [(x[0]/self.ml, x[1]/self.ml) for x in labels_]
        return {'input':image, 'target':labels}

