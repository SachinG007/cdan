import matplotlib
matplotlib.use('Agg')
import io
import json
import os
import pickle

import numpy as np
import pandas as pd
import scipy.stats
import pathlib
import PIL.Image
import torch
import torchvision
from torch.utils.data import TensorDataset, DataLoader, Dataset
import torchvision.transforms as transforms
import torch.utils.data as torchdata
import matplotlib.pyplot as plt

np.random.seed(1)

class CustomTensorDataset(Dataset):
    """TensorDataset with support of transforms.
    """
    def __init__(self, tensors, transform=None):
        assert all(tensors[0].size(0) == tensor.size(0) for tensor in tensors)
        self.tensors = tensors
        self.transform = transform

    def __getitem__(self, index):
        x = self.tensors[0][index]

        if self.transform:
            # import pdb;pdb.set_trace()
            x = x.permute(2,0,1)
            # np.transpose( x.numpy(), (2, 0, 1))
            # if self.transform!=None:
            x = self.transform(x)

        y = self.tensors[1][index]

        return x, y

    def __len__(self):
        return self.tensors[0].size(0)
       
def load_new_test_data(version_string='', load_tinyimage_indices=False):
    data_path = os.path.join(os.path.dirname(__file__), '../datasets/')
    filename = 'cifar10.1'
    if version_string == '':
        version_string = 'v7'
    if version_string in ['v4', 'v6', 'v7']:
        filename += '_' + version_string
    else:
        raise ValueError('Unknown dataset version "{}".'.format(version_string))
    label_filename = filename + '_labels.npy'
    imagedata_filename = filename + '_data.npy'
    label_filepath = os.path.abspath(os.path.join(data_path, label_filename))
    imagedata_filepath = os.path.abspath(os.path.join(data_path, imagedata_filename))
    print('Loading labels from file {}'.format(label_filepath))
    assert pathlib.Path(label_filepath).is_file()
    labels = np.load(label_filepath)
    print('Loading image data from file {}'.format(imagedata_filepath))
    assert pathlib.Path(imagedata_filepath).is_file()
    imagedata = np.load(imagedata_filepath)
    assert len(labels.shape) == 1
    assert len(imagedata.shape) == 4
    assert labels.shape[0] == imagedata.shape[0]
    assert imagedata.shape[1] == 32
    assert imagedata.shape[2] == 32
    assert imagedata.shape[3] == 3
    if version_string == 'v6' or version_string == 'v7':
        assert labels.shape[0] == 2000
    elif version_string == 'v4':
        assert labels.shape[0] == 2021

    if not load_tinyimage_indices:
        return imagedata, labels
    else:
        ti_indices_data_path = os.path.join(os.path.dirname(__file__), '../other_data/')
        ti_indices_filename = 'cifar10.1_' + version_string + '_ti_indices.json'
        ti_indices_filepath = os.path.abspath(os.path.join(ti_indices_data_path, ti_indices_filename))
        print('Loading Tiny Image indices from file {}'.format(ti_indices_filepath))
        assert pathlib.Path(ti_indices_filepath).is_file()
        with open(ti_indices_filepath, 'r') as f:
            tinyimage_indices = json.load(f)
        assert type(tinyimage_indices) is list
        assert len(tinyimage_indices) == labels.shape[0]
        return imagedata, labels, tinyimage_indices


def CIFAR10_1(transform):
    version = 'v4'
    images, labels = load_new_test_data(version)
    images = images/255.0
    num_images = images.shape[0]

    ##Train test split
    all_idx = np.arange(0,2021)
    train_idx_array = np.random.choice(np.arange(0, 2021), replace=False, size=(1021))
    test_idx_array = [x for x in all_idx if x not in train_idx_array]

    train_images = images[train_idx_array]
    test_images = images[test_idx_array]

    train_labels = labels[train_idx_array]
    test_labels = labels[test_idx_array]

    (unique, counts) = np.unique(train_labels, return_counts=True)
    frequencies = np.asarray((unique, counts)).T
    print(frequencies)

    (unique, counts) = np.unique(test_labels, return_counts=True)
    frequencies = np.asarray((unique, counts)).T
    print(frequencies)


    tensor_train_images = torch.Tensor(train_images) # transform to torch tensor
    tensor_test_images = torch.Tensor(test_images) # transform to torch tensor
    tensor_train_labels = torch.Tensor(train_labels) # transform to torch tensor
    tensor_test_labels = torch.Tensor(test_labels) # transform to torch tensor

    # transform=transforms.Compose([transforms.Normalize((0.437, 0.4437, 0.4728), (0.1980, 0.2010, 0.1970))])

    train_dataset = CustomTensorDataset((tensor_train_images,tensor_train_labels), transform) # create your datset
    # train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)

    test_dataset = CustomTensorDataset((tensor_test_images,tensor_test_labels), transform) # create your datset
    # test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    return train_dataset, test_dataset