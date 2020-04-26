# Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Callable, Iterable, Optional, Union, List

import torch
import math
import os
from glob import glob
import numpy as np

from tqdm import tqdm
import pandas as pd

from kaolin.rep.TriangleMesh import TriangleMesh
from kaolin.transforms import transforms as tfs


#Based off of modelnet.py and its ModelNet dataset class
class Scan2CAD(object):
    """ Dataset class for the Scan2CAD dataset.

    Args:
        data_frame (pd Dataframe): Dataframe containing directory - CAD ID pairs
            -Column Names: 'Filepath', 'ID'
        split (str, optional): Split to load ('train' vs 'test',
            default: 'train').
        transform (callable, optional): A function/transform to apply on each
            loaded example.
        device (str or torch.device, optional): Device to use (cpu,
            cuda, cuda:1, etc.).  Default: 'cpu'

    Examples:
        >>> dataset = ModelNet(basedir='data/ModelNet')
        >>> train_loader = DataLoader(dataset, batch_size=10, shuffle=True, num_workers=8)
        >>> obj, label = next(iter(train_loader))
    """

    def __init__(self, data_frame: pd.DataFrame,
                 split: Optional[str] = 'train',
                 transform: Optional[Callable] = None,
                 device: Optional[Union[torch.device, str]] = 'cpu'):

        #To ensure same split every time
        np.random.seed(42)

        split = split.lower()
        assert split in ['train' , 'validation','test']

        self.split = split
        self.transform = transform
        self.device = device

        filepaths = data_frame['Filepath']
        cad_ids = data_frame['ID']
        self.num_classes = cad_ids.nunique()
        self.unique_labels = cad_ids.unique()
        self.label_map = {self.unique_labels[i] : i for i in range(len(self.unique_labels))}

        #Does the data split
        ct_cad_ids = cad_ids.value_counts()
        s = ct_cad_ids.to_frame(name='Count')
        single_ct_labels = s[s['Count'] == 1]
        single_ct_labels = single_ct_labels.index.tolist()
        single_indices = []
        for index, row in data_frame.iterrows():
            if(row[1] in single_ct_labels):
                single_indices.append(index)
        
        rest_of_data_frame = data_frame.drop(index = single_indices)
        train_frac = 0.6 - (len(data_frame) - len(rest_of_data_frame))/ len(data_frame)

        num_train_samples = math.floor(train_frac * len(rest_of_data_frame))
        num_val_samples = math.floor(0.2*len(rest_of_data_frame))
        num_test_samples = len(rest_of_data_frame) - (num_train_samples + num_val_samples)

        shuffled_indices = np.random.choice(range(0,len(rest_of_data_frame)),
                                                len(rest_of_data_frame), 
                                                replace = False ).tolist()

        
        train_sample_indices = shuffled_indices[0:num_train_samples]
        val_indices = shuffled_indices[num_train_samples : num_train_samples + num_val_samples]
        test_indices = shuffled_indices[len(rest_of_data_frame)-num_test_samples:]

        #creates train and validation set
        self.train_data_frame = pd.concat([data_frame.iloc[single_indices], rest_of_data_frame.iloc[train_sample_indices]])
        self.train_filepaths =  self.train_data_frame['Filepath']
        self.train_cad_ids = self.train_data_frame['ID']

        self.validation_data_frame = rest_of_data_frame.iloc[val_indices]
        self.validation_filepaths =  self.validation_data_frame['Filepath']
        self.validation_cad_ids = self.validation_data_frame['ID']
        
        self.test_data_frame = rest_of_data_frame.iloc[test_indices]
        self.test_filepaths =  self.test_data_frame['Filepath']
        print(self.test_filepaths)
        self.test_cad_ids = self.test_data_frame['ID']

        train_indices_lst = self.train_data_frame.index.tolist()
        val_indices_lst = self.validation_data_frame.index.tolist()
        test_indices_lst = self.test_data_frame.index.tolist()


        #Saves dataframe just in case :)
        self.data_frame = data_frame

    def __len__(self):
        if(self.split == 'train'):
            return len(self.train_cad_ids)
        elif(self.split == 'validation'):
            return len(self.validation_cad_ids)
        else:
            return len(self.test_cad_ids)
    
    def get_num_classes(self):
        #Returns numclasses in ENTIRE DATASET
        return self.num_classes
    
    def set_split(self,split):
        split = split.lower()
        assert split in ['train' , 'validation','test']
        self.split = split

    def __getitem__(self, index):
        """Returns the item at index idx. """
        if(self.split == 'train'):
            data = TriangleMesh.from_off(self.train_filepaths[index])
            cad_id = self.train_cad_ids[index]
            label = self.label_map[cad_id]
        
        elif(self.split == 'validation'):
            data = TriangleMesh.from_off(self.validation_filepaths[index])
            cad_id = self.validation_cad_ids[index]
            label = self.label_map[cad_id]
        
        elif(self.split == 'test'):
            data = TriangleMesh.from_off(self.test_filepaths[index])
            cad_id = self.test_cad_ids[index]
            label = self.label_map[cad_id]

        label = torch.tensor(label, dtype=torch.long, device=self.device)
        data.to(self.device)
        
        if self.transform:
            data = self.transform(data)

        return data, label
    


