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

        split = split.lower()
        assert split in ['train' ,'test']
        self.data_frame = data_frame
        self.transform = transform
        self.device = device
        self.names = []
        self.filepaths = self.data_frame['Filepath']
        self.cad_ids = self.data_frame['ID']
        self.unique_labels = self.cad_ids.unique()
        self.label_map = {self.unique_labels[i] : i for i in range(len(self.unique_labels))}
        
        if(split == 'train'):
            ct_cad_ids = self.cad_ids.value_counts()
            s = ct_cad_ids.to_frame(name='Count')
            single_ct_labels = s[s['Count'] == 1]
            single_ct_labels = single_ct_labels.index.tolist()
            drop_indices = []
            for index, row in self.data_frame.iterrows():
                if(row[1] in single_ct_labels):
                    drop_indices.append(index)

            rest_of_data_frame = self.data_frame.drop(index = drop_indices)
            print(len(rest_of_data_frame))
            print(len(self.data_frame))
            assert 3==2
        
        #gets test set
        else:
            print("help")

    def __len__(self):
        return len(self.cad_ids)
    
    def num_classes(self):
        return self.cad_ids.nunique()

    def __getitem__(self, index):
        """Returns the item at index idx. """
        data = TriangleMesh.from_off(self.filepaths[index])
        cad_id = self.cad_ids[index]
        label = self.label_map[cad_id]

        label = torch.tensor(label, dtype=torch.long, device=self.device)
        data.to(self.device)
        
        if self.transform:
            data = self.transform(data)

        return data, label


