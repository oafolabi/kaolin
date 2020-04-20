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
from tqdm import tqdm
import pandas as pd

from kaolin.rep.TriangleMesh import TriangleMesh
from kaolin.transforms import transforms as tfs

#Based off of modelnet.py and its ModelNet dataset class
class Scan2CAD(object):
    """ Dataset class for the Scan2CAD dataset.

    Args:
        data_frame (pd Dataframe): Dataframe containing directory - CAD ID pairs
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

        assert split.lower() in ['train', 'test']

        self.data_frame = data_frame
        self.transform = transform
        self.device = device
        self.names = []
        self.filepaths = data_frame.iloc[:, 0]
        self.cad_ids = data_frame.iloc[:,1]
        print("File PAth")
        print(self.filepaths[0])
        print("CAD ID")
        print(self.cad_ids[0])

        # for path in self.filepaths:
        #     if(not os.path.exists(path)):
        #         raise ValueError('OFF file not found at "{0}".'.format(basedir))


        # if not os.path.exists(basedir):
        #     raise ValueError('ModelNet was not found at "{0}".'.format(basedir))

        # available_categories = [p for p in os.listdir(basedir) if os.path.isdir(os.path.join(basedir, p))]

        # for cat_idx, category in enumerate(categories):
        #     assert category in available_categories, 'object class {0} not in list of available classes: {1}'.format(
        #         category, available_categories)

        #     cat_paths = glob(os.path.join(basedir, category, split.lower(), '*.off'))

        #     self.cat_idxs += [cat_idx] * len(cat_paths)
        #     self.names += [os.path.splitext(os.path.basename(cp))[0] for cp in cat_paths]
        #     self.filepaths += cat_paths

    def __len__(self):
        return len(self.cad_ids)
    
    def num_classes(self):
        return self.cad_ids.nunique()

    def __getitem__(self, index):
        """Returns the item at index idx. """
        data = TriangleMesh.from_off(self.filepaths[index])
        cad_id = self.cad_ids[index]
        data.to(self.device)
        if self.transform:
            data = self.transform(data)

        return data, cad_id


