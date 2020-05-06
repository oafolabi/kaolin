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

from kaolin.rep.TriangleMesh import TriangleMesh
from kaolin.transforms import transforms as tfs


class Redwood(object):
    r""" Dataset class for the Redwood dataset.

    Args:
        basedir (str): Path to the base directory of the ModelNet dataset.
        split (str, optional): Split to load ('train' vs 'test',
            default: 'train').
        categories (iterable, optional): List of categories to load
            (default: ['chair']).
        transform (callable, optional): A function/transform to apply on each
            loaded example.
        device (str or torch.device, optional): Device to use (cpu,
            cuda, cuda:1, etc.).  Default: 'cpu'

    Examples:
        >>> dataset = Redwod(basedir='data/ModelNet')
        >>> train_loader = DataLoader(dataset, batch_size=10, shuffle=True, num_workers=8)
        >>> obj, label = next(iter(train_loader))
    """

    def __init__(self, basedir: str,
                 split: Optional[str] = 'train',
                 categories: Optional[Iterable] = ['09637'],
                 transform: Optional[Callable] = None,
                 device: Optional[Union[torch.device, str]] = 'cpu'):

        assert split.lower() in ['train', 'test']

        self.basedir = basedir
        self.transform = transform
        self.device = device
        self.categories = categories
        self.names = []
        self.filepaths = []
        self.cat_idxs = []

        if not os.path.exists(basedir):
            raise ValueError('ModelNet2 was not found at "{0}".'.format(basedir))

        available_categories = [p for p in os.listdir(basedir) if os.path.isdir(os.path.join(basedir, p))]

        for cat_idx, category in enumerate(categories):
            assert category in available_categories, 'object class {0} not in list of available classes: {1}'.format(
                category, available_categories)

            cat_paths = glob(os.path.join(basedir, category, 'pre_aligned_mesh.obj'))

            self.cat_idxs += [cat_idx] * len(cat_paths)
            self.names += [os.path.splitext(os.path.basename(cp))[0] for cp in cat_paths]
            self.filepaths += cat_paths

    def __len__(self):
        return len(self.names)

    def __getitem__(self, index):
        """Returns the item at index idx. """
        category = torch.tensor(self.cat_idxs[index], dtype=torch.long, device=self.device)
        data = TriangleMesh.from_obj(self.filepaths[index])
        data.to(self.device)
        if self.transform:
            data = self.transform(data)

        return data, category