#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os

import cv2
import h5py
import numpy as np

from extract.transforms import random_sized_crop, color_norm
import torch.utils.data
import torch.nn.functional as F

# Per-channel mean and SD values in BGR order
_MEAN = [0.406, 0.456, 0.485]
_SD = [0.225, 0.224, 0.229]


def read_imlist(imlist_fn):
    with open(imlist_fn, 'r') as file:
        imlist = file.read().splitlines()
    return imlist


class FeatureStorage:
    def __init__(self, save_dir, desc_name, split, extension, global_desc_dim, local_desc_dim, dataset_size, save_type, topk=400):
        self.save_dir = save_dir
        self.desc_name = desc_name
        self.dataset_size = dataset_size
        self.save_type = save_type
        self.pointer = 0
        self.storage = {}

        if 'global' in save_type:
            global_storage = torch.FloatStorage.from_file(
                os.path.join(save_dir, f'{desc_name}{split}_global{extension}.pt'),
                True, dataset_size * global_desc_dim)
            self.storage['global'] = torch.FloatTensor(global_storage).reshape(dataset_size, global_desc_dim)

        if 'cls' in save_type:
            global_cls_storage = torch.FloatStorage.from_file(
                os.path.join(save_dir, f'{desc_name}_cls{split}_global{extension}.pt'),
                True, dataset_size * global_desc_dim)
            self.storage['cls'] = torch.FloatTensor(global_cls_storage).reshape(dataset_size, global_desc_dim)

        if 'local' in save_type:
            self.hdf5_file = h5py.File(os.path.join(save_dir, f'{desc_name}{split}_local{extension}.hdf5'), 'w')
            self.storage['local'] = self.hdf5_file.create_dataset('features', shape=[dataset_size, topk, local_desc_dim + 5], dtype=np.float32)
            self.storage['file_names'] = self.hdf5_file.create_dataset('file_names', shape=[dataset_size], dtype=h5py.special_dtype(vlen=str))
            
    
    def set_names(self, names_list):
        assert len(names_list) == self.dataset_size
        self.storage['file_names'][:] = names_list
        
    def __del__(self):
        if 'local' in self.storage:
            self.hdf5_file.close()


    def save(self, feats, save_type, file_name=''):
        if save_type in self.storage:
            self.storage[save_type][self.pointer:self.pointer + len(feats)] = feats            

    def update_pointer(self, size):
        self.pointer += size


class DataSet(torch.utils.data.Dataset):
    """Common dataset."""

    def __init__(self, name, data_path_base, scale_list, data_path, imsize=None, patch_size=None,
                 gnd=None, train=False, norm=True):
        assert os.path.exists(
            data_path_base), "Data path '{}' not found".format(data_path_base)
        self.data_path_base = data_path_base
        self.data_path = data_path
        self.gnd = gnd
        self.name = name
        self._scale_list = np.asarray(scale_list)
        self.imsize = imsize
        self.ps = patch_size
        self.train = train
        self.norm = norm
        self._construct_db()

    def _construct_db(self):
        """Constructs the db."""
        # Compile the split data path
        # if self.name == 'gldv2' and self.train:
        #     samples = [(line.split(',')[0], int(line.split(',')[1]), int(line.split(',')[2]), int(line.split(',')[3]))
        #                for
        #                line in self.data_path]
        #     self.categories = sorted(list(set([int(entry[1]) for entry in samples])))
        #     cat_to_label = dict(zip(self.categories, range(len(self.categories))))
        #     samples = [(entry[0], cat_to_label[entry[1]], entry[2], entry[3]) for entry in samples]
        #     self.targets = np.asarray([entry[1] for entry in samples])

        # self.data_path = [os.path.join(self.data_path_base, i.split(',')[0]) for i in self.data_path]
        self.data_path = [i.split(',')[0] for i in self.data_path]
        self.n = len(self.data_path)
    
    @staticmethod
    def rescale_tensor(tensor, max_size=2000):
        _, _, n, m = tensor.shape
        max_dim = max(n, m)

        if max_dim <= max_size:
            return tensor

        scale_factor = max_size / max_dim
        new_size = (int(n * scale_factor), int(m * scale_factor))
        rescaled_tensor = F.interpolate(tensor, size=new_size, mode='bilinear', align_corners=False)

        return rescaled_tensor

    def _prepare_im(self, im):
        """Prepares the image for network input."""
        if self.train:
            im = random_sized_crop(im, self.imsize)
        im = im.transpose([2, 0, 1])
        # [0, 255] -> [0, 1]
        im = im / 255.0
        # Color normalization
        im = torch.from_numpy(im).unsqueeze(0)
        im = self.rescale_tensor(im)[0]

        if self.norm:
            im = color_norm(im, _MEAN, _SD)
        return im

    def quantization_factor(self, side, scale):
        new_side = scale * side
        quantize_to = max(round(new_side / self.ps), 1.0)
        return scale / ((new_side / self.ps) / quantize_to)

    def __getitem__(self, indices):
        # Load the image
        im_list, scale_list = [], []
        for index in indices:
            try:
                img_path = os.path.join(self.data_path_base, self.data_path[index])
                if img_path.endswith('txt'):
                    pass
                img = cv2.imread(img_path)

                if not self.train and self.imsize is not None:
                    scales = self._scale_list * (self.imsize / max(img.shape))
                else:
                    scales = self._scale_list

                if self.gnd is not None:
                    bbx = self.gnd[index]["bbx"]
                    img = img[int(bbx[1]):int(bbx[3]), int(bbx[0]):int(bbx[2])]

                for scale in scales:
                    if not self.train and self.ps is not None:
                        scale_x = self.quantization_factor(img.shape[1], scale)
                        scale_y = self.quantization_factor(img.shape[0], scale)
                    else:
                        scale_x, scale_y = scale, scale

                    if scale_x < 1.0 or scale_y < 1.0:
                        im = cv2.resize(img, dsize=(0, 0), fx=scale_x, fy=scale_y, interpolation=cv2.INTER_AREA)
                    elif scale_x > 1.0 or scale_y > 1.0:
                        im = cv2.resize(img, dsize=(0, 0), fx=scale_x, fy=scale_y, interpolation=cv2.INTER_LINEAR)
                    else:
                        im = img

                    im_np = im.astype(np.float32, copy=False)
                    im_list.append(im_np)
                    scale_list.append((scale_x, scale_y))

            except Exception as e:
                print(f'error: {e}, file: {self.data_path[index]}')

        for idx in range(len(im_list)):
            im_list[idx] = self._prepare_im(im_list[idx])

        if self.train:
            return im_list, self.targets[indices]
        else:
            return im_list, scale_list

    def __len__(self):
        return len(self.data_path)
