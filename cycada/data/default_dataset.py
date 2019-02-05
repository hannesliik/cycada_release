import os.path
from glob import glob
import json

import numpy as np
import scipy.io
import torch
import torch.utils.data as data
from PIL import Image

from .data_loader import register_data_params, register_dataset_obj
from .data_loader import DatasetParams
from .cityscapes import id2label as LABEL2TRAIN

def colors2labels(arr, label2color, ignore_label=255):
    out = ignore_label * np.ones(arr.shape, dtype=np.uint8)
    for label, color in label2color.items():
        out[arr == id] = int(label)
    return out

@register_data_params('default')
class DefaultParams(DatasetParams):
    num_channels = 3
    image_size = 1024
    mean = 0.5
    std = 0.5
    num_cls = 19
    target_transform = None


@register_dataset_obj('default')
class DefaultDataset(data.Dataset):
    """
    Dataset Structure:
    /../data_dir/
        /some_other_dataset/
        /dataset_name/  (root)
            remap.json
            /train/
                /images/
                    *.png
                /labels/
                    *.png
            /test/
                /images/
                    *.png
                /labels/
                    *.png
            /val/
                /images/
                    *.png
                /labels/
                    *.png
    """

    def __init__(self, root, num_cls=19, split='train', remap_labels=True,
                 transform=None, target_transform=None):
        self.root = root
        self.split = split
        self.data_dir = os.path.join(self.root, self.split)
        assert os.path.exists(self.data_dir), f"{self.data_dir} does not exist"

        self.remap_labels = remap_labels
        self.img_ids, self.label_ids = self.collect_ids()
        self.transform = transform
        self.target_transform = target_transform

        if remap_labels:
            l2c_path = os.path.join(self.data_dir, "label2color.json")
            assert os.path.exists(l2c_path), f"Label to color mapping does not exist at {l2c_path}"
            with open(l2c_path, "r") as fp:
                self.label2color = json.load(fp)

    def collect_ids(self):
        # Iterate through all the images and remember their paths
        # It is presumed, that in the labels directory, if it exists, the images have the same ordering

        img_ids = glob(self.data_dir + "/images/*.jpg") + glob(self.data_dir + "/images/*.png")
        label_ids = glob(self.data_dir + "/labels/*.jpg") + glob(self.data_dir + "/labels/*.png")
        print(f"{len(img_ids)} images and {len(label_ids)} labels")
        return img_ids, label_ids

    def img_path(self, id):
        return self.img_ids[id]

    def label_path(self, id):
        return self.label_ids[id]

    def __getitem__(self, index):
        img_path = self.img_path(index)
        label_path = self.label_path(index)
        img = Image.open(img_path).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
        target = Image.open(label_path)
        if self.remap_labels:
            target = np.asarray(target)
            target = colors2labels(target, self.label2color)
            target = Image.fromarray(target, 'L')
        if self.target_transform is not None:
            target = self.target_transform(target)
        return img, target

    def __len__(self):
        return len(self.ids)
