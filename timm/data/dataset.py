""" Quick n Simple Image Folder, Tarfile based DataSet

Hacked together by / Copyright 2020 Ross Wightman
"""
import torch.utils.data as data
import os
import torch
import logging
import random

from PIL import Image
from shutil import copyfile
from .parsers import create_parser

_logger = logging.getLogger(__name__)


_ERROR_RETRY = 50


class ImageDataset(data.Dataset):

    def __init__(
            self,
            root,
            parser=None,
            class_map=None,
            load_bytes=False,
            transform=None,
            target_transform=None,
    ):
        if parser is None or isinstance(parser, str):
            parser = create_parser(parser or '', root=root, class_map=class_map)
        self.parser = parser
        self.load_bytes = load_bytes
        self.transform = transform
        self.target_transform = target_transform
        self._consecutive_errors = 0

    def __getitem__(self, index):
        img, target = self.parser[index]
        try:
            img = img.read() if self.load_bytes else Image.open(img).convert('RGB')
        except Exception as e:
            _logger.warning(f'Skipped sample (index {index}, file {self.parser.filename(index)}). {str(e)}')
            self._consecutive_errors += 1
            if self._consecutive_errors < _ERROR_RETRY:
                return self.__getitem__((index + 1) % len(self.parser))
            else:
                raise e
        self._consecutive_errors = 0
        if self.transform is not None:
            img = self.transform(img)
        if target is None:
            target = -1
        elif self.target_transform is not None:
            target = self.target_transform(target)
        return img, target

    def __len__(self):
        return len(self.parser)

    def filename(self, index, basename=False, absolute=False):
        return self.parser.filename(index, basename, absolute)

    def filenames(self, basename=False, absolute=False):
        return self.parser.filenames(basename, absolute)

    def class_map(self):
        return self.parser.classmap()

class ImageDatasetV2(data.Dataset):
    def __init__(self, root=None, transform=None, flag=2, is_training=False):
        self.root = root
        self.transform = transform
        self.is_training = is_training
        if (flag == 2):
            self.dataset, self.id_class, self.id_num, self.filenames_list = self.get_dataset_v2(self.root)
        elif (flag == 3):
            self.dataset, self.id_class, self.id_num, self.filenames_list = self.get_dataset_v3(self.root)
        elif (flag == 4):
            self.dataset, self.id_class, self.id_num, self.filenames_list = self.get_dataset_v4(self.root)


    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        img_path, label = self.dataset[index]
        img = Image.open(img_path).convert('RGB')   # RGB  L

        if self.transform is not None:
            img = self.transform(img)
        return img, label

    # 增加按照txt读取方式
    def get_dataset_v2(self, root):
        id_class = {'another': 0, 'phone': 1, 'sigurate': 2}
        id_num = {'another': 0, 'phone': 0, 'sigurate': 0}
        datasets = []
        filenames = []
        for path in root:
            with open(path, 'r') as f:
                lines = f.readlines()
                for line in lines:
                    line = line.strip()
                    name = line.split('/')[-2]
                    label = id_class[name]
                    id_num[name] += 1
                    datasets.append((line, label))
        print(id_num)

        random.shuffle(datasets)
        for d in datasets:
            # filename = os.path.join(d[0].split("/")[-2], d[0].split("/")[-1])
            filename = os.path.join(d[0])
            filenames.append(filename)

        return datasets, id_class, id_num, filenames

    # 将another类别进一步细分为假抽烟、假打电话、其他共三类
    def get_dataset_v3(self, root):
        id_class = {'another': 0, 'phone': 1, 'sigurate': 2, 'fake_phone': 3, 'fake_sigurate': 4}
        id_num = {'another': 0, 'phone': 0, 'sigurate': 0, 'fake_phone': 0, 'fake_sigurate': 0}
        datasets = []
        filenames = []
        for path in root:
            with open(path, 'r') as f:
                lines = f.readlines()
                for line in lines:
                    line = line.strip()
                    name = line.split('/')[-2]
                    if (name == 'another'):
                        imgname = line.split('/')[-1]
                        fake_flag = imgname.split("_")[0]
                        if (fake_flag == 'sigurate'):
                            name = "fake_sigurate"
                        elif (fake_flag == 'phone'):
                            name = "fake_phone"
                    label = id_class[name]
                    id_num[name] += 1
                    datasets.append((line, label))
        print(id_num)

        random.shuffle(datasets)
        for d in datasets:
            # filename = os.path.join(d[0].split("/")[-2], d[0].split("/")[-1])
            filename = os.path.join(d[0])
            filenames.append(filename)
        return datasets, id_class, id_num, filenames

    # 只取抽烟和假抽烟两类
    def get_dataset_v4(self, root):
        id_class = {'sigurate': 0, 'fake_sigurate': 1}
        id_num = {'sigurate': 0, 'fake_sigurate': 0}
        datasets = []
        filenames = []
        for path in root:
            with open(path, 'r') as f:
                lines = f.readlines()
                for line in lines:
                    line = line.strip()
                    name = line.split('/')[-2]
                    if (name == 'another'):
                        imgname = line.split('/')[-1]
                        fake_flag = imgname.split("_")[0]
                        if (fake_flag == 'sigurate'):
                            name = "fake_sigurate"
                        else:
                            continue
                    elif (name == 'sigurate'):
                        pass
                    else:
                        continue
                    label = id_class[name]
                    id_num[name] += 1
                    datasets.append((line, label))
        print(id_num)

        random.shuffle(datasets)
        for d in datasets:
            # filename = os.path.join(d[0].split("/")[-2], d[0].split("/")[-1])
            filename = os.path.join(d[0])
            filenames.append(filename)
            '''
            # 保存新数据集
            img_path = d[0]
            savePath = "/home/chenpengfei/dataset/DSMhands5"
            if (self.is_training):
                savePath = os.path.join(savePath, "train")
            else:
                savePath = os.path.join(savePath, "val")
            if (d[1] == 0):
                c_savePath = os.path.join(savePath, "sigurate")
                if not os.path.exists(c_savePath):
                    os.makedirs(c_savePath)
                filename = img_path.split("/")[-1]
                filepath = os.path.join(c_savePath, filename)
                copyfile(img_path, filepath)
            elif (d[1] == 1):
                c_savePath = os.path.join(savePath, "fake_sigurate")
                if not os.path.exists(c_savePath):
                    os.makedirs(c_savePath)
                filename = img_path.split("/")[-1]
                filepath = os.path.join(c_savePath, filename)
                copyfile(img_path, filepath)
            '''
        return datasets, id_class, id_num, filenames

    def class_map(self):
        return self.id_class

    def filename(self, index, basename=False, absolute=False):
        return self.filenames[index]

    def filenames(self, basename=False, absolute=False):
        return self.filenames_list


class IterableImageDataset(data.IterableDataset):

    def __init__(
            self,
            root,
            parser=None,
            split='train',
            is_training=False,
            batch_size=None,
            repeats=0,
            download=False,
            transform=None,
            target_transform=None,
    ):
        assert parser is not None
        if isinstance(parser, str):
            self.parser = create_parser(
                parser, root=root, split=split, is_training=is_training,
                batch_size=batch_size, repeats=repeats, download=download)
        else:
            self.parser = parser
        self.transform = transform
        self.target_transform = target_transform
        self._consecutive_errors = 0

    def __iter__(self):
        for img, target in self.parser:
            if self.transform is not None:
                img = self.transform(img)
            if self.target_transform is not None:
                target = self.target_transform(target)
            yield img, target

    def __len__(self):
        if hasattr(self.parser, '__len__'):
            return len(self.parser)
        else:
            return 0

    def filename(self, index, basename=False, absolute=False):
        assert False, 'Filename lookup by index not supported, use filenames().'

    def filenames(self, basename=False, absolute=False):
        return self.parser.filenames(basename, absolute)


class AugMixDataset(torch.utils.data.Dataset):
    """Dataset wrapper to perform AugMix or other clean/augmentation mixes"""

    def __init__(self, dataset, num_splits=2):
        self.augmentation = None
        self.normalize = None
        self.dataset = dataset
        if self.dataset.transform is not None:
            self._set_transforms(self.dataset.transform)
        self.num_splits = num_splits

    def _set_transforms(self, x):
        assert isinstance(x, (list, tuple)) and len(x) == 3, 'Expecting a tuple/list of 3 transforms'
        self.dataset.transform = x[0]
        self.augmentation = x[1]
        self.normalize = x[2]

    @property
    def transform(self):
        return self.dataset.transform

    @transform.setter
    def transform(self, x):
        self._set_transforms(x)

    def _normalize(self, x):
        return x if self.normalize is None else self.normalize(x)

    def __getitem__(self, i):
        x, y = self.dataset[i]  # all splits share the same dataset base transform
        x_list = [self._normalize(x)]  # first split only normalizes (this is the 'clean' split)
        # run the full augmentation on the remaining splits
        for _ in range(self.num_splits - 1):
            x_list.append(self._normalize(self.augmentation(x)))
        return tuple(x_list), y

    def __len__(self):
        return len(self.dataset)
