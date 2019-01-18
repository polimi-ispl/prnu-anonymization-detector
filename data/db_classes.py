import os
from torch.utils.data import Dataset
import numpy as np
import pandas as pd
from PIL import Image
import torch
from sklearn.model_selection import train_test_split
from imageio import imread
from patch import PatchExtractor
from params import db_path
from skimage.restoration import denoise_wavelet


class TrainDB(Dataset):
    def __init__(self, patch_size: tuple, subsample: float, transform_pre=None, transform_post=None, patch_stride: tuple = None):
        """

        :param patch_size:
        :param subsample:
        :param transform_pre: Transformation applied before casting to torch tensor
        :param transform_post: Transformation applied after casting to torch tensor
        :param patch_stride:
        """
        self.seed = 21
        self.csv_path = None
        self.original_db = None
        self.split = False
        self.train_db = None
        self.val_db = None
        self.is_train = None
        self.is_val = None
        self.train_idx = None
        self.val_idx = None
        self.test_idx = None

        self.current_db = None
        self.current_idx = None

        self.image_size = (512, 512, 3)
        self.patch_size = patch_size
        self.patch_stride = patch_stride

        self.patch_idx = None
        self.patch_idx_cum = None

        self.patch_idx_train = None
        self.patch_idx_cum_train = None
        self.patch_idx_val = None
        self.patch_idx_cum_val = None

        self.transform_pre = None
        self.transform_post = None

    def __len__(self):
        return len(self.patch_idx)

    def __getitem__(self, idx: int):
        cpi = self.patch_idx[idx]
        less_index_list = self.patch_idx_cum[self.patch_idx_cum <= idx]
        lil = less_index_list[-1] if len(less_index_list) != 0 else 0
        spi = idx - lil

        img = np.asarray(Image.open(self.current_db.loc[self.current_idx[cpi]].ProbePath)).astype(np.uint8)
        pe = PatchExtractor.PatchExtractor(dim=self.patch_size, stride=self.patch_stride, indexes=[spi])
        img = pe.extract(img)  # side effect: shape = (1,) + shape

        img = img.astype(np.float) / 255
        img = np.squeeze(img)

        if self.transform_pre == 'wv_fft':
            img_den = denoise_wavelet(img, multichannel=True)
            img_noise = img - img_den
            img_noise_fft = np.abs((np.fft.fft2(img_noise, (self.patch_size[0] * 2,
                                                            self.patch_size[1] * 2),
                                                axes=[0, 1])))[:self.patch_size[0], :self.patch_size[1]]
            img = img_noise_fft

        label = np.zeros((1,)).astype(np.uint8) if self.current_db.loc[self.current_idx[cpi]].Provenance == 'original' \
            else np.ones((1, )).astype(np.uint8)
        label = torch.from_numpy(label).float()

        img = torch.from_numpy(img.transpose((2, 0, 1))).float()

        if self.transform_post:
            img = self.transform_post(img)

        return img, label

    def __compute_patch_idx_lenght(self):
        patch_idx = []
        for i in range(len(self.current_db)):
            try:
                img_size = imread(self.current_db.loc[self.current_idx[i]].ProbePath).shape
            except IOError:
                raise NotImplementedError
            n_patch = PatchExtractor.count_patches(img_size, self.patch_size, self.patch_stride)
            patch_idx += [i] * n_patch

        return np.asarray(patch_idx), np.cumsum(np.bincount(patch_idx))

    def train(self):
        assert self.split is True
        self.is_train = True
        self.is_val = False
        self.current_db = self.train_db
        self.current_idx = self.train_idx
        if self.patch_idx_cum_train is None:
            self.patch_idx_train, self.patch_idx_cum_train = self.__compute_patch_idx_lenght()
        self.patch_idx = self.patch_idx_train
        self.patch_idx_cum = self.patch_idx_cum_train

    def val(self):
        assert self.split is True
        self.is_train = False
        self.is_val = True
        self.current_db = self.val_db
        self.current_idx = self.val_idx
        if self.patch_idx_cum_val is None:
            self.patch_idx_val, self.patch_idx_cum_val = self.__compute_patch_idx_lenght()
        self.patch_idx = self.patch_idx_val
        self.patch_idx_cum = self.patch_idx_cum_val

    def generate_split(self, train_size: float):
        train_idx, val_idx = train_test_split(self.original_db.index, train_size=train_size, random_state=self.seed)

        self.train_db = self.original_db.loc[train_idx]
        self.val_db = self.original_db.loc[val_idx]
        self.train_idx = train_idx
        self.val_idx = val_idx
        self.split = True


class D1(TrainDB):
    def __init__(self, patch_size, subsample: float = None, transform_pre=None, transform_post=None, patch_stride: tuple=None):
        super().__init__(patch_size, subsample, transform_pre, transform_post, patch_stride)

        self.csv_path = os.path.join(db_path, 'orig_mand_train.csv')
        self.original_db = pd.read_csv(self.csv_path)

        if subsample:
            self.original_db = self.original_db.sample(frac=subsample, random_state=self.seed)

        if transform_pre:
            self.transform_pre = transform_pre

        if transform_post:
            self.transform_post = transform_post


class D2(TrainDB):
    def __init__(self, patch_size, subsample: float = None, transform_pre=None, transform_post=None, patch_stride: tuple=None):
        super().__init__(patch_size, subsample, transform_pre, transform_post, patch_stride)

        self.csv_path = os.path.join(db_path, 'orig_kirch_train.csv')
        self.original_db = pd.read_csv(self.csv_path)

        if subsample:
            self.original_db = self.original_db.sample(frac=subsample, random_state=self.seed)

        if transform_pre:
            self.transform_pre = transform_pre

        if transform_post:
            self.transform_post = transform_post


class D3(TrainDB):
    def __init__(self, patch_size, subsample: float = None, transform_pre=None, transform_post=None, patch_stride: tuple=None):
        super().__init__(patch_size, subsample, transform_pre, transform_post, patch_stride)

        self.csv_path = os.path.join(db_path, 'orig_mand_kirch_train.csv')
        self.original_db = pd.read_csv(self.csv_path)

        if subsample:
            self.original_db = self.original_db.sample(frac=subsample, random_state=self.seed)

        if transform_pre:
            self.transform_pre = transform_pre

        if transform_post:
            self.transform_post = transform_post


class TestDB(Dataset):
    def __init__(self, patch_size: tuple, subsample: float, transform_pre=None, transform_post=None, patch_stride: tuple = None):
        """
        :param subsample: fraction of the entire dataset to consider
        :param transform: optional transformation to be applied on each image
        """
        self.seed = 21
        self.csv_path = None
        self.original_db = None
        self.original_idx = None

        self.current_db = None
        self.current_idx = None

        self.image_size = (512, 512, 3)
        self.patch_size = patch_size
        self.patch_stride = patch_stride

        self.patch_idx = None
        self.patch_idx_cum = None

        self.transform = None

    def __len__(self):
        return len(self.patch_idx)

    def __getitem__(self, idx: int):
        cpi = self.patch_idx[idx]
        less_index_list = self.patch_idx_cum[self.patch_idx_cum <= idx]
        lil = less_index_list[-1] if len(less_index_list) != 0 else 0
        spi = idx - lil

        img = np.asarray(Image.open(self.current_db.loc[self.current_idx[cpi]].ProbePath)).astype(np.uint8)
        pe = PatchExtractor.PatchExtractor(dim=self.patch_size, stride=self.patch_stride, indexes=[spi])
        img = pe.extract(img)  # side effect: shape = (1,) + shape

        img = img.astype(np.float) / 255
        img = np.squeeze(img)

        if self.transform_pre == 'wv_fft':
            img_den = denoise_wavelet(img, multichannel=True)
            img_noise = img - img_den
            img_noise_fft = np.abs((np.fft.fft2(img_noise, (self.patch_size[0] * 2,
                                                            self.patch_size[1] * 2),
                                                axes=[0, 1])))[:self.patch_size[0], :self.patch_size[1]]
            img = img_noise_fft

        label = np.zeros((1,)).astype(np.uint8) if self.current_db.loc[self.current_idx[cpi]].Provenance == 'original' \
            else np.ones((1, )).astype(np.uint8)
        label = torch.from_numpy(label).float()

        img = torch.from_numpy(img.transpose((2, 0, 1))).float()

        if self.transform:
            img = self.transform(img)

        return img, label

    def compute_patch_idx_lenght(self):
        patch_idx = []
        for i in range(len(self.current_db)):
            try:
                img_size = imread(self.current_db.loc[self.current_idx[i]].ProbePath).shape
            except IOError:
                raise NotImplementedError
            n_patch = PatchExtractor.count_patches(img_size, self.patch_size, self.patch_stride)
            patch_idx += [i] * n_patch
        self.patch_idx = np.asarray(patch_idx)
        self.patch_idx_cum = np.cumsum(np.bincount(self.patch_idx))


class Test1(TestDB):
    def __init__(self, patch_size, subsample: float = None, transform_pre=None, transform_post=None, patch_stride: tuple=None):
        super().__init__(patch_size, subsample, transform_pre, transform_post, patch_stride)

        self.csv_path = os.path.join(db_path, 'orig_test.csv')
        self.original_db = pd.read_csv(self.csv_path)

        if subsample:
            self.original_db = self.original_db.sample(frac=subsample, random_state=self.seed)

        if transform_pre:
            self.transform_pre = transform_pre

        if transform_post:
            self.transform_post = transform_post

        self.current_db = self.original_db
        self.current_idx = self.current_db.index
        super().compute_patch_idx_lenght()


class Test2(TestDB):
    def __init__(self, patch_size, subsample: float = None, transform_pre=None, transform_post=None, patch_stride: tuple = None):
        super().__init__(patch_size, subsample, transform_pre, transform_post, patch_stride)

        self.csv_path = os.path.join(db_path, 'mand_test.csv')
        self.original_db = pd.read_csv(self.csv_path)

        if subsample:
            self.original_db = self.original_db.sample(frac=subsample, random_state=self.seed)

        if transform_pre:
            self.transform_pre = transform_pre

        if transform_post:
            self.transform_post = transform_post

        self.current_db = self.original_db
        self.current_idx = self.current_db.index
        super().compute_patch_idx_lenght()


class Test3(TestDB):
    def __init__(self, patch_size, subsample: float = None, transform_pre=None, transform_post=None, patch_stride: tuple = None):
        super().__init__(patch_size, subsample, transform_pre, transform_post, patch_stride)

        self.csv_path = os.path.join(db_path, 'kirch_test.csv')
        self.original_db = pd.read_csv(self.csv_path)

        if subsample:
            self.original_db = self.original_db.sample(frac=subsample, random_state=self.seed)

        if transform_pre:
            self.transform_pre = transform_pre

        if transform_post:
            self.transform_post = transform_post

        self.current_db = self.original_db
        self.current_idx = self.current_db.index
        super().compute_patch_idx_lenght()


class TestOS(TestDB):
    def __init__(self, patch_size, subsample: float = None, transform_pre=None, transform_post=None, patch_stride: tuple = None):
        super().__init__(patch_size, subsample, transform_pre, transform_post, patch_stride)

        self.csv_path = os.path.join(db_path, 'orig_mand_test.csv')
        self.original_db = pd.read_csv(self.csv_path)

        if subsample:
            self.original_db = self.original_db.sample(frac=subsample, random_state=self.seed)

        if transform_pre:
            self.transform_pre = transform_pre

        if transform_post:
            self.transform_post = transform_post

        self.current_db = self.original_db
        self.current_idx = self.current_db.index
        super().compute_patch_idx_lenght()


class TestOK(TestDB):
    def __init__(self, patch_size, subsample: float = None, transform_pre=None, transform_post=None, patch_stride: tuple = None):
        super().__init__(patch_size, subsample, transform_pre, transform_post, patch_stride)

        self.csv_path = os.path.join(db_path, 'orig_kirch_test.csv')
        self.original_db = pd.read_csv(self.csv_path)

        if subsample:
            self.original_db = self.original_db.sample(frac=subsample, random_state=self.seed)

        if transform_pre:
            self.transform_pre = transform_pre

        if transform_post:
            self.transform_post = transform_post

        self.current_db = self.original_db
        self.current_idx = self.current_db.index
        super().compute_patch_idx_lenght()