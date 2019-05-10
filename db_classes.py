import os
from torch.utils.data import Dataset
import numpy as np
import pandas as pd
from PIL import Image, ImageEnhance
import torch
from sklearn.model_selection import train_test_split
from imageio import imread
from patch import PatchExtractor
from params import db_path
from skimage.restoration import denoise_wavelet
from prnu.prnu.functions import wiener_adaptive, zero_mean_total
import io
from scipy.io import loadmat
from prnu.prnu import extract_single
from skimage import color


class TrainDB(Dataset):
    def __init__(self, patch_size: tuple, subsample: float, transform_pre: str = None, transform_post=None,
                 patch_stride: tuple = None):
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

        self.transform_pre = transform_pre
        self.transform_post = transform_post

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

        if self.transform_pre == 'wv':
            img_den = denoise_wavelet(img, multichannel=True)

            img = img_den

        elif self.transform_pre == 'wv_fft':
            img_den = denoise_wavelet(img, multichannel=True)
            img_noise = img - img_den
            img_noise_fft = np.abs((np.fft.fft2(img_noise, (self.patch_size[0] * 2,
                                                            self.patch_size[1] * 2),
                                                axes=[0, 1])))[:self.patch_size[0], :self.patch_size[1]]

            img = img_noise_fft

        elif self.transform_pre == 'wv_fft_wiener1':

            img_den = denoise_wavelet(img, multichannel=True)
            img_noise = img - img_den
            img_noise_fft = np.abs((np.fft.fft2(img_noise, (self.patch_size[0] * 2,
                                                            self.patch_size[1] * 2),
                                                axes=[0, 1])))[:self.patch_size[0], :self.patch_size[1]]

            W = np.zeros_like(img_noise_fft)

            for c in range(img_noise_fft.ndim):
                W_c = zero_mean_total(img_noise_fft[:, :, c])
                W_c_std = W_c.std(ddof=1)
                W[:, :, c] = wiener_adaptive(W_c, W_c_std ** 2).astype(np.float32)
            img = W

        elif self.transform_pre == 'wv_fft_wiener2':

            img_den = denoise_wavelet(img, multichannel=True)
            img_noise = img - img_den
            img_noise_fft = np.abs((np.fft.fft2(img_noise, (self.patch_size[0] * 2,
                                                            self.patch_size[1] * 2),
                                                axes=[0, 1])))[:self.patch_size[0], :self.patch_size[1]]

            W = np.zeros_like(img_noise_fft)

            for c in range(img_noise_fft.ndim):
                W_c = zero_mean_total(img_noise_fft[:, :, c])
                W_c_std = W_c.std(ddof=1)
                W[:, :, c] = wiener_adaptive(W_c, (W_c_std ** 2) * 0.77, window_size_list=[3]).astype(np.float32)

            img = W

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
    def __init__(self, patch_size, subsample: float = None, transform_pre=None, transform_post=None, transform_test: str = None, patch_stride=None):
        super().__init__(patch_size=patch_size, subsample=subsample, transform_pre=transform_pre,
                         transform_post=transform_post, patch_stride=patch_stride)

        self.csv_path = os.path.join(db_path, 'orig_mand_train.csv')
        self.original_db = pd.read_csv(self.csv_path)

        if subsample:
            self.original_db = self.original_db.sample(frac=subsample, random_state=self.seed)

        if transform_pre:
            self.transform_pre = transform_pre

        if transform_post:
            self.transform_post = transform_post


class D2(TrainDB):
    def __init__(self, patch_size, subsample: float = None, transform_pre=None, transform_post=None, transform_test: str = None, patch_stride: tuple=None):
        super().__init__(patch_size=patch_size, subsample=subsample, transform_pre=transform_pre,
                         transform_post=transform_post, patch_stride=patch_stride)

        self.csv_path = os.path.join(db_path, 'orig_kirch_train.csv')
        self.original_db = pd.read_csv(self.csv_path)

        if subsample:
            self.original_db = self.original_db.sample(frac=subsample, random_state=self.seed)

        if transform_pre:
            self.transform_pre = transform_pre

        if transform_post:
            self.transform_post = transform_post


class D3(TrainDB):
    def __init__(self, patch_size, subsample: float = None, transform_pre=None, transform_post=None, transform_test: str = None, patch_stride: tuple=None):
        super().__init__(patch_size=patch_size, subsample=subsample, transform_pre=transform_pre,
                         transform_post=transform_post, patch_stride=patch_stride)

        self.csv_path = os.path.join(db_path, 'orig_mand_kirch_train.csv')
        self.original_db = pd.read_csv(self.csv_path)

        if subsample:
            self.original_db = self.original_db.sample(frac=subsample, random_state=self.seed)

        if transform_pre:
            self.transform_pre = transform_pre

        if transform_post:
            self.transform_post = transform_post


class M(TrainDB):
    def __init__(self, patch_size, subsample: float = None, transform_pre=None, transform_post=None, transform_test: str = None, patch_stride: tuple=None):
        super().__init__(patch_size=patch_size, subsample=subsample, transform_pre=transform_pre,
                         transform_post=transform_post, patch_stride=patch_stride)

        self.csv_path = os.path.join(db_path, 'orig_mand_kirch_train_no_D200_0.csv')
        self.original_db = pd.read_csv(self.csv_path)

        if subsample:
            self.original_db = self.original_db.sample(frac=subsample, random_state=self.seed)

        if transform_pre:
            self.transform_pre = transform_pre

        if transform_post:
            self.transform_post = transform_post


class TestDB(Dataset):
    def __init__(self, patch_size: tuple, subsample: float, transform_pre: str = None, transform_post=None,
                 transform_test: str = None, patch_stride: tuple = None):
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

        self.transform_pre = transform_pre
        self.transform_post = transform_post
        self.transform_test = transform_test

    def _random_trans(self, img: Image):
        """

        :param img: original PIL.Image in range [0, 255]
        :return: transformed PIL.Image in range [0, 255]
        """

        case = np.random.randint(4)
        img_trans = None

        if case == 0:
            # jpeg
            buffer = io.BytesIO()
            factor = np.random.choice([70, 75, 80, 85, 90])
            img.save(buffer, 'JPEG', quality=int(factor))
            img_trans = Image.open(buffer)

        elif case == 1:
            # gamma
            factor = np.random.choice([0.5, 0.6, 0.7, 0.8, 0.9])
            img_trans = ((np.asarray(img).astype(np.float) / 255) ** factor) * 255
            img_trans = Image.fromarray(img_trans.astype(np.uint8))

        elif case == 2:
            # brightness
            factor = np.random.choice([0.5, 0.6, 0.7, 0.8, 0.9])
            enhancer = ImageEnhance.Brightness(img)
            img_trans = enhancer.enhance(factor)

        elif case == 3:
            # contrast
            factor = np.random.choice([0.5, 0.6, 0.7, 0.8, 0.9])
            enhancer = ImageEnhance.Contrast(img)
            img_trans = enhancer.enhance(factor)

        return img_trans

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

        img = np.squeeze(img)

        if self.transform_test:
            img = self._random_trans(Image.fromarray(img))

        img = np.asarray(img).astype(np.float) / 255

        if self.transform_pre == 'wv':
            img_den = denoise_wavelet(img, multichannel=True)

            img = img_den

        elif self.transform_pre == 'wv_fft':
            img_den = denoise_wavelet(img, multichannel=True)
            img_noise = img - img_den
            img_noise_fft = np.abs((np.fft.fft2(img_noise, (self.patch_size[0] * 2,
                                                            self.patch_size[1] * 2),
                                                axes=[0, 1])))[:self.patch_size[0], :self.patch_size[1]]

            img = img_noise_fft

        elif self.transform_pre == 'wv_fft_wiener1':

            img_den = denoise_wavelet(img, multichannel=True)
            img_noise = img - img_den
            img_noise_fft = np.abs((np.fft.fft2(img_noise, (self.patch_size[0] * 2,
                                                            self.patch_size[1] * 2),
                                                axes=[0, 1])))[:self.patch_size[0], :self.patch_size[1]]

            W = np.zeros_like(img_noise_fft)

            for c in range(img_noise_fft.ndim):
                W_c = zero_mean_total(img_noise_fft[:, :, c])
                W_c_std = W_c.std(ddof=1)
                W[:, :, c] = wiener_adaptive(W_c, W_c_std ** 2).astype(np.float32)

            img = W

        elif self.transform_pre == 'wv_fft_wiener2':

            img_den = denoise_wavelet(img, multichannel=True)
            img_noise = img - img_den
            img_noise_fft = np.abs((np.fft.fft2(img_noise, (self.patch_size[0] * 2,
                                                            self.patch_size[1] * 2),
                                                axes=[0, 1])))[:self.patch_size[0], :self.patch_size[1]]

            W = np.zeros_like(img_noise_fft)

            for c in range(img_noise_fft.ndim):
                W_c = zero_mean_total(img_noise_fft[:, :, c])
                W_c_std = W_c.std(ddof=1)
                W[:, :, c] = wiener_adaptive(W_c, (W_c_std ** 2) * 0.77, window_size_list=[3]).astype(np.float32)

            img = W

        label = np.zeros((1,)).astype(np.uint8) if self.current_db.loc[self.current_idx[cpi]].Provenance == 'original' \
            else np.ones((1, )).astype(np.uint8)
        label = torch.from_numpy(label).float()

        img = torch.from_numpy(img.transpose((2, 0, 1))).float()

        if self.transform_post:
            img = self.transform_post(img)

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
    def __init__(self, patch_size, subsample: float = None, transform_pre=None, transform_post=None, transform_test: str = None, patch_stride: tuple=None):
        super().__init__(patch_size=patch_size, subsample=subsample, transform_pre=transform_pre,
                         transform_post=transform_post, transform_test=transform_test, patch_stride=patch_stride)

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
    def __init__(self, patch_size, subsample: float = None, transform_pre=None, transform_post=None, transform_test: str = None, patch_stride: tuple = None):
        super().__init__(patch_size=patch_size, subsample=subsample, transform_pre=transform_pre,
                         transform_post=transform_post, transform_test=transform_test, patch_stride=patch_stride)

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
    def __init__(self, patch_size, subsample: float = None, transform_pre=None, transform_post=None, transform_test: str = None, patch_stride: tuple = None):
        super().__init__(patch_size=patch_size, subsample=subsample, transform_pre=transform_pre,
                         transform_post=transform_post, transform_test=transform_test, patch_stride=patch_stride)

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
    def __init__(self, patch_size, subsample: float = None, transform_pre=None, transform_post=None, transform_test: str = None, patch_stride: tuple = None):
        super().__init__(patch_size=patch_size, subsample=subsample, transform_pre=transform_pre,
                         transform_post=transform_post, transform_test=transform_test, patch_stride=patch_stride)

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
    def __init__(self, patch_size, subsample: float = None, transform_pre=None, transform_post=None, transform_test: str = None, patch_stride: tuple = None):
        super().__init__(patch_size=patch_size, subsample=subsample, transform_pre=transform_pre,
                         transform_post=transform_post, transform_test=transform_test, patch_stride=patch_stride)

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


class TestOS_D200_0(TestDB):
    def __init__(self, patch_size, subsample: float = None, transform_pre=None, transform_post=None, transform_test: str = None, patch_stride: tuple = None):
        super().__init__(patch_size=patch_size, subsample=subsample, transform_pre=transform_pre,
                         transform_post=transform_post, transform_test=transform_test, patch_stride=patch_stride)

        self.csv_path = os.path.join(db_path, 'orig_mand_test_D200_0.csv')
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


class TestOK_D200_0(TestDB):
    def __init__(self, patch_size, subsample: float = None, transform_pre=None, transform_post=None, transform_test: str = None, patch_stride: tuple = None):
        super().__init__(patch_size=patch_size, subsample=subsample, transform_pre=transform_pre,
                         transform_post=transform_post, transform_test=transform_test, patch_stride=patch_stride)

        self.csv_path = os.path.join(db_path, 'orig_kirch_test_D200_0.csv')
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


class TestTransDB(Dataset):
    def __init__(self, patch_size: tuple, subsample: float, transform_pre: str = None, transform_post=None,
                 transform_test: str = None, patch_stride: tuple = None):
        """
        :param subsample: fraction of the entire dataset to consider
        :param transform: optional transformation to be applied on each image
        """
        self.seed = 21
        self.csv_path = None
        self.prnu_csv_path = None
        self.original_db = None
        self.prnu_db = None
        self.original_idx = None

        self.current_db = None
        self.current_idx = None

        self.image_size = (512, 512, 3)
        self.patch_size = patch_size
        self.patch_stride = patch_stride

        self.patch_idx = None
        self.patch_idx_cum = None

        self.transform_pre = transform_pre
        self.transform_post = transform_post
        self.transform_test = transform_test

    @staticmethod
    def _random_trans(img: Image):
        """

        :param img: original PIL.Image in range [0, 255]
        :return: transformed PIL.Image in range [0, 255]
        """

        case = np.random.randint(4)
        img_trans = None

        if case == 0:
            # jpeg
            buffer = io.BytesIO()
            factor = np.random.choice([70, 75, 80, 85, 90])
            img.save(buffer, 'JPEG', quality=int(factor))
            img_trans = Image.open(buffer)

        elif case == 1:
            # gamma
            factor = np.random.choice([0.5, 0.6, 0.7, 0.8, 0.9])
            img_trans = ((np.asarray(img).astype(np.float) / 255) ** factor) * 255
            img_trans = Image.fromarray(img_trans.astype(np.uint8))

        elif case == 2:
            # brightness
            factor = np.random.choice([0.5, 0.6, 0.7, 0.8, 0.9])
            enhancer = ImageEnhance.Brightness(img)
            img_trans = enhancer.enhance(factor)

        elif case == 3:
            # contrast
            factor = np.random.choice([0.5, 0.6, 0.7, 0.8, 0.9])
            enhancer = ImageEnhance.Contrast(img)
            img_trans = enhancer.enhance(factor)

        return img_trans

    @staticmethod
    def ncc(k1: np.ndarray, k2: np.ndarray) -> float:
        k1 = k1.copy().reshape(k1.shape[0] * k1.shape[1], -1)
        k2 = k2.copy().reshape(k2.shape[0] * k2.shape[1], -1)

        k1_norm = np.linalg.norm(k1)
        k2_norm = np.linalg.norm(k2)

        # ncc = np.matmul(k1, k2.T)
        _ncc = np.sum(k1 * k2, axis=0)
        _ncc = _ncc / (k1_norm * k2_norm + np.finfo(float).eps)

        return _ncc[0]

    def __len__(self):
        return len(self.patch_idx)

    def __getitem__(self, idx: int):
        cpi = self.patch_idx[idx]
        less_index_list = self.patch_idx_cum[self.patch_idx_cum <= idx]
        lil = less_index_list[-1] if len(less_index_list) != 0 else 0
        spi = idx - lil

        img = np.asarray(Image.open(self.current_db.loc[self.current_idx[cpi]].ProbePath)).astype(np.uint8)
        k = loadmat(self.prnu_db[self.prnu_db.Device == self.current_db.loc[self.current_idx[cpi]].Device].PrnuPath.item())['prnu']
        pe = PatchExtractor.PatchExtractor(dim=self.patch_size, stride=self.patch_stride, indexes=[spi])
        img = pe.extract(img)  # side effect: shape = (1,) + shape
        k = np.stack([k, k, k], axis=2)
        k = pe.extract(k)

        img = np.squeeze(img)
        img_gray = (color.rgb2gray(img)*255).astype(np.uint8)
        k = np.squeeze(k)
        k = k[:, :, 0]
        w_pre = extract_single(img_gray)

        ncc_pre = self.ncc(w_pre, k * img_gray)

        if self.transform_test:
            img = self._random_trans(Image.fromarray(img))

        img = np.asarray(img).astype(np.uint8)

        img_gray = (color.rgb2gray(img)*255).astype(np.uint8)
        w_post = extract_single(img_gray)
        ncc_post = self.ncc(w_post, k * img_gray)

        img = img.astype(np.float) / 255

        if self.transform_pre == 'wv':
            img_den = denoise_wavelet(img, multichannel=True)

            img = img_den

        elif self.transform_pre == 'wv_fft':
            img_den = denoise_wavelet(img, multichannel=True)
            img_noise = img - img_den
            img_noise_fft = np.abs((np.fft.fft2(img_noise, (self.patch_size[0] * 2,
                                                            self.patch_size[1] * 2),
                                                axes=[0, 1])))[:self.patch_size[0], :self.patch_size[1]]

            img = img_noise_fft

        elif self.transform_pre == 'wv_fft_wiener1':

            img_den = denoise_wavelet(img, multichannel=True)
            img_noise = img - img_den
            img_noise_fft = np.abs((np.fft.fft2(img_noise, (self.patch_size[0] * 2,
                                                            self.patch_size[1] * 2),
                                                axes=[0, 1])))[:self.patch_size[0], :self.patch_size[1]]

            W = np.zeros_like(img_noise_fft)

            for c in range(img_noise_fft.ndim):
                W_c = zero_mean_total(img_noise_fft[:, :, c])
                W_c_std = W_c.std(ddof=1)
                W[:, :, c] = wiener_adaptive(W_c, W_c_std ** 2).astype(np.float32)

            img = W

        elif self.transform_pre == 'wv_fft_wiener2':

            img_den = denoise_wavelet(img, multichannel=True)
            img_noise = img - img_den
            img_noise_fft = np.abs((np.fft.fft2(img_noise, (self.patch_size[0] * 2,
                                                            self.patch_size[1] * 2),
                                                axes=[0, 1])))[:self.patch_size[0], :self.patch_size[1]]

            W = np.zeros_like(img_noise_fft)

            for c in range(img_noise_fft.ndim):
                W_c = zero_mean_total(img_noise_fft[:, :, c])
                W_c_std = W_c.std(ddof=1)
                W[:, :, c] = wiener_adaptive(W_c, (W_c_std ** 2) * 0.77, window_size_list=[3]).astype(np.float32)

            img = W

        label = np.zeros((1,)).astype(np.uint8) if self.current_db.loc[self.current_idx[cpi]].Provenance == 'original' \
            else np.ones((1, )).astype(np.uint8)
        label = torch.from_numpy(label).float()

        img = torch.from_numpy(img.transpose((2, 0, 1))).float()

        if self.transform_post:
            img = self.transform_post(img)

        return img, label, ncc_pre, ncc_post

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


class Test1Trans(TestTransDB):
    def __init__(self, patch_size, subsample: float = None, transform_pre=None, transform_post=None, transform_test: str = None, patch_stride: tuple = None):
        super().__init__(patch_size=patch_size, subsample=subsample, transform_pre=transform_pre,
                         transform_post=transform_post, transform_test=transform_test, patch_stride=patch_stride)

        self.csv_path = os.path.join(db_path, 'orig_test.csv')
        self.prnu_csv_path = os.path.join(db_path, 'prnu.csv')
        self.original_db = pd.read_csv(self.csv_path)
        self.prnu_db = pd.read_csv(self.prnu_csv_path)

        if subsample:
            self.original_db = self.original_db.sample(frac=subsample, random_state=self.seed)

        if transform_pre:
            self.transform_pre = transform_pre

        if transform_post:
            self.transform_post = transform_post

        self.current_db = self.original_db
        self.current_idx = self.current_db.index
        super().compute_patch_idx_lenght()