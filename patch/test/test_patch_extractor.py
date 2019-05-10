import numpy as np
import unittest
from PatchExtractor import PatchExtractor, mid_intensity_high_texture, count_patches, patch_array_shape
from PIL import Image

class TestPatchExtractor(unittest.TestCase):

    def test_init_with_rand(self):
        pe = PatchExtractor((24,24,24),
                            offset=(2,2,2),
                            stride=(8,8,8),
                            threshold=0.1,
                            rand=True)
        self.assertIsInstance(pe.dim, tuple)
        self.assertIsInstance(pe.offset, tuple)
        self.assertIsInstance(pe.stride, tuple)
        self.assertIsNone(pe.function_handler)
        self.assertIsInstance(pe.threshold, float)
        self.assertIsNone(pe.indexes)
        self.assertIsInstance(pe.rand, bool)

    def test_init_with_function(self):
        pe = PatchExtractor((24, 24, 24),
                            offset=(2, 2, 2),
                            stride=(8, 8, 8),
                            threshold=0.1,
                            function=mid_intensity_high_texture)
        self.assertIsInstance(pe.dim, tuple)
        self.assertIsInstance(pe.offset, tuple)
        self.assertIsInstance(pe.stride, tuple)
        self.assertEqual(pe.rand, False)
        self.assertIsInstance(pe.threshold, float)
        self.assertIsNone(pe.indexes)
        self.assertTrue(callable(pe.function_handler))

    def test_init_with_num(self):
        pe = PatchExtractor((24, 24, 24),
                            offset=(2, 2, 2),
                            stride=(8, 8, 8),
                            threshold=0.1,
                            function=mid_intensity_high_texture,
                            num=15)
        self.assertIsInstance(pe.dim, tuple)
        self.assertIsInstance(pe.offset, tuple)
        self.assertIsInstance(pe.stride, tuple)
        self.assertEqual(pe.rand, False)
        self.assertIsInstance(pe.threshold, float)
        self.assertIsNone(pe.indexes)
        self.assertTrue(callable(pe.function_handler))
        self.assertIsNone(pe.indexes)
        self.assertIsInstance(pe.num, int)

    def test_init_with_indexes(self):
        pe = PatchExtractor((24, 24, 24),
                            offset=(2, 2, 2),
                            stride=(8, 8, 8),
                            threshold=0.1,
                            function=mid_intensity_high_texture,
                            indexes=np.arange(0, 100, 3))
        self.assertIsInstance(pe.dim, tuple)
        self.assertIsInstance(pe.offset, tuple)
        self.assertIsInstance(pe.stride, tuple)
        self.assertEqual(pe.rand, False)
        self.assertIsInstance(pe.threshold, float)
        self.assertTrue(callable(pe.function_handler))
        self.assertIsNone(pe.num)
        self.assertIsInstance(pe.indexes, np.ndarray)

    def test_extract_output_shape_grayscale(self):
        im = np.asarray(Image.open('data/img_gray.png'))

        patch_shape = (32, 32)
        pe = PatchExtractor(patch_shape)
        patch = pe.extract(im)

        stride = (17, 17)
        pe_stride = PatchExtractor(patch_shape, stride=stride)
        patch_stride = pe_stride.extract(im)

        self.assertEqual(patch.shape, (16, 16, 32, 32))
        self.assertEqual(patch_stride.shape, (29, 29, 32, 32))

    def test_extract_output_shape_color(self):
        im = np.asarray(Image.open('data/img_color.png'))

        patch_shape = (32, 32, 2)
        pe = PatchExtractor(patch_shape)
        patch = pe.extract(im)

        stride = (17, 17, 2)
        pe_stride = PatchExtractor(patch_shape, stride=stride)
        patch_stride = pe_stride.extract(im)

        self.assertEqual(patch.shape, (16, 16, 1, 32, 32, 2))
        self.assertEqual(patch_stride.shape, (29, 29, 1, 32, 32, 2))

    def test_extract_output_shape_4d(self):
        im = np.load('data/img_4d.npy')

        patch_shape = (9, 32, 32, 3)
        pe = PatchExtractor(patch_shape)
        patch = pe.extract(im)

        stride = (7, 17, 17, 2)
        pe_stride = PatchExtractor(patch_shape, stride=stride)
        patch_stride = pe_stride.extract(im)

        self.assertEqual(patch.shape, (3, 2, 2, 1, 9, 32, 32, 3))
        self.assertEqual(patch_stride.shape, (4, 2, 2, 1, 9, 32, 32, 3))

    def test_extract_num(self):
        im = np.asarray(Image.open('data/img_color.png'))

        patch_shape = (32, 32, 2)
        num = 8

        pe_seq = PatchExtractor(patch_shape, num=num, rand=False)
        patch_seq = pe_seq.extract(im)

        pe_rand = PatchExtractor(patch_shape, num=num, rand=True)
        patch_rand = pe_rand.extract(im)

        self.assertEqual(patch_seq.ndim, 1 + im.ndim)
        self.assertEqual(patch_seq.shape[0], num)
        self.assertEqual(patch_rand.ndim, 1 + im.ndim)
        self.assertEqual(patch_rand.shape[0], num)

    def test_extract_indexes(self):
        im = np.asarray(Image.open('data/img_color.png'))

        patch_shape = (32, 32, 2)
        indexes = np.arange(0, 42, 3)

        pe = PatchExtractor(patch_shape, indexes=indexes)
        patch = pe.extract(im)

        self.assertEqual(patch.ndim, 1 + im.ndim)
        self.assertEqual(patch.shape[0], len(indexes))

    def test_extract_dtype(self):
        im = np.load('data/img_4d.npy').astype(np.float)

        patch_shape = (8, 15, 29, 2)
        patch_stride = (7, 26, 31, 3)
        pe_str = PatchExtractor(patch_shape, stride=patch_stride)
        patch_str = pe_str.extract(im)

        self.assertEqual(patch_str.dtype, patch_str.dtype)

    def test_mid_intensity_high_texture_threshold(self):
        im = np.asarray(Image.open('data/img_color.png'))

        patch_shape = (32, 32, 2)
        func = mid_intensity_high_texture
        threshold = 0.8
        pe = PatchExtractor(patch_shape, function=func, threshold=threshold)
        patch = pe.extract(im)

        p = patch[0]
        self.assertEqual(len(patch), 147)
        self.assertAlmostEqual(mid_intensity_high_texture(p), 0.926545988463008)

    def test_count_patches(self):
        im = np.asarray(Image.open('data/img_color.png'))

        patch_shape = (15, 29, 1)
        patch_stride = (34, 5, 2)

        pe = PatchExtractor(patch_shape, stride=patch_stride)
        patch = pe.extract(im)

        self.assertEqual(count_patches(im.shape, patch_shape, patch_stride), np.prod(patch.shape[:3]))

    def test_reconstruct_grayscale(self):
        im = np.asarray(Image.open('data/img_gray.png'))

        patch_shape = (15, 29)
        pe = PatchExtractor(patch_shape)
        patch = pe.extract(im)
        im_recon = pe.reconstruct(patch)

        patch_stride = (2, 3)
        pe_str = PatchExtractor(patch_shape, stride=patch_stride)
        patch_str = pe_str.extract(im)
        im_recon_str = pe_str.reconstruct(patch_str)

        im_recon_gt = np.asarray(Image.open('data/img_gray_recon_15_29.png'))
        im_recon_str_gt = np.asarray(Image.open('data/img_gray_recon_15_29_str_2_3.png'))
        self.assertTrue(np.allclose(im_recon_gt, im_recon))
        self.assertTrue(np.allclose(im_recon_str_gt, im_recon_str))

    def test_reconstruct_color(self):
        im = np.asarray(Image.open('data/img_color.png'))

        patch_shape = (15, 29, 2)
        pe = PatchExtractor(patch_shape)
        patch = pe.extract(im)
        im_recon = pe.reconstruct(patch)

        patch_stride = (2, 3, 1)
        pe_str = PatchExtractor(patch_shape, stride=patch_stride)
        patch_str = pe_str.extract(im)
        im_recon_str = pe_str.reconstruct(patch_str)

        im_recon_gt = np.asarray(Image.open('data/img_color_recon_15_29_2.png'))
        im_recon_str_gt = np.asarray(Image.open('data/img_color_recon_15_29_2_str_2_3_1.png'))
        self.assertTrue(np.allclose(im_recon_gt, im_recon))
        self.assertTrue(np.allclose(im_recon_str_gt, im_recon_str))

    def test_reconstruct_4d(self):
        im = np.load('data/img_4d.npy')

        patch_shape = (8, 15, 29, 2)
        pe = PatchExtractor(patch_shape)
        patch = pe.extract(im)
        im_recon = pe.reconstruct(patch)

        patch_stride = (7, 26, 31, 3)
        pe_str = PatchExtractor(patch_shape, stride=patch_stride)
        patch_str = pe_str.extract(im)
        im_recon_str = pe_str.reconstruct(patch_str)

        im_recon_gt = np.load('data/img_4d_recon_8_15_29_2.npy')
        im_recon_str_gt = np.load('data/img_4d_recon_8_15_29_2_str_7_26_31_1.npy')
        self.assertTrue(np.allclose(im_recon_gt, im_recon))
        self.assertTrue(np.allclose(im_recon_str_gt, im_recon_str, equal_nan=True))

    def test_reconstruct_dtype(self):
        im = np.load('data/img_4d.npy').astype(np.float)

        patch_shape = (8, 15, 29, 2)
        patch_stride = (7, 26, 31, 3)
        pe_str = PatchExtractor(patch_shape, stride=patch_stride)
        patch_str = pe_str.extract(im)
        im_recon_str = pe_str.reconstruct(patch_str)

        self.assertEqual(patch_str.dtype, im_recon_str.dtype)

    def test_patch_array_shape(self):
        im = np.asarray(Image.open('data/img_color.png'))

        patch_shape = (15, 29, 1)
        patch_stride = (34, 5, 2)

        pe = PatchExtractor(patch_shape, stride=patch_stride)
        patch = pe.extract(im)

        self.assertEqual(patch_array_shape(im.shape, patch_shape, patch_stride), patch.shape)
