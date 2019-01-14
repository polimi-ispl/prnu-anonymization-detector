import os
import numpy as np
import pandas as pd
from params import mandelli_path, kirchner_path, original_path, prnu_path


def main():
    np.random.seed(21)

    camera_names = os.listdir(original_path)
    idx_perm = np.random.permutation(len(os.listdir(os.path.join(original_path, camera_names[0]))))
    idx_set_1 = idx_perm[len(idx_perm) // 2:]
    idx_set_2 = idx_perm[:len(idx_perm) // 2 + 1]

    img_names = {}
    for c in camera_names:
        img_names[c] = os.listdir(os.path.join(original_path, c))

    pass

    img_orig_mand_train = []
    # img_orig_mand_test = []
    img_orig_kirc_train = []
    # img_orig_kirc_test = []
    img_orig_mand_kirch_train = []
    # img_orig_mand_kirch_test = []
    for c in camera_names:
        img_orig_mand_train += [np.asarray(map(lambda x: os.path.join(original_path, x), os.listdir(os.path.join(original_path, c))))[idx_set_1]]
        img_orig_mand_train += [np.asarray(os.listdir(os.path.join(mandelli_path, c)))[idx_set_1]]
        img_orig_kirc_train += [np.asarray(os.listdir(os.path.join(original_path, c)))[idx_set_1]]
        img_orig_kirc_train += [np.asarray(os.listdir(os.path.join(kirchner_path, c)))[idx_set_1]]
        img_orig_mand_kirch_train += [np.asarray(os.listdir(os.path.join(original_path, c)))[idx_set_1]]
        img_orig_mand_kirch_train += [np.asarray(os.listdir(os.path.join(mandelli_path, c)))[idx_set_1][:50]]
        img_orig_mand_kirch_train += [np.asarray(os.listdir(os.path.join(kirchner_path, c)))[idx_set_1][50:]]

    pass


if __name__ == '__main__':
    main()
