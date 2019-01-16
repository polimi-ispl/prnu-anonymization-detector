import os
import numpy as np
import pandas as pd
from params import mandelli_path, kirchner_path, original_path, db_path


def create_df(path_arr):
    df = pd.DataFrame(data=path_arr, columns=['ProbePath'])
    df['Device'] = df['ProbePath'].map(lambda x: x.split('/')[-1].rsplit('_', 1)[0])
    df['Provenance'] = df['ProbePath'].map(lambda x: x.split('/')[-2])
    return df


def main():
    np.random.seed(21)

    # split idx
    idx_perm = np.random.permutation(len(os.listdir(original_path)))
    idx_set_train = idx_perm[:len(idx_perm) // 2]
    idx_set_test = idx_perm[len(idx_perm) // 2:]

    # complete lists
    mand_path_list = np.asarray([os.path.join(mandelli_path, x) for x in os.listdir(mandelli_path)])
    kirch_path_list = np.asarray([os.path.join(kirchner_path, x) for x in os.listdir(kirchner_path)])
    orig_path_list = np.asarray([os.path.join(original_path, x) for x in os.listdir(original_path)])

    # train-test split
    orig_mand_path_list_train = np.concatenate([orig_path_list[idx_set_train], mand_path_list[idx_set_train]])
    orig_kirch_path_list_train = np.concatenate([orig_path_list[idx_set_train], kirch_path_list[idx_set_train]])
    orig_mand_kirch_path_list_train = np.concatenate([orig_path_list[idx_set_train],
                                                mand_path_list[idx_set_train[:len(idx_set_train) // 2]],
                                                kirch_path_list[idx_set_train[len(idx_set_train) // 2:]]])
    orig_path_list_test = orig_path_list[idx_set_test]
    mand_path_list_test = mand_path_list[idx_set_test]
    kirch_path_list_test = kirch_path_list[idx_set_test]
    orig_mand_path_list_test = np.concatenate([orig_path_list[idx_set_test], mand_path_list[idx_set_test]])
    orig_kirch_path_list_test = np.concatenate([orig_path_list[idx_set_test], kirch_path_list[idx_set_test]])

    # creating DFs
    orig_mand_df_train = create_df(orig_mand_path_list_train)
    orig_kirch_df_train = create_df(orig_kirch_path_list_train)
    orig_mand_kirch_df_train = create_df(orig_mand_kirch_path_list_train)
    orig_df_test = create_df(orig_path_list_test)
    mand_df_test = create_df(mand_path_list_test)
    kirch_df_test = create_df(kirch_path_list_test)
    orig_mand_df_test = create_df(orig_mand_path_list_test)
    orig_kirch_df_test = create_df(orig_kirch_path_list_test)

    # saving DFs
    orig_mand_df_train.to_csv(os.path.join(db_path, 'orig_mand_train.csv'), index=None)
    orig_kirch_df_train.to_csv(os.path.join(db_path, 'orig_kirch_train.csv'), index=None)
    orig_mand_kirch_df_train.to_csv(os.path.join(db_path, 'orig_mand_kirch_train.csv'), index=None)
    orig_df_test.to_csv(os.path.join(db_path, 'orig_test.csv'), index=None)
    mand_df_test.to_csv(os.path.join(db_path, 'mand_test.csv'), index=None)
    kirch_df_test.to_csv(os.path.join(db_path, 'kirch_test.csv'), index=None)
    orig_mand_df_test.to_csv(os.path.join(db_path, 'orig_mand_test.csv'), index=None)
    orig_kirch_df_test.to_csv(os.path.join(db_path, 'orig_kirch_test.csv'), index=None)

    # saving idx
    np.save(os.path.join(db_path, 'idx_train.npy'), idx_set_train)
    np.save(os.path.join(db_path, 'idx_test.npy'), idx_set_test)


if __name__ == '__main__':
    main()
