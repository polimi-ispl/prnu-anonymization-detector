import os
from glob import glob
import numpy as np
import argparse
from tqdm import tqdm
from torchvision import transforms, models
from torch.utils.data import DataLoader
import torch
import warnings

from params import default_batch_size, default_num_workers, default_subsample, \
                   default_patch_size, default_train_size, default_patch_stride, runs_path, \
    results_path
import db_classes as db_classes

warnings.filterwarnings("ignore")
torch.multiprocessing.set_sharing_strategy('file_system')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', help='GPU to use (0 based)', required=False, default=0, type=int)
    parser.add_argument('--db', help='Database to use during training', required=True, type=str)
    parser.add_argument('--transform_pre', help='Apply transformation before casting to Tensor', type=str)
    parser.add_argument('--transform_test', help='Transformation to be applied only in test phase', action='store_true')
    parser.add_argument('--model', help='Network model to be trained', required=True, type=str)
    parser.add_argument('--runs', nargs='+', help='Run code for selected models (one or more)', required=True, type=str)
    parser.add_argument('--subsample', help='Fraction of image to keep subsampling db', type=float)
    parser.add_argument('--patch_size', help='Patch size', type=int)
    parser.add_argument('--patch_stride', help='Patch stride', type=int)
    parser.add_argument('--num_workers', help='Parallel job in data loading', required=False, type=int)
    parser.add_argument('--batch_size', help='Training image batch size', required=False, type=int)
    parser.add_argument('--test_on_train', action='store_true')
    parser.add_argument('--debug', help='Debug flag for visualization', required=False, action='store_true')

    args = parser.parse_args()

    device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() else 'cpu')
    db_name = args.db
    transform_pre = args.transform_pre
    transform_test = args.transform_test
    model_name = args.model
    run_list = args.runs
    subsample = args.subsample if args.subsample is not None else default_subsample
    patch_size = args.patch_size if args.patch_size is not None else default_patch_size
    patch_stride = args.patch_stride if args.patch_stride is not None else default_patch_stride
    num_workers = args.num_workers if args.num_workers is not None else default_num_workers
    batch_size = args.batch_size if args.batch_size is not None else default_batch_size
    test_on_train_flag = args.test_on_train
    debug = args.debug

    # transform function as needed from torch model zoo
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    # Inizialize database and dataloader
    db_class = getattr(db_classes, db_name)

    if test_on_train_flag:
        db = db_class(patch_size=patch_size, patch_stride=patch_stride, transform_pre=transform_pre,
                      transform_post=normalize, subsample=subsample)
        db.generate_split(train_size=default_train_size)
        db.train()
    else:
        db = db_class(patch_size=patch_size, patch_stride=patch_stride, transform_pre=transform_pre,
                      transform_post=normalize, transform_test=transform_test, subsample=subsample)

    dl_test = DataLoader(db, batch_size=batch_size, num_workers=num_workers, shuffle=False, drop_last=True)

    # initialize network
    print('Loading network weights')
    model = None
    if model_name == 'resnet':
        model = models.resnet18(pretrained=False)
        # craft network last layer to be a two-class classifier
        num_ftrs = model.fc.in_features
        model.fc = torch.nn.Linear(num_ftrs, 1)

    elif model_name == 'alexnet':
        model = models.alexnet(pretrained=False)
        # craft network last layer to be a two-class classifier
        num_ftrs = model.classifier._modules['6'].in_features
        model.classifier._modules['6'] = torch.nn.Linear(num_ftrs, 1)

    elif model_name == 'vgg':
        model = models.vgg16_bn(pretrained=False)
        # craft network last layer to be a two-class classifier
        num_ftrs = model.classifier._modules['6'].in_features
        model.classifier._modules['6'] = torch.nn.Linear(num_ftrs, 1)

    model.to(device)
    s = torch.nn.Sigmoid()

    for run_name in run_list:
        print('\n\n')
        print('Testing model {} with dataset {}'.format(run_name, db.__class__.__name__))
        if transform_pre:
            print('\nApply preprocessing {}'.format(transform_pre))
        if transform_test:
            print('\nApply random test preprocessing')

        print('\n\n')

        # Load weights
        run_folder = glob(os.path.join(runs_path, '*-{}'.format(run_name)))[0]
        try:
            model.load_state_dict(torch.load(os.path.join(run_folder, 'model_best.pth')))
        except FileNotFoundError:
            print('No model weights found in {}.\nExiting'.format(run_folder))
            return 1
        label = []
        pred = []
        ncc_pre = []
        ncc_post = []
        for i_batch, sample_batched in tqdm(enumerate(dl_test), desc='Testing',
                                            total=len(dl_test), unit='batch'):
            # load data
            X = sample_batched[0].to(device)
            y = sample_batched[1].to(device)
            ncc_pre_batch = sample_batched[2]
            ncc_post_batch = sample_batched[3]

            y_hat = s(model(X))

            label += [y.cpu().numpy()]
            pred += [y_hat.detach().cpu().numpy()]
            ncc_pre += [ncc_pre_batch.numpy()]
            ncc_post += [ncc_post_batch.numpy()]

        label = np.squeeze(np.asarray(label)).reshape(-1, 1)
        pred = np.squeeze(np.asarray(pred)).reshape(-1, 1)
        ncc_pre = np.squeeze(np.asarray(ncc_pre)).reshape(-1, 1)
        ncc_post = np.squeeze(np.asarray(ncc_post)).reshape(-1, 1)

        # fpr, tpr, thr = roc_curve(label, pred, pos_label=0)

        # if debug:
        #     print('FPR: {} \n TPR: {}\n'.format(fpr, tpr))

        # Save result to npy
        res_folder_name = os.path.join(results_path, run_name + '_' + db.__class__.__name__)
        if transform_test:
            res_folder_name += '_trans'
        os.makedirs(res_folder_name, exist_ok=True)
        np.save(os.path.join(res_folder_name, 'result.npy'),
                {'label': label,
                 'pred': pred,
                 'ncc_pre': ncc_pre,
                 'ncc_post': ncc_post})

    return 0


if __name__ == '__main__':
    main()
