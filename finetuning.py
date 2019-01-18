import os
import random
import string
import numpy as np
import pandas as pd
import argparse
from glob import glob
from tqdm import tqdm
from torchvision import transforms, models
from torchvision.models.resnet import model_urls as resnet_model_urls
from torchvision.models.alexnet import model_urls as alexnet_model_urls
from torchvision.models.vgg import model_urls as vgg_model_urls
from torch.utils.data import DataLoader
from torch.utils import model_zoo
import torch
from tensorboardX import SummaryWriter
import warnings


from params import db_path, default_batch_size, default_lr, default_n_epochs, default_num_workers, default_subsample, \
                   default_patch_size, default_train_size, model_path, default_patch_stride, runs_path, \
                   log_period_iter, log_start_iter, early_stop
import data.db_classes as db_classes

warnings.filterwarnings("ignore")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', help='GPU to use (0 based)', required=False, default=0, type=int)
    parser.add_argument('--db', help='Database to use during training', required=True, type=str)
    parser.add_argument('--transform_pre', help='Apply transformation before casting to Tensor', type=str)
    parser.add_argument('--model', help='Network model to be trained', required=True, type=str)
    parser.add_argument('--subsample', help='Fraction of image to keep subsampling db', type=float)
    parser.add_argument('--patch_size', help='Patch size', type=int)
    parser.add_argument('--patch_stride', help='Patch stride', type=int)
    parser.add_argument('--num_workers', help='Parallel job in data loading', required=False, type=int)
    parser.add_argument('--batch_size', help='Training image batch size', required=False, type=int)
    parser.add_argument('--lr', help='Learning rate', required=False, type=float)
    parser.add_argument('--n_epochs', help='Number of training epochs', required=False, type=int)
    parser.add_argument('--continue_train', help='resume training from best epoch of #### run', type=str)
    parser.add_argument('--debug', help='Debug flag for visualization', required=False, action='store_true')

    args = parser.parse_args()

    device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() else 'cpu')
    db_name = args.db
    transform_pre = args.transform_pre
    model_name = args.model
    subsample = args.subsample if args.subsample is not None else default_subsample
    patch_size = args.patch_size if args.patch_size is not None else default_patch_size
    patch_stride = args.patch_stride if args.patch_stride is not None else default_patch_stride
    num_workers = args.num_workers if args.num_workers is not None else default_num_workers
    batch_size = args.batch_size if args.batch_size is not None else default_batch_size
    lr = args.lr if args.lr is not None else default_lr
    n_epochs = args.n_epochs if args.n_epochs is not None else default_n_epochs
    continue_train_run = args.continue_train
    debug = args.debug

    # transform function as needed from torch model zoo
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    # Inizialize database and dataloader
    db_class = getattr(db_classes, db_name)
    db = db_class(patch_size=patch_size, patch_stride=patch_stride, transform_pre=transform_pre, transform_post=normalize, subsample=subsample)
    db.generate_split(train_size=default_train_size)
    dl_train = DataLoader(db, batch_size=batch_size, num_workers=num_workers, shuffle=True, drop_last=True)

    # initialize network
    print('Loading network weights')
    model = None
    if model_name == 'resnet':
        model = models.resnet18(pretrained=False)
        model.load_state_dict(model_zoo.load_url(resnet_model_urls['resnet18'], model_dir=model_path))

        # craft network last layer to be a two-class classifier
        num_ftrs = model.fc.in_features
        model.fc = torch.nn.Linear(num_ftrs, 1)

        model = ResNet(BasicBlock, [2, 2, 2])
        model.load_state_dict(model_zoo.load_url(resnet_model_urls['resnet18'], model_dir=model_path), strict=False)

    elif model_name == 'alexnet':
        model = models.alexnet(pretrained=False)
        model.load_state_dict(model_zoo.load_url(alexnet_model_urls['alexnet'], model_dir=model_path))

        # craft network last layer to be a two-class classifier
        num_ftrs = model.classifier._modules['6'].in_features
        model.classifier._modules['6'] = torch.nn.Linear(num_ftrs, 1)

    elif model_name == 'vgg':
        # model = models.vgg16_bn(pretrained=False, init_weights=False)
        # model.load_state_dict(model_zoo.load_url(vgg_model_urls['vgg16'], model_dir=model_path))
        model = models.vgg16_bn(pretrained=True)

        # craft network last layer to be a two-class classifier
        num_ftrs = model.classifier._modules['6'].in_features
        model.classifier._modules['6'] = torch.nn.Linear(num_ftrs, 1)

    # Load weights
    if continue_train_run is not None:
        run_folder = glob(os.path.join(runs_path, '*-{}'.format(continue_train_run)))[0]
        try:
            model.load_state_dict(torch.load(os.path.join(run_folder, 'model_best.pth')))
        except FileNotFoundError:
            print('No models weights found in {}.\nTraining from scratch...'.format(run_folder))

    # craft network last layer to be a two-class classifier
    # num_ftrs = model.fc.in_features
    # model.fc = torch.nn.Linear(num_ftrs, np.prod(patch_size[:2]))
    model.to(device)

    # define criterion
    criterion = torch.nn.BCELoss()
    s = torch.nn.Sigmoid()

    # define optimizer and lr decay
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

    if debug:
        # Prepare Tensorboard writer
        writer_dir = os.path.join(runs_path,
                                  model.__class__.__name__ + '_' +
                                  '{}'.format(db.__class__.__name__) + '_' +
                                  'transform_{}'.format(transform_pre) + '_' +
                                  'subsample_{}'.format(subsample) + '_' +
                                  'patch_{}'.format(patch_size) + '_' +
                                  'stride_{}'.format(patch_stride) + '_' +
                                  'batch_{}'.format(batch_size) + '_' +
                                  'lr_{}'.format(lr) + '_' +
                                  'nepochs_{}'.format(n_epochs))
        run_name = [random.choice(string.ascii_letters + string.digits) for _ in range(6)]
        writer_dir += '-' + ''.join(run_name)
        while os.path.exists(writer_dir):
            run_name = [random.choice(string.ascii_letters + string.digits) for _ in range(6)]
            writer_dir = writer_dir.rsplit('-', 1)[0]
            writer_dir += '-' + ''.join(run_name)
        writer = SummaryWriter(writer_dir)
        db.train_db.to_csv(os.path.join(writer_dir, 'train.csv'))
        db.val_db.to_csv(os.path.join(writer_dir, 'val.csv'))
        print('\n\n')
        print('Finetuning model {} on db {}, run {}'.format(model.__class__.__name__, db.__class__.__name__,
                                                                    ''.join(run_name)))
        if transform_pre:
            print('\nApply preprocessing {}'.format(transform_pre))

        print('\n\n')

    min_val_loss = np.inf
    early_stop_counter = 0
    early_stop_flag = False

    for epoch in range(n_epochs):
        if not early_stop_flag:
            # -----------------
            #  Train
            # -----------------
            db.train()
            lr_scheduler.step()
            model.train()
            running_loss_train = 0
            for i_batch, sample_batched in tqdm(enumerate(dl_train), desc='Train Epoch {}'.format(epoch + 1),
                                                total=len(dl_train), unit='batch'):
                overall_iter = int(i_batch + epoch * len(dl_train))

                # load data
                X = sample_batched[0].to(device)
                y = sample_batched[1].to(device)

                # zero the gradients
                optimizer.zero_grad()

                # forward
                y_hat = s(model(X))
                loss = criterion(y_hat, y.view(batch_size, -1))

                # backward
                loss.backward()
                optimizer.step()

                # statistics
                running_loss_train += loss.item()

                if debug and overall_iter % log_period_iter == log_start_iter:
                    writer.add_scalar('loss/train', loss.item() , overall_iter)

            epoch_loss_train = running_loss_train / len(db)
            print('Train Loss: {:.4f}'.format(epoch_loss_train))
            # ---------------------
            #  Validation
            # ---------------------
            db.val()
            model.eval()
            dl_val = DataLoader(db, batch_size=batch_size, num_workers=num_workers, shuffle=True)
            running_loss_val = 0
            for i_batch, sample_batched in tqdm(enumerate(dl_val), desc='Val Epoch {}'.format(epoch + 1),
                                                total=len(dl_val), unit='batch'):

                # load data
                X = sample_batched[0].to(device)
                y = sample_batched[1].to(device)

                # forward
                y_hat = s(model(X))
                loss = criterion(y_hat, y.view(batch_size, -1))

                # statistics
                running_loss_val += loss.item()

            epoch_loss_val = running_loss_val / len(db)
            print('Val Loss: {:.4f}'.format(epoch_loss_val))

            if debug:
                writer.add_scalar('loss/val', epoch_loss_val, overall_iter)
                writer.add_image('X', X[0].detach(), overall_iter)
                # writer.add_image('y', y[0].detach().view(patch_size[0], patch_size[1]))
                # writer.add_image('y_hat', y_hat[0].detach().view(patch_size[0], patch_size[1]))

            if min_val_loss - epoch_loss_val > 1e-4:
                if min_val_loss == np.inf:
                    print('Val_loss {}. \nSaving models...'.format(epoch_loss_val))
                else:
                    print('Val_loss improved by {0:.6f}. \nSaving models...'.format(min_val_loss - epoch_loss_val))
                torch.save(model.state_dict(), os.path.join(writer_dir, 'model_best.pth'))
                min_val_loss = epoch_loss_val

                # save results to csv
                train_results = pd.DataFrame({'model': model.__class__.__name__,
                                              'subsample': subsample,
                                              'patch': patch_size[0],
                                              'stride': patch_stride[0],
                                              'batch': batch_size,
                                              'lr': lr,
                                              'val_loss': epoch_loss_val,
                                              'epoch': epoch + 1,
                                              'run_name': ''.join(run_name)}, index=[0])
                train_results.to_csv(os.path.join(writer_dir, 'train_results.csv'))
                early_stop_counter = 0

            else:
                early_stop_counter += 1

            if debug:
                writer.add_scalar('Epoch', epoch + 1, overall_iter)

            if early_stop_counter == early_stop:
                early_stop_flag = True
                print('\nEarly stopping due to non-decreasing validation loss for {} epochs\n'.format(early_stop))

    return 0


if __name__ == '__main__':
    main()
