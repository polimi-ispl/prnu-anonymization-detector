import os

# paths
root_path = os.getcwd()
dataset_path = os.path.join(root_path, 'dataset')
mandelli_path = os.path.join(dataset_path, 'mandelli')
kirchner_path = os.path.join(dataset_path, 'kirchner')
original_path = os.path.join(dataset_path, 'original')
prnu_path = os.path.join(dataset_path, 'prnu')
data_path = os.path.join(root_path, 'data')
db_path = os.path.join(data_path, 'db')
model_path = os.path.join(data_path, 'model')
runs_path = os.path.join(data_path, 'runs')
results_path = os.path.join(data_path, 'results')

# default params
default_subsample = 1
default_patch_size = (224, 224, 3)
default_patch_stride = (32, 32, 3)
default_num_workers = 40
default_batch_size = 48
default_lr = 0.01
default_n_epochs = 30
default_train_size = 0.7
early_stop = 10

# visualization params
log_period_iter = 7
log_start_iter = 1