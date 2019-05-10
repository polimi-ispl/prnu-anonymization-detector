# PRNU ANONYMIZATION DETECTOR
IMAGE ANONYMIZATION DETECTION WITH DEEP HANDCRAFTED FEATURES (ICIP 2019)


## Prerequisites
+ Python 3.6
+ [Pytorch 0.4](https://pytorch.org/)
+ Numpy
+ Scipy
+ Pandas
+ PIL
+ Scikit-image
+ Scikit-learn
+ tqdm
+ [TensorboardX](https://github.com/lanpa/tensorboardX) (optional)
+ Jupyter (optional)
  
## Configuration
Change `root_path` in `params.py` according to your project root folder.

All the command below are meant to be run in your project root folder.

## Build db
```
python generate_db.py 
```

## Finetuning (training) a model
```
python finetuning.py --gpu 0 --db D1 --model resnet --num_workers 24 --batch_size 24 --transform_pre wv_fft_wiener2
```

This will generate a run folder under the folder specified in `runs_path` variable of `params.py`.
The run name is composed of all the train parameters value and a 6-digits alphanumeric code, `$RUN` from now on.

## Cross dataset testing
```
python test.py --gpu 0 --db TestOS --model resnet --num_workers 1 --batch_size 1 --transform_pre wv_fft_wiener2 --runs $RUN
``` 

## Leave one out testing
```
python test.py --gpu 0 --db D3_no_D200_0 --model resnet --num_workers 1 --batch_size 1 --transform_pre wv_fft_wiener2 --test_on_train --runs $RUN
``` 

## Transformation testing
```
python test.py --gpu 0 --db TestOS --model resnet --num_workers 1 --batch_size 1 --transform_pre wv_fft_wiener2 --transform_test --runs $RUN
``` 

## Paper plots
```
jupyter notebook notebook/paper_figures.ipynb
```
