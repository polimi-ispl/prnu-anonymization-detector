# IMAGE ANONYMIZATION DETECTION WITH DEEP HANDCRAFTED FEATURES
This is the original code implementation of the paper:

*N. Bonettini, D. Güera, L. Bondi, P. Bestagini, E.J. Delp, S. Tubaro,
"Image Anonymization Detection With Deep Handcrafted Features",
IEEE International Conference on Image Processing (ICIP), 2019*

Please, cite this if you use this code for your research.

## Clone the repository
PRNU functions are contained into the submodule [prnu](https://github.com/polimi-ispl/prnu-python/tree/ebf0ec76e1aea8683d76707011ee16b29eb0619a). In order to clone the repository with the submodule you need to add the `--recurse-submodules` flag to the `clone` command, for instance:

```
git clone --recurse-submodules https://github.com/polimi-ispl/prnu-anonymization-detector.git
```

Alternatively, you can clone the [prnu](https://github.com/polimi-ispl/prnu-python/tree/ebf0ec76e1aea8683d76707011ee16b29eb0619a) repository into the empty `prnu` folder.

## Prerequisites
+ Python 3.6
+ [Pytorch](https://pytorch.org/)
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
This will generate several .csv files in `data/db`, containing the paths
of the images in `data/dataset`. For convenience, we included a very
reduced subset of the images we considered for this work, alongside their
matching PRNUs. The original images are from
[Dresden dataset](http://forensics.inf.tu-dresden.de/ddimgdb/).

If you want to use your own set of images, just put them in `dataset` and
run `generate_db.py`.

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
