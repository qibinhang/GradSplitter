# Reusing Convolutional Neural Network Models through Modularization and Composition

## Introduction
This repository includes the source codes and experimental data in our paper entitled "Reusing Convolutional Neural Network Models through Modularization and Composition". 
It can be used to modularize a trained N-class CNN model into N smaller modules. 
Then the modules can be reused to (1) build new composed CNN models to satisfy the requirements of a new tasks, and (2) build more accurate CNN models for the original task.


## Requirements
+ argparse 1.4.0<br>
+ numpy 1.19.2<br>
+ python 3.8.10<br>
+ pytorch 1.8.1<br>
+ torchvision 0.9.0<br>
+ scikit-learn 0.22<br>
+ tqdm 4.61.0<br>
+ GPU with CUDA support is also needed


## How to install
Install the dependent packages via pip:

    $ pip install argparse==1.4.0 numpy==1.19.2 scikit-learn==0.22 tqdm==4.61.0
    
Install pytorch according to your environment, see https://pytorch.org/.


## Downloading experimental data
We provide the sixty trained CNN models and the corresponding modules. \
One can download `data/` from [here](https://mega.nz/folder/5dEh1KzL#NUiCn0I6U-fx7KRGnNGdcQ) and then reuse the modules to reproduce the results following the description below. 


## Training CNN models
1. modify `root_dir` in `src/global_configure.py`.
2. run `python run_train.py` in `script/` to train a set of CNN models.


## Modularizing trained CNN models
1. run `python run_splitter.py` in `script/` to modularize the trained CNN models.
2. run `python select_modules.py` in `script/` to select modules.


## Reusing modules to build more accurate CNN models
1. run `python run_evaluate_modules.py` in `script/` to evaluate all modules.
2. run `python run_module_reuse_for_accurate_model.py` in `script/` to reuse modules to build a more accurate CNN model.


## Reusing modules to build new CNN models for new tasks
1. run `python run_module_reuse_for_new_task.py` in `script/` to build a composed CNN model for the new task.
2. run `python run_train_model_for_new_task.py` in `script/` to train a new CNN model from scratch for the new task.

