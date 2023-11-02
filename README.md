# Reusing Convolutional Neural Network Models through Modularization and Composition

## Introduction
With the widespread success of deep learning technologies, many trained deep neural network (DNN) models are now publicly available. However, directly reusing the public DNN models for new tasks often fails due to mismatching functionality or performance. Inspired by the notion of modularization and composition in software reuse, we investigate the possibility of improving the reusability of DNN models in a more fine-grained manner. Specifically, we propose two modularization approaches named CNNSplitter and GradSplitter, which can decompose a trained convolutional neural network (CNN) model for N-class classification into N small reusable modules. Each module recognizes one of the N classes and contains a part of the convolution kernels of the trained CNN model. Then, the resulting modules can be reused to patch existing CNN models or build new CNN models through composition. The main difference between CNNSplitter and GradSplitter lies in their search methods: the former relies on a genetic algorithm to explore search space, while the latter utilizes a gradient-based search method. Our experiments with three representative CNNs on three widely-used public datasets demonstrate the effectiveness of the proposed approaches. Compared with CNNSplitter, GradSplitter incurs less accuracy loss, produces much smaller modules (19.88% fewer kernels), and achieves better results on patching weak models. In particular, experiments on GradSplitter show that (1) by patching weak models, the average improvement in terms of precision, recall, and F1-score is 17.13%, 4.95%, and 11.47%, respectively, and (2) for a new task, compared with the models trained from scratch, reusing modules achieves similar accuracy (the average loss of accuracy is only 2.46%) without a costly training process. Our approaches provide a viable solution to the rapid development and improvement of CNN models.


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
One can download `data/` from [here](https://mega.nz/folder/ADMjESyC#LkCOzE0qVHs8DOXkN3l_WA) and then reuse the modules to reproduce the results following the description below. 


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
