PSA: I will update the project to fix outstanding issues by 1Q2024.maintenance issues are due to lack of nvidia gpu (for tensorflow) 


 PassGAN Evaluation
=======================

This is the release code repository for the evaluation of PassGAN(https://arxiv.org/pdf/1709.00440.pdf) built using Tensorflow 2, Python 3.7, Keras and Numpy to the described specification.
It contains my Tensorflow 2 implementation of an Improved Wasserstein GAN (IWGAN) with the intent of comparing the results found in the aformentioned paper.
GCP has the fastest cold start time roughly taking 10minutes from start to train depending on the dataset download speed. Simply grab the container in a VM, install Python 3.7, the latest pip (20 and above) and install requirements. You should be ready to go. Training time on an 80% dataset takes almost a week on a V100, but is characteristically IWGAN stable.


## Getting Started

python GAN.py -dataset rock_you -batch_size 64 -layer_dim 128 && tensorboard --log_dir logs/gradient_tape

### Prerequisites

System Software

```
Ubuntu 19.10+ or suitable Docker environment https://www.docker.com/get-started
```

```
TENSORFLOW-GPU 2.1 https://www.tensorflow.org/
```

```
Jetbrains-Pycharm or equivalent
```

### Installing
The easiest way to get started

```
Setup environment according to the TENSORFLOW setup document included: Ubuntu required
```

```
Pull the repository
```

```
Import packages via Pycharm packet manager
```

```
Run project with python GAN.py to pull dataset and begin training
```
