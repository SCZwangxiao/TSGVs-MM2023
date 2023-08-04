# Installation

We provide some tips for OpenTSGV installation in this file.

<!-- TOC -->

- [Installation](#installation)
  - [Requirements](#requirements)
  - [Prepare environment](#prepare-environment)
  - [Install OpenTSGV](#install-opentsgv)

<!-- TOC -->

## Requirements

- Linux
- Python 3.7.9
- PyTorch 1.10.2
- Torchvision 0.11.3
- CUDA 11.4
- GCC 5+
- [mmcv](https://github.com/open-mmlab/mmcv) (Version is depended on cuda and pytorch versions)
- Numpy
- Scipy
- Transformers
- future
- Tensorboard
- h5py

> **_NOTE:_**
You need to run `pip uninstall mmcv` first if you have mmcv installed.
If mmcv and mmcv-full are both installed, there will be `ModuleNotFoundError`.

## Prepare environment

a. Create a conda virtual environment and activate it.

```shell
conda create -n tsgv python=3.7 -y
conda activate tsgv
```

b. Install PyTorch and torchvision following the [official instructions](https://pytorch.org/), e.g.,

```shell
conda install pytorch torchvision -c pytorch
```

> **_NOTE:_**
Make sure that your compilation CUDA version and runtime CUDA version match.
You can check the supported CUDA version for precompiled packages on the [PyTorch website](https://pytorch.org/).

`E.g.1` If you have CUDA 10.1 installed under `/usr/local/cuda` and would like to install PyTorch 1.5,
you need to install the prebuilt PyTorch with CUDA 10.1.

```shell
conda install pytorch cudatoolkit=10.1 torchvision -c pytorch
```

`E.g.2` If you have CUDA 9.2 installed under `/usr/local/cuda` and would like to install PyTorch 1.3.1.,
you need to install the prebuilt PyTorch with CUDA 9.2.

```shell
conda install pytorch=1.3.1 cudatoolkit=9.2 torchvision=0.4.2 -c pytorch
```

If you build PyTorch from source instead of installing the prebuilt package, you can use more CUDA versions such as 9.0.
:::

## Install OpenTSGV

You can install OpenTSGV manually:

a. Install mmcv-full, we recommend you to install the pre-built package as below.

```shell
# pip install mmcv-full -f https://download.openmmlab.com/mmcv/dist/{cu_version}/{torch_version}/index.html
pip install mmcv-full -f https://download.openmmlab.com/mmcv/dist/cu102/torch1.10.0/index.html
```

mmcv-full is only compiled on PyTorch 1.x.0 because the compatibility usually holds between 1.x.0 and 1.x.1. If your PyTorch version is 1.x.1, you can install mmcv-full compiled with PyTorch 1.x.0 and it usually works well.

```
# We can ignore the micro version of PyTorch
pip install mmcv-full -f https://download.openmmlab.com/mmcv/dist/cu102/torch1.10/index.html
```

See [here](https://github.com/open-mmlab/mmcv#installation) for different versions of MMCV compatible to different PyTorch and CUDA versions.

Optionally you can choose to compile mmcv from source by the following command

```shell
git clone https://github.com/open-mmlab/mmcv.git
cd mmcv
MMCV_WITH_OPS=1 pip install -e .  # package mmcv-full, which contains cuda ops, will be installed after this step
# OR pip install -e .  # package mmcv, which contains no cuda ops, will be installed after this step
cd ..
```

Or directly run

```shell
pip install mmcv-full
# alternative: pip install mmcv
```

**Important:** You need to run `pip uninstall mmcv` first if you have mmcv installed. If mmcv and mmcv-full are both installed, there will be `ModuleNotFoundError`.

b. Clone the OpenTSGV repository.

```shell
git clone **
cd **
```

c. Install build requirements and then install OpenTSGV.

```shell
pip install -r requirements/build.txt
pip install -v -e .  # or "python setup.py develop"
```

> **_NOTE:_**
The git commit id will be written to the version number with step b, e.g. 0.6.0+2e7045c. The version will also be saved in trained models.
   It is recommended that you run step b each time you pull some updates from github. If C++/CUDA codes are modified, then this step is compulsory.