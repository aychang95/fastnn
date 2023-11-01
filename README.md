# Fast Neural Networks (FastNN)

A framework for deploying serializable and optimizable neural net models at scale in production via. the NVIDIA Triton Inference Server.

<p align="center">
    <a href="https://hub.docker.com/r/aychang/fastnn">
        <img src="https://img.shields.io/docker/cloud/build/aychang/fastnn"
    </a>
    <a href="https://badge.fury.io/py/fastnn">
        <img src="https://badge.fury.io/py/fastnn.svg">
    </a>
    <a href="https://github.com/aychang95/fastnn/blob/master/LICENSE">
        <img src="https://img.shields.io/github/license/aychang95/fastnn">
    </a>
</p>

## [**FastNN Docker Release Selector (Ubuntu 18.04)**](https://andrewchang.dev/fastnn/index.html#fastnn-docker-release-selector-ubuntu-1804)

## [Documentation](https://andrewchang.dev/fastnn) - Guides, Models, API References

## Features:
  - **Data Processing**
    - Intuitive data processing modules for encoding human-readible data into tensors compatible with deep learning libraries
  - **Model Exporting**
    - FastNN torch modules and tools for exporting models via. `TorchScript` tracing and scripting to a production environment. Now includes Text Generation.
  - **Model Zoo**
    - Various exported models hosted in this repo via. git-lfs and AWS S3. Includes models from the HuggingFace's Transformers and 
    TorchVision
  - **Model Deployment**
    - Deploy models using Triton Inference Server on CPU/GPU-compatible server(s) with helm or docker
  - **FastNN Client**
    - Client wrapper for Triton Inference Server's client module for programmatic requests with python


## Pre-Requisites:

Git LFS is required if you'd like to use any of the models provided by FastNN in `./model_repository`.

Cloning this repository without Git LFS will clone a repository with LFS pointers, not the actual model.

After the repository is cloned and Git LFS is installed, set up with `git lfs install`.

Download specific models with:

```sh
git lfs pull --include="model_repository/<path-to-model-dir>" --exclude=""
```

Download ALL models with:

```sh
git lfs pull

# Or
#git lfs pull --include="*" --exclude=""
```


## Quickstart and Installation:

### *Pre-requisites:*

Requirements: Python 3.9+, PyTorch 2+, Triton Client

Optional: CUDA Compatible GPU, NVIDIA Drivers, cudnn (PyTorch pre-built wheels)

1. To install PyTorch with TorchVision, please refer to the installation instructions on their web page [here](https://pytorch.org/get-started/locally/#start-locally).

2. The tritonclient package wheels are not hosted on the public PyPI server. We need to add the address of NVIDA's private python package index to the environment. You can complete these steps and install the tritonclient package by running the following.

```sh
# If you cloned this repo, you can just uncomment and run the one line below
#sh ./scripts/install_triton_client.
pip install nvidia-pyindex
pip install tritonclient[all]
```

### **Install via. pip**

Once the above requirements are set, you can install fastnn with the command below:

```sh
pip install fastnn
```

If you are comfortable with the latest default stable releases of PyTorch you can skip step 1 in the pre-requisites and run `pip install fastnn[torch]` instead.


### **Install from source with Poetry for development**

You will need to install poetry by referring to the installation instructions on their web page [here](https://python-poetry.org/docs/#installation).

After cloning the repository, just run `poetry install` to install with the .lock file.

Activate the virtual environment with the command below:

```sh
poetry shell
```


### **Docker**

Official FastNN images are hosted on [Docker Hub](https://hub.docker.com/r/aychang/fastnn).

Select FastNN package and image versions by referencing the [documentation](https://andrewchang.dev/fastnn/index.html#fastnn-docker-release-selector-ubuntu-1804). Development and runtime environments are available.

Jupyter lab and notebook servers are accessible with notebook examples and terminal access `localhost:8888` with every image.


Run the latest FastNN image by running below:

```sh
docker run --gpus all --rm -it -p 8888:8888 aychang/fastnn:latest
```

Run images with specific configurations as can see in the example command below:

```sh
docker run --gpus all --rm -it -p 8888:8888 aychang/fastnn:0.1.0-cuda11.0-runtime-ubuntu18.04-py3.7

```


## Triton Inference Server: Local or Kubernetes Cluster


### **Local Deployment**

Requirements:
  - Docker 19.03+

GPU Inference Requirements:

  - NVIDIA CUDA-Compatible GPU
  
  - [NVIDIA Container Toolkit (nvidia-docker)](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html)

Local deployment of the Triton Server uses the EXPLICIT model control mode. Local models must be explicitly specified with the `--load-model` 
argument in `./scripts/start_triton_local.sh`

```sh

export MODEL_REPOSITORY=$(pwd)/model_repository
sh ./scripts/start_triton_local.sh

```


### **Helm Chart install in Kubernetes Cluster**

Requirements: kubectl 1.17+, Helm 3+, Kubernetes 1.17+

You can currently install the local FastNN helm chart with the following instuctions:

```sh

cd ./k8s
helm install fastnn .
export MODEL_REPOSITORY=$(pwd)/model_repository

```

Note: The current local helm chart installation deploys Triton using the NONE model control mode. All models available in the S3 Model Zoo will be deployed...good luck. 
Deployed models and model control mode can be edited in the helm chart deployment configuration file.

# License

This project is licensed under the terms of the MIT license.
