# Fast Neural Networks (FastNN)

A framework for deploying serializable and optimizable neural net models at scale in production via. the NVIDIA Triton Inference Server.


## **FastNN Docker Release Selector (Ubuntu 18.04)**
    
!!! info

    === "Stable (0.0.1)"
    
        === "Runtime"
    
            === "Python 3.7"
            
                === "CUDA 10.2"
    
                    `docker run --gpus all --rm -it -p 8888:8888 aychang95/fastnn:0.0.1-cuda10.2-runtime-ubuntu18.04-py3.7`
    
                === "CUDA 11.0"
    
                    `docker run --gpus all --rm -it -p 8888:8888 aychang95/fastnn:0.0.1-cuda11.0-runtime-ubuntu18.04-py3.7`
    
            === "Python 3.8"
    
                === "CUDA 10.2"
    
                    `docker run --gpus all --rm -it -p 8888:8888 aychang95/fastnn:0.0.1-cuda10.2-runtime-ubuntu18.04-py3.8`
    
                === "CUDA 11.0"
    
                    `docker run --gpus all --rm -it -p 8888:8888 aychang95/fastnn:0.0.1-cuda11.0-runtime-ubuntu18.04-py3.8`
    
        === "Development"
    
            === "Python 3.7"
            
                === "CUDA 10.2"
    
                    `docker run --gpus all --rm -it -p 8888:8888 aychang95/fastnn:0.0.1-cuda10.2-devel-ubuntu18.04-py3.7`
    
                === "CUDA 11.0"
    
                    `docker run --gpus all --rm -it -p 8888:8888 aychang95/fastnn:0.0.1-cuda11.0-devel-ubuntu18.04-py3.7`
    
            === "Python 3.8"
    
                === "CUDA 10.2"
    
                    `docker run --gpus all --rm -it -p 8888:8888 aychang95/fastnn:0.0.1-cuda10.2-devel-ubuntu18.04-py3.8`
    
                === "CUDA 11.0"
    
                    `docker run --gpus all --rm -it -p 8888:8888 aychang95/fastnn:0.0.1-cuda11.0-devel-ubuntu18.04-py3.8`
    
    === "Nightly"
    
        === "Runtime"
    
            === "Python 3.7"
            
                === "CUDA 10.2"
    
                    `docker run --gpus all --rm -it -p 8888:8888 aychang95/fastnn:latest-cuda10.2-runtime-ubuntu18.04-py3.7`
    
                === "CUDA 11.0"
    
                    `docker run --gpus all --rm -it -p 8888:8888 aychang95/fastnn:latest-cuda11.0-runtime-ubuntu18.04-py3.7`
    
            === "Python 3.8"
    
                === "CUDA 10.2"
    
                    `docker run --gpus all --rm -it -p 8888:8888 aychang95/fastnn:latest-cuda10.2-runtime-ubuntu18.04-py3.8`
    
                === "CUDA 11.0"
    
                    `docker run --gpus all --rm -it -p 8888:8888 aychang95/fastnn:latest-cuda11.0-runtime-ubuntu18.04-py3.8`
    
        === "Development"
    
            === "Python 3.7"
            
                === "CUDA 10.2"
    
                    `docker run --gpus all --rm -it -p 8888:8888 aychang95/fastnn:latest-cuda10.2-devel-ubuntu18.04-py3.7`
    
                === "CUDA 11.0"
    
                    `docker run --gpus all --rm -it -p 8888:8888 aychang95/fastnn:latest-cuda11.0-devel-ubuntu18.04-py3.7`
    
            === "Python 3.8"
    
                === "CUDA 10.2"
    
                    `docker run --gpus all --rm -it -p 8888:8888 aychang95/fastnn:latest-cuda10.2-devel-ubuntu18.04-py3.8`
    
                === "CUDA 11.0"
    
                    `docker run --gpus all --rm -it -p 8888:8888 aychang95/fastnn:latest-cuda11.0-devel-ubuntu18.04-py3.8`




## [Documentation](https://aychang95.github.com/fastnn) - Guides, Models, API References

## Features:
  - **Data Processing**
    - Intuitive data processing modules for encoding human-readible data into tensors compatible with deep learning libraries
  - **Model Exporting**
    - FastNN torch modules and tools for exporting models via. `TorchScript` tracing and scripting to a production environment
  - **Model Zoo**
    - Various exported models hosted in this repo via. git-lfs and AWS S3. Includes models from the HuggingFace's Transformers and 
    TorchVision
  - **Model Deployment**
    - Deploy models with CPU/GPU-compatible servers with Triton Inference Server
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


## Quickstart:

### **Install via. pip**

Requirements: Python 3.7+
Optional: CUDA Compatible GPU, NVIDIA Drivers, cudnn (PyTorch pre-built wheels)

Run the below in your command line.
```sh
# Uncomment if you want to add NVIDIA's python package index and install FastNN/Triton Inference Server client

#sh ./scripts/install_triton_client.py

pip install fastnn
```


### **Docker**

Images hosted on Docker Hub [here](https://hub.docker.com/r/aychang/fastnn).

Access notebooks and terminal on `localhost:8888`

```sh

docker run --gpus all --rm -it -p 8888:8888 aychang95/fastnn:0.0.1-cuda11.0-runtime-ubuntu18.04-py3.7

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


