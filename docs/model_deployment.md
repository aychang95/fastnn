# Deploy Models with Triton Inference Server

**Pre-requisites:**

Clone the fastnn repository.

Add NVIDIA's python package index and install FastNN/Triton Inference Server client

`sh ./scripts/install_triton_client.py`


## **Local Deployment**

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


## **Helm Chart install in Kubernetes Cluster**

Requirements: kubectl 1.17+, Helm 3+, Kubernetes 1.17+

You can currently install the local FastNN helm chart with the following instuctions:

```sh

cd ./k8s
helm install fastnn .
export MODEL_REPOSITORY=$(pwd)/model_repository

```

Note: The current local helm chart installation deploys Triton using the NONE model control mode. All models available in the S3 Model Zoo will be deployed...good luck. 
Deployed models and model control mode can be edited in the helm chart deployment configuration file.