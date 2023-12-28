# EZ-FastChat

## 1. Installation
```
conda create -n EZFastChat python=3.9
conda activate EZFastChat
pip install -r requirements.txt
pip install torch torchvision torchaudio
cd fastchat
pip install "fschat[model_worker,webui]"
```

[cuda12.2](https://developer.nvidia.com/cuda-12-2-0-download-archive) and [cuDNN](https://developer.nvidia.com/rdp/cudnn-download) is required. 
Please install the cuDNN following its [installation guide](https://docs.nvidia.com/deeplearning/cudnn/install-guide/index.html#install-linux)


## 2. Data
Refer to the [guide](https://github.com/puar-playground/EZ-FastChat/tree/main/data).
