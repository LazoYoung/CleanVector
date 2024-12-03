# Setup project
## Prerequisite
- Python 3.10
- [CUDA Toolkit](https://developer.nvidia.com/cuda-toolkit)
- [NVIDIA cuDNN](https://developer.nvidia.com/cudnn)

## Install modules
If you don't use anaconda, use pip to install the modules. (Not recommended)

Make sure the CUDA Toolkit 12.4 is installed before you proceed!

```shell
conda install pytorch torchvision torchaudio pytorch-cuda=12.4 -c pytorch -c nvidia
pip install accelerate diffusers["torch"] transformers
pip install xformers --index-url https://download.pytorch.org/whl/cu124
```

## Update submodules
```shell
git submodule init
git submodule update
```

## Troubleshoot
- [If system fails to load libcudnn_cnn_infer.so.8 (WSL only)](https://github.com/microsoft/WSL/issues/8587)
