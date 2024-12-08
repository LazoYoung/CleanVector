# Project
![Architecture](https://file.notion.so/f/f/930fdaf7-2edf-449c-a195-a8b9fdede966/352b02f4-d6f9-40b7-9479-d5985620a8e8/7a5e0b17-a84c-4045-903c-5ebd9eb149bb.png?table=block&id=1562655b-ecda-80bf-af2d-c60e5562896e&spaceId=930fdaf7-2edf-449c-a195-a8b9fdede966&expirationTimestamp=1733731200000&signature=gRMpQZg084w5xFgOpWjm0S5Ln53mKXv1qxgy6LX0V1c&downloadName=Diagram.png)

Diffusion-based vector image generator. Aims for a practical application.
[Click here to see more.](https://www.notion.so/parkcymil/14c2655becda80e8a041d63471a814ee)

## Prerequisite
- Python 3.9
- [CUDA Toolkit](https://developer.nvidia.com/cuda-toolkit)
- [NVIDIA cuDNN](https://developer.nvidia.com/cudnn)

## Install modules
If you don't use anaconda, use pip to install the modules. (Not recommended)

Make sure the CUDA Toolkit 12.4 is installed before you proceed!

```shell
conda install pytorch torchvision torchaudio pytorch-cuda=12.4 -c pytorch -c nvidia
conda install -c conda-forge accelerate
conda install -c conda-forge diffusers
conda install conda-forge::transformers
pip install xformers --index-url https://download.pytorch.org/whl/cu124
pip install compel sentencepiece protobuf nltk
```

## Update submodules
```shell
git submodule update --init --recursive
```

Refer [this document](https://github.com/BachiLi/diffvg?tab=readme-ov-file#install) to install and build DiffVG.


## Troubleshoot
- [If system fails to load libcudnn_cnn_infer.so.8 (WSL only)](https://github.com/microsoft/WSL/issues/8587)
