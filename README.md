# Mirror-NeRF: Learning Neural Radiance Fields for Mirrors with Whitted-Style Ray Tracing

### [Project Page](https://zju3dv.github.io/Mirror-NeRF/) | [Video](https://youtu.be/gyGK9wnU0tc) | [Paper](https://arxiv.org/pdf/2308.03280.pdf)
<div align=center>
<img src="https://github.com/zjy-zju/open_access_assets/raw/main/Mirror-NeRF/teaser.gif" width="100%"/>
</div>

> [Mirror-NeRF: Learning Neural Radiance Fields for Mirrors with Whitted-Style Ray Tracing](https://arxiv.org/pdf/2308.03280.pdf)  
> 
> [[Junyi Zeng](https://zjy-zju.github.io/), [Chong Bao](https://chobao.github.io/)<sup>Co-Authors</sup>], [Rui Chen](https://github.com/rabbitchenrui), [Zilong Dong](https://scholar.google.com/citations?user=GHOQKCwAAAAJ&hl=en&oi=ao), [Guofeng Zhang](http://www.cad.zju.edu.cn/home/gfzhang/), [Hujun Bao](http://www.cad.zju.edu.cn/home/bao/), [Zhaopeng Cui](https://zhpcui.github.io/). 
> 
> ACM Multimedia 2023
> 


## Hardware

* OS: Ubuntu 18.04
* NVIDIA GPU with **CUDA>=11.1** (tested with 1 RTX3090)


## Installation

We have tested the code on Python 3.8.0 and PyTorch 1.8.1, while a newer version of pytorch should also work. 

The steps of installation are as follows:

* Clone this repo by `git clone --recursive https://github.com/zju3dv/Mirror-NeRF`
* Python>=3.8 (installation via [Anaconda](https://www.anaconda.com/) is recommended, use `conda create -n mirror_nerf python=3.8` to create a conda environment and activate it by `conda activate mirror_nerf`)
* Python libraries
    * Install requirements by `pip install -r requirements.txt`
    * Install PyTorch 1.8.1 by `pip install torch==1.8.1+cu111 torchvision==0.9.1+cu111  -f https://download.pytorch.org/whl/torch_stable.html`
* [Optional] If you want to use [tiny-cuda-nn](https://github.com/NVlabs/tiny-cuda-nn) for acceleration, install [tiny-cuda-nn PyTorch extension](https://github.com/NVlabs/tiny-cuda-nn#pytorch-extension) by `pip install git+https://github.com/NVlabs/tiny-cuda-nn/#subdirectory=bindings/torch`


## Data

We support synthetic datasets (`datasets/blender.py`) and real datasets (`datasets/real_arkit.py`).

For customized datasets with camera poses reconstructed by [COLMAP](https://colmap.github.io/), you can refer to `datasets/real_colmap.py`.

### Data download

Download our captured synthetic and real **datasets** from [here](https://drive.google.com/drive/folders/1fBeIAI64v5GRoIo7eceGnbqS0sF07YN_?usp=sharing).

Download our **pretrained models** on the synthetic and real datasets from [here](https://drive.google.com/drive/folders/1BFAMPw1T64es-o-2rMzBNbj01VxJFF4b?usp=sharing).

> Notes: For pretrained models with "tcnn" in the filename, select `MODEL_TYPE="nerf_tcnn"` in `run.sh`.

For pretrained models of D-NeRF, please refer to [D-NeRF](https://github.com/albertpumarola/D-NeRF#download-pre-trained-weights) repository.


## Running

We integrate training, testing and applications in one script `run.sh`,

```
bash run.sh {MODE} {GPU_ID}
```

`MODE`: 

> `1` for evaluation (Novel View Synthesis), 
> 
> `2` for extracting mesh, 
> 
> `3` for placing new mirrors (application), 
> 
> `4` for reflecting newly placed objects (application),
> 
> `5` or `52` for controlling mirror roughness (application), 
> 
> `6` for reflection substitution (application),
> 
> other numbers like `0` for training.


**Please configure the settings in `run.sh` before running.**

For example, choose the `DATASET` and `MODEL_TYPE`.

> Notes: For acceleration, use `MODEL_TYPE="nerf_tcnn"`.
> 
> For scenes with accurate camera poses (like synthetic scenes), `MODEL_TYPE="nerf"` is recommended.
> 
> For some real captures with inaccurate camera poses, `MODEL_TYPE="nerf_tcnn"` is recommended.

For evaluation and applications, specify the `LOG` (necessary), `SUBSTITUTION_LOG` (for reflection substitution) and `OBJ_CKPT_PATH` (for reflecting newly placed objects).

**For more configurations, see `opt.py`.**

The results of training will be automatically stored in the `log/` directory.

The results of evaluation and application will be automatically stored in the `results/` directory.


## Citing
```
@inproceedings{zeng2023mirror-nerf,
    title={Mirror-NeRF: Learning Neural Radiance Fields for Mirrors with Whitted-Style Ray Tracing},
    author={Zeng, Junyi and Bao, Chong and Chen, Rui and Dong, Zilong and Zhang, Guofeng and Bao, Hujun and Cui, Zhaopeng},
    booktitle={Proceedings of the 31st ACM International Conference on Multimedia},
    pages={4606--4615},
    year={2023}
}
```

## Acknowledgement

This repository is developed upon [nerf_pl](https://github.com/kwea123/nerf_pl).

And we use parts of the code of [D-NeRF](https://github.com/albertpumarola/D-NeRF) for object models used in the application of reflecting newly placed objects, and parts of the code of [torch-ngp](https://github.com/ashawkey/torch-ngp) for using tiny-cuda-nn if acceleration is needed.

Thanks for these great projects!