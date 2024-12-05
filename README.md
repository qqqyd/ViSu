# Boosting Semi-Supervised Scene Text Recognition via Viewing and Summarizing

## Introduction
This is a pytorch implementation for paper [ViSu](https://arxiv.org/abs/2411.15585).

## Installation

### Requirements
- Python==3.9
- Pytorch==1.10.1
- CUDA==11.1

```bash
git clone https://github.com/qqqyd/ViSu.git
cd ViSu/

conda create --name visu python=3.9 -y
conda activate visu
pip install torch==1.10.1+cu111 torchvision==0.11.2+cu111 -f https://download.pytorch.org/whl/cu111/torch_stable.html
pip install mmcv-full==1.7.0 -f https://download.openmmlab.com/mmcv/dist/cu111/torch1.10/index.html
pip install -r requirements.txt
```

## Training
* Put the synthetic and evaluation datasets in ```datasets/rec_data/``` and Union14M in ```datasets/Union14M/```. You can download synthetic and evaluation datasets [here](https://github.com/baudm/parseq) and download Union14M [here](https://github.com/Mountchicken/Union14M). The structure of the datasets folder is shown below.

```
datasets
├── rec_data
│   ├── test
│   │   ├── ArT
│   │   ├── COCOv1.4
│   │   ├── CUTE80
│   │   ├── IC13_1015
│   │   ├── IC13_1095
│   │   ├── IC13_857
│   │   ├── IC15_1811
│   │   ├── IC15_2077
│   │   ├── IIIT5k
│   │   ├── SVT
│   │   ├── SVTP
│   │   ├── Uber
│   │   ├── WordArt
│   └── train
│       ├── real
│       └── synth
└── Union14M
    ├── Union14M-L
    │   ├── train_lmdb
    │   └── Union14M-Benchmarks
    └── Union14M-U
        ├── boo32_lmdb
        ├── cc_lmdb
        └── openvino_lmdb

```

* Download the [fonts](https://drive.google.com/drive/folders/1ZlKC7u0aHwAGQ_Oloe8U-bNk_t5LJmx_) and put them in ```fonts/```.

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nnodes=1 --nproc_per_node=4 --master_addr 127.0.0.1 --master_port 20001 train.py --config configs/config.py --workdir output/workdir
```

## Testing

Pretrained model is available [here](https://drive.google.com/drive/folders/1ZlKC7u0aHwAGQ_Oloe8U-bNk_t5LJmx_).

```bash
python test.py --config configs/config.py --checkpoint path/to/checkpoint --with_proj
```
