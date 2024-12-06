<h1 align="center"> <a href=https://arxiv.org/abs/2308.07102>Temporal Sentence Grounding in Streaming Videos</a></h2>

## Introduction

Implementation of paper `Temporal Sentence Grounding in Streaming Videos` (Accepted by ACM Multimedia 2023).

## Installation

Please refer to [install.md](docs/install.md) for installation.

## Data Preparation

Please refer to [data_preparation.md](docs/data_preparation.md) for a general knowledge of data preparation.

## Reproduce the results
- ActivityNet Captions
```bash
# If you have 8 gpus
export CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7"
./tools/dist_train.sh configs/tpn/benchmark/anet_mtl_16_16_tpn_dec2_rnn_dot_s8_l64_b8*64_kd04.py 8 --validate --test-best

# If you have fewer gpus (like 2), you have three options:
# a) Increase batch size
export CUDA_VISIBLE_DEVICES="0,1"
./tools/dist_train.sh configs/tpn/benchmark/anet_mtl_16_16_tpn_dec2_rnn_dot_s8_l64_b8*64_kd04.py 2 --validate --test-best --cfg-options data.videos_per_gpu=256

# b) Decrease learning rate if GPU memory not enough for a)
export CUDA_VISIBLE_DEVICES="0,1"
./tools/dist_train.sh configs/tpn/benchmark/anet_mtl_16_16_tpn_dec2_rnn_dot_s8_l64_b8*64_kd04.py 2 --validate --test-best --cfg-options optimizer.lr=2.5e-6

# c) combine the a) and b)
```
- TACoS
```bash
export CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7"
./tools/dist_train.sh configs/tpn/benchmark/tacos_mtl_16_tpn_dec1_rnn_dot_s8_l64_b8*64_kd02.py 8 --validate --test-best
```
- MAD
```bash
# Train
export CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7"
./tools/dist_train.sh configs/tpn/benchmark/mad_mtl_16_16_tpn_dec2_rnn_dot_s8_l32_b8*128_kd02.py 8 --validate
# Test
export CUDA_VISIBLE_DEVICES="0,1"
for i in {0..19} # split the dataset into 20 chunks to avoid OOM
do
    ./tools/dist_test.sh work_dirs/mad_mtl_16_16_tpn_dec2_rnn_dot_s8_l32_b8*128_kd02/mad_mtl_16_16_tpn_dec2_rnn_dot_s8_l32_b8*128_kd02.py \
    work_dirs/mad_mtl_16_16_tpn_dec2_rnn_dot_s8_l32_b8*128_kd02/best_R@1,IoU=0.3_epoch_13.pth 2 --eval R@N,IoU=M \
    --out work_dirs/mad_mtl_16_16_tpn_dec2_rnn_dot_s8_l32_b8*128_kd02/best_pred_${i}_20.pkl \
    --cfg-options data.workers_per_gpu=10 data.test.portion=[$i,20]
done
```

## Citation
```
@inproceedings{tian2017tsgvs,
  title={{Temporal} {Sentence} in {Streaming} {Videos}},
  author={Tian Gan, Xiao Wang, Yan Sun, Jianlong Wu, Qingpei Guo, Liqiang Nie},
  booktitle={{International} {Conference} on {Multimedia}},
  publisher= {{ACM}},
  year={2023}
}
```
