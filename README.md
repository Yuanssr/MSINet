# MSINet: A Mask Structure Inference Network for Scene Text Image Super-Resolution

Official PyTorch implementation of **MSINet**, published in *IEEE Transactions
on Circuits and Systems for Video Technology (TCSVT), 2026*.

Runmin Wang, Shengrong Yuan, Shan Ye, Xingdong Song, Han Xu, Shengyou Qian,
Changxin Gao, and Nong Sang

[[Paper / DOI](https://doi.org/10.1109/TCSVT.2026.3667545)]

## Installation

The supplied environment is defined in `environment.yml`: Python 3.8,
PyTorch 1.13.1, torchvision 0.14.1, CUDA 11.7, and timm 0.3.2.

```bash
conda env create -f environment.yml
conda activate msinet
```

Alternatively, install a compatible PyTorch/torchvision build for your platform
and then run `pip install -r requirements.txt`.

## Dataset preparation

Obtain [TextZoom](https://github.com/JasonBoy1/TextZoom) from its official
source and keep its LMDB layout. The loader expects `num-samples`, `label-*`,
`image_hr-*`, and `image_lr-*` keys. Configure the following structure (paths
may be changed in YAML):

```text
data/TextZoom/
├── train1/
├── train2/
└── test/
    ├── easy/
    ├── medium/
    └── hard/
```

The six additional STR benchmarks reported in the paper use LMDB directories
for ICDAR 2013, ICDAR 2015, CUTE80, IIIT5K, SVT, and SVTP. They are not
redistributed here. Add their directories to `TRAIN.VAL.val_data_dir` only when
running the corresponding evaluation protocol.

## Model zoo and dependencies
Download the upstream initialization weights from their project pages and prepare the remaining
checkpoints as described below. The MSINet and TextZoom-adapted checkpoints are
available in the [Baidu Netdisk `ckpt` folder (extraction code: `f5aw`)](https://pan.baidu.com/s/1lZGkd8t85VFxCfVNJmJaew?pwd=f5aw).

| Artifact | Role | Download |
|---|---|---|
| Union14M MAE-S | Upstream initialization, ViT-S/4 | [Official Union14M Google Drive](https://drive.google.com/file/d/1F_wK7iAYzyz-T7-4D_dNQo1ORZ2Kesyw/view) |
| Union14M MAE-B | Upstream initialization, ViT-B/4 | [Official Union14M Google Drive](https://drive.google.com/file/d/1OSGZ7Jt7q4Pj3M9cUQznJpRDSDjzBVuy/view) |
| TextZoom-adapted MAE-S (`checkpoint-49.pth`) | Optional MAE adaptation | [Baidu Netdisk (`f5aw`)](https://pan.baidu.com/s/1lZGkd8t85VFxCfVNJmJaew?pwd=f5aw) |
| TextZoom-adapted MAE-B (`vit_base_checkpoint-49.pth`) | MAE initialization for MSINet-B training | [Baidu Netdisk (`f5aw`)](https://pan.baidu.com/s/1lZGkd8t85VFxCfVNJmJaew?pwd=f5aw) |
| MSINet-S (`masktextsr_small.pth`) | Final SR model, without TCM weights | [Baidu Netdisk (`f5aw`)](https://pan.baidu.com/s/1lZGkd8t85VFxCfVNJmJaew?pwd=f5aw) |
| MSINet-B (`msinet_base_no_tcm.pth`) | Final SR model, without TCM weights | [Baidu Netdisk (`f5aw`)](https://pan.baidu.com/s/1lZGkd8t85VFxCfVNJmJaew?pwd=f5aw) |
| ABINet | TCM confidence-map generator | [Official Google Drive](https://drive.google.com/file/d/1mYM_26qHUom_5NU7iutHneB_KHlLjL5y/view?usp=sharing) · [Official Baidu Netdisk, code `kwck`](https://pan.baidu.com/s/1b3vyvPwvh_75FkPlp87czQ) |
| Text-Focus recognizer | Training loss dependency | [Official LEMMA Google Drive](https://drive.google.com/file/d/1HRpzveBbnJPQn3-k_y2Y1YY4PcraWOFP/view?usp=drive_link) |
| CRNN | Evaluation recognizer | [Official CRNN Dropbox](https://www.dropbox.com/scl/fi/b5as2fmdfg2wnmepvf8vy/crnn.pth?rlkey=3j634rn1efnjoeav1hpk44a8t&dl=0) |
| MORAN | Optional evaluation recognizer | [Official repository](https://github.com/Canjie-Luo/MORAN_v2) |
| ASTER | Optional evaluation recognizer | [Official repository](https://github.com/ayumiymk/aster.pytorch) |
| ABINet | Optional evaluation recognizer | [Official repository](https://github.com/FangShancheng/ABINet) |
| MATRN | Optional evaluation recognizer | [Official repository](https://github.com/byeonghu-na/MATRN) |
| PARSeq | Optional evaluation recognizer | [Official repository](https://github.com/baudm/parseq) |
| TATT | Base SR initialization for MSINet-B training | [Baidu Netdisk (`f5aw`)](https://pan.baidu.com/s/1lZGkd8t85VFxCfVNJmJaew?pwd=f5aw) |

Place files under `weights/` as referenced by
`config/msinet_small.yaml` or `config/msinet_base.yaml`.

The released MSINet-S and MSINet-B checkpoints contain the MAE/SR weights and
omit `Confidence_generator.*`. TCM remains enabled by default and loads its
weights from the official ABINet checkpoint. Extract the ABINet `workdir.zip`
and place its files at the paths already specified in
`config/msinet_small.yaml` and `config/msinet_base.yaml`:

```text
weights/abinet/best-train-abinet.pth
weights/abinet/pretrain-vision-model.pth
weights/abinet/pretrain-language-model.pth
```

The corresponding `ConfidenceMapGenerator` settings in each YAML file are:

```yaml
ConfidenceMapGenerator:
  full_ckpt: "weights/abinet/best-train-abinet.pth"
  vision:
    checkpoint: "weights/abinet/pretrain-vision-model.pth"
  language:
    checkpoint: "weights/abinet/pretrain-language-model.pth"
```


## MAE weight preparation

The Union14M-U MAE recipe uses 32×128 RGB images, 4×4 patches, random 75%
masking, normalized-pixel reconstruction loss, AdamW, and base learning rate
1.5e-4. MSINet training starts from MAE checkpoints, so reproducing the full
Union14M-U pre-training run is optional.

The training recipe includes an additional self-supervised adaptation stage on
TextZoom HR training images. This stage is separate from MSINet SR training.
Commands and settings are documented in [`mae/README.md`](mae/README.md).

## Training

Edit dataset and checkpoint paths in the selected YAML, then run from any
directory:

```bash
python /path/to/MSINet/main.py \
  --config msinet_small.yaml \
  --mae-checkpoint /path/to/vit_small_checkpoint-19.pth \
  --output-dir /path/to/outputs \
  --seed 1234
```

For MSINet-B:

```bash
python /path/to/MSINet/main.py \
  --config msinet_base.yaml \
  --mae-checkpoint /path/to/vit_base_checkpoint-49.pth \
  --sr-checkpoint /path/to/TATT_best_acc.pth \
  --output-dir /path/to/outputs \
  --seed 1234
```

The supplied configurations use AdamW, a 1e-3 main parameter-group learning
rate, a 1e-5 MAE/base-SR group learning rate, batch size 16 for MSINet-S, and
batch size 8 for MSINet-B. Both configurations specify 500 training epochs.

Use `--checkpoint` to initialize model weights from an existing checkpoint.

## Evaluation

The evaluation loader performs tensor-only safe loading and requires a complete
checkpoint with exact state-dict keys. Run, for example:

```bash
python main.py --config msinet_small.yaml --test \
  --checkpoint /path/to/msinet_small.pth \
  --textzoom-root /path/to/TextZoom \
  --recognizer-checkpoint /path/to/crnn.pth \
  --test_model CRNN --seed 1234
```

For MSINet-B:

```bash
python main.py --config msinet_base.yaml --test \
  --checkpoint /path/to/msinet_base.pth \
  --textzoom-root /path/to/TextZoom \
  --recognizer-checkpoint /path/to/crnn.pth \
  --test_model CRNN --seed 1234
```

Replace `CRNN` with `MORAN`, `ASTER`, `ABINet`, `MATRN`, or `PARSeq` when the
corresponding implementation and checkpoint are available. TextZoom Overall is
sample-weighted using 1,619 Easy, 1,411 Medium, and 1,343 Hard images; it is not
the arithmetic mean of the three subset accuracies. PSNR and SSIM use the same
split weighting in the current evaluation code.

## Main results

The following TextZoom recognition and image-quality results are reported in
the paper.

| Model | Recognizer | Easy | Medium | Hard | Overall |
|---|---|---:|---:|---:|---:|
| MSINet-S | CRNN | 71.4 | 61.9 | 49.0 | 61.5 |
| MSINet-S | MORAN | 82.0 | 67.5 | 52.5 | 68.3 |
| MSINet-S | ASTER | 84.6 | 70.5 | 54.9 | 70.9 |
| MSINet-B | CRNN | 72.1 | 64.8 | 52.3 | 63.7 |
| MSINet-B | MORAN | 83.1 | 69.4 | 55.9 | 70.3 |
| MSINet-B | ASTER | 85.9 | 71.4 | 58.2 | 72.7 |

| Model | Easy PSNR/SSIM | Medium PSNR/SSIM | Hard PSNR/SSIM | Overall PSNR/SSIM |
|---|---:|---:|---:|---:|
| MSINet-S | 24.56 / 0.8994 | 19.76 / 0.7010 | 20.75 / 0.7809 | 21.84 / 0.7990 |
| MSINet-B | 25.11 / 0.9055 | 20.49 / 0.7232 | 21.43 / 0.7970 | 22.49 / 0.8100 |


## Acknowledgements

This codebase builds on [Union14M / MAERec](https://github.com/Mountchicken/Union14M)
for masked autoencoder pre-training and [LEMMA](https://github.com/csguoh/LEMMA)
for the STISR training/evaluation framework. It also incorporates or interfaces
with [MAE](https://github.com/facebookresearch/mae),
[TextZoom](https://github.com/JasonBoy1/TextZoom), TSRN, STT, MORAN, ASTER,
CRNN, ABINet, MATRN, and PARSeq. See `THIRD_PARTY_NOTICES.md` for component
attributions and license terms.

## Citation

```bibtex
@article{wang2026msinet,
  author  = {Runmin Wang and Shengrong Yuan and Shan Ye and Xingdong Song and
             Han Xu and Shengyou Qian and Changxin Gao and Nong Sang},
  title   = {{MSINet}: A Mask Structure Inference Network for Scene Text Image
             Super-Resolution},
  journal = {IEEE Transactions on Circuits and Systems for Video Technology},
  year    = {2026},
  volume  = {36},
  number  = {6},
  pages   = {7894--7908},
  doi     = {10.1109/TCSVT.2026.3667545}
}
```

## License

MSINet-authored code is released under Apache-2.0; see `LICENSE`. Included
third-party components retain their respective licenses and attribution
notices. See `THIRD_PARTY_NOTICES.md` for component scope and license links.
Checkpoint and dataset terms are separate from the source-code licenses.
