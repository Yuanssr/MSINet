# MAE pre-training and TextZoom adaptation

This directory contains the MAE pre-training implementation used by MSINet.
The model operates on 32×128 RGB text images with 4×4 patches and predicts
masked image patches.

## Union14M-U pre-training

The pre-training recipe uses random 75% masking, normalized pixel targets,
AdamW, and base learning rate 1.5e-4. Example command for four GPUs:

```bash
python -m torch.distributed.launch --nproc_per_node=4 mae/main_pretrain.py \
  --model mae_vit_small_patch4 --batch_size 256 --epochs 20 \
  --mask_ratio 0.75 --norm_pix_loss --blr 1.5e-4 \
  --data_path /path/to/Union14M-U/openvino /path/to/Union14M-U/book32 \
              /path/to/Union14M-U/CC \
  --output_dir outputs/mae_small --log_dir outputs/mae_small
```

For MSINet training, users may initialize from the released Union14M
checkpoints linked in the main README instead of running this full pre-training
stage.

## TextZoom adaptation

TextZoom adaptation uses the HR images in `train1` and `train2` as both MAE
input and reconstruction target. It is a self-supervised stage that precedes
MSINet SR training. The following commands continue from the Union14M MAE
checkpoint at epoch 19 and write the adapted checkpoint at epoch 49.

Small variant:

```bash
python mae/main_pretrain.py \
  --model mae_vit_small_patch4 --batch_size 64 --epochs 50 \
  --start_epoch 20 --mask_ratio 0.75 --norm_pix_loss --blr 1.5e-4 \
  --resume /path/to/vit_small_checkpoint-19.pth \
  --data_path /path/to/TextZoom/train1 /path/to/TextZoom/train2 \
  --output_dir outputs/mae_small_textzoom \
  --log_dir outputs/mae_small_textzoom
```

With one process, the effective learning rate is 3.75e-5.

Base variant:

```bash
python mae/main_pretrain.py \
  --model mae_vit_base_patch4 --batch_size 32 --epochs 50 \
  --start_epoch 20 --mask_ratio 0.75 --blr 1.5e-4 \
  --resume /path/to/vit_base_checkpoint-19.pth \
  --data_path /path/to/TextZoom/train1 /path/to/TextZoom/train2 \
  --output_dir outputs/mae_base_textzoom \
  --log_dir outputs/mae_base_textzoom
```

Use `outputs/mae_base_textzoom/checkpoint-49.pth` to initialize MSINet-B and
`outputs/mae_small_textzoom/checkpoint-49.pth` when choosing the optional
TextZoom-adapted MAE-S initialization. The MSINet-S configuration can also be
initialized directly from the Union14M MAE-S checkpoint.
