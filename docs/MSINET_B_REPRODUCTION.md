# MSINet-B CRNN reproduction

This document records an end-to-end MSINet-B evaluation with the code and
protocol provided in this repository. The MSINet checkpoint omits TCM weights;
the ABINet checkpoints are loaded separately through the YAML configuration.

## Protocol

- Model: MSINet-B with TATT base SR branch and ABINet TCM.
- MSINet checkpoint: 660 state-dict entries; the 455 `Confidence_generator.*`
  entries are supplied by the separately loaded ABINet checkpoint.
- ABINet initialization: vision, language, and full-model checkpoints from the
  official ABINet `workdir.zip` package.
- TCM ratios: `[0.4, 0.3, 0.2, 0.1]` (66/256 masked patches after integer
  truncation).
- Recognizer: CRNN.
- Dataset: TextZoom Easy (1,619), Medium (1,411), Hard (1,343).
- Overall aggregation: sample-count weighted.
- Seed: 1234.
- Hardware used for this check: one NVIDIA GeForce RTX 3080.

```bash
python main.py --config msinet_base.yaml --test \
  --checkpoint /path/to/msinet_base.pth \
  --textzoom-root /path/to/TextZoom \
  --recognizer-checkpoint /path/to/crnn.pth \
  --test_model CRNN --seed 1234
```

## Results

| Source | Easy | Medium | Hard | Overall |
|---|---:|---:|---:|---:|
| Paper | 72.1 | 64.8 | 52.3 | 63.7 |
| Reproduced | 73.13 | 65.49 | 52.64 | 64.372 |

| Source | Easy PSNR/SSIM | Medium PSNR/SSIM | Hard PSNR/SSIM | Overall PSNR/SSIM |
|---|---:|---:|---:|---:|
| Paper | 25.11 / 0.9055 | 20.49 / 0.7232 | 21.43 / 0.7970 | 22.49 / 0.8100 |
| Reproduced | 25.13 / 0.9058 | 20.50 / 0.7233 | 21.41 / 0.7971 | 22.4931 / 0.8135 |

The reproduced overall accuracy is 0.672 percentage points above the paper
result. TCM samples patches randomly within confidence strata, so the seed is
included in the evaluation command.
