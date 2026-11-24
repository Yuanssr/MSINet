# Third-Party Notices

The Apache-2.0 license in the repository root applies to original MSINet code.
It does not replace the licenses below. Copyright and license notices for
third-party code are retained in this repository.

## LEMMA

- Source: [csguoh/LEMMA](https://github.com/csguoh/LEMMA)
- License: Apache-2.0; the upstream license text is retained as `LICENSE.txt`.
- Incorporated framework code includes the STISR training/evaluation interfaces,
  dataset and utility code, and related model components adapted from LEMMA.

## Union14M / MAERec

- Source: [Mountchicken/Union14M](https://github.com/Mountchicken/Union14M)
- License: MIT; see `third_party/Union14M_LICENSE.txt`.
- Incorporated code: `mae/datasets/lmdb_dataset.py` and the Union14M-adapted
  MAE implementation in `model/models_mae.py`.
- The Union14M repository also incorporates code from Facebook Research MAE;
  MAE-derived portions are identified separately below and retain the MAE
  license terms.

## Facebook Research MAE

- Source: [facebookresearch/mae](https://github.com/facebookresearch/mae)
- License: CC BY-NC 4.0. The [license text and conditions](https://creativecommons.org/licenses/by-nc/4.0/legalcode) are available from Creative Commons.
- MAE-derived files in this repository include `mae/models_mae.py`,
  `mae/main_pretrain.py`, `mae/engine_pretrain.py`, `mae/util/`,
  `mae/visualize.py`, and the MAE portions of `model/models_mae.py` and
  `utils/pos_embed.py`. These files retain attribution to Facebook Research MAE.
- Modifications were made for the MSINet/Union14M image size, training recipe,
  and model integration. The CC BY-NC 4.0 terms apply to these MAE-derived
  portions; they are not relicensed by the root Apache-2.0 license.

## ABINet

- Source: [FangShancheng/ABINet](https://github.com/FangShancheng/ABINet)
- License: the upstream BSD 2-Clause license, titled “ABINet for non-commercial
  purposes”; a copy is retained as `third_party/ABINet_LICENSE.txt`.
- Incorporated code: `model/ABINet/` and ABINet-derived portions of
  `model/confidence_map_generator.py`. The upstream project describes the code
  as intended for academic research/non-commercial use; see its
  [README](https://github.com/FangShancheng/ABINet#license).
- ABINet model checkpoints are separate artifacts and are not covered by this
  source-code license notice.

## Scene text recognizers

- [MORAN](https://github.com/Canjie-Luo/MORAN_v2): MIT; license text retained
  as `third_party/MORAN_LICENSE.txt`. The upstream README states the software
  is for academic research purposes.
- [ASTER](https://github.com/ayumiymk/aster.pytorch): MIT, copyright Mingkun
  Yang; license text retained as `third_party/ASTER_LICENSE.txt`. The upstream
  README notes its academic-research intent and asks product users to contact
  the authors about possible patent issues.
- [MATRN](https://github.com/byeonghu-na/MATRN): MIT, copyright wp03052;
  license text retained as `third_party/MATRN_LICENSE.txt`.
- [PARSeq](https://github.com/baudm/parseq): Apache-2.0 for the upstream
  project’s majority code. The upstream project gives separate licenses for
  bundled ABINet/CRNN models; this repository’s copied PARSeq modules are
  attributed to PARSeq, while ABINet and CRNN files follow their own notices.
- [CRNN / NAVER](https://github.com/clovaai/deep-text-recognition-benchmark):
  Apache-2.0. The source header is retained in `model/crnn/model.py`.

## Facebook DETR transformer

- Source: [facebookresearch/detr](https://github.com/facebookresearch/detr)
- License: Apache-2.0.
- `model/transformer_v2.py` retains its Facebook copyright notice and DETR
  attribution.

## Other upstream code

- The MPN-COV source file carried an “All rights reserved” notice without a
  redistribution license and is not used by the released training/evaluation
  code. It is excluded from the current source snapshot; its local original is
  preserved outside the Git index.
- Checkpoints and datasets have separate terms. This repository does not grant
  redistribution rights to third-party weights or datasets.

## License scope

`LICENSE` and `CITATION.cff` identify Apache-2.0 as the license for original
MSINet-authored code. Files identified above remain subject to their listed
upstream licenses and notices. Users of those files must follow the applicable
terms for each component.
