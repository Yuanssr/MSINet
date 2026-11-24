import yaml
import argparse
import json
import os
import random

import numpy as np
import torch
from easydict import EasyDict


def main(config, args):
    # Keep heavyweight model imports out of argument parsing so that
    # ``python main.py --help`` works without initializing recognizers.
    from interfaces.super_resolution import TextSR

    Mission = TextSR(config, args)
    if args.test:
        Mission.test()
    else:
        Mission.train()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Train or evaluate MSINet.')
    parser.add_argument('--config', type=str, default='msinet_small.yaml',
                        help='YAML file name under config/ or an absolute path.')
    parser.add_argument('--test', action='store_true', default=False)
    parser.add_argument('--STN', dest='STN', action='store_true', default=True,
                        help='Enable the legacy TPS spatial transformer (default).')
    parser.add_argument('--no-stn', dest='STN', action='store_false',
                        help='Disable the legacy TPS spatial transformer.')
    parser.add_argument('--srb', type=int, default=5,
                        help='Legacy SRB argument retained for checkpoint compatibility.')
    parser.add_argument('--mask', dest='mask', action='store_true', default=True,
                        help='Enable the fourth-channel text mask (default).')
    parser.add_argument('--no-mask', dest='mask', action='store_false',
                        help='Disable the fourth-channel text mask.')
    parser.add_argument('--test_model', type=str, default='CRNN', choices=['ASTER', "CRNN", "MORAN",'ABINet','MATRN','PARSeq'])
    parser.add_argument('--learning_rate', type=float, default=None,
                        help='Override TRAIN.lr from the YAML file.')
    parser.add_argument('--lr_position', type=float, default=1e-5,
                        help='Learning rate for the MAE and base SR parameter groups.')
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Override TRAIN.resume (model checkpoint or warm start).')
    parser.add_argument('--mae-checkpoint', type=str, default=None,
                        help='Override MaskRefine.mae_vit.checkpoint (MAE initialization).')
    parser.add_argument('--sr-checkpoint', type=str, default=None,
                        help='Override Pretrain_SR_Model.checkpoint (base STISR initialization).')
    parser.add_argument('--output-dir', type=str, default=None,
                        help='Override TRAIN.ckpt_dir.')
    parser.add_argument('--textzoom-root', type=str, default=None,
                        help='TextZoom root containing train1, train2, and test/.')
    parser.add_argument('--recognizer-checkpoint', type=str, default=None,
                        help='Override the selected evaluation recognizer checkpoint.')
    parser.add_argument('--seed', type=int, default=None,
                        help='Set Python, NumPy, and PyTorch random seeds.')
    parser.add_argument('--print-config', action='store_true',
                        help='Validate, print the resolved configuration, and exit.')
    args = parser.parse_args()

    project_root = os.path.dirname(os.path.abspath(__file__))
    config_path = args.config if os.path.isabs(args.config) else os.path.join(project_root, 'config', args.config)
    # The legacy implementation resolves charset files and several model
    # resources relative to the repository root.
    os.chdir(project_root)
    with open(config_path, 'r', encoding='utf-8') as stream:
        config = yaml.safe_load(stream)
    config = EasyDict(config)

    if args.learning_rate is not None:
        config.TRAIN.lr = args.learning_rate
    if args.checkpoint is not None:
        config.TRAIN.resume = args.checkpoint
    if args.mae_checkpoint is not None:
        config.MaskRefine.mae_vit.checkpoint = args.mae_checkpoint
    if args.sr_checkpoint is not None:
        config.Pretrain_SR_Model.checkpoint = args.sr_checkpoint
    if args.output_dir is not None:
        config.TRAIN.ckpt_dir = args.output_dir
    if args.textzoom_root is not None:
        root = os.path.abspath(args.textzoom_root)
        config.TRAIN.train_data_dir = [os.path.join(root, 'train1'), os.path.join(root, 'train2')]
        config.TRAIN.VAL.val_data_dir = [
            os.path.join(root, 'test', 'easy'),
            os.path.join(root, 'test', 'medium'),
            os.path.join(root, 'test', 'hard'),
        ]
    if args.recognizer_checkpoint is not None:
        if args.test_model == 'CRNN':
            config.TRAIN.VAL.crnn_pretrained = args.recognizer_checkpoint
        elif args.test_model == 'MORAN':
            config.TRAIN.VAL.moran_pretrained = args.recognizer_checkpoint
        elif args.test_model == 'ASTER':
            config.TRAIN.VAL.rec_pretrained = args.recognizer_checkpoint
        else:
            config[args.test_model].full_ckpt = args.recognizer_checkpoint

    # A final MSINet checkpoint contains the MAE and SR branches. If it does not
    # contain the TCM weights, initialize those from the configured ABINet
    # checkpoint before loading the MSINet weights.
    if args.test and config.TRAIN.resume:
        config.MaskRefine.mae_vit.checkpoint = None
        config.Pretrain_SR_Model.checkpoint = ''
    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)

    if args.print_config:
        plain_config = json.loads(json.dumps(config))
        print(yaml.safe_dump(plain_config, sort_keys=False))
        raise SystemExit(0)

    from setup import Logger
    Logger.init('logs', 'MaskTextSR', 'train')
    Logger.enable_file()
    main(config, args)
