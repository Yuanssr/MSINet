import logging
import torch
from model.models_mae import mae_vit_small_patch4_dec512d8b as mae_vit_small
from model.models_mae import mae_vit_base_patch4_dec512d8b as mae_vit_base
from utils.visualize import visualize_tensor,unpatchify
from torch import nn
import torch.nn.functional as F
import yaml
import os
from easydict import EasyDict

import torch
import torch.nn as nn


class MaskRefineModule(nn.Module):
    '''
        input:
        [N,3,H,W]

        output:
        refine_sr[N,3,H,W],
        restruction[N,3,H,W],
        mask[N,L]

    '''

    def __init__(self, config, freeze_mae=False):
        super(MaskRefineModule, self).__init__()
        self.freeze_mae=freeze_mae
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if config.mae_vit.type =="base":
            self.mae = mae_vit_base().to(self.device)
        else:
            self.mae = mae_vit_small().to(self.device)
        if config.mae_vit.checkpoint is not None:
            logging.info(f'Read mae_vit model from {config.mae_vit.checkpoint}.')
            self.load(config.mae_vit.checkpoint)

        self.SR_Module = LightWeightSR(config.srb_nums).to(self.device)

        if self.freeze_mae:
            self.mae_freeze()

    def load(self, source, device=None, strict=True):
        state = torch.load(source, map_location=device)
        self.mae.load_state_dict(state['model'], strict=strict)

    def mae_freeze(self):
        for param in self.mae.parameters():
            param.requires_grad = False
    
    def forward(self, sr_images,text_prob=None):
        if self.freeze_mae:
            with torch.no_grad():  # 冻结模型参数
                restruction, mask, _ = self.mae(sr_images,text_prob)
        else:
            restruction, mask, _ = self.mae(sr_images,text_prob)
        restruction = unpatchify(restruction)
        refine_sr = self.SR_Module(restruction)
        return refine_sr, restruction, mask


class LightWeightSR(nn.Module):
    '''
    input:[N,3,H,W]

    output:[N,3,H,W]
    '''
    def __init__(self, srb_nums,channel=64,hidden_units=32,in_planes = 3):
        super(LightWeightSR, self).__init__()
        self.block1 = nn.Sequential(
            nn.Conv2d(in_planes, 2 * hidden_units, kernel_size=9, padding=4),
            nn.PReLU()
        )
        self.srb_nums=srb_nums
        for i in range(srb_nums):
            setattr(self, 'block%d' % (i+2), RecurrentResidualBlock(2 * hidden_units))

        setattr(self, 'block%d' % (srb_nums + 2),
                nn.Sequential(
                    nn.Conv2d(2 * hidden_units, 2 * hidden_units, kernel_size=3, padding=1),
                    nn.BatchNorm2d(2 * hidden_units)
                ))
        setattr(self, 'block%d' % (srb_nums + 3), nn.Conv2d(2 * hidden_units, in_planes, kernel_size=9, padding=4))

    def forward(self,x):
        block = {'1': self.block1(x)}
        for i in range(self.srb_nums+1):
            block[str(i + 2)] = getattr(self, 'block%d' % (i + 2))(block[str(i + 1)])
        block[str(self.srb_nums + 3)] = getattr(self, 'block%d' % (self.srb_nums + 3)) \
            ((block['1'] + block[str(self.srb_nums+2)]))
        output = torch.tanh(block[str(self.srb_nums + 3)])
        return output

class RecurrentResidualBlock(nn.Module):
    def __init__(self, channels):
        super(RecurrentResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.gru1 = GruBlock(channels, channels)
        # self.prelu = nn.ReLU()
        self.prelu = mish()
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)
        self.gru2 = GruBlock(channels, channels)

    def forward(self, x):
        residual = self.conv1(x)
        residual = self.bn1(residual)
        residual = self.prelu(residual)
        residual = self.conv2(residual)
        residual = self.bn2(residual)
        residual = self.gru1(residual.transpose(-1, -2)).transpose(-1, -2)

        return self.gru2(x + residual)


class GruBlock(nn.Module):
    def __init__(self, in_channels,out_channels):
        super(GruBlock, self).__init__()
        assert out_channels % 2 == 0
        self.gru = nn.GRU(out_channels, out_channels // 2, bidirectional=True, batch_first=True)

    def forward(self, x):
        x = x.permute(0, 2, 3, 1).contiguous()
        b = x.size()
        x = x.view(b[0] * b[1], b[2], b[3])
        self.gru.flatten_parameters()
        x, _ = self.gru(x)
        x = x.view(b[0], b[1], b[2], b[3])
        x = x.permute(0, 3, 1, 2)
        return x


class mish(nn.Module):
    def __init__(self, ):
        super(mish, self).__init__()
        self.activated = True

    def forward(self, x):
        if self.activated:
            x = x * (torch.tanh(F.softplus(x)))
        return x


def unpatchify(x, patch_size: int = 4, shape: tuple = (32, 128)):
    """
    x: (N, L, patch_size**2 *3)
    imgs: (N, 3, H, W)
    """
    p = patch_size
    h = shape[0] // p
    w = shape[1] // p
    x = x.reshape(shape=(x.shape[0], h, w, p, p, 3))
    x = torch.einsum('nhwpqc->nchpwq', x)
    imgs = x.reshape(shape=(x.shape[0], 3, h * p, w * p))
    return imgs


if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config_path = os.path.join(os.path.dirname(__file__), '..', 'config', 'msinet_small.yaml')
    config = yaml.safe_load(open(config_path, 'r', encoding='utf-8'))
    config = EasyDict(config)
    img = torch.rand(16, 3, 32, 128).to(device)
    model=MaskRefineModule(config.MaskRefine)
    refine_sr,restruction,mask=model(img)
    print(refine_sr.shape)
    print(restruction.shape)
    print(mask.shape)

