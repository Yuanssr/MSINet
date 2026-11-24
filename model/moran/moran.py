import torch.nn as nn
from .morn import MORN
from .asrn_res import ASRN
import sys
import torch
from utils import utils_moran
import string

class MORAN(nn.Module):

    def __init__(self, nc, nclass, nh, targetH, targetW, BidirDecoder=False,
                 inputDataType='torch.cuda.FloatTensor', maxBatch=256, CUDA=True):
        super(MORAN, self).__init__()
        self.MORN = MORN(nc, targetH, targetW, inputDataType, maxBatch, CUDA)
        self.ASRN = ASRN(targetH, nc, nclass, nh, BidirDecoder, CUDA)

    def forward(self, x, test=True, debug=True):
        x,length, text, text_rev=parse_moran_data(x)
        if debug:
            x_rectified, demo = self.MORN(x, test, debug=debug)
            preds = self.ASRN(x_rectified, length, text, text_rev, test)
            return preds, demo
        else:
            x_rectified = self.MORN(x, test, debug=debug)
            preds = self.ASRN(x_rectified, length, text, text_rev, test)
            return preds



def parse_moran_data(imgs_input):
    batch_size = imgs_input.shape[0]
    alphabet_moran = ':'.join(string.digits + string.ascii_lowercase + '$')
    converter_moran = utils_moran.strLabelConverterForAttention(alphabet_moran, ':')
    imgs_input = torch.nn.functional.interpolate(imgs_input, (32, 128), mode='bicubic')
    R = imgs_input[:, 0:1, :, :]
    G = imgs_input[:, 1:2, :, :]
    B = imgs_input[:, 2:3, :, :]
    tensor = 0.299 * R + 0.587 * G + 0.114 * B
    text = torch.LongTensor(batch_size * 5)
    length = torch.IntTensor(batch_size)
    max_iter = 20
    t, l = converter_moran.encode(['0' * max_iter] * batch_size)
    utils_moran.loadData(text, t)
    utils_moran.loadData(length, l)
    return tensor, length, text, text