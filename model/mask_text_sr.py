import logging
from torch import nn
from model.Mask_Refine_Module import MaskRefineModule
from model.lemma import LEMMA
import torch
import os
from model.Mask_Learning_branch import Attention_Map_Maker
from model.Position_aware_module import PositionAwareModule
from model.ABINet.abinet import ABINet_model
from model.confidence_map_generator import ConfidenceMapGenerator
import setup
from utils.metrics import get_string_abinet
from model import tsrn,tbsrn

abi_charset = setup.CharsetMapper()

class MaskTextSR(nn.Module):
    def __init__(self,
                 scale_factor=2,
                 width=128,
                 height=32,
                 STN=False,
                 srb_nums=5,
                 mask=True,
                 cfg=None):
        super(MaskTextSR, self).__init__()
        if cfg.Pretrain_SR_Model.name == 'LEMMA':
            self.pretrain_sr_model=LEMMA(scale_factor=scale_factor, width=width, height=height,STN=STN, mask=mask, srb_nums=srb_nums,cfg=cfg)
        elif cfg.Pretrain_SR_Model.name == 'TATT':
            self.pretrain_sr_model = tsrn.TSRN_TL_TRANS(scale_factor=scale_factor, width=width, height=height,
                                 STN=STN, mask=mask, srb_nums=srb_nums,
                                 hidden_units=32)
        elif cfg.Pretrain_SR_Model.name == 'STT':
            self.pretrain_sr_model = tbsrn.TBSRN(scale_factor=scale_factor, width=width, height=height,
                                 STN=STN, mask=mask, srb_nums=srb_nums,
                                 hidden_units=32)
        elif cfg.Pretrain_SR_Model.name == 'TSRN':
            self.pretrain_sr_model = tsrn.TSRN(scale_factor=scale_factor, width=width, height=height,
                                                        STN=STN, mask=mask, srb_nums=srb_nums,
                                                        hidden_units=32)
        elif cfg.Pretrain_SR_Model.name == 'BICUBIC':
            self.pretrain_sr_model = tsrn.BICUBIC(scale_factor=scale_factor, width=width, height=height,
                                                        STN=STN, mask=mask, srb_nums=srb_nums,
                                                        hidden_units=32)
        else:
            pass
        self.MaskRefine = MaskRefineModule(cfg.MaskRefine)
        #self.Attention_Maker=Attention_Map_Maker(cfg.PositionAware)
        self.config=cfg
        self.Confidence_generator=ConfidenceMapGenerator(cfg.ConfidenceMapGenerator)

        resume=cfg.Pretrain_SR_Model.checkpoint
        if not resume == '':
            logging.info('loading pre-trained STISR model from %s ' % resume)
            if os.path.isdir(resume):
                model_dict = torch.load(resume)['state_dict_G']
                self.pretrain_sr_model.load_state_dict(model_dict, strict=False)
            else:
                loaded_state_dict = torch.load(resume)['state_dict_G']
                self.pretrain_sr_model.load_state_dict(loaded_state_dict)


    def forward(self, x):
        pos_prior=0
        if self.config.Pretrain_SR_Model.name=='LEMMA':
            sr_img, pos_prior=self.pretrain_sr_model(x)
        elif self.config.Pretrain_SR_Model.name == 'TATT':
            sr_img, _=self.pretrain_sr_model(x)
        elif self.config.Pretrain_SR_Model.name == 'STT':

            sr_img=self.pretrain_sr_model(x[:,:,:,:])
        elif self.config.Pretrain_SR_Model.name == 'TSRN':
            sr_img=self.pretrain_sr_model(x)
        elif self.config.Pretrain_SR_Model.name == 'BICUBIC':
            sr_img=self.pretrain_sr_model(x)
        else:
            pass
        #Mask Learning Forward
        #sr_prior=self.Attention_Maker(sr_img[:,:3,:,:].detach())
        #MaskRefine Forward\confidence_map=self.Confidence_generator(sr_img)
        confidence_map=self.Confidence_generator(sr_img)[0]
        refine_sr, restruction, mask=self.MaskRefine(sr_img[:,:3,:,:],confidence_map)

        return sr_img,refine_sr, restruction, mask, pos_prior