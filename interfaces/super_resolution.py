import torch
import os
from datetime import datetime
import time
import pickle
import copy
from utils import util, ssim_psnr
from IPython import embed
import torch.nn.functional as F
import cv2
from interfaces import base
from utils.metrics import get_string_aster, get_string_crnn, Accuracy,get_string_abinet,get_string_parseq
from utils.util import str_filt
import numpy as np
from ptflops import get_model_complexity_info
import editdistance
import time
import logging
from dataset.dataset import Lable2Tensor
from torch import optim as optim
import lpips
import setup
import matplotlib.pyplot as plt
import sys
from model.parseq.parseq_tokenizer import get_parseq_tokenize
from model.Position_aware_module import PositionAwareModule
from model.lemma import showPIL,show_batch_images,show_batch_images_mask
from model.models_mae import mae_vit_small_patch4_dec512d8b as mae_vit
from model.Mask_Learning_branch import Attention_Map_Maker
from torch.utils.tensorboard import SummaryWriter
from utils.visualize import visualize_tensor,unpatchify,visualize_text_distribution
from model.ABINet.abinet import ABINet_model
from loss.text_focus_loss import StyleLoss
import matplotlib
matplotlib.use("Agg")


def save_image(image, path, index):
    if path is not None and not os.path.exists(path):
        os.makedirs(path)
    plt.imshow(image)
    plt.axis('off')
    plt.savefig(f'{path}/{index}.png', bbox_inches='tight', pad_inches=0)
    plt.close()  # 关闭图像，释放内存

curr_device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
parseq_tokenizer = get_parseq_tokenize()
abi_charset = setup.CharsetMapper()
label2tensor = Lable2Tensor()
lpips_vgg = lpips.LPIPS(net="vgg").to(curr_device)

class TextSR(base.TextBase):
    def loss_stablizing(self, loss_set, keep_proportion=0.7):
        # acsending
        sorted_val, sorted_ind = torch.sort(loss_set)
        batch_size = loss_set.shape[0]
        # print("batch_size:", loss_set, batch_size)
        loss_set[sorted_ind[int(keep_proportion * batch_size)]:] = 0.0
        return loss_set

    def model_inference(self, images_lr, images_hr, model_list, aster, i):
        ret_dict = {}
        ret_dict['duration']= 0.
        images_sr = []
        before = time.time()

        cascade_images,refine_sr, restruction, mask, pos_prior= model_list[0](images_lr)

        #cascade_images, pos_prior = model_list[0](images_lr)
        after = time.time()
        #print("fps:", 1//(after - before))
        ret_dict["duration"] += (after - before)
        #images_sr.append(refine_sr)
        images_sr.append(refine_sr)
        ret_dict["images_sr"] = images_sr

        # mask
        mask = mask.unsqueeze(-1).repeat(1, 1, 48)
        mask = unpatchify(mask)
        img_lr = F.interpolate(images_lr[:,:3,:,:], scale_factor=2, mode='bicubic', align_corners=True)
        # Masked image

        masked_img = img_lr * (1 - mask)

        ret_dict["masked_img"] = masked_img
        #show_batch_images(cascade_images)
        """
        vis_tensor = visualize_tensor(
         images_lr[:, :3, ...], images_hr[:, :3, ...], restruction[:, :3, ...], mask,
         refine_sr)
        plt.imshow(vis_tensor.permute(1,2,0).cpu())
        plt.show()
        plt.axis('off')
        """
        #sys.exit()
        # Assuming sr_image and hr_image are already defined and loaded tensors

        #style_loss = StyleLoss()

        # Visualize the Gram matrices of the super-resolved image
        #style_loss.plot_multiple_grams(images_lr, images_hr, refine_sr)

        return ret_dict



    def train(self):
        cfg = self.config.TRAIN
        train_dataset, train_loader = self.get_train_data()
        val_dataset_list, val_loader_list = self.get_val_data()
        model_dict = self.load_checkpoint()
        model, image_crit,best_acc= model_dict['model'], model_dict['crit'],model_dict['best_acc']
        model_list = [model]
        log_writer = SummaryWriter(log_dir=os.path.join(cfg.ckpt_dir, 'logs'))  # 创建TensorBoard的SummaryWriter对象
        log_freq=cfg.log_freq
        vis_freq=cfg.vis_freq
        # ------- init text recognizer for eval here --------
        test_bible = {}
        if self.args.test_model == "CRNN":
            crnn, aster_info = self.CRNN_init()
            crnn.eval()
            test_bible["CRNN"] = {
                        'model': crnn,
                        'data_in_fn': self.parse_crnn_data,
                        'string_process': get_string_crnn
                    }
        elif self.args.test_model == "ASTER":
            aster_real, aster_real_info = self.Aster_init() # init ASTER model
            aster_info = aster_real_info
            test_bible["ASTER"] = {
                'model': aster_real,
                'data_in_fn': self.parse_aster_data,
                'string_process': get_string_aster
            }

        elif self.args.test_model == "MORAN":
            moran = self.MORAN_init()
            if isinstance(moran, torch.nn.DataParallel):
                moran.device_ids = [0]
            test_bible["MORAN"] = {
                'model': moran,
                'data_in_fn': self.parse_moran_data,
                'string_process': get_string_crnn
            }

        elif self.args.test_model == 'ABINet':
            abinet = self.ABINet_init()
            test_bible["ABINet"] = {
                'model': abinet,
                'data_in_fn': self.parse_abinet_data,
                'string_process': get_string_abinet
            }

        elif self.args.test_model == 'MATRN':
            matrn = self.MATRN_init()
            test_bible["MATRN"] = {
                'model': matrn,
                'data_in_fn': self.parse_abinet_data,
                'string_process': get_string_abinet
            }

        elif self.args.test_model == 'PARSeq':
            parseq = self.PARSeq_init()
            test_bible["PARSeq"] = {
                'model': parseq,
                'data_in_fn': self.parse_parseq_data,
                'string_process': get_string_parseq
            }

        optimizer_G = self.optimizer_init(model_list[0])
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer_G, 400,0.5)

        if not os.path.exists(cfg.ckpt_dir):
            os.makedirs(cfg.ckpt_dir)
        best_history_acc = dict(
            zip([val_loader_dir.split('/')[-1] for val_loader_dir in self.config.TRAIN.VAL.val_data_dir],
                [0] * len(val_loader_list)))
        best_model_acc = copy.deepcopy(best_history_acc)
        best_model_psnr = copy.deepcopy(best_history_acc)
        best_model_ssim = copy.deepcopy(best_history_acc)
        converge_list = []

        for model in model_list:
            model.train()

        for epoch in range(cfg.epochs):
            # ----------------start training here -------------------
            for j, data in (enumerate(train_loader)):
                iters = len(train_loader) * epoch + j + 1
                for model in model_list:
                    for p in model.parameters():
                        p.requires_grad = True

                images_hr, _, images_lr, _, _, label_strs, label_vecs, weighted_mask, weighted_tics = data
                images_lr = images_lr.to(self.device) # [1,4,16,64]
                images_hr = images_hr.to(self.device) # [1,4,32,128]
                label_tensor = label2tensor.getlabletensor(label_strs)
                loss_img = 0.

                cascade_images,refine_sr, restruction, mask, pos_prior =model_list[0](images_lr) # forward process
                #cascade_images, pos_prior = model_list[0](images_lr)  # forward process
                loss, mse_loss, attention_loss, recognition_loss,restruction_loss,refine_loss,finetune_loss = image_crit(cascade_images, images_hr,
                                                                                             label_tensor,restruction,refine_sr, pos_prior)
                #loss, mse_loss, attention_loss, recognition_loss,finetune_loss = image_crit(cascade_images, images_hr, label_tensor,pos_prior)



                loss_img_each  =  loss.mean() * 100
                loss_img += loss_img_each
                loss_im = loss_img
                optimizer_G.zero_grad()
                loss_im.backward()
                for model in model_list:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 0.25)
                optimizer_G.step()
                #show_batch_images(restruction)
                #if iters % 5== 0:
                    #plt.show()
                    #sys.exit()
                if iters % log_freq == 0:

                    log_writer.add_scalar('total_loss', loss_im.item(), iters)
                    log_writer.add_scalar('mse_loss', mse_loss.mean().item() * 100, iters)
                    log_writer.add_scalar('restruction_loss', restruction_loss.mean().item() * 100, iters)
                    log_writer.add_scalar('refine_loss', refine_loss.mean().item() * 100, iters)
                    log_writer.add_scalar('trans_attn', attention_loss.mean().item() * 1000, iters)
                    log_writer.add_scalar('trans_recg_loss', recognition_loss.mean().item() * 100 * 0.0005, iters)
                #if iters % vis_freq == 0:
                    #vis_tensor = visualize_tensor(
                        #images_lr[:, :3, ...], images_hr[:, :3, ...], cascade_images[:, :3, ...], mask,
                        #refine_sr,confidence_map)
                    #log_writer.add_image('train_vis', vis_tensor, iters)

                if iters % cfg.displayInterval == 0:
                    logging.info('[{}]\t'
                                 'Epoch: [{}][{}/{}]\t'
                                 'total_loss {:.3f}\t'
                                 'mse_loss {:.3f}\t'
                                 'restruction_loss {:.3f}\t'
                                 'refine_loss {:.3f}\t'
                                 'finetune_loss {:.3f}\t'
                                 'trans_attn {:.3f}\t'
                                 'trans_recg_loss {:.3f}\t'
                                 .format(datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                                         epoch, j + 1, len(train_loader),
                                         float(loss_im.data),
                                         mse_loss.mean().item() *100,
                                         restruction_loss.mean().item()*100,
                                         refine_loss.mean().item() * 100,
                                         finetune_loss.mean().item() if type(finetune_loss)!=int else finetune_loss,
                                         attention_loss.mean().item()*1000,
                                         recognition_loss.mean().item() * 100*0.0005,
                                         ))

                if iters % cfg.VAL.valInterval == 0:
                    logging.info('======================================================')
                    current_acc_dict = {}
                    psnr_dict={}
                    ssim_dict={}
                    for k, val_loader in enumerate(val_loader_list):
                        data_name = self.config.TRAIN.VAL.val_data_dir[k].split('/')[-1]
                        logging.info('evaling %s' % data_name)
                        for model in model_list:
                            model.eval()
                            for p in model.parameters():
                                p.requires_grad = False

                        metrics_dict = self.eval(
                            model_list,
                            val_loader,
                            image_crit,
                            iters,
                            [test_bible[self.args.test_model], None, None],
                            aster_info,
                            data_name,
                        )
                        psnr_dict[data_name]=metrics_dict['psnr_avg']
                        ssim_dict[data_name]=metrics_dict['ssim_avg']

                        for model in model_list:
                            for p in model.parameters():
                                p.requires_grad = True
                            model.train()

                        converge_list.append({'iterator': iters,
                                              'acc': metrics_dict['accuracy'],
                                              'psnr': metrics_dict['psnr_avg'],
                                              'ssim': metrics_dict['ssim_avg']})
                        acc = metrics_dict['accuracy']
                        current_acc_dict[data_name] = float(acc)
                        if acc > best_history_acc[data_name]:
                            best_history_acc[data_name] = float(acc)
                            best_history_acc['epoch'] = epoch
                            logging.info('best_%s = %.2f%%*' % (data_name, best_history_acc[data_name] * 100))

                        else:
                            logging.info('best_%s = %.2f%%' % (data_name, best_history_acc[data_name] * 100))

                        log_writer.add_scalar('accuracy/{}'.format(data_name), metrics_dict['accuracy'], iters)


                    if sum(current_acc_dict.values()) > best_acc:
                        best_acc = sum(current_acc_dict.values())
                        best_model_acc = current_acc_dict
                        best_model_acc['epoch'] = epoch
                        best_model_psnr[data_name] = metrics_dict['psnr_avg']
                        best_model_ssim[data_name] = metrics_dict['ssim_avg']
                        best_model_info = {'accuracy': best_model_acc, 'psnr': best_model_psnr, 'ssim': best_model_ssim}
                        logging.info('saving best model')
                        logging.info('avg_acc {:.4f}%'.format(100*(current_acc_dict['easy']*1619
                                                              +current_acc_dict['medium']*1411
                                                              +current_acc_dict['hard']*1343)/(1343+1411+1619)))
                        logging.info('=============')
                        logging.info('bset psnr {:.4f}%'.format(1*(psnr_dict['easy']*1619
                                                              +psnr_dict['medium']*1411
                                                              +psnr_dict['hard']*1343)/(1343+1411+1619)))
                        logging.info('best ssim {:.4f}%'.format(1 * (ssim_dict['easy'] * 1619
                                                                + ssim_dict['medium'] * 1411
                                                                + ssim_dict['hard'] * 1343) / (1343 + 1411 + 1619)))
                        self.save_checkpoint(model_list, epoch, iters, best_history_acc, best_model_info, True, converge_list,best_acc, recognizer=None,)

                if iters % cfg.saveInterval == 0:
                    best_model_info = {'accuracy': best_model_acc, 'psnr': best_model_psnr, 'ssim': best_model_ssim}
                    self.save_checkpoint(model_list, epoch, iters, best_history_acc, best_model_info, False, converge_list, best_acc,recognizer=None)
            lr_scheduler.step()
        log_writer.close()


    def eval(self, model_list, val_loader, image_crit, index, aster, aster_info,data_name=None):
        self.string_process_fn=aster[0]["string_process"]

        n_correct_lr = 0
        n_correct_hr = 0
        sum_images = 0
        metric_dict = {
                       'psnr_lr': [],
                       'ssim_lr': [],
                       'ssi_lr': [],
                       'cnt_psnr_lr': [],
                       'cnt_ssim_lr': [],
                       'cnt_ssi_lr': [],
                       'psnr': [],
                       'ssim': [],
                        'ssi': [],
                       'cnt_psnr': [],
                       'cnt_ssim': [],
                       'cnt_ssi': [],
                       'accuracy': 0.0,
                       'psnr_avg': 0.0,
                       'ssim_avg': 0.0,
                        'ssi_avg': 0.0,
                        'edis_LR': [],
                        'edis_SR': [],
                        'edis_HR': [],
                        'LPIPS_VGG_LR': [],
                        'LPIPS_VGG_SR': []
                       }

        counters = {0: 0}
        image_counter = 0
        rec_str = ""

        sr_infer_time = 0
        with torch.no_grad():

            for i, data in (enumerate(val_loader)):

                images_hrraw, images_lrraw, images_HRy, images_lry, label_strs, label_vecs_gt = data
                
                images_lr = images_lrraw.to(self.device)
                images_hr = images_hrraw.to(self.device)

                #show_batch_images(images_lr)
                #show_batch_images(images_hr)



                val_batch_size = images_lr.shape[0]
                ret_dict = self.model_inference(images_lr, images_hr, model_list, aster, i)
                #plt.show()
                #if i>5:
                    #sys.exit()
                sr_infer_time += ret_dict["duration"]

                images_sr = ret_dict["images_sr"]
                masked_img=ret_dict["masked_img"]
                """
                save_image(masked_img[0].permute(1,2,0).cpu(), f'outputs/mask_vis/{data_name}/mask_image', i)

                #show_batch_images(images_sr[0])
                #plt.show()
                #sys.exit()
                
                batch_size = images_lr.shape[0]
                for batch_i in range(batch_size):
                    # 获取每个批次的图像数据
                    lr_image = images_lr[batch_i, :3, ...].permute(1, 2, 0).cpu()
                    hr_image = images_hr[batch_i, :3, ...].permute(1, 2, 0).cpu()
                    sr_image = images_sr[0][batch_i, :3, ...].permute(1, 2, 0).cpu()

                    # 保存图像
                    save_image(lr_image, f'./output_dir/msinet_small_vis/{data_name}/LR', index)
                    save_image(hr_image, f'./output_dir/msinet_small_vis/{data_name}/HR', index)
                    save_image(sr_image, f'./output_dir/msinet_small_vis/{data_name}/SR', index)

                    index += 1
                """
                # == 首先解析输入图像为文本识别器的输入形式 =======
                aster_dict_lr = aster[0]["data_in_fn"](images_lr[:, :3, :, :])
                aster_dict_hr = aster[0]["data_in_fn"](images_hr[:, :3, :, :])
                # ==== 之后是将解析后的图像用于文本识别  这里只对LR和HR进行识别====
                if self.args.test_model == "MORAN":
                    # LR
                    aster_output_lr = aster[0]["model"](
                        aster_dict_lr[0],
                        aster_dict_lr[1],
                        aster_dict_lr[2],
                        aster_dict_lr[3],
                        test=True,
                        debug=True
                    )
                    # HR
                    aster_output_hr = aster[0]["model"](
                        aster_dict_hr[0],
                        aster_dict_hr[1],
                        aster_dict_hr[2],
                        aster_dict_hr[3],
                        test=True,
                        debug=True
                    )
                else:
                    aster_output_lr = aster[0]["model"](aster_dict_lr)
                    aster_output_hr = aster[0]["model"](aster_dict_hr)

                # ============对SR进行图像解析和通过文本识别器 ==============
                if type(images_sr) == list:
                    predict_result_sr = []
                    image = images_sr[0]
                    aster_dict_sr = aster[0]["data_in_fn"](image[:, :3, :, :])

                    if self.args.test_model == "MORAN":
                        # aster_output_sr = aster[0]["model"](*aster_dict_sr)
                        aster_output_sr = aster[0]["model"](
                            aster_dict_sr[0],
                            aster_dict_sr[1],
                            aster_dict_sr[2],
                            aster_dict_sr[3],
                            test=True,
                            debug=True
                        )
                    else:
                        aster_output_sr = aster[0]["model"](aster_dict_sr)

                    # outputs_sr = aster_output_sr.permute(1, 0, 2).contiguous()

                    if self.args.test_model == "CRNN":

                        predict_result_sr_ = self.string_process_fn(aster_output_sr)

                    elif self.args.test_model == "ASTER":
                        predict_result_sr_, _ = self.string_process_fn(
                            aster_output_sr['output']['pred_rec'],
                            aster_dict_sr['rec_targets'],
                            dataset=aster_info
                        )
                    elif self.args.test_model == "MORAN":
                        preds, preds_reverse = aster_output_sr[0]
                        _, preds = preds.max(1)
                        sim_preds = self.converter_moran.decode(preds.data, aster_dict_sr[1].data)
                        predict_result_sr_ = [pred.split('$')[0] for pred in sim_preds]

                    elif self.args.test_model in ["ABINet",'MATRN']:
                        # 对识别字典做处理
                        predict_result_sr_ = aster[0]['string_process'](aster_output_sr, abi_charset)[0]

                    elif self.args.test_model in ['PARSeq']:
                        # 对parseq的识别字典logits做处理
                        predict_result_sr_ = aster[0]['string_process'](aster_output_sr, parseq_tokenizer)

                    predict_result_sr.append(predict_result_sr_)

                    img_lr = torch.nn.functional.interpolate(images_lr, images_hr.shape[-2:], mode="bicubic")
                    img_sr = torch.nn.functional.interpolate(images_sr[-1], images_hr.shape[-2:], mode="bicubic")

                    metric_dict['psnr'].append(self.cal_psnr(img_sr[:, :3], images_hr[:, :3]))
                    metric_dict['ssim'].append(self.cal_ssim(img_sr[:, :3], images_hr[:, :3]))
                    
                    metric_dict["LPIPS_VGG_SR"].append(lpips_vgg(img_sr[:, :3], images_hr[:, :3]).data[0].reshape(-1)[0])

                    metric_dict['psnr_lr'].append(self.cal_psnr(img_lr[:, :3], images_hr[:, :3]))
                    metric_dict['ssim_lr'].append(self.cal_ssim(img_lr[:, :3], images_hr[:, :3]))
                    

                    metric_dict["LPIPS_VGG_LR"].append(lpips_vgg(img_lr[:, :3], images_hr[:, :3]).data[0].reshape(-1)[0])



                if self.args.test_model == "CRNN":#之前是只对SR的识别结果进行的处理，这里将HR和LR同样进行处理

                    predict_result_lr =self.string_process_fn(aster_output_lr)

                    predict_result_hr = self.string_process_fn(aster_output_hr)
                elif self.args.test_model == "ASTER":
                    predict_result_lr, _ = self.string_process_fn(
                        aster_output_lr['output']['pred_rec'],
                        aster_dict_lr['rec_targets'],
                        dataset=aster_info
                    )
                    predict_result_hr, _ =self.string_process_fn(
                        aster_output_hr['output']['pred_rec'],
                        aster_dict_hr['rec_targets'],
                        dataset=aster_info
                    )
                elif self.args.test_model == "MORAN":
                    ### LR ###
                    preds, preds_reverse = aster_output_lr[0]
                    _, preds = preds.max(1)
                    sim_preds = self.converter_moran.decode(preds.data, aster_dict_lr[1].data)
                    predict_result_lr = [pred.split('$')[0] for pred in sim_preds]

                    ### HR ###
                    preds, preds_reverse = aster_output_hr[0]
                    _, preds = preds.max(1)
                    sim_preds = self.converter_moran.decode(preds.data, aster_dict_hr[1].data)
                    predict_result_hr = [pred.split('$')[0] for pred in sim_preds]

                elif self.args.test_model in  ['ABINet','MATRN']:
                    predict_result_lr =self.string_process_fn(aster_output_lr, abi_charset)[0]
                    predict_result_hr = self.string_process_fn(aster_output_hr, abi_charset)[0]
                elif self.args.test_model in ['PARSeq']:
                    predict_result_lr = self.string_process_fn(aster_output_lr, parseq_tokenizer)
                    predict_result_hr = self.string_process_fn(aster_output_hr, parseq_tokenizer)
                filter_mode = 'lower'

                """
                if i == 406:
                    print('i   :',i)
                    print('LR',predict_result_lr[0])
                    print('HR',predict_result_hr[0])
                    print('SR',predict_result_sr[0][0])
                    print('Lable',label_strs[0])
                    sys.exit()
                """

                for batch_i in range(images_lr.shape[0]):
                    label = label_strs[batch_i]
                    image_counter += 1
                    rec_str += str(image_counter) + ".jpg," + label + "\n"

                    if str_filt(predict_result_sr[0][batch_i], filter_mode) == str_filt(label, filter_mode):
                        counters[0] += 1
                    if str_filt(predict_result_lr[batch_i], filter_mode) == str_filt(label, filter_mode):
                        n_correct_lr += 1
                    if str_filt(predict_result_hr[batch_i], filter_mode) == str_filt(label, filter_mode):
                        n_correct_hr += 1
                sum_images += val_batch_size
            torch.cuda.empty_cache()


            # 已经把整个测试集跑完
            psnr_avg = sum(metric_dict['psnr']) / (len(metric_dict['psnr']) + 1e-10)
            ssim_avg = sum(metric_dict['ssim']) / (len(metric_dict['ssim']) + 1e-10)
            ssi_avg = sum(metric_dict['ssi']) / (len(metric_dict['ssi']) + 1e-10)

            psnr_avg_lr = sum(metric_dict['psnr_lr']) / (len(metric_dict['psnr_lr']) + 1e-10)
            ssim_avg_lr = sum(metric_dict['ssim_lr']) / (len(metric_dict['ssim_lr']) + 1e-10)
            ssi_avg_lr = sum(metric_dict['ssi_lr']) / (len(metric_dict['ssi_lr']) + 1e-10)
            lpips_vgg_lr = sum(metric_dict["LPIPS_VGG_LR"]) / (len(metric_dict['LPIPS_VGG_LR']) + 1e-10)
            lpips_vgg_sr = sum(metric_dict["LPIPS_VGG_SR"]) / (len(metric_dict['LPIPS_VGG_SR']) + 1e-10)

            logging.info('[{}]\t'
                  'loss_rec {:.3f}| loss_im {:.3f}\t'
                  'PSNR {:.2f} | SSIM {:.4f}\t'
                  'LPIPS {:.4f}  | SSI {:.4f}\t'
                  .format(datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                          0, 0,
                          float(psnr_avg), float(ssim_avg), float(lpips_vgg_sr),float(ssi_avg)))

            logging.info('[{}]\t'
                  'PSNR_LR {:.2f} | SSIM_LR {:.4f}\t'
                  'LPIPS_LR {:.4f}| SSI_LR {:.4f}\t'
                  .format(datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                          float(psnr_avg_lr), float(ssim_avg_lr), float(lpips_vgg_lr),float(ssi_avg_lr)))

            logging.info('save display images')

            accuracy = round(counters[0] / sum_images, 4)# loader已经跑完了，把全部的测试集图片和全部正确个数相除

            accuracy_lr = round(n_correct_lr / sum_images, 4)
            accuracy_hr = round(n_correct_hr / sum_images, 4)
            psnr_avg = round(psnr_avg.item(), 6)
            ssim_avg = round(ssim_avg.item(), 6)
           


            logging.info('sr_accuray_iter' + ': %.2f%%' % (accuracy * 100))


            logging.info('lr_accuray: %.2f%%' % (accuracy_lr * 100))
            logging.info('hr_accuray: %.2f%%' % (accuracy_hr * 100))
            metric_dict['accuracy'] = accuracy
            metric_dict['psnr_avg'] = psnr_avg
            metric_dict['ssim_avg'] = ssim_avg
        

            inference_time = sum_images / sr_infer_time
            logging.info("AVG inference:{}".format(inference_time))
            logging.info("sum_images:{}".format(sum_images))

            return metric_dict



    def test(self):
        total_acc = {}
        total_ssi = {}
        total_psnr = {}
        total_ssim = {}
        #torch.manual_seed(0)
        val_dataset_list, val_loader_list = self.get_val_data()
        model_dict = self.generator_init()
        model, image_crit = model_dict['model'], model_dict['crit']
        model_list = [model]
        logging.info('Using text recognizer {}'.format(self.args.test_model))
        test_bible = {}
        aster_info = None
        if self.args.test_model == "CRNN":
            crnn, aster_info = self.CRNN_init()
            crnn.eval()
            test_bible["CRNN"] = {
                'model': crnn,
                'data_in_fn': self.parse_crnn_data,
                'string_process': get_string_crnn
            }
        elif self.args.test_model == "ASTER":
            aster_real, aster_real_info = self.Aster_init()  # init ASTER model
            aster_info = aster_real_info
            test_bible["ASTER"] = {
                'model': aster_real,
                'data_in_fn': self.parse_aster_data,
                'string_process': get_string_aster
            }

        elif self.args.test_model == "MORAN":
            moran = self.MORAN_init()
            aster_info = None
            if isinstance(moran, torch.nn.DataParallel):
                moran.device_ids = [0]
            test_bible["MORAN"] = {
                'model': moran,
                'data_in_fn': self.parse_moran_data,
                'string_process': get_string_crnn
            }
        elif self.args.test_model == 'ABINet':
            abinet = self.ABINet_init()
            test_bible["ABINet"] = {
                'model': abinet,
                'data_in_fn': self.parse_abinet_data,
                'string_process': get_string_abinet
            }
        elif self.args.test_model == 'MATRN':
            matrn = self.MATRN_init()
            test_bible["MATRN"] = {
                'model': matrn,
                'data_in_fn': self.parse_abinet_data,
                'string_process': get_string_abinet
            }
        elif self.args.test_model == 'PARSeq':
            parseq = self.PARSeq_init()
            test_bible["PARSeq"] = {
                'model': parseq,
                'data_in_fn': self.parse_parseq_data,
                'string_process': get_string_parseq
            }


        for k, val_loader in enumerate(val_loader_list):
            data_name = self.config.TRAIN.VAL.val_data_dir[k].split('/')[-1]
            logging.info('testing %s' % data_name)
            for model in model_list:
                model.eval()
                for p in model.parameters():
                    p.requires_grad = False

            metrics_dict = self.eval(
                model_list,
                val_loader,
                image_crit,
                0,
                [test_bible[self.args.test_model], None, None],
                aster_info,
                data_name
            )
            acc = float(metrics_dict['accuracy'])
            
            psnr=float(metrics_dict['psnr_avg'] )
            ssim= float(metrics_dict['ssim_avg'])
            total_acc[data_name]=acc
            
            total_psnr[data_name] = psnr
            total_ssim[data_name] = ssim
            logging.info('best_%s = %.2f%%' % (data_name, acc * 100))
        logging.info('avg_acc(Easy,Medium,Hard) is {:.3f}'.format(100*(total_acc['easy']*1619
                                                              +total_acc['medium']*1411
                                                              +total_acc['hard']*1343)/(1343+1411+1619)))
        
        logging.info('avg_psnr(Easy,Medium,Hard) is {:.4f}'.format( (total_psnr['easy'] * 1619
                                                                         + total_psnr['medium'] * 1411
                                                                         + total_psnr['hard'] * 1343) / (
                                                                          1343 + 1411 + 1619)))
        logging.info('avg_ssim(Easy,Medium,Hard) is {:.4f}'.format(  (total_ssim['easy'] * 1619
                                                                         + total_ssim['medium'] * 1411
                                                                         + total_ssim['hard'] * 1343) / (
                                                                          1343 + 1411 + 1619)))
        logging.info('Test with recognizer {} finished!'.format(self.args.test_model))


    def get_mask(self, image):

        img_hr = image.data.numpy() * 255

        img_hr_gray = cv2.cvtColor(img_hr[..., :3].astype(np.uint8), cv2.COLOR_RGB2GRAY)

        # print("img_hr_gray: ", np.unique(img_hr_gray), img_hr_gray.shape)
        kernel = np.ones((5, 5), np.uint8)
        hr_canny = cv2.Canny(img_hr_gray, 20, 150)
        hr_canny = cv2.dilate(hr_canny, kernel, iterations=1)
        hr_canny = cv2.GaussianBlur(hr_canny, (5, 5), 1)

        weighted_mask = 0.4 + (hr_canny / 255.0) * 0.6
        return torch.tensor(weighted_mask).float().unsqueeze(0)
