# References:
# MAE: https://github.com/facebookresearch/mae
# --------------------------------------------------------

from functools import partial
import numpy as np
import torch
import torch.nn as nn
from timm.models.vision_transformer import Block, PatchEmbed

from utils.pos_embed import get_2d_sincos_pos_embed




class MaskedAutoencoderViT(nn.Module):
    """ Masked Autoencoder with VisionTransformer backbone
    """

    def __init__(self,
                 img_size=(32, 128),
                 patch_size=4,
                 in_chans=3,
                 embed_dim=768,
                 depth=12,
                 num_heads=12,
                 decoder_embed_dim=512,
                 decoder_depth=8,
                 decoder_num_heads=16,
                 mlp_ratio=4.,
                 norm_layer=nn.LayerNorm,
                 norm_pix_loss=False):
        super().__init__()

        # --------------------------------------------------------------------------
        # MAE encoder specifics
        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans,
                                      embed_dim)
        num_patches = self.patch_embed.num_patches
        self.patch_size = patch_size
        self.img_size = img_size
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(
            torch.zeros(1, num_patches + 1, embed_dim),
            requires_grad=False)  # fixed sin-cos embedding

        self.blocks = nn.ModuleList([
            Block(
                embed_dim,
                num_heads,
                mlp_ratio,
                qkv_bias=True,
                #qk_scale=None,
                norm_layer=norm_layer) for i in range(depth)
        ])
        self.norm = norm_layer(embed_dim)
        # ---------------------------------------------------------------------

        # ---------------------------------------------------------------------
        # MAE decoder specifics
        self.decoder_embed = nn.Linear(embed_dim, decoder_embed_dim, bias=True)

        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))

        self.decoder_pos_embed = nn.Parameter(
            torch.zeros(1, num_patches + 1, decoder_embed_dim),
            requires_grad=False)  # fixed sin-cos embedding

        self.decoder_blocks = nn.ModuleList([
            Block(
                decoder_embed_dim,
                decoder_num_heads,
                mlp_ratio,
                qkv_bias=True,
                #qk_scale=None,
                norm_layer=norm_layer) for i in range(decoder_depth)
        ])

        self.decoder_norm = norm_layer(decoder_embed_dim)
        self.decoder_pred = nn.Linear(
            decoder_embed_dim, patch_size**2 * in_chans,
            bias=True)  # decoder to patch
        # ---------------------------------------------------------------------

        self.norm_pix_loss = norm_pix_loss

        self.initialize_weights()

    def initialize_weights(self):
        # initialization
        # initialize (and freeze) pos_embed by sin-cos embedding
        pos_embed = get_2d_sincos_pos_embed(
            self.pos_embed.shape[-1],
            int(self.patch_embed.num_patches**.5),
            cls_token=True)
        self.pos_embed.data.copy_(
            torch.from_numpy(pos_embed).float().unsqueeze(0))

        decoder_pos_embed = get_2d_sincos_pos_embed(
            self.decoder_pos_embed.shape[-1],
            int(self.patch_embed.num_patches**.5),
            cls_token=True)
        self.decoder_pos_embed.data.copy_(
            torch.from_numpy(decoder_pos_embed).float().unsqueeze(0))

        # initialize patch_embed like nn.Linear (instead of nn.Conv2d)
        w = self.patch_embed.proj.weight.data
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

        # timm's trunc_normal_(std=.02) is effectively normal_(std=0.02) as cutoff is too big (2.) # noqa
        torch.nn.init.normal_(self.cls_token, std=.02)
        torch.nn.init.normal_(self.mask_token, std=.02)

        # initialize nn.Linear and nn.LayerNorm
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def patchify(self, imgs):
        """
        imgs: (N, 3, H, W)
        x: (N, L, patch_size**2 *3)
        """
        p = self.patch_embed.patch_size[0]  # p = 8
        assert imgs.shape[2] % p == 0 and imgs.shape[3] % p == 0

        h = imgs.shape[2] // p
        w = imgs.shape[3] // p
        x = imgs.reshape(shape=(imgs.shape[0], 3, h, p, w, p))
        x = torch.einsum('nchpwq->nhwpqc', x)
        x = x.reshape(shape=(imgs.shape[0], h * w, p**2 * 3))
        return x

    def unpatchify(self, x):
        """
        x: (N, L, patch_size**2 *3)
        imgs: (N, 3, H, W)
        """
        p = self.patch_size
        h = self.img_size[0] // p
        w = self.img_size[1] // p
        x = x.reshape(shape=(x.shape[0], h, w, p, p, 3))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], 3, h * p, w * p))
        return imgs

    def text_prob_masking(self,x, text_prob, mask_ratios=[0.4,0.3,0.2,0.1]):
        """
        x: [N, L, D], sequence-
        text_prob:[N,L]
        """
        N, L, D = x.shape  # batch, length, dim
        
        text_prob_index = torch.argsort(text_prob, dim=1, descending=True)
        L0 = L // len(mask_ratios)
        len_keep = 0
        ids_shuffle_keep = []
        ids_shuffle_unkeep = []

        for i, mask_ratio in enumerate(mask_ratios):
            sub_text_prob_index = text_prob_index[:, i * L0:(i + 1) * L0]
            per_len_keep = int(L0 * (1 - mask_ratio))
            len_keep += per_len_keep
            noise = torch.rand(N, L0, device=x.device)  # noise in [0, 1]
            per_ids_shuffle_index = torch.argsort(noise, dim=1)
            per_ids_shuffle = torch.gather(sub_text_prob_index, dim=1, index=per_ids_shuffle_index)
            per_ids_keep = per_ids_shuffle[:, :per_len_keep]
            per_ids_unkeep = per_ids_shuffle[:, per_len_keep:]
            ids_shuffle_keep.append(per_ids_keep)
            ids_shuffle_unkeep.append(per_ids_unkeep)

        ids_shuffle_keep = torch.cat(ids_shuffle_keep, dim=1)
        ids_shuffle_unkeep = torch.cat(ids_shuffle_unkeep, dim=1)
        ids_shuffle = torch.cat((ids_shuffle_keep, ids_shuffle_unkeep), dim=1)
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # keep the first subset
        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(
            x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

        # generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([N, L], device=x.device)
        mask[:, :len_keep] = 0
        # unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore)

        return x_masked, mask, ids_restore


    def random_masking(self, x, mask_ratio):
        """
        Perform per-sample random masking by per-sample shuffling.
        Per-sample shuffling is done by argsort random noise.
        x: [N, L, D], sequence
        """
        N, L, D = x.shape  # batch, length, dim
        len_keep = int(L * (1 - mask_ratio))

        noise = torch.rand(N, L, device=x.device)  # noise in [0, 1]

        # sort noise for each sample
        ids_shuffle = torch.argsort(
            noise, dim=1)  # ascend: small is keep, large is remove
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # keep the first subset
        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(
            x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

        # generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([N, L], device=x.device)
        mask[:, :len_keep] = 0
        # unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore)

        return x_masked, mask, ids_restore

    def masking(self, x, mask_ratio):
        """
        Perform per-sample random masking by per-sample shuffling.
        Per-sample shuffling is done by argsort random noise.
        x: [N, L, D], sequence
        """
        N, L, D = x.shape  # batch, length, dim
        len_keep = int(L * (1 - mask_ratio))

        patch_matrix = torch.tensor(list(range(L)), device=x.device).unsqueeze(-1).view(8,-1)  #[8,32]
        #perm_indices = torch.randperm(patch_matrix.shape[1])
        idx=torch.tensor([1,4,5,7,9,10,13,15,17,18,21,23,25,27,29,31,0,2,3,6,8,11,12,14,16,19,20,22,24,26,28,30], device=x.device)
        patch_matrix_shuffled = patch_matrix[:, idx]
        ids_shuffle=patch_matrix_shuffled.T.contiguous().view(256,-1).squeeze(-1).unsqueeze(0).repeat(32,1)   #[256]
        #ids_shuffle=torch.ones([N, L], device=x.device)



        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # keep the first subset
        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(
            x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

        # generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([N, L], device=x.device)
        mask[:, :len_keep] = 0
        # unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore)

        return x_masked, mask, ids_restore

    def high_freq_masking(self, x, imgs, mask_ratio):
            """
            根据图像的高频分量进行掩码。
            x: [N, L, D], patch 序列
            imgs: [N, 3, H, W], 原始图像（用于提取高频）
            """
            N, L, D = x.shape
            len_keep = int(L * (1 - mask_ratio))

            # 1. 计算高频（使用拉普拉斯算子提取边缘）
            # 将图像转为灰度图 [N, 1, H, W]
            gray_imgs = imgs.mean(dim=1, keepdim=True)
            
            # 定义拉普拉斯卷积核
            kernel = torch.tensor([[-1, -1, -1], 
                                [-1,  8, -1], 
                                [-1, -1, -1]], dtype=torch.float32, device=x.device).view(1, 1, 3, 3)
            
            # 提取高频特征 [N, 1, H, W]
            high_freq = torch.abs(torch.nn.functional.conv2d(gray_imgs, kernel, padding=1))

            # 2. 计算每个 Patch 的平均高频能量
            p = self.patch_size
            h, w = self.img_size[0] // p, self.img_size[1] // p
            
            # 将高频图切分为 patch [N, 1, h, p, w, p]
            hp = high_freq.reshape(N, 1, h, p, w, p)
            # 重新排列并计算 patch 内的均值 [N, L] (L = h*w)
            hp = hp.permute(0, 2, 4, 1, 3, 5).reshape(N, L, -1)
            patch_energy = hp.mean(dim=-1) # [N, L]

            # 3. 排序并选择高频区域进行 Masking
            # 我们要掩盖高频区域，所以保留能量较低的区域 (small energy is keep)
            ids_shuffle = torch.argsort(patch_energy, dim=1, descending=False) 
            ids_restore = torch.argsort(ids_shuffle, dim=1)

            # 保持低频部分
            ids_keep = ids_shuffle[:, :len_keep]
            x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

            # 生成掩码: 0 是保留, 1 是移除
            mask = torch.ones([N, L], device=x.device)
            mask[:, :len_keep] = 0
            mask = torch.gather(mask, dim=1, index=ids_restore)

            return x_masked, mask, ids_restore

    def grid_masking(self, x, mask_ratio):
        N, L, D = x.shape  # batch, length, dim
        len_keep = int(L * (1 - mask_ratio))

        p = self.patch_size
        h = self.img_size[0] // p
        w = self.img_size[1] // p

        # Define grid step dynamically based on mask_ratio
        step = max(1, int(1 / mask_ratio))  # Ensure step is at least 1

        # Initialize the mask
        mask = torch.zeros([N, h, w], device=x.device)

        # Apply grid masking with dynamic step size
        mask[:, ::int(step/2), ::int(step/2)] = 1  # Zero out patches based on the step size

        # Unravel the 2D mask to match the patch sequence
        mask = mask.view(N, L)
        ids_shuffel = torch.argsort(mask, dim=1)
        ids_restore = torch.argsort(ids_shuffel, dim=1)
        ids_keep = ids_shuffel[:, :len_keep]
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

        return x_masked, mask, ids_restore

    def block_masking(self, x, mask_ratio):
        N, L, D = x.shape  # batch, length, dim
        len_keep = int(L * (1 - mask_ratio))
        p = self.patch_size
        h = self.img_size[0] // p
        w = self.img_size[1] // p
        # Define block size and ensure that the block does not exceed the image dimensions

        block_h = int(h*mask_ratio**0.5)
        block_w= int(w*mask_ratio**0.5)

        mask = torch.zeros([N, h, w], device=x.device)

        for i in range(N):
            top_left_h = np.random.randint(0, h - block_h)
            top_left_w = np.random.randint(0, w - block_w)

            mask[i, top_left_h:top_left_h + block_h, top_left_w:top_left_w + block_w] = 1

        mask = mask.view(N, L)
        ids_shuffel = torch.argsort(mask, dim=1)
        ids_restore= torch.argsort(ids_shuffel, dim=1)
        ids_keep = ids_shuffel[:, :len_keep]
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

        return x_masked, mask, ids_restore

    def forward_encoder(self, x, mask_ratio, text_prob, mask_type='tcm', imgs=None):
            
        # ... 后续 cls token 和 transformer 逻辑保持不变
        # embed patches
        x = self.patch_embed(x)

        # add pos embed w/o cls token
        x = x + self.pos_embed[:, 1:, :]

        # masking: length -> length * mask_ratio
        # 扩展掩码逻辑
        if mask_type == 'high_freq' and imgs is not None:
            x, mask, ids_restore = self.high_freq_masking(x, imgs, mask_ratio)
        elif text_prob is not None:
            x, mask, ids_restore = self.text_prob_masking(x, text_prob)
        else:
            x, mask, ids_restore = self.random_masking(x, mask_ratio)

        # append cls token
        cls_token = self.cls_token + self.pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        # apply Transformer blocks
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)

        return x, mask, ids_restore

    def forward_decoder(self, x, ids_restore):
        # embed tokens
        x = self.decoder_embed(x)

        # append mask tokens to sequence
        mask_tokens = self.mask_token.repeat(
            x.shape[0], ids_restore.shape[1] + 1 - x.shape[1], 1)
        x_ = torch.cat([x[:, 1:, :], mask_tokens], dim=1)  # no cls token
        x_ = torch.gather(
            x_,
            dim=1,
            index=ids_restore.unsqueeze(-1).repeat(1, 1,
                                                   x.shape[2]))  # unshuffle
        x = torch.cat([x[:, :1, :], x_], dim=1)  # append cls token

        # add pos embed
        x = x + self.decoder_pos_embed

        # apply Transformer blocks
        for blk in self.decoder_blocks:
            x = blk(x)
        x = self.decoder_norm(x)

        # predictor projection
        x = self.decoder_pred(x)

        # remove cls token
        x = x[:, 1:, :]

        return x

    def forward_loss(self, imgs, pred, mask):
        """
        imgs: [N, 3, H, W]
        pred: [N, L, p*p*3]
        mask: [N, L], 0 is keep, 1 is remove,
        """
        target = self.patchify(imgs)
        if self.norm_pix_loss:

            mean = target.mean(dim=-1, keepdim=True)
            var = target.var(dim=-1, keepdim=True)
            target = (target - mean) / (var + 1.e-6)**.5

        loss = (pred - target)**2
        loss = loss.mean(dim=-1)  # [N, L], mean loss per patch

        loss = (loss * mask).sum() / mask.sum()  # mean loss on removed patches
        return loss

    def forward(self, imgs,text_prob, mask_ratio=0.25):
        latent, mask, ids_restore = self.forward_encoder(imgs,mask_ratio,text_prob)
        pred = self.forward_decoder(latent, ids_restore)  # [N, L, p*p*3]

        return  pred, mask, imgs


def mae_vit_small_patch4_dec512d8b(**kwargs):
    model = MaskedAutoencoderViT(
        img_size=(32, 128),
        patch_size=4,
        embed_dim=384,
        depth=12,
        num_heads=6,
        decoder_embed_dim=512,
        decoder_depth=8,
        decoder_num_heads=16,
        mlp_ratio=4.,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs)
    return model


def mae_vit_base_patch4_dec512d8b(**kwargs):
    model = MaskedAutoencoderViT(
        img_size=(32, 128),
        patch_size=4,
        embed_dim=768,
        depth=12,
        num_heads=12,
        decoder_embed_dim=512,
        decoder_depth=8,
        decoder_num_heads=16,
        mlp_ratio=4,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs)
    return model


# set recommended archs
mae_vit_small_patch4 = mae_vit_small_patch4_dec512d8b  # decoder: 512 dim, 8 blocks # noqa
mae_vit_base_patch4 = mae_vit_base_patch4_dec512d8b  # decoder: 512 dim, 8 blocks # noqa



