import math
import logging
from functools import partial
from collections import OrderedDict
from copy import Error, deepcopy
from re import S
from numpy.lib.arraypad import pad
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
import torch.fft
from torch.nn.modules.container import Sequential
import random
from math import sqrt
from functools import partial, reduce
from operator import mul

from PerturbStyle.DSU import DSU
from PerturbStyle.MixStyle import MixStyle

_logger = logging.getLogger(__name__)


def _cfg(url='', **kwargs):
    return {
        'url': url,
        'num_classes': 1000, 'input_size': (3, 224, 224), 'pool_size': None,
        'crop_pct': .9, 'interpolation': 'bicubic',
        'mean': IMAGENET_DEFAULT_MEAN, 'std': IMAGENET_DEFAULT_STD,
        'first_conv': 'patch_embed.proj', 'classifier': 'head',
        **kwargs
    }

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


def dropchannel(x, drop_prob=0.5, p=0.33):
    if random.random() < drop_prob:
        return x
    # x: BxNxC
    keep_prob = 1 - p
    shape = (x.shape[0],) + (1,) * (x.ndim - 2) + (x.shape[-1],)
    # shape = (x.shape[0], 1, x.shape[-1])
    # random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
    random_tensor = torch.empty(shape).bernoulli_(keep_prob).cuda()
    if keep_prob > 0.0:
        random_tensor.div_(keep_prob)
    return x * random_tensor


def droptoken(x, drop_prob=0.5, p=0.33):
    if random.random() < drop_prob:
        return x
    # x: BxNxC
    keep_prob = 1 - p
    shape = (x.shape[0], x.shape[1], 1)
    random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
    if keep_prob > 0.0:
        random_tensor.div_(keep_prob)
    return x * random_tensor


class GlobalFilter(nn.Module):
    def __init__(self, dim, h=14, w=8, drop_mode=0, drop_p=0.1, drop_token_or_channel=0,
                 Fourier_flag=0, Fourier_swap=0, mask_radio=0.1, mask_alpha=0.5,
                 noise_mode=1, severity=0.1, sig=0.1, domain_mix=0, mix_test=0, drop_whole=0, global_filter=1,
                 low_or_high=0, uncertainty_model=0, perturb_prob=0.5, beta_flag=0, statistics_mode=0,
                 Fourier_high_enhance=0, Fourier_drop_flag=0, Fourier_drop_apply_p=0., Fourier_drop_p=0.,
                 noise_mix_flag=0, uncertainty_factor=1.0, uncertainty_sample=0, noise_unif_oneside=0,
                 noise_type=0, noise_layer_flag=0, gauss_or_uniform=0, miu_mean_flag=0,
                 only_low_high_flag=0):
        super().__init__()
        self.complex_weight = nn.Parameter(torch.randn(h, w, dim, 2, dtype=torch.float32) * 0.02)
        self.w = w
        self.h = h
        self.drop_mode = drop_mode
        self.drop_p = drop_p
        self.drop_token_or_channel = drop_token_or_channel

        self.Fourier_flag = Fourier_flag
        self.Fourier_swap = Fourier_swap
        self.mask_radio = mask_radio

        self.noise_mode = noise_mode
        self.severity = severity
        self.sig = sig
        self.noise_unif_oneside = noise_unif_oneside
        self.noise_type = noise_type

        self.noise_layer_flag = noise_layer_flag

        self.domain_mix = domain_mix
        self.mix_test = mix_test

        self.drop_whole = drop_whole
        self.alpha = mask_alpha

        self.global_filter = global_filter
        self.low_or_high = low_or_high

        self.eps = 1e-6
        self.factor = uncertainty_factor
        self.uncertainty_model = uncertainty_model
        self.uncertainty_sample = uncertainty_sample
        self.p = perturb_prob
        self.gauss_or_uniform = gauss_or_uniform

        self.beta_flag = beta_flag
        self.beta = torch.distributions.Beta(self.alpha, self.alpha)

        self.statistics_mode = statistics_mode
        self.Fourier_high_enhance = Fourier_high_enhance

        self.Fourier_drop_flag = Fourier_drop_flag
        self.Fourier_drop_apply_p = Fourier_drop_apply_p
        self.Fourier_drop_p = Fourier_drop_p

        self.noise_mix_flag = noise_mix_flag
        self.miu_mean_flag = miu_mean_flag

        self.only_low_high_flag = only_low_high_flag


    def spectrum_mix(self, img_fft, alpha=1.0, ratio=1.0, Fourier_swap=0, domain_mix=0, low_or_high=0,
                     drop_flag=0, drop_apply_p=0.5, drop_p=0.5, high_pass_enhance=0, only_low_high_flag=0):
        """Input image size: ndarray of [H, W, C]"""
        if random.random() > self.p:
            return img_fft

        batch_size, h, w, c = img_fft.shape
        if Fourier_swap == 1:
            lam = 1.0
        else:
            if self.beta_flag == 1:
                lam = self.beta.sample((batch_size, 1, 1, 1)).cuda()
            else:
                lam = (torch.rand(batch_size, 1, 1, 1) * alpha).cuda()
                # lam = np.random.uniform(0, alpha)

        img_abs, img_pha = torch.abs(img_fft), torch.angle(img_fft)
        img_abs = torch.fft.fftshift(img_abs, dim=(1, 2))
        h_crop = int(h * sqrt(ratio))
        w_crop = int(w * sqrt(ratio))
        h_start = h // 2 - h_crop // 2
        # w_start = w // 2 - w_crop // 2
        w_start = w - w_crop

        img1_abs_ = img_abs.clone()

        batch_size = img_fft.shape[0]
        if domain_mix == 1 or domain_mix == 2:
            # use different domain to conduct mix
            perm = torch.arange(batch_size) # 0, b-1
            perm_a, perm_b, perm_c = perm.chunk(3)   # split into three parts
            domain_batch_size = batch_size // 3
            perm_a = perm_a[torch.randperm(domain_batch_size)]
            perm_b = perm_b[torch.randperm(domain_batch_size)]
            perm_c = perm_c[torch.randperm(domain_batch_size)]
            if domain_mix == 1:
                if random.random() < 0.5:
                    perm = torch.cat([perm_b, perm_c, perm_a], dim=0)
                else:
                    perm = torch.cat([perm_c, perm_a, perm_b], dim=0)
            else:
                perm = torch.cat([perm_a, perm_b, perm_c], dim=0)
        else:
            perm = torch.randperm(batch_size)
        img2_abs_ = img1_abs_[perm]

        if low_or_high == 0:
            if only_low_high_flag == 1:
                img_abs[:, h_start:h_start + h_crop, w_start:, :] = 0.
            else:
                if drop_flag and random.random() < drop_apply_p:
                    drop_prob = torch.ones(size=(batch_size, 1, 1, 1)) * drop_p
                    lam = torch.bernoulli(drop_prob).cuda()
                    img_abs[:, h_start:h_start + h_crop, w_start:, :] = \
                        lam * img1_abs_[:, h_start:h_start + h_crop, w_start:, :].mean() + \
                        (1 - lam) * img1_abs_[:, h_start:h_start + h_crop, w_start:, :]
                else:
                    img_abs[:, h_start:h_start + h_crop, w_start:, :] \
                        = lam * img2_abs_[:, h_start:h_start + h_crop, w_start:, :] + \
                          (1 - lam) * img1_abs_[:, h_start:h_start + h_crop, w_start:, :]

            img_abs_high = img_abs.clone()
            img_abs_high[:, h_start:h_start + h_crop, w_start:, :] = 0
            if high_pass_enhance == 1:
                img_abs = img_abs + img_abs_high
            img_abs_high = torch.fft.ifftshift(img_abs_high, dim=(1, 2))

            img_abs = torch.fft.ifftshift(img_abs, dim=(1, 2))
        else:
            img_abs_high = img_abs.clone()
            img_abs_high[:, h_start:h_start + h_crop, w_start:, :] = 0
            img_abs_high = torch.fft.ifftshift(img_abs_high, dim=(1, 2))

            img_abs = torch.fft.ifftshift(img_abs, dim=(1, 2))

            if only_low_high_flag == 1:
                img_abs[:, h_start:h_start + h_crop, w_start:, :] = 0.
            else:
                img_abs[:, h_start:h_start + h_crop, w_start:, :] \
                    = lam * img2_abs_[:, h_start:h_start + h_crop, w_start:, :] + \
                      (1 - lam) * img1_abs_[:, h_start:h_start + h_crop, w_start:, :]

        img_high = img_abs_high * (np.e ** (1j * img_pha))
        img_mix = img_abs * (np.e ** (1j * img_pha))
        return img_mix, img_high

    def spectrum_mix_statistics(self, img_fft, alpha=0.3, ratio=1.0, domain_mix=0, statistics_mode=0):
        """Input image size: ndarray of [H, W, C]"""
        """statistics_mode: 0: mean and var; 1: histogram"""
        if random.random() > self.p:
            return img_fft

        batch_size, h, w, c = img_fft.shape
        if alpha == 0:  # swap style
            lam = 1.0
        else:
            if self.beta_flag == 1:
                lam = self.beta.sample((batch_size, 1, 1, 1)).cuda()
            else:
                lam = (torch.rand(batch_size, 1, 1, 1) * alpha).cuda()
                # lam = np.random.uniform(0, alpha)

        img_abs, img_pha = torch.abs(img_fft), torch.angle(img_fft)
        img_abs = torch.fft.fftshift(img_abs, dim=(1, 2))
        h_crop = int(h * sqrt(ratio))
        w_crop = int(w * sqrt(ratio))
        h_start = h // 2 - h_crop // 2
        # w_start = w // 2 - w_crop // 2
        w_start = w - w_crop

        if statistics_mode == 0:
            # 1: compute the mean and var of low-frequency amplitude
            abs_miu1 = torch.mean(img_abs[:, h_start:h_start + h_crop, w_start:, :], dim=(1, 2), keepdim=True)
            abs_var1 = torch.var(img_abs[:, h_start:h_start + h_crop, w_start:, :], dim=(1, 2), keepdim=True)
            abs_sig1 = (abs_var1 + self.eps).sqrt()

            abs_miu1 = abs_miu1.detach()
            abs_sig1 = abs_sig1.detach()
        else:
            # 1: compute the sorted value and index
            # abs_view = img_abs[:, h_start:h_start + h_crop, w_start:, :].view(batch_size, -1, c)
            abs_view = img_abs[:, h_start:h_start + h_crop, w_start:, :].reshape(batch_size, -1, c)
            value_abs, index_abs = torch.sort(abs_view, dim=1)

        # 2: permute along the batch dimension
        batch_size = img_fft.shape[0]
        if domain_mix == 1 or domain_mix == 2:
            # use different domain to conduct mix
            perm = torch.arange(batch_size) # 0, b-1
            perm_a, perm_b, perm_c = perm.chunk(3)   # split into three parts
            domain_batch_size = batch_size // 3
            perm_a = perm_a[torch.randperm(domain_batch_size)]
            perm_b = perm_b[torch.randperm(domain_batch_size)]
            perm_c = perm_c[torch.randperm(domain_batch_size)]
            if domain_mix == 1:
                if random.random() < 0.5:
                    perm = torch.cat([perm_b, perm_c, perm_a], dim=0)
                else:
                    perm = torch.cat([perm_c, perm_a, perm_b], dim=0)
            else:
                perm = torch.cat([perm_a, perm_b, perm_c], dim=0)
        else:
            perm = torch.randperm(batch_size)

        if statistics_mode == 0:
            # 3: mix the statistics between randomly-selected two samples
            abs_miu2, abs_sig2 = abs_miu1[perm], abs_sig1[perm]
            beta = lam * abs_miu2 + (1 - lam) * abs_miu1
            gamma = lam * abs_sig2 + (1 - lam) * abs_sig2
            img_abs[:, h_start:h_start + h_crop, w_start:, :] = gamma * (
                    img_abs[:, h_start:h_start + h_crop, w_start:, :] - abs_miu1) / abs_sig1 + beta
        else:
            # 3: mix the histogram between randomly-selected two samples
            inverse_index = torch.argsort(index_abs, dim=1)
            abs_view_copy = torch.gather(value_abs[perm], dim=1, index=inverse_index)

            abs_view = abs_view.view(batch_size, h_crop, -1, c)
            abs_view_copy = abs_view_copy.view(batch_size, h_crop, -1, c)
            new_abs = abs_view + (abs_view_copy - abs_view.detach()) * (1 - lam)
            img_abs[:, h_start:h_start + h_crop, w_start:, :] = new_abs

        img_abs = torch.fft.ifftshift(img_abs, dim=(1, 2))
        img_mix = img_abs * (np.e ** (1j * img_pha))
        return img_mix

    def gaussian_noise(self, feat, mean, sig):
        noise = torch.randn(feat.shape) * sig + mean
        feat_noised = feat + noise.cuda()
        # feat_noise = np.uint8(np.clip(feat_noise, 0, 255))
        return feat_noised

    def uniform_noise(self, feat, severity, one_side=0):
        if one_side == 0:
            noise = (torch.rand(feat.shape) - 0.5) * 2. * severity + 1.     # 1-severity ~ 1+severity
        else:
            noise = - torch.rand(feat.shape) * severity + 1     # 1-severity ~ 1
        feat_noised = feat * noise.cuda()
        return feat_noised

    def _reparameterize(self, mu, std, epsilon_norm):
        # epsilon = torch.randn_like(std) * self.factor
        epsilon = epsilon_norm * self.factor
        mu_t = mu + epsilon * std
        return mu_t

    def spectrum_noise(self, img_fft, alpha=1.0, ratio=1.0, noise_mode=1, severity=0.1, sig=0.1, drop_whole=0,
                       low_or_high=0, uncertainty_model=0, gauss_or_uniform=0, miu_mean_flag=0):
        """Input image size: ndarray of [H, W, C]"""
        """noise_mode: 1 amplitude; 2: phase 3:both"""
        """uncertainty_model: 1 batch-wise modeling 2: channel-wise modeling 3:token-wise modeling"""
        if random.random() > self.p:
            return img_fft

        batch_size, h, w, c = img_fft.shape
        if alpha == 0:
            lam = 1.0
        else:
            if self.beta_flag == 1:
                lam = self.beta.sample((batch_size, 1, 1, 1)).cuda()
            else:
                lam = (torch.rand(batch_size, 1, 1, 1) * alpha).cuda()

        img_abs, img_pha = torch.abs(img_fft), torch.angle(img_fft)
        img_abs = torch.fft.fftshift(img_abs, dim=(1, 2))
        h_crop = int(h * sqrt(ratio))
        w_crop = int(w * sqrt(ratio))
        h_start = h // 2 - h_crop // 2
        # w_start = w // 2 - w_crop // 2
        w_start = w - w_crop

        # img_abs_ = img_abs.clone().detach()
        img_abs_ = img_abs.clone()
        if noise_mode == 1:
            if uncertainty_model != 0:
                if uncertainty_model == 1:
                    # batch level modeling
                    miu = torch.mean(img_abs_[:, h_start:h_start + h_crop, w_start:, :], dim=(1, 2), keepdim=True)
                    var = torch.var(img_abs_[:, h_start:h_start + h_crop, w_start:, :], dim=(1, 2), keepdim=True)
                    sig = (var + self.eps).sqrt()  # Bx1x1xC

                    var_of_miu = torch.var(miu, dim=0, keepdim=True)
                    var_of_sig = torch.var(sig, dim=0, keepdim=True)
                    sig_of_miu = (var_of_miu + self.eps).sqrt().repeat(miu.shape[0], 1, 1, 1)
                    sig_of_sig = (var_of_sig + self.eps).sqrt().repeat(miu.shape[0], 1, 1, 1)  # Bx1x1xC

                    if gauss_or_uniform == 0:
                        if self.uncertainty_sample == 0:
                            epsilon_norm_miu = torch.randn_like(sig_of_miu) # N(0,1)
                            epsilon_norm_sig = torch.randn_like(sig_of_sig)
                        else:
                            epsilon_norm_miu = torch.randn_like(sig_of_miu)
                            epsilon_norm_sig = epsilon_norm_miu
                        if miu_mean_flag == 1:
                            miu_mean = torch.mean(miu, dim=0, keepdim=True)
                            sig_mean = torch.mean(sig, dim=0, keepdim=True)
                        else:
                            miu_mean = miu
                            sig_mean = sig
                        beta = self._reparameterize(mu=miu_mean, std=sig_of_miu, epsilon_norm=epsilon_norm_miu)
                        gamma = self._reparameterize(mu=sig_mean, std=sig_of_sig, epsilon_norm=epsilon_norm_sig)
                    elif gauss_or_uniform == 1:
                        epsilon_norm_miu = torch.rand_like(sig_of_miu) * 2 - 1. # U(-1,1)
                        epsilon_norm_sig = torch.rand_like(sig_of_sig) * 2 - 1.
                        beta = self._reparameterize(mu=miu, std=sig_of_miu, epsilon_norm=epsilon_norm_miu)
                        gamma = self._reparameterize(mu=sig, std=sig_of_sig, epsilon_norm=epsilon_norm_sig)
                    else:
                        epsilon_norm_miu = torch.randn_like(sig_of_miu)  # N(0,1)
                        epsilon_norm_sig = torch.randn_like(sig_of_sig)
                        beta = self._reparameterize(mu=miu, std=1., epsilon_norm=epsilon_norm_miu)
                        gamma = self._reparameterize(mu=sig, std=1., epsilon_norm=epsilon_norm_sig)

                    # adjust statistics for each sample
                    img_abs[:, h_start:h_start + h_crop, w_start:, :] = gamma * (
                            img_abs[:, h_start:h_start + h_crop, w_start:, :] - miu) / sig + beta

                    # inter-sample mixing
                    if self.noise_mix_flag == 1:
                        img1_abs_ = img_abs.clone()
                        batch_size = img_fft.shape[0]
                        perm = torch.randperm(batch_size)
                        img2_abs_ = img1_abs_[perm]
                        img_abs[:, h_start:h_start + h_crop, w_start:, :] \
                            = lam * img2_abs_[:, h_start:h_start + h_crop, w_start:, :] + \
                              (1 - lam) * img1_abs_[:, h_start:h_start + h_crop, w_start:, :]

                elif uncertainty_model == 2:
                    # element level modeling
                    miu_of_elem = torch.mean(img_abs_[:, h_start:h_start + h_crop, w_start:, :], dim=0, keepdim=True)
                    var_of_elem = torch.var(img_abs_[:, h_start:h_start + h_crop, w_start:, :], dim=0, keepdim=True)
                    sig_of_elem = (var_of_elem + self.eps).sqrt()   # 1xHxWxC

                    if gauss_or_uniform == 0:
                        if miu_mean_flag == 1:
                            img_abs[:, h_start:h_start + h_crop, w_start:, :] = miu_of_elem.repeat(batch_size, 1, 1, 1)
                        epsilon_sig = torch.randn_like(img_abs[:, h_start:h_start + h_crop, w_start:, :])   # BxHxWxC N(0,1)
                        gamma = epsilon_sig * sig_of_elem * self.factor
                    elif gauss_or_uniform == 1:
                        epsilon_sig = torch.rand_like(img_abs[:, h_start:h_start + h_crop, w_start:, :]) * 2 - 1.  # U(-1,1)
                        gamma = epsilon_sig * sig_of_elem * self.factor
                    else:
                        epsilon_sig = torch.randn_like(
                            img_abs[:, h_start:h_start + h_crop, w_start:, :])  # BxHxWxC N(0,1)
                        gamma = epsilon_sig * self.factor

                    img_abs[:, h_start:h_start + h_crop, w_start:, :] = img_abs[:, h_start:h_start + h_crop, w_start:, :] + gamma

            else:
                if drop_whole == 1:
                    img_abs[:, h_start:h_start + h_crop, w_start:, :] = img_abs[:, h_start:h_start + h_crop, w_start:, :].mean(dim=(1, 2), keepdim=True)
                else:
                    if self.noise_type == 0:
                        img_abs[:, h_start:h_start + h_crop, w_start:, :] = \
                            self.uniform_noise(img_abs[:, h_start:h_start + h_crop, w_start:, :], severity=severity,
                                               one_side=self.noise_unif_oneside)
                    else:
                        img_abs[:, h_start:h_start + h_crop, w_start:, :] = \
                            self.gaussian_noise(img_abs[:, h_start:h_start + h_crop, w_start:, :], mean=0, sig=sig)
            img_abs = torch.fft.ifftshift(img_abs, dim=(1, 2))
        elif noise_mode == 2:
            if drop_whole == 1:
                img_pha = img_pha.mean()
            else:
                if self.noise_type == 0:
                    img_pha[:, h_start:h_start + h_crop, w_start:, :] = \
                        self.uniform_noise(img_pha[:, h_start:h_start + h_crop, w_start:, :], severity=severity,
                                           one_side=self.noise_unif_oneside)
                else:
                    img_pha[:, h_start:h_start + h_crop, w_start:, :] = self.gaussian_noise(
                        img_pha[:, h_start:h_start + h_crop, w_start:, :], mean=0, sig=sig)
            img_abs = torch.fft.ifftshift(img_abs, dim=(1, 2))
        elif noise_mode == 3:
            img_abs[:, h_start:h_start + h_crop, w_start:, :] = self.uniform_noise(
                img_abs[:, h_start:h_start + h_crop, w_start:, :], severity=severity)
            img_pha[:, h_start:h_start + h_crop, w_start:, :] = self.gaussian_noise(
                img_abs[:, h_start:h_start + h_crop, w_start:, :], mean=0, sig=sig)
            img_abs = torch.fft.ifftshift(img_abs, dim=(1, 2))
        else:
            pass
        img_mix = img_abs * (np.e ** (1j * img_pha))
        return img_mix

    def forward(self, x, FourierMix_training=False, layer_index=0, spatial_size=None):
        if self.global_filter == 0:
            return x

        B, N, C = x.shape
        if spatial_size is None:
            a = b = int(math.sqrt(N))
        else:
            a, b = spatial_size

        if self.drop_mode == 1:
            if self.drop_token_or_channel == 1:
                x = dropchannel(x, p=self.drop_p)
            else:
                x = droptoken(x, p=self.drop_p)
        x = x.view(B, a, b, C)

        x = x.to(torch.float32)

        x = torch.fft.rfft2(x, dim=(1, 2), norm='ortho')
        x_mix = None
        x_high = None
        if FourierMix_training or self.mix_test:
            if self.noise_mode != 0 and self.noise_layer_flag == 1:
                x = self.spectrum_noise(x, alpha=self.alpha, ratio=self.mask_radio, noise_mode=self.noise_mode,
                                        severity=self.severity, sig=self.sig, drop_whole=self.drop_whole,
                                        uncertainty_model=self.uncertainty_model,
                                        gauss_or_uniform=self.gauss_or_uniform,
                                        miu_mean_flag=self.miu_mean_flag)

            if self.Fourier_flag == 1:
                x, x_high = self.spectrum_mix(x, alpha=self.alpha, ratio=self.mask_radio,
                                                  Fourier_swap=self.Fourier_swap,
                                                  domain_mix=self.domain_mix, low_or_high=self.low_or_high,
                                                  drop_flag=self.Fourier_drop_flag,
                                                  drop_apply_p=self.Fourier_drop_apply_p,
                                                  drop_p=self.Fourier_drop_p,
                                                  high_pass_enhance=self.Fourier_high_enhance)
            elif self.Fourier_flag == 2:
                x_mix = self.spectrum_mix_statistics(x, alpha=self.alpha, ratio=self.mask_radio, domain_mix=self.domain_mix,
                                                 statistics_mode=self.statistics_mode)
            else:
                x_mix = x

        weight = torch.view_as_complex(self.complex_weight)
        x = x * weight
        if self.Fourier_high_enhance == 2 and x_high != None:
            x = (x + x_high) / 2.

        if self.drop_mode == 2:
            if self.drop_token_or_channel == 1:
                x = dropchannel(x, p=self.drop_p)
            else:
                x = droptoken(x, p=self.drop_p)
        x = torch.fft.irfft2(x, s=(a, b), dim=(1, 2), norm='ortho')
        # x.real x.imag
        # torch.angle() torch.abs()

        x = x.reshape(B, N, C)
        if self.drop_mode == 3:
            # dropout after Fourier
            if self.drop_token_or_channel == 1:
                x = dropchannel(x, p=self.drop_p)
            else:
                x = droptoken(x, p=self.drop_p)

        return x

class Block(nn.Module):
    def __init__(self, dim, mlp_ratio=4., drop=0., drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, h=14, w=8,
                 fdrop_mode=0, fdrop_p=0.1, fdrop_t_or_c=0,
                 Fourier_flag=0, Fourier_swap=0, mask_radio=0.1, mask_alpha=0.5,
                 noise_mode=1, severity=0.1, sig=0.1, domain_mix=0, mix_test=0, drop_whole=0, global_filter=1,
                 low_or_high=0, uncertainty_model=0, perturb_prob=0.5, beta_flag=0, statistics_mode=0,
                 Fourier_high_enhance=0, Fourier_drop_flag=0, Fourier_drop_apply_p=0., Fourier_drop_p=0.,
                 noise_mix_flag=0, uncertainty_factor=1.0, uncertainty_sample=0, noise_unif_oneside=0,
                 noise_type=0, noise_layers=None, gauss_or_uniform=0, miu_mean_flag=0,
                 only_low_high_flag=0):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.filter = GlobalFilter(dim, h=h, w=w, drop_mode=fdrop_mode, drop_p=fdrop_p, drop_token_or_channel=fdrop_t_or_c,
                                   Fourier_flag=Fourier_flag, Fourier_swap=Fourier_swap, mask_radio=mask_radio,
                                   mask_alpha=mask_alpha,
                                   noise_mode=noise_mode, severity=severity, sig=sig, domain_mix=domain_mix,
                                   mix_test=mix_test, drop_whole=drop_whole, global_filter=global_filter,
                                   low_or_high=low_or_high, uncertainty_model=uncertainty_model, perturb_prob=perturb_prob,
                                   beta_flag=beta_flag, statistics_mode=statistics_mode, Fourier_high_enhance=Fourier_high_enhance,
                                   Fourier_drop_flag=Fourier_drop_flag, Fourier_drop_apply_p=Fourier_drop_apply_p,
                                   Fourier_drop_p=Fourier_drop_p, noise_mix_flag=noise_mix_flag,
                                   uncertainty_factor=uncertainty_factor, uncertainty_sample=uncertainty_sample,
                                   noise_unif_oneside=noise_unif_oneside, noise_type=noise_type, noise_layer_flag=1,
                                   gauss_or_uniform=gauss_or_uniform, miu_mean_flag=miu_mean_flag,
                                   only_low_high_flag=only_low_high_flag)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, input):
        x, FourierMix_training = input
        x = x + self.drop_path(self.mlp(self.norm2(self.filter(self.norm1(x), FourierMix_training))))
        # Drop_path: In residual architecture, drop the current block for randomly seleted samples
        return x

class BlockLayerScale(nn.Module):
    def __init__(self, dim, mlp_ratio=4., drop=0., drop_path=0., act_layer=nn.GELU, 
                norm_layer=nn.LayerNorm, h=14, w=8, init_values=1e-5, fdrop_mode=0, fdrop_p=0.1,
                 fdrop_t_or_c=0,
                Fourier_flag=0, Fourier_swap=0, mask_radio=0.1, mask_alpha=0.5,
                 noise_mode=1, severity=0.1, sig=0.1, domain_mix=0, mix_test=0, drop_whole=0, global_filter=1,
                 low_or_high=0, uncertainty_model=0, perturb_prob=0.5, beta_flag=0, statistics_mode=0,
                 Fourier_high_enhance=0, Fourier_drop_flag=0, Fourier_drop_apply_p=0., Fourier_drop_p=0.,
                 noise_mix_flag=0, uncertainty_factor=1.0, uncertainty_sample=0, noise_unif_oneside=0,
                 noise_type=0, layer_index=0, noise_layers=[0, 1, 2, 3], gauss_or_uniform=0,
                 miu_mean_flag=0, only_low_high_flag=0):
        super().__init__()
        self.norm1 = norm_layer(dim)

        if layer_index in noise_layers:
            noise_layer_flag = 1
        else:
            noise_layer_flag = 0
        self.filter = GlobalFilter(dim, h=h, w=w, drop_mode=fdrop_mode, drop_p=fdrop_p, drop_token_or_channel=fdrop_t_or_c,
                                   Fourier_flag=Fourier_flag, Fourier_swap=Fourier_swap, mask_radio=mask_radio,
                                   mask_alpha=mask_alpha,
                                   noise_mode=noise_mode, severity=severity, sig=sig, domain_mix=domain_mix,
                                   mix_test=mix_test, drop_whole=drop_whole, global_filter=global_filter,
                                   low_or_high=low_or_high, uncertainty_model=uncertainty_model, perturb_prob=perturb_prob,
                                   beta_flag=beta_flag, statistics_mode=statistics_mode, Fourier_high_enhance=Fourier_high_enhance,
                                   Fourier_drop_flag=Fourier_drop_flag, Fourier_drop_apply_p=Fourier_drop_apply_p,
                                   Fourier_drop_p=Fourier_drop_p, noise_mix_flag=noise_mix_flag,
                                   uncertainty_factor=uncertainty_factor, uncertainty_sample=uncertainty_sample,
                                   noise_unif_oneside=noise_unif_oneside, noise_type=noise_type,
                                   noise_layer_flag=noise_layer_flag, gauss_or_uniform=gauss_or_uniform,
                                   miu_mean_flag=miu_mean_flag, only_low_high_flag=only_low_high_flag)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        self.gamma = nn.Parameter(init_values * torch.ones((dim)), requires_grad=True)

        self.layer_index = layer_index  # where is the block in

    def forward(self, input):
        x, FourierMix_training = input
        x = x + self.drop_path(self.gamma * self.mlp(self.norm2(self.filter(self.norm1(x), FourierMix_training, self.layer_index))))
        return x, FourierMix_training

class PatchEmbed(nn.Module):
    """ Image to Patch Embedding
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768, domain_mix=0, mix_alpha=1.0,
                 domain_or_random=0, Fourier_mix=0, Fourier_radio=0.5, Fourier_alpha=1.0, Fourier_domainmix=0):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0])
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

        self.domain_mix = domain_mix
        self.mix_alpha = mix_alpha
        self.embed_dim = embed_dim
        self.beta = torch.distributions.Beta(self.mix_alpha, self.mix_alpha)

        self.domain_or_random = domain_or_random
        self.Fourier_mix = Fourier_mix
        self.Fourier_radio = Fourier_radio
        self.Fourier_alpha = Fourier_alpha
        self.Fourier_domainmix = Fourier_domainmix

    def init_weights(self, stop_grad_conv1=0):
        val = math.sqrt(6. / float(3 * reduce(mul, self.patch_size, 1) + self.embed_dim))
        nn.init.uniform_(self.proj.weight, -val, val)
        nn.init.zeros_(self.proj.bias)

        if stop_grad_conv1:
            self.proj.weight.requires_grad = False
            self.proj.bias.requires_grad = False

    def fourier_mix(self, img, alpha=1.0, radio=1.0, domain_mix=0):
        batch_size, c, h, w = img.shape
        # if alpha == 0:
        #     lam = 1.0
        # else:
        #     if self.beta_flag == 1:
        #         lam = self.beta.sample((batch_size, 1, 1, 1)).cuda()
        #     else:
        #         lam = (torch.rand(batch_size, 1, 1, 1) * alpha).cuda()
        #         # lam = np.random.uniform(0, alpha)
        # lam = (torch.rand(batch_size, 1, 1, 1) * alpha).cuda()
        lam = self.beta.sample((batch_size, 1, 1, 1)).cuda()

        img_fft = torch.fft.rfft2(img, dim=(2, 3), norm='ortho')
        img_abs, img_pha = torch.abs(img_fft), torch.angle(img_fft)
        img_abs = torch.fft.fftshift(img_abs, dim=(2, 3))
        h_crop = int(h * sqrt(radio))
        w_crop = int(w * sqrt(radio))
        h_start = h // 2 - h_crop // 2
        w_start = w - w_crop
        img1_abs_ = img_abs.clone()
        if domain_mix == 1 or domain_mix == 2:
            # use different domain to conduct mix
            perm = torch.arange(batch_size)  # 0, b-1
            perm_a, perm_b, perm_c = perm.chunk(3)  # split into three parts
            domain_batch_size = batch_size // 3
            perm_a = perm_a[torch.randperm(domain_batch_size)]
            perm_b = perm_b[torch.randperm(domain_batch_size)]
            perm_c = perm_c[torch.randperm(domain_batch_size)]
            if domain_mix == 1:
                if random.random() < 0.5:
                    perm = torch.cat([perm_b, perm_c, perm_a], dim=0)
                else:
                    perm = torch.cat([perm_c, perm_a, perm_b], dim=0)
            else:
                perm = torch.cat([perm_a, perm_b, perm_c], dim=0)
        else:
            perm = torch.randperm(batch_size)
        img2_abs_ = img1_abs_[perm]
        img_abs[:, h_start:h_start + h_crop, w_start:, :] \
            = lam * img2_abs_[:, h_start:h_start + h_crop, w_start:, :] + \
              (1 - lam) * img1_abs_[:, h_start:h_start + h_crop, w_start:, :]
        img_abs = torch.fft.ifftshift(img_abs, dim=(2, 3))
        img_mix = img_abs * (np.e ** (1j * img_pha))
        img_mix = torch.fft.irfft2(img_mix, s=(h, w), dim=(2, 3), norm='ortho')
        return img_mix

    def forward(self, x, domain_mix=0, mix_alpha=0.1, domain_or_random=0):
        B, C, H, W = x.shape
        # FIXME look at relaxing size constraints
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        if self.Fourier_mix == 1 and self.training:
            x = self.fourier_mix(x, alpha=self.Fourier_alpha, radio=self.Fourier_radio,
                                 domain_mix=self.Fourier_domainmix)
        x = self.proj(x).flatten(2).transpose(1, 2) # BxCxHxW -> BxNxC , N=(224/4)^2=3136, C=64

        # swap the patch between different domains
        if domain_mix != 0 and self.training:
            B, N, C = x.shape
            # generate Mask: BxN
            # lam = self.beta.sample((B, 1, 1)).cuda()    # BxNx1
            lam = np.random.beta(mix_alpha, mix_alpha)
            random_mask = lam + torch.rand((B, N, 1), dtype=x.dtype, device=x.device)
            random_mask.floor_()  # binarize

            x1_ = x.clone()
            # use different domain to conduct mix
            if domain_or_random == 0:
                perm = torch.arange(B)  # 0, b-1
                perm_a, perm_b, perm_c = perm.chunk(3)  # split into three parts
                domain_batch_size = B // 3
                perm_a = perm_a[torch.randperm(domain_batch_size)]
                perm_b = perm_b[torch.randperm(domain_batch_size)]
                perm_c = perm_c[torch.randperm(domain_batch_size)]
                if random.random() < 0.5:
                    perm = torch.cat([perm_b, perm_c, perm_a], dim=0)
                else:
                    perm = torch.cat([perm_c, perm_a, perm_b], dim=0)
            else:
                perm = torch.randperm(B)
            x2_ = x1_[perm]

            if domain_mix == 1:
                x = x1_ * random_mask + x2_ * (1 - random_mask)
            elif domain_mix == 2:
                x = x1_ * lam + x2_ * (1 - lam)
            # return x, [lam.squeeze(), perm]
            return x, [lam, perm]
        # x = self.proj(x)
        # x = x.flatten(2)
        # x = x.transpose(1, 2)
        return x, [None, None]


class DownLayer(nn.Module):
    """ Image to Patch Embedding
    """
    def __init__(self, img_size=56, dim_in=64, dim_out=128):
        super().__init__()
        self.img_size = img_size
        self.dim_in = dim_in
        self.dim_out = dim_out
        self.proj = nn.Conv2d(dim_in, dim_out, kernel_size=2, stride=2)
        self.num_patches = img_size * img_size // 4

    def forward(self, x, domain_mix=0, mix_alpha=0.1, domain_or_random=0):
        B, N, C = x.size()
        x = x.view(B, self.img_size, self.img_size, C).permute(0, 3, 1, 2)
        x = self.proj(x).permute(0, 2, 3, 1)
        x = x.reshape(B, -1, self.dim_out)

        if domain_mix != 0 and self.training:
            B, N, C = x.shape
            lam = np.random.beta(mix_alpha, mix_alpha)
            random_mask = lam + torch.rand((B, N, 1), dtype=x.dtype, device=x.device)
            random_mask.floor_()  # binarize

            x1_ = x.clone()
            if domain_or_random == 0:
                perm = torch.arange(B)  # 0, b-1
                perm_a, perm_b, perm_c = perm.chunk(3)  # split into three parts
                domain_batch_size = B // 3
                perm_a = perm_a[torch.randperm(domain_batch_size)]
                perm_b = perm_b[torch.randperm(domain_batch_size)]
                perm_c = perm_c[torch.randperm(domain_batch_size)]
                if random.random() < 0.5:
                    perm = torch.cat([perm_b, perm_c, perm_a], dim=0)
                else:
                    perm = torch.cat([perm_c, perm_a, perm_b], dim=0)
            else:
                perm = torch.randperm(B)
            x2_ = x1_[perm]

            if domain_mix == 1:
                x = x1_ * random_mask + x2_ * (1 - random_mask)
            elif domain_mix == 2:
                x = x1_ * lam + x2_ * (1 - lam)
            # return x, [lam.squeeze(), perm]
            return x, [lam, perm]
        return x, [None, None]


class GFNet(nn.Module):

    def __init__(self, img_size=224, patch_size=16, in_chans=3, num_classes=1000, embed_dim=768, depth=12,
                 mlp_ratio=4., representation_size=None, uniform_drop=False,
                 drop_rate=0., drop_path_rate=0., norm_layer=None, 
                 dropcls=0, fdrop_mode=0, fdrop_p=0.1):
        """
        Args:
            img_size (int, tuple): input image size
            patch_size (int, tuple): patch size
            in_chans (int): number of input channels
            num_classes (int): number of classes for classification head
            embed_dim (int): embedding dimension
            depth (int): depth of transformer
            num_heads (int): number of attention heads
            mlp_ratio (int): ratio of mlp hidden dim to embedding dim
            qkv_bias (bool): enable bias for qkv if True
            qk_scale (float): override default qk scale of head_dim ** -0.5 if set
            representation_size (Optional[int]): enable and set representation layer (pre-logits) to this value if set
            drop_rate (float): dropout rate
            attn_drop_rate (float): attention dropout rate
            drop_path_rate (float): stochastic depth rate
            hybrid_backbone (nn.Module): CNN backbone to use in-place of PatchEmbed module
            norm_layer: (nn.Module): normalization layer
        """
        super().__init__()
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)

        self.patch_embed = PatchEmbed(
                img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches

        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)

        h = img_size // patch_size
        w = h // 2 + 1

        if uniform_drop:
            print('using uniform droppath with expect rate', drop_path_rate)
            dpr = [drop_path_rate for _ in range(depth)]  # stochastic depth decay rule
        else:
            print('using linear droppath with expect rate', drop_path_rate * 0.5)
            dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        # dpr = [drop_path_rate for _ in range(depth)]  # stochastic depth decay rule
        
        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim, mlp_ratio=mlp_ratio,
                drop=drop_rate, drop_path=dpr[i], norm_layer=norm_layer, h=h, w=w, fdrop_mode=fdrop_mode, fdrop_p=fdrop_p)
            for i in range(depth)])
        
        self.norm = norm_layer(embed_dim)

        # Representation layer
        if representation_size:
            self.num_features = representation_size
            self.pre_logits = nn.Sequential(OrderedDict([
                ('fc', nn.Linear(embed_dim, representation_size)),
                ('act', nn.Tanh())
            ]))
        else:
            self.pre_logits = nn.Identity()

        # Classifier head
        self.head = nn.Linear(self.num_features, num_classes) if num_classes > 0 else nn.Identity()

        if dropcls > 0:
            print('dropout %.2f before classifier' % dropcls)
            self.final_dropout = nn.Dropout(p=dropcls)
        else:
            self.final_dropout = nn.Identity()

        trunc_normal_(self.pos_embed, std=.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token'}

    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes, global_pool=''):
        self.num_classes = num_classes
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()

    def forward_features(self, x):
        B = x.shape[0]
        x = self.patch_embed(x)
        x = x + self.pos_embed
        x = self.pos_drop(x)

        for blk in self.blocks:
            x = blk(x)

        x = self.norm(x).mean(1)
        return x

    def forward(self, x):
        x = self.forward_features(x)
        x = self.final_dropout(x)
        x = self.head(x)
        return x


class GFNetPyramid(nn.Module):

    def __init__(self, img_size=224, patch_size=4, num_classes=1000, embed_dim=[64, 128, 256, 512], depth=[2,2,10,4],
                 mlp_ratio=[4, 4, 4, 4],
                 drop_rate=0., drop_path_rate=0., norm_layer=None, init_values=0.001, no_layerscale=False, dropcls=0,
                 fdrop_mode=0, fdrop_p=0.1, fdrop_t_or_c=0,
                 cdrop_mode=0, cdrop_p=0.1, cdrop_layers=[1, 2, 3],
                 Fourier_flag=0, Fourier_swap=0, mask_radio=0.1, mask_alpha=0.5, noise_mode=1, severity=0.1, sig=0.1,
                 domain_mix=0, mix_test=0, drop_whole=0, global_filter=1, low_or_high=0, uncertainty_model=0,
                 perturb_prob=0.5, beta_flag=0, statistics_mode=0, patch_domain_mix=0, patch_mix_alpha=1.0,
                 patch_embed_fix=0, patch_domain_or_random=0,
                 patch_Fourier_mix=0, patch_Fourier_radio=0.5, patch_Fourier_alpha=1.0, patch_Fourier_domainmix=0,
                 Fourier_high_enhance=0, Fourier_drop_flag=0, Fourier_drop_apply_p=0., Fourier_drop_p=0.,
                 noise_mix_flag=0, uncertainty_factor=1.0, uncertainty_sample=0, noise_unif_oneside=0,
                 noise_type=0, noise_layers=[0, 1, 2, 3], gauss_or_uniform=0,
                 MixStyle_flag=0, MixStyle_mix=0, MixStyle_layers=[0, 1, 2],
                 DSU_flag=0, miu_mean_flag=0, only_low_high_flag=0):
        """
        Args:
            img_size (int, tuple): input image size
            patch_size (int, tuple): patch size
            in_chans (int): number of input channels
            num_classes (int): number of classes for classification head
            embed_dim (int): embedding dimension
            depth (int): depth of transformer
            num_heads (int): number of attention heads
            mlp_ratio (int): ratio of mlp hidden dim to embedding dim
            qkv_bias (bool): enable bias for qkv if True
            qk_scale (float): override default qk scale of head_dim ** -0.5 if set
            representation_size (Optional[int]): enable and set representation layer (pre-logits) to this value if set
            drop_rate (float): dropout rate
            attn_drop_rate (float): attention dropout rate
            drop_path_rate (float): stochastic depth rate
            hybrid_backbone (nn.Module): CNN backbone to use in-place of PatchEmbed module
            norm_layer: (nn.Module): normalization layer
        """
        super().__init__()
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim[-1]  # num_features for consistency with other models
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)

        self.patch_embed = nn.ModuleList()
        
        patch_embed = PatchEmbed(
                img_size=img_size, patch_size=patch_size, in_chans=3, embed_dim=embed_dim[0],
            domain_mix=patch_domain_mix, mix_alpha=patch_mix_alpha, domain_or_random=patch_domain_or_random,
        Fourier_mix=patch_Fourier_mix, Fourier_radio=patch_Fourier_radio, Fourier_alpha=patch_Fourier_alpha,
            Fourier_domainmix=patch_Fourier_domainmix)
        num_patches = patch_embed.num_patches

        patch_embed.init_weights(stop_grad_conv1=patch_embed_fix)
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim[0]))

        self.patch_embed.append(patch_embed)

        sizes = [56, 28, 14, 7]
        for i in range(4):
            sizes[i] = sizes[i] * img_size // 224

        for i in range(3):
            patch_embed = DownLayer(sizes[i], embed_dim[i], embed_dim[i+1])
            num_patches = patch_embed.num_patches
            self.patch_embed.append(patch_embed)

        self.pos_drop = nn.Dropout(p=drop_rate)
        self.blocks = nn.ModuleList()

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depth))]  # stochastic depth decay rule
        cur = 0
        for i in range(4):
            h = sizes[i]
            w = h // 2 + 1

            if no_layerscale:
                print('using standard block')
                blk = nn.Sequential(*[
                    Block(
                    dim=embed_dim[i], mlp_ratio=mlp_ratio[i],
                    drop=drop_rate, drop_path=dpr[cur + j], norm_layer=norm_layer, h=h, w=w,
                        fdrop_mode=fdrop_mode, fdrop_p=fdrop_p, fdrop_t_or_c=fdrop_t_or_c,
                    Fourier_flag=Fourier_flag, Fourier_swap=Fourier_swap, mask_radio=mask_radio,
                        mask_alpha=mask_alpha,
                    noise_mode=noise_mode, severity=severity, sig=sig, domain_mix=domain_mix, mix_test=mix_test,
                    drop_whole=drop_whole, global_filter=global_filter, low_or_high=low_or_high,
                        uncertainty_model=uncertainty_model, perturb_prob=perturb_prob, beta_flag=beta_flag,
                    statistics_mode=statistics_mode, Fourier_high_enhance=Fourier_high_enhance,
                    Fourier_drop_flag=Fourier_drop_flag, Fourier_drop_apply_p=Fourier_drop_apply_p,
                        Fourier_drop_p=Fourier_drop_p, noise_mix_flag=noise_mix_flag,
                        uncertainty_factor=uncertainty_factor, uncertainty_sample=uncertainty_sample,
                        noise_unif_oneside=noise_unif_oneside, noise_type=noise_type, gauss_or_uniform=gauss_or_uniform,
                        miu_mean_flag=miu_mean_flag, only_low_high_flag=only_low_high_flag
                    )
                for j in range(depth[i])
                ])
            else:
                print('using layerscale block')
                blk = nn.Sequential(*[
                    BlockLayerScale(
                    dim=embed_dim[i], mlp_ratio=mlp_ratio[i],
                    drop=drop_rate, drop_path=dpr[cur + j], norm_layer=norm_layer, h=h, w=w, init_values=init_values,
                    fdrop_mode=fdrop_mode, fdrop_p=fdrop_p, fdrop_t_or_c=fdrop_t_or_c,
                        Fourier_flag=Fourier_flag, Fourier_swap=Fourier_swap, mask_radio=mask_radio,
                        mask_alpha=mask_alpha,
                    noise_mode=noise_mode, severity=severity, sig=sig, domain_mix=domain_mix, mix_test=mix_test,
                    drop_whole=drop_whole, global_filter=global_filter, low_or_high=low_or_high,
                        uncertainty_model=uncertainty_model, perturb_prob=perturb_prob, beta_flag=beta_flag,
                    statistics_mode=statistics_mode, Fourier_high_enhance=Fourier_high_enhance,
                        Fourier_drop_flag=Fourier_drop_flag, Fourier_drop_apply_p=Fourier_drop_apply_p,
                        Fourier_drop_p=Fourier_drop_p, noise_mix_flag=noise_mix_flag,
                        uncertainty_factor=uncertainty_factor, uncertainty_sample=uncertainty_sample,
                        noise_unif_oneside=noise_unif_oneside, noise_type=noise_type, layer_index=i,
                        noise_layers=noise_layers, gauss_or_uniform=gauss_or_uniform, miu_mean_flag=miu_mean_flag,
                        only_low_high_flag=only_low_high_flag
                    )
                for j in range(depth[i])
                ])
            self.blocks.append(blk)
            cur += depth[i]

        # Classifier head
        self.norm = norm_layer(embed_dim[-1])

        self.head = nn.Linear(self.num_features, num_classes)

        if dropcls > 0:
            print('dropout %.2f before classifier' % dropcls)
            self.final_dropout = nn.Dropout(p=dropcls)
        else:
            self.final_dropout = nn.Identity()

        trunc_normal_(self.pos_embed, std=.02)
        self.apply(self._init_weights)

        self.cdrop_mode = cdrop_mode
        self.cdrop_p = cdrop_p
        self.cdrop_layers = cdrop_layers

        self.FourierMix_training = False
        self.patch_domain_mix = patch_domain_mix
        self.patch_mix_alpha = patch_mix_alpha
        self.patch_domain_or_random = patch_domain_or_random

        self.MixStyle_flag = MixStyle_flag
        if self.MixStyle_flag == 1:
            if MixStyle_mix == 0:
                self.MixStyle = MixStyle(mix='random')
            else:
                self.MixStyle = MixStyle(mix='crossdomain')
            self.MixStyle_layers = MixStyle_layers

        self.DSU_flag = DSU_flag
        if self.DSU_flag == 1:
            self.DSU = DSU()


    def Fourier_training(self):
        self.FourierMix_training = True

    def Fourier_eval(self):
        self.FourierMix_training = False

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token'}

    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes, global_pool=''):
        self.num_classes = num_classes
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()

    def forward_features(self, x, patch_layer=0):
        for i in range(4):
            # if i == 0:
            #     x, [lam, perm] = self.patch_embed[i](x, domain_mix=0, mix_alpha=0.1, domain_or_random=0)
            #     x = x + self.pos_embed
            # else:
            #     x, [lam, perm] = self.patch_embed[i](x, domain_mix=0, mix_alpha=0.1, domain_or_random=0)
            # x = self.patch_embed[i](x)

            if patch_layer == i:
                x, [lam, perm] = self.patch_embed[i](x, domain_mix=self.patch_domain_mix,
                                                     mix_alpha=self.patch_mix_alpha,
                                                     domain_or_random=self.patch_domain_or_random)
            else:
                x, _ = self.patch_embed[i](x, domain_mix=0, mix_alpha=self.patch_mix_alpha,
                                                     domain_or_random=self.patch_domain_or_random)
            if i == 0:
                x = x + self.pos_embed
            x, _ = self.blocks[i]((x, self.FourierMix_training))

            if self.cdrop_mode == 1 and (i+1) in self.cdrop_layers:
                x = dropchannel(x, p=self.cdrop_p)

            if self.MixStyle_flag == 1 and i in self.MixStyle_layers:
                x = self.MixStyle(x)
            if self.DSU_flag == 1:
                x = self.DSU(x)

        x = self.norm(x).mean(1)
        return x, [lam, perm]

    def forward(self, x, patch_layer=0):
        x, [lam, perm] = self.forward_features(x, patch_layer=patch_layer)
        x = self.final_dropout(x)
        x = self.head(x)
        return x, [lam, perm]

def resize_pos_embed(posemb, posemb_new):
    # Rescale the grid of position embeddings when loading from state_dict. Adapted from
    # https://github.com/google-research/vision_transformer/blob/00883dd691c63a6830751563748663526e811cee/vit_jax/checkpoint.py#L224
    _logger.info('Resized position embedding: %s to %s', posemb.shape, posemb_new.shape)
    ntok_new = posemb_new.shape[1]
    if True:
        posemb_tok, posemb_grid = posemb[:, :1], posemb[0, 1:]
        ntok_new -= 1
    else:
        posemb_tok, posemb_grid = posemb[:, :0], posemb[0]
    gs_old = int(math.sqrt(len(posemb_grid)))
    gs_new = int(math.sqrt(ntok_new))
    _logger.info('Position embedding grid-size from %s to %s', gs_old, gs_new)
    posemb_grid = posemb_grid.reshape(1, gs_old, gs_old, -1).permute(0, 3, 1, 2)
    posemb_grid = F.interpolate(posemb_grid, size=(gs_new, gs_new), mode='bilinear')
    posemb_grid = posemb_grid.permute(0, 2, 3, 1).reshape(1, gs_new * gs_new, -1)
    posemb = torch.cat([posemb_tok, posemb_grid], dim=1)
    return posemb

def checkpoint_filter_fn(state_dict, model):
    """ convert patch embedding weight from manual patchify + linear proj to conv"""
    out_dict = {}
    if 'model' in state_dict:
        # For deit models
        state_dict = state_dict['model']
    for k, v in state_dict.items():
        if 'patch_embed.proj.weight' in k and len(v.shape) < 4:
            # For old models that I trained prior to conv based patchification
            O, I, H, W = model.patch_embed.proj.weight.shape
            v = v.reshape(O, -1, H, W)
        elif k == 'pos_embed' and v.shape != model.pos_embed.shape:
            # To resize pos embedding when using model at different size from pretrained weights
            v = resize_pos_embed(v, model.pos_embed)
        out_dict[k] = v
    return out_dict
