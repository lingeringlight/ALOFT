from torchvision import transforms
import random
import torch
import numpy as np
from math import sqrt
import os

from PIL import Image



def colorful_spectrum_mix(img1, img2, alpha, ratio=1.0, Fourier_swap=0, Fourier_phase=0):
    """Input image size: ndarray of [H, W, C]"""
    if Fourier_swap == 1:
        lam = 1.0
    else:
        lam = np.random.uniform(0, alpha)

    img1 = np.array(img1, dtype=float)
    img2 = np.array(img2, dtype=float)
    assert img1.shape == img2.shape
    h, w, c = img1.shape
    h_crop = int(h * sqrt(ratio))
    w_crop = int(w * sqrt(ratio))
    h_start = h // 2 - h_crop // 2
    w_start = w // 2 - w_crop // 2

    img1_fft = np.fft.fft2(img1, axes=(0, 1))
    img2_fft = np.fft.fft2(img2, axes=(0, 1))
    img1_abs, img1_pha = np.abs(img1_fft), np.angle(img1_fft)
    img2_abs, img2_pha = np.abs(img2_fft), np.angle(img2_fft)

    if Fourier_phase == 1:
        cont_abs_1 = img1_abs.mean()
        cont_abs_2 = img2_abs.mean()
        img1_pha = cont_abs_1 * (np.e ** (1j * img1_pha))
        img2_pha = cont_abs_2 * (np.e ** (1j * img2_pha))
        img1_pha = np.real(np.fft.ifft2(img1_pha, axes=(0, 1)))
        img2_pha = np.real(np.fft.ifft2(img2_pha, axes=(0, 1)))
        img1_pha = np.uint8(np.clip(img1_pha, 0, 255))
        img2_pha = np.uint8(np.clip(img2_pha, 0, 255))
        return img1_pha, img2_pha, lam
        # img1_pha = np.uint8(np.clip(img1_pha, 0, 255))
        # img2_pha = np.uint8(np.clip(img2_pha, 0, 255))
        # return img1_pha, img2_pha, lam

    img1_abs = np.fft.fftshift(img1_abs, axes=(0, 1))
    img2_abs = np.fft.fftshift(img2_abs, axes=(0, 1))

    img1_abs_ = np.copy(img1_abs)
    img2_abs_ = np.copy(img2_abs)
    img1_abs[h_start:h_start + h_crop, w_start:w_start + w_crop] = \
        lam * img2_abs_[h_start:h_start + h_crop, w_start:w_start + w_crop] + (1 - lam) * img1_abs_[
                                                                                          h_start:h_start + h_crop,
                                                                                          w_start:w_start + w_crop]
    img2_abs[h_start:h_start + h_crop, w_start:w_start + w_crop] = \
        lam * img1_abs_[h_start:h_start + h_crop, w_start:w_start + w_crop] + (1 - lam) * img2_abs_[
                                                                                          h_start:h_start + h_crop,
                                                                                          w_start:w_start + w_crop]

    img1_abs = np.fft.ifftshift(img1_abs, axes=(0, 1))
    img2_abs = np.fft.ifftshift(img2_abs, axes=(0, 1))

    img21 = img1_abs * (np.e ** (1j * img1_pha))
    img12 = img2_abs * (np.e ** (1j * img2_pha))
    img21 = np.real(np.fft.ifft2(img21, axes=(0, 1)))
    img12 = np.real(np.fft.ifft2(img12, axes=(0, 1)))
    img21 = np.uint8(np.clip(img21, 0, 255))
    img12 = np.uint8(np.clip(img12, 0, 255))

    return img21, img12, lam


def filter_pass(img1, S=10, high_or_low=0):
    """Input image size: ndarray of [H, W, C]"""

    img1 = np.array(img1, dtype=float)
    h, w, c = img1.shape
    h_start = h // 2 - S // 2
    w_start = w // 2 - S // 2

    img1_fft = np.fft.fft2(img1, axes=(0, 1))

    img1_abs, img1_pha = np.abs(img1_fft), np.angle(img1_fft)
    img1_abs = np.fft.fftshift(img1_abs, axes=(0, 1))
    if high_or_low == 1:
        #     low-pass
        # masks = torch.zeros_like(img1_abs)
        masks = np.zeros_like(img1_abs)
        h_start = h // 2 - S // 2
        w_start = w // 2 - S // 2
        masks[h_start:h_start + S, w_start:w_start + S, :] = 1
        img1_abs = img1_abs * masks
    else:
        # high-pass
        # masks = torch.ones_like(img1_abs)
        masks = np.ones_like(img1_abs)
        h_start = S // 2
        w_start = S // 2
        masks[h_start:(h_start + h - S), w_start:w_start + h - S, :] = 0
        img1_abs = img1_abs * masks
    img1_abs = np.fft.ifftshift(img1_abs, axes=(0, 1))
    img1 = img1_abs * (np.e ** (1j * img1_pha))

    img1 = np.real(np.fft.ifft2(img1, axes=(0, 1)))
    img1 = np.uint8(np.clip(img1, 0, 255))
    return img1


class FourierTransform:
    def __init__(self, args, alpha=1.0, dataset_list=None, base_dir=None):
        self.alpha = alpha
        # self.from_domain = args.from_domain
        # self.Fourier_swap = args.Fourier_swap
        # self.Fourier_phase = args.Fourier_phase
        self.dataset_list = dataset_list
        self.base_dir = base_dir
        self.dataset = args.data

        self.filter_flag = args.freq_analyse
        self.filter_S = args.freq_analyse_S
        self.high_or_low = args.freq_analyse_high_or_low

        domain_path = os.path.join(self.base_dir, self.dataset_list[0])
        if self.dataset == "VLCS":
            domain_path = os.path.join(domain_path, "full")
        self.class_names = sorted(os.listdir(domain_path))

        self.pre_transform = transforms.Compose(
            [transforms.RandomResizedCrop(args.image_size, scale=(args.min_scale, 1.0)),
             transforms.RandomHorizontalFlip(args.random_horiz_flip),
             transforms.ColorJitter(brightness=args.jitter, contrast=args.jitter, saturation=args.jitter,
                                    hue=min(0.5, args.jitter))
             ]
        )
        self.post_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    def __call__(self, img, domain_label):
        img_o = self.pre_transform(img)
        if self.filter_flag == 1:
            img_s2o = filter_pass(img_o, S=self.filter_S, high_or_low=self.high_or_low)
            domain_s = None
            lam = None
        else:
            img_s, label_s, domain_s = self.sample_image(domain_label)
            img_s2o, img_o2s, lam = colorful_spectrum_mix(img_o, img_s, alpha=self.alpha, Fourier_swap=self.Fourier_swap,
                                                          Fourier_phase=self.Fourier_phase)
        img_s2o = self.post_transform(img_s2o)

        return img_s2o, domain_s, lam

    def sample_image(self, domain_label):
        if self.from_domain == 'all':
            domain_idx = random.randint(0, len(self.dataset_list) - 1)
        elif self.from_domain == 'inter':
            domains = list(range(len(self.dataset_list)))
            domains.remove(domain_label)
            domain_idx = random.sample(domains, 1)[0]
        elif self.from_domain == 'intra':
            domain_idx = domain_label
        else:
            raise ValueError("Not implemented")
        other_domain_name = self.dataset_list[domain_idx]
        class_idx = random.randint(0, len(self.class_names)-1)
        other_class_name = self.class_names[class_idx]
        base_dir_domain = os.path.join(self.base_dir, other_domain_name)
        if self.dataset == "VLCS":
            base_dir_domain = os.path.join(base_dir_domain, "full")
        base_dir_domain_class = os.path.join(base_dir_domain, other_class_name)
        other_id = np.random.choice(os.listdir(base_dir_domain_class))
        other_img = Image.open(os.path.join(base_dir_domain_class, other_id)).convert('RGB')

        return self.pre_transform(other_img), class_idx, domain_idx