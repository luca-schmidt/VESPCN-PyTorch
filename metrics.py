import torch
import torch.nn.functional as F
import numpy as np
import cv2
from config import mean_value, in_channels



def denormalize(img, comp_idx, mean=mean_value):
    if in_channels==2:
        if not isinstance(mean, tuple):
            raise ValueError("For in_channels==2, mean must be a tuple.")
        mean = mean[comp_idx]
    elif in_channels==1:
        if isinstance(mean, tuple):
            raise ValueError("For in_channels==1, mean must be an integer.")
    else: 
        pass
    return img + mean


def calc_psnr(img1, img2, max=255.0):
    img1 = img1.mul(255.0).clamp(0.0, 255.0)
    img2 = img2.mul(255.0).clamp(0.0, 255.0)
    mse = torch.mean((img1 - img2) ** 2)
    psnr = 10.0 * torch.log10((max ** 2) / (mse))
    return psnr


def calc_mse(img1, img2): 
    mse = torch.mean((img1 - img2) ** 2)
    return mse


def calc_mae(img1, img2):
  loss = torch.nn.L1Loss()
  return loss(img1, img2)



def ssim(img1, img2):
    C1 = (0.01 * 255)**2
    C2 = (0.03 * 255)**2

    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)

    kernel = cv2.getGaussianKernel(11, 1.5)
    window = np.outer(kernel, kernel.transpose())

    mu1 = cv2.filter2D(img1, -1, window)[5:-5, 5:-5]  
    mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]
    mu1_sq = mu1**2
    mu2_sq = mu2**2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = cv2.filter2D(img1**2, -1, window)[5:-5, 5:-5] - mu1_sq
    sigma2_sq = cv2.filter2D(img2**2, -1, window)[5:-5, 5:-5] - mu2_sq
    sigma12 = cv2.filter2D(img1 * img2, -1, window)[5:-5, 5:-5] - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) *
                                                            (sigma1_sq + sigma2_sq + C2))
    return ssim_map.mean()


def calc_ssim(img1, img2):
    img1 = img1.mul(255.0).clamp(0.0, 255.0)
    img2 = img2.mul(255.0).clamp(0.0, 255.0)

    img1=img1.cpu().squeeze(0).squeeze(0).numpy()
    img2 = img2.cpu().squeeze(0).squeeze(0).numpy()

    if not img1.shape == img2.shape:
        raise ValueError('Input images must have the same dimensions.')
    if img1.ndim == 2:
        return ssim(img1, img2)
    elif img1.ndim == 3:
        if img1.shape[2] == 3:
            ssims = []
            for i in range(3):
                ssims.append(ssim(img1, img2))
            return np.array(ssims).mean()
        elif img1.shape[2] == 1:
            return ssim(np.squeeze(img1), np.squeeze(img2))
    else:
        raise ValueError('Wrong input image dimensions.')