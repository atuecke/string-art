from dataclasses import dataclass
from typing import Tuple, List
from enum import Enum, auto
from skimage.metrics import structural_similarity as ssim
import sys
import numpy as np

import torch
import torchvision.transforms as T
from pytorch_msssim import ms_ssim
import lpips

print(sys.executable, torch.__file__)

# 1) Pick your device once
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 2) Instantiate LPIPS once, move to device, set to eval
lpips_model = lpips.LPIPS(net="vgg").to(device)
lpips_model.eval()

class AccuracyMethod(Enum):
    LINE_COST = "Line Cost"
    IMAGE_COST = "Image Cost"
    AVG_LINE_LENGTH = "Line Length"
    SSIM = "SSIM"
    MS_SSIM = "MS_SSIM"
    LPIPS = "LPIPS"

@dataclass
class StringArtAccuracy:
    type: str
    datapoints: List[Tuple[int, float]]

def eval_ssim(input_img: np.ndarray, target_img: np.ndarray):
    """
    Structural Similarity Matrix
    Compares images based on luminance, contrast, and structure
    Range: [0, 1], where 1 is a perfect match
    """
    return ssim(input_img, target_img, data_range=1)

def eval_ms_ssim(input_img: np.ndarray, target_img: np.ndarray):
    """
    Multi-Scale Structural Similarity
    Extends SSIM by computing the index at multiple image scales
    Range: [0, 1], where 1 is a perfect match

    """

    # to tensor, shape (1, 1, H, W)
    t1 = torch.from_numpy(input_img).float().unsqueeze(0).unsqueeze(0).to(device)
    t2 = torch.from_numpy(target_img).float().unsqueeze(0).unsqueeze(0).to(device)

    #ms_ssim expects inputs in [0, 1]
    score = ms_ssim(t1, t2, data_range=1.0, size_average=True)
    return score.item()

def eval_lpips(input_img: np.ndarray, target_img: np.ndarray):
    """
    Learned Perceptual Image Patch Similarity
    Measures distance in deep-network feature space. Both images are passed through a pre-trained CNN
    Range: [0, inf] where 0 is a perfect match

    """

    # to tensor (1,1,H,W), then repeat to (1,3,H,W)
    t1 = torch.from_numpy(input_img).float().unsqueeze(0).unsqueeze(0).repeat(1,3,1,1).to(device)
    t2 = torch.from_numpy(target_img).float().unsqueeze(0).unsqueeze(0).repeat(1,3,1,1).to(device)

    # map [0,1] → [-1,1]
    t1 = t1 * 2 - 1
    t2 = t2 * 2 - 1

    with torch.no_grad():
        dist = lpips_model(t1, t2)
    return dist.item()
    
    