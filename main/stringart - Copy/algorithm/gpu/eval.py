import numpy as np
import cupy as cp
from tqdm import tqdm
from enum import Enum, auto
from skimage.metrics import structural_similarity as ssim
import math

from stringart.algorithm.lines import FlatGpuStringLines
from stringart.algorithm.costmethod import CostMethod


def find_pixelwise_cost(method: CostMethod, flat_lines: FlatGpuStringLines):
    ""
    total_darkness_values = flat_lines.string_art_values + flat_lines.string_darkness
    weighted_difference = (total_darkness_values - flat_lines.base_image) * flat_lines.importance_values
    return weighted_difference



def find_best_line(method: CostMethod, flat_lines: FlatGpuStringLines):
    ""
    #cp.copyto(flat_lines.pixelwise_cost, find_pixelwise_cost(method=method, flat_lines=flat_lines))
    flat_lines.pixelwise_cost = find_pixelwise_cost(method=method, flat_lines=flat_lines)

    sums = cp.array([cp.mean(flat_lines.pixelwise_cost[slice]).item() for slice in flat_lines.slices])

    best_line_idx = cp.argmin(sums).item()
    loss = sums[best_line_idx]

    return best_line_idx, loss