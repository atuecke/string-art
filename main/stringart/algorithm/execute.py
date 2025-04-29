import numpy as np
from tqdm import tqdm
from enum import Enum
from skimage.metrics import structural_similarity as ssim
import math
import cupy as cp

from stringart.core.stringimage import StringArtImage
from stringart.preprocessing.importancemaps import ImportanceMap
from stringart.algorithm.costmethod import CostMethod
from stringart.algorithm.cpu.exec_cpu import create_string_art_cpu
from stringart.algorithm.cpu.exec_cpu_unoptimized import create_string_art_cpu_unoptimized
from stringart.preprocessing.image import BaseImage

def create_string_art(first_anchor: int, base_img: BaseImage, string_art_img: StringArtImage, line_pixel_dict: dict, line_darkness_dict: dict, iterations: int, cost_method: CostMethod = CostMethod.MEAN, max_darkness: float = None, eval_interval: int = None, importance_map: ImportanceMap = None, gpu_accel = False, use_prev_anchor: bool = True, random_neighbor: bool = False, profiling = False):
    if gpu_accel:
        ""
    else:
        return create_string_art_cpu(
            first_anchor=first_anchor,
            base_img=base_img,
            string_art_img=string_art_img,
            line_pixel_dict=line_pixel_dict,
            line_darkness_dict=line_darkness_dict,
            iterations=iterations,
            cost_method=cost_method,
            max_darkness=max_darkness,
            eval_interval=eval_interval,
            importance_map=importance_map,
            use_prev_anchor=use_prev_anchor,
            random_neighbor=random_neighbor,
            profiling=profiling
        )
        # return create_string_art_cpu_unoptimized(
        #     first_anchor=first_anchor,
        #     base_img=base_img,
        #     string_art_img=string_art_img,
        #     line_pixel_dict=line_pixel_dict,
        #     line_darkness_dict=line_darkness_dict,
        #     iterations=iterations,
        #     cost_method=cost_method,
        #     max_darkness=max_darkness,
        #     eval_interval=eval_interval,
        #     importance_map=importance_map,
        #     use_prev_anchor=use_prev_anchor,
        #     random_neighbor=random_neighbor,
        # )