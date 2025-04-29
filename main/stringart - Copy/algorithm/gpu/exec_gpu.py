import numpy as np
import cupy as cp
from tqdm import tqdm
from enum import Enum
from skimage.metrics import structural_similarity as ssim
import math

from stringart.core.stringimage import StringArtImage
from stringart.preprocessing.importancemaps import ImportanceMap
from stringart.algorithm.lines import FlatGpuStringLines, GpuStringLine
from stringart.algorithm.costmethod import CostMethod
from stringart.algorithm.gpu.eval import find_pixelwise_cost, find_best_line



def create_string_art_gpu(first_anchor: int, base_img: np.ndarray, string_art_img: StringArtImage, line_pixel_dict: dict, line_darkness_dict: dict, iterations: int, cost_method: CostMethod = CostMethod.MEAN, max_darkness: float = None, eval_interval: int = None, importance_map: ImportanceMap = None):
    """
    Creates the completed string art

    Args:
        first_anchor: The starting anchor
        base_img: The origonal image, used for finding the best line
        string_art_img: The string art image that is supposed to be made TODO: Create the string art image in in the function
        line_pixel_dict: The ditionary of line pixels for each anchor combination
        line_darkness_dict: The dictionary of line darkness values for each anchor combenation, cooresponds to line_pixel_dict
        iterations: The number of strings drawn. Range vaires with anchor count and image size
        loss_method: The method that the function uses to determine the loss of a line
        max_darkness: The maximum darkness of a line in the string art image. If set to none, there is no limit
        eval_interval: Evalutate the entire image on this interval, saved in the returned string_art_img
        importance_map: The image map of importance values for each pixel

    Returns:
        string_art_img: The final string art image
        difference_img: The difference image used in this creation
    """

    anchor_line_idx = {}
    if not importance_map:
        importance_map = ImportanceMap(img=np.ones_like(base_img))
    
    
    
    print("Finding pixel count...")
    total_num_pixels = 0
    for line in tqdm(line_pixel_dict.values()):
        total_num_pixels += len(line)
    
    print(f"Done! The total number of pixels is {total_num_pixels}")

    print("Creating line arrays...")

    flat_lines = FlatGpuStringLines(
        base_image=cp.empty(total_num_pixels),
        importance_values=cp.empty(total_num_pixels),
        string_pixels=cp.empty(total_num_pixels),
        string_darkness=cp.empty(total_num_pixels),
        pixelwise_cost=cp.ones(total_num_pixels),
        string_art_values=cp.empty(total_num_pixels)
    )
    lines = []

    print("Done!")

    print("Filing line arrays... ")
    
    prev_slice_end = 0
    _pixel_map = [[[] for _ in range(string_art_img.img.shape[0])] for _ in range(string_art_img.img.shape[1])]
    

    for idx, anchors in enumerate(tqdm(line_pixel_dict.keys())):
        pixels = line_pixel_dict[anchors]
        x_coords = pixels[:, 0]
        y_coords = pixels[:, 1]

        lines_slice = slice(prev_slice_end, prev_slice_end+len(pixels))
        prev_slice_end += len(pixels)

        flat_lines.base_image[lines_slice] = cp.array(base_img[x_coords, y_coords])
        flat_lines.importance_values[lines_slice] = cp.array(importance_map.img[x_coords, y_coords])
        flat_lines.string_darkness[lines_slice] = cp.array(line_darkness_dict[anchors])
        flat_lines.string_art_values[lines_slice] = cp.array(string_art_img.img[x_coords, y_coords])
        flat_lines.slices.append(lines_slice)

        lines.append(GpuStringLine(string_darkness=line_darkness_dict[anchors], string_pixels=pixels, line_slice=lines_slice))

        pixel_indices = np.arange(lines_slice.start, lines_slice.stop)
        for x, y, pixel_idx in zip(x_coords, y_coords, pixel_indices):
            _pixel_map[x][y].append(pixel_idx)
    

    
    print("Creating pixel index map...")
    for idx, line in enumerate(tqdm(lines)):
        x_coords = line.string_pixels[:, 0]
        y_coords = line.string_pixels[:, 1]

        line_map = []
        for x, y in zip(x_coords, y_coords):
            line_map.append(_pixel_map[x][y])

        line.string_pixel_map = line_map



    for iter in tqdm(range(iterations)):
        best_line_idx, loss = find_best_line(method=cost_method, flat_lines=flat_lines)
        best_line: GpuStringLine = lines[best_line_idx]

        #flat_lines.string_art_values[flat_lines.slices[best_line_idx]] += 0.2
        for idx, pixel_list in enumerate(best_line.string_pixel_map):
            new_values = cp.clip(flat_lines.string_art_values[pixel_list] + best_line.string_darkness[idx], 0, max_darkness)
            flat_lines.string_art_values[pixel_list] = new_values

            #same = cp.in1d(flat_lines.string_art_values[flat_lines.slices[best_line_idx]], cp.array(pixel_list))
            #if(same.any()): print(same)


        pixels = best_line.string_pixels
        x_coords = pixels[:, 0]
        y_coords = pixels[:, 1]
        new_values = np.clip(string_art_img.img[x_coords, y_coords] + best_line.string_darkness, 0, max_darkness)
        string_art_img.img[x_coords, y_coords] = new_values


        print(f"best line idx: {best_line_idx, loss}")

    
    return string_art_img


