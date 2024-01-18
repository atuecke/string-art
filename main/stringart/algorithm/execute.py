import numpy as np
from tqdm import tqdm
from enum import Enum
from skimage.metrics import structural_similarity as ssim
import math

from stringart.core.stringimage import StringArtImage
from stringart.preprocessing.importancemaps import ImportanceMap
from stringart.algorithm.lines import StringLine
from stringart.algorithm.eval import CostMethod, find_cost

def create_string_art(first_anchor: int, base_img: np.ndarray, string_art_img: StringArtImage, line_pixel_dict: dict, line_darkness_dict: dict, iterations: int, cost_method: CostMethod = CostMethod.MEAN, max_darkness: float = None, eval_interval: int = None, importance_map: ImportanceMap = None):
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
    lines = []
    if not importance_map:
        importance_map = ImportanceMap(img=np.ones_like(base_img))
    
    print("Building line arrays")
    for idx, anchors in enumerate(line_pixel_dict.keys()):
        pixels = line_pixel_dict[anchors]
        x_coords = pixels[:, 0]
        y_coords = pixels[:, 1]

        anchor_line_idx[anchors] = idx
        lines.append(StringLine(
            base_image=base_img[x_coords, y_coords],
            importance_values=importance_map.img[x_coords, y_coords],
            string_pixels=pixels,
            string_darkness=line_darkness_dict[anchors]
        ))
    print("Done!")
   
    if not importance_map:
        importance_map = np.ones(string_art_img.img.shape)
    importance_map = importance_map.img
                    
    def find_best_line(previous_anchor_idx: int):
        """
        Starts at a specified anchor and find the loss for every string leading to every other anchor, updating the best loss whenever a better one is found

        Args:
            previous_anchor_idx: The previous anchor, which was the end anchor for the last line

        Returns:
            best_anchors: The tuple of the two best anchors
            best_loss: The best line loss found
        """
        best_loss = np.inf #TODO set this to the starting loss and make the algorithm terminate when there isn't a possible improvement
        best_anchors = None
        best_end_anchor = None
        best_start_anchor = None
        for start_anchor_idx in get_neighbors(string_art_img.anchors, previous_anchor_idx): #Finds the neighbors of the start anchor
            for end_anchor_idx in range(len(string_art_img.anchors)):
                both_anchors = tuple(sorted((start_anchor_idx, end_anchor_idx))) #Makes sure to get the right order for the indices, set in make_line_dict().
                if both_anchors not in line_pixel_dict: continue
                line_idx = anchor_line_idx[both_anchors]
                temp_loss = find_cost(method=cost_method, line=lines[line_idx], string_art_img=string_art_img.img)
                if(temp_loss < best_loss): #Updates best loss if temp loss is better
                    best_loss = temp_loss
                    best_anchors = both_anchors
                    best_end_anchor = end_anchor_idx
                    best_start_anchor = start_anchor_idx
        return best_anchors, best_loss, best_start_anchor, best_end_anchor

    previous_anchor_idx = first_anchor
    for iter in tqdm(range(iterations)):
        best_anchors, best_loss, best_start_anchor, best_end_anchor = find_best_line(previous_anchor_idx=previous_anchor_idx)

        #Load the best line and add it to the string art image
        best_string_pixels, best_string_darkness_values = line_pixel_dict[best_anchors], line_darkness_dict[best_anchors]
        x_coords = best_string_pixels[:, 0]
        y_coords = best_string_pixels[:, 1]
        
        if max_darkness:
            new_values = np.clip(string_art_img.img[x_coords, y_coords] + best_string_darkness_values, 0, max_darkness)
            string_art_img.img[x_coords, y_coords] = new_values
        else:
            new_values = string_art_img.img[x_coords, y_coords] + best_string_darkness_values
            string_art_img.img[x_coords, y_coords] = new_values
        string_art_img.string_path.append((best_start_anchor, best_end_anchor))
        previous_anchor_idx = best_end_anchor
        string_art_img.loss_list.append(best_loss)

        #Evaluate the string art
        if eval_interval:
            if iter%eval_interval == 0:
                similarity = compare_images(string_art_img.img, base_img, method="ssim")
                string_art_img.similarities.append((iter, similarity))
    
    return string_art_img

def get_neighbors(arr: list, idx):
    # Calculate indices for neighbors with wrapping
    left_idx = (idx - 1) % len(arr)
    right_idx = (idx + 1) % len(arr)

    return left_idx, right_idx

def compare_images(input_img, target_img, method):
    """
    """
    if method == "ssim":
        similarity = ssim(input_img, target_img, data_range=1)
    elif method == "psnr":
        mse = np.mean((input_img - target_img) ** 2)
        if mse == 0:
            similarity = float('inf')
        max_pixel = 255.0
        psnr = 20 * math.log10(max_pixel / math.sqrt(mse))
        similarity = psnr

    
    return similarity