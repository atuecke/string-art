import numpy as np
from tqdm import tqdm
from enum import Enum, auto
from skimage.metrics import structural_similarity as ssim
import math

from stringart.algorithm.lines import StringLine

class CostMethod(Enum):
    DIFFERENCE = auto()
    MEAN = auto()
    SMSE = auto()
    SMRSE = auto()
    MEDIAN = auto()

def find_cost(method: CostMethod, line: StringLine, string_art_img: np.ndarray):
        
    x_coords = line.string_pixels[:, 0]
    y_coords = line.string_pixels[:, 1]

    string_art_values = string_art_img[x_coords, y_coords]
    total_darkness_values = string_art_values + line.string_darkness

    match method:
        case CostMethod.DIFFERENCE:
            diff_values = difference(line.base_image, string_art_values)
            weighted_difference = line.importance_values * (difference(line.base_image, total_darkness_values) - diff_values)
            return(np.sum(weighted_difference)/len(x_coords))
        
        case CostMethod.MEAN:
            weighted_difference = (total_darkness_values - line.base_image)*line.importance_values
            return(np.sum(weighted_difference)/len(x_coords))
        
        case CostMethod.SMSE: #signed mean squared error
            weighted_difference = (total_darkness_values - line.base_image)*line.importance_values
            signed_squared_difference = np.sign(weighted_difference) * weighted_difference**2
            return(np.sum(signed_squared_difference)/len(x_coords))
        
        case CostMethod.SMRSE: #signed root mean squared error THIS GIVES SAME RESULT AS MEAN ERROR AND IDK WHY
            weighted_difference = (total_darkness_values - line.base_image)*line.importance_values
            signed_squared_difference = np.sign(weighted_difference) * (weighted_difference**2)
            mean_difference = np.sum(signed_squared_difference)/len(x_coords)
            return(np.sqrt(abs(mean_difference))*np.sign(mean_difference))
        
        case CostMethod.MEDIAN:
            weighted_difference = (total_darkness_values - line.base_image)*line.importance_values
            return(np.median(weighted_difference))


def difference(a, b):
        """
        Calculates the absolute difference between two pixel values
        Args:
            a: The first value/values
            b: The second value/values
        """
        return np.abs(a - b)