import numpy as np
from tqdm import tqdm
from enum import Enum, auto
import math


from stringart.algorithm.costmethod import CostMethod
from numba import njit

def find_cost(method: CostMethod, string_art_img: np.ndarray):


    match method:
        case CostMethod.DIFFERENCE:
            ""
        
        case CostMethod.MEAN:
            ""
        
        case CostMethod.SMSE: #signed mean squared error
            ""
        
        case CostMethod.SMRSE: #signed root mean squared error THIS GIVES SAME RESULT AS MEAN ERROR AND IDK WHY
            ""
        
        case CostMethod.MEDIAN:
            ""
     
@njit
def update_costs_jit_mean(
    costs: np.ndarray,
    pixel_ptr: np.ndarray,
    flat_line_ids: np.ndarray,
    flat_weights: np.ndarray,
    pix: np.ndarray,
    delta: np.ndarray
):
    """
    JIT-compiled inner loop for cost updates:
      costs[j2] += delta[i] * weight[j2,i] for each pixel i on the new line.
    """
    for i in range(pix.shape[0]):
        p = pix[i]
        d = delta[i]
        for k in range(pixel_ptr[p], pixel_ptr[p+1]):
            costs[flat_line_ids[k]] += d * flat_weights[k]

def difference(a, b):
        """
        Calculates the absolute difference between two pixel values
        Args:
            a: The first value/values
            b: The second value/values
        """
        return np.abs(a - b)