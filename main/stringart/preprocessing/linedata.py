from tqdm import tqdm
from pathlib import Path
import numpy as np
from numba import njit
from collections import defaultdict
from typing import List, Tuple, Callable, Any, Dict
import os

from stringart.preprocessing.stringartblueprint import StringArtBlueprint
from stringart.core.lines import draw_line, StringLineCSRMapping


def preprocess_line_data(data_folder:str, string_art_blueprint: StringArtBlueprint, closest_neighbors: int = 10):

    shape = string_art_blueprint.img.shape
    folder_path = f"{data_folder}/line_data/{shape[0]}x{shape[1]}-{len(string_art_blueprint.anchors)}/"
    

    if Path(folder_path).exists():
        line_csr_mapping = StringLineCSRMapping.load(folder_path=folder_path)
        return line_csr_mapping
    else:
        print("No line data folder found, generating new one...")
        line_csr_mapping = generate_line_data(anchors=string_art_blueprint.anchors, line_darkness_multiplier=string_art_blueprint.line_darkness, folder_path=folder_path, mask=string_art_blueprint.mask, closest_neighbors=closest_neighbors)
        print("Saving new line data")
        os.makedirs(folder_path, exist_ok=True)
        line_csr_mapping.save(folder_path=folder_path)
        return line_csr_mapping
        
    
    

def generate_line_data(anchors: List[Any], line_darkness_multiplier: float, folder_path: str, mask: np.ndarray, closest_neighbors: int = 10):
    """
    Precompute the CSR-style line mapping for string-art:
      - line_ptr: pointers into flat arrays per line
      - line_pix: flat pixel indices for each line
      - line_dark: per-pixel darkness values for each line
      - anchor_to_lines: mapping each anchor idx -> list of line indices incident
    
    Args:
        anchors: list of anchor objects, each with .coordinates (x,y) tuple
        mask: optional mask array for draw_line
        line_darkness_multiplier: global multiplier for darkness
    
    Returns:
        line_ptr: np.ndarray shape (L+1,), int32
        line_pix: np.ndarray shape (total_pixels,), int32
        line_dark: np.ndarray shape (total_pixels,), float32
        anchor_to_lines: dict mapping anchor_idx -> list of line_idx
    """

    def is_within_range(arr, idx1:int, idx2:int):
        """
        Calculate the distance between indices in both directions considering wrapped boundaries and returns true or false if they are within closes_nieghbors to one another

        Args:
            arr: The array used to find neighbors, only the length of the array is used
            idx1: The first index
            idx2: The second index
        
        Returns:
            A boolean: true if the indicies are within the range, false if not
        """
        forward_distance = (idx2 - idx1) % len(arr)
        backward_distance = (idx1 - idx2) % len(arr)

        # Check if either forward or backward distance is less than or equal to x
        return forward_distance <= closest_neighbors or backward_distance <= closest_neighbors or idx1 == idx2

    anchor_line_pairs = []
    line_pixels_list = []
    line_darkness_list = []
    L = 0

    # 1) generate all unique lines
    for i in tqdm(range(len(anchors)), desc="Generating all unique lines"):
        for j in range(i+1, len(anchors)):
            if is_within_range(anchors, i, j): continue

            p0 = anchors[i].coordinates
            p1 = anchors[j].coordinates
            pixels, darkness = draw_line(p0, p1, line_darkness_multiplier, mask)
            anchor_line_pairs.append((i, j))
            line_pixels_list.append(pixels)
            line_darkness_list.append(darkness)
            L += 1

    # 2) build CSR pointer array
    print("Building the CSR pointer array")
    lengths = [pix.shape[0] for pix in line_pixels_list]
    line_ptr = np.zeros(L+1, dtype=np.int32)
    np.cumsum(lengths, out=line_ptr[1:])

    total = line_ptr[-1]
    line_pix = np.empty(total, dtype=np.int32)
    line_dark = np.empty(total, dtype=np.float32)

    # 3) fill flat arrays
    print("Filling the flat arrays")
    idx = 0
    for k, (pixels, darkness) in enumerate(zip(line_pixels_list, line_darkness_list)):
        n = lengths[k]
        # flatten 2D coords into 1D indices
        flat = pixels[:,0] * mask.shape[1] + pixels[:,1]
        line_pix[idx:idx+n] = flat
        line_dark[idx:idx+n] = darkness
        idx += n

    # 4) build anchor_to_lines mapping
    print("Building the anchor to line mapping")
    anchor_to_lines: Dict[int, List[Tuple[int,int]]] = defaultdict(list)
    for line_idx, (i, j) in enumerate(anchor_line_pairs):
        # for anchor i, record (this line_idx, other end j)
        anchor_to_lines[i].append((line_idx, j))
        # for anchor j, record (this line_idx, other end i)
        anchor_to_lines[j].append((line_idx, i))

    # 5) build pixel to line mapping CSR arrays
    print("Building pixel to line CSR arrays")
    H, W = mask.shape
    pixel_to_line_ptr, pixel_line_indices, pixel_line_weights_default = build_pixel_to_line_mapping_jit(line_ptr=line_ptr, line_pix=line_pix, H=H, W=W)

    
    line_csr_mapping = StringLineCSRMapping(line_ptr=line_ptr,
                                            line_pix=line_pix,
                                            line_dark=line_dark,
                                            anchor_to_lines=anchor_to_lines,
                                            pixel_to_line_ptr=pixel_to_line_ptr,
                                            pixel_line_indices=pixel_line_indices,
                                            pixel_line_weights_default=pixel_line_weights_default,
                                            folder_path=folder_path)
    
    line_csr_mapping
    return line_csr_mapping


@njit
def build_pixel_to_line_mapping_jit(
line_ptr: np.ndarray,    # int32[L+1]
    line_pix: np.ndarray,    # int32[E]
    H: int,
    W: int
):
    """
    JIT-compiled: Build pixel→line CSR mapping plus uniform weights (1/line_length)
    when importance is constant (all ones).

    Args:
      line_ptr: CSR pointer array for lines (shape L+1)
      line_pix: Flattened pixel indices for each line entry (shape E)
      H, W:     Canvas dimensions

    Returns:
      pixel_to_line_ptr   : int32 array shape (P+1)
      pixel_line_indices  : int32 array shape (E,)
      pixel_line_weights  : float32 array shape (E,), weight = 1/line_length_j
    """
    P = H * W
    E = line_pix.shape[0]
    L = line_ptr.shape[0] - 1

    # 1) Count entries per pixel
    counts = np.zeros(P, np.int32)
    for k in range(E):
        counts[line_pix[k]] += 1

    # 2) Prefix-sum to build pixel_to_line_ptr
    pixel_to_line_ptr = np.empty(P+1, np.int32)
    pixel_to_line_ptr[0] = 0
    for p in range(P):
        pixel_to_line_ptr[p+1] = pixel_to_line_ptr[p] + counts[p]

    # 3) Allocate output arrays
    pixel_line_indices = np.empty(E, np.int32)
    pixel_line_weights = np.empty(E, np.float32)

    # 4) Precompute inverse lengths for each line
    inv_len = np.empty(L, np.float32)
    for j in range(L):
        start = line_ptr[j]
        end = line_ptr[j+1]
        inv_len[j] = 1.0 / (end - start)

    # 5) Reset counts to use as write cursors
    for p in range(P):
        counts[p] = 0

    # 6) Fill in the reverse mapping with indices and uniform weights
    for j in range(L):
        start = line_ptr[j]
        end = line_ptr[j+1]
        w = inv_len[j]
        for k in range(start, end):
            p = line_pix[k]
            dst = pixel_to_line_ptr[p] + counts[p]
            pixel_line_indices[dst] = j
            pixel_line_weights[dst] = w
            counts[p] += 1

    return pixel_to_line_ptr, pixel_line_indices, pixel_line_weights