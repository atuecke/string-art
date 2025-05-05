import numpy as np
import math
import random
from enum import Enum
from typing import List, Tuple, Dict
from collections import defaultdict

from tqdm import tqdm
from numba import njit
from skimage.metrics import structural_similarity as ssim

from stringart.algorithm.accuracy_eval import StringArtAccuracy, AccuracyMethod, eval_ssim, eval_ms_ssim, eval_lpips
from stringart.core.stringimage import StringImage
from stringart.preprocessing.stringartblueprint import StringArtBlueprint
from stringart.preprocessing.importancemaps import ImportanceMap
from stringart.core.lines import StringLineCSRMapping, draw_line
from stringart.algorithm.costmethod import CostMethod
from stringart.algorithm.cost_eval import find_cost
from stringart.preprocessing.image import BaseImage
from stringart.algorithm.cost_eval import update_costs_jit_mean


def create_string_art(
    first_anchor: int,
    blueprint: StringArtBlueprint,
    line_csr_mapping: StringLineCSRMapping,
    iterations: int,
    cost_method: CostMethod = CostMethod.MEAN,
    max_darkness: float = None,
    importance_map: ImportanceMap = None,
    use_prev_anchor: bool = True,
    random_neighbor: bool = False,
    profiling: bool = False,
    accuracy_methods: List[Tuple[AccuracyMethod, int]] = None
) -> StringImage:
    """
    Generate string art by iteratively selecting and adding optimal strings.

    Steps:
      1. Build internal data structures (lines, cost vector, pixel→line mapping).
      2. On each iteration, choose the best next string based on current costs.
      3. Update the canvas and all line costs incrementally.
      4. Optionally evaluate similarity at intervals.

    Args:
        first_anchor: Index of the starting anchor.
        base_img: Reference image used to compute cost improvements.
        string_art_img: Canvas object storing the current string art state.
        line_pixel_dict: Maps anchor-pair tuples to pixel coordinates arrays.
        line_darkness_dict: Maps anchor-pair tuples to per-pixel darkness values.
        iterations: Number of strings to add.
        cost_method: Method to compute line cost (e.g. mean difference).
        max_darkness: Clamp maximum darkness per pixel if specified.
        importance_map: Per-pixel importance weighting; defaults to uniform.
        use_prev_anchor: If True, next line starts from last end anchor.
        random_neighbor: If True, randomly choose among two possible next anchors.
        profiling: If True, use pure-Python update for profiling (slower).
        accuracy_methods: A list of methods of evaluating accuracy. Ex. [(LINE_COST, 5)] evaluates the line cost accuracy every five iterations

    Returns:
        string_art_img: The final StringArtImage with all added strings.
    """
    string_art_img = StringImage(blueprint=blueprint, anchors=blueprint.anchors, line_darkness=blueprint.line_darkness, mask=blueprint.mask)
    
    #--- Prepare line objects ---
    line_ptr = line_csr_mapping.line_ptr
    line_pix = line_csr_mapping.line_pix
    line_dark = line_csr_mapping.line_dark
    pixel_to_line_ptr = line_csr_mapping.pixel_to_line_ptr
    pixel_line_indices = line_csr_mapping.pixel_line_indices

    anchor_to_lines = line_csr_mapping.anchor_to_lines

    # Load the defualt pixel line weights if no importance map is present, otherwise generates new one
    pixel_line_weights = None
    if not importance_map:
        print("Importance map not found, using uniform map.")
        importance_map = ImportanceMap(img=np.ones_like(blueprint.img))
        pixel_line_weights = line_csr_mapping.load_default_pixel_line_weights()
    else:
        # Build pixel -> (line, weight) mapping via JIT for speed
        print("Building pixel-to-line and line-to-pixel mapping for the weight (CSR)…")
        imp_flat  = importance_map.img.reshape(-1)
        pixel_line_weights = build_pixel_weights_direct_jit(
            line_ptr,
            line_pix,
            imp_flat,
            pixel_to_line_ptr
        )

    base_flat = blueprint.img.reshape(-1)
    imp_flat  = importance_map.img.reshape(-1)

    print("Calculating initial line costs")
    costs = init_line_costs_csr_jit(line_ptr=line_ptr, line_pix=line_pix, base_flat=base_flat, imp_flat=imp_flat)

    #--- Inner helper to choose the next best line ---
    def find_best_line(prev_anchor: int) -> Tuple[Tuple[int,int], float, int, int]:
        """
        Select the optimal next string based on current costs.

        Returns best_anchors (sorted tuple), best_cost, start_anchor, end_anchor.
        """
        # Determine candidate start anchors (neighbors)
        nbrs = get_neighbors(
            string_art_img.anchors, prev_anchor, random_neighbor=random_neighbor
        )

        # Gather (line_idx, start, end) for each candidate line
        candidates: List[Tuple[int,int,int]] = []
        for start in nbrs:
            for line_idx, end in anchor_to_lines[start]:
                candidates.append((line_idx, start, end))

        if not candidates:
            raise RuntimeError(f"No candidate lines from anchor {prev_anchor}")

        # Find the candidate with minimum cost
        line_idxs = np.array([c[0] for c in candidates], dtype=np.int32)
        costs_subset = costs[line_idxs]
        best_pos = int(np.argmin(costs_subset))

        best_j, best_start, best_end = candidates[best_pos]
        best_cost = float(costs[best_j])
        best_anchors = tuple(sorted((best_start, best_end)))
        return best_j, best_anchors, best_cost, best_start, best_end

    #--- Begin iterative string placement ---
    canvas_flat = string_art_img.img.reshape(-1)
    previous_anchor_idx = (
        first_anchor if use_prev_anchor else random.randint(0, len(string_art_img.anchors)-1)
    )

    accuracy_list = defaultdict(lambda: ([], [])) # initialize a tuple of empty lists for each new key
    line_len_list = []

    for iteration in tqdm(
        range(iterations), desc=f"Creating string art for {blueprint.img_path}"
    ):
        # 1) Choose next string
        best_j, best_anchors, best_cost, best_start, best_end = find_best_line(previous_anchor_idx)

        # 2) Extract pixels & darkness values for chosen line
        s_ptr, e_ptr = line_ptr[best_j], line_ptr[best_j+1]
        pix_flat = line_pix[s_ptr:e_ptr]
        darkness_vals = line_dark[s_ptr:e_ptr]

        # 3) Update canvas and compute per-pixel deltas
        old_vals = canvas_flat[pix_flat]
        temp = old_vals + darkness_vals
        new_vals = np.clip(temp, 0, max_darkness) if max_darkness is not None else temp
        delta_vals = new_vals - old_vals
        canvas_flat[pix_flat] = new_vals

        # 4) Incrementally update line costs
        update_costs(
            costs, pixel_to_line_ptr, pixel_line_indices, pixel_line_weights,
            pix_flat, delta_vals, profiling=profiling
        )

        # 5) Record and advance anchor
        string_art_img.string_path.append((best_start, best_end))
        previous_anchor_idx = best_end

        # Different accuracy evaluations
        for accuracy_type, freq in accuracy_methods:

            if(accuracy_type == AccuracyMethod.AVG_LINE_LENGTH):
                # to find the avg line lengths, you need to find line lengths for every iteration
                start_coords = blueprint.anchors[best_start].coordinates
                end_coords = blueprint.anchors[best_end].coordinates
                line = draw_line(start_coords, end_coords, multiplier=1, mask=blueprint.mask)
                line_len_list.append(len(line[0]))

            if(iteration % freq) != 0: continue
            accuracy_list[accuracy_type][0].append(iteration)
            match accuracy_type:
                case AccuracyMethod.LINE_COST:
                    accuracy_list[accuracy_type][1].append(best_cost)
                case AccuracyMethod.IMAGE_COST:
                    accuracy_num = np.mean(string_art_img.img - blueprint.img)
                    accuracy_list[accuracy_type][1].append(accuracy_num)
                case AccuracyMethod.SSIM:
                    accuracy_num = eval_ssim(string_art_img.img, blueprint.img)
                    accuracy_list[accuracy_type][1].append(accuracy_num)
                case AccuracyMethod.MS_SSIM:
                    accuracy_num = eval_ms_ssim(string_art_img.img, blueprint.img)
                    accuracy_list[accuracy_type][1].append(accuracy_num)
                case AccuracyMethod.LPIPS:
                    accuracy_num = eval_lpips(string_art_img.img, blueprint.img)
                    accuracy_list[accuracy_type][1].append(accuracy_num)

        string_art_img.best_anchors_list.append(best_anchors)

    

    for accuracy_type, freq in accuracy_methods:
        if(accuracy_type == AccuracyMethod.AVG_LINE_LENGTH):
            line_avgs = find_line_len_averages(line_len_list, freq)
            accuracy_list[AccuracyMethod.AVG_LINE_LENGTH][1].extend(line_avgs)

    string_art_img.accuracy_list = dict(accuracy_list)
    return string_art_img


#------------------------------------------------------------------------------
# Helper routines
#------------------------------------------------------------------------------

def find_line_len_averages(line_lengths, block_size):
    """
    Compute the average of each consecutive block_size elements in `line_lengths`.
    If the last block is shorter than block_size, take the last block_size
    elements (overlapping the previous block) and average those.
    """
    n = len(line_lengths)
    averages = []
    
    for start in range(0, n, block_size):
        # if we have a full block, just slice it
        if start + block_size <= n:
            chunk = line_lengths[start:start + block_size]
        else:
            # at the end: take the last `block_size` elements instead
            chunk = line_lengths[-block_size:]
        
        averages.append(sum(chunk) / len(chunk))
        
        # once we hit the tail-case, we’re done
        if start + block_size >= n:
            break
    
    return averages

def update_costs(
    costs: np.ndarray,
    pixel_ptr: np.ndarray,
    flat_line_ids: np.ndarray,
    flat_weights: np.ndarray,
    pix: np.ndarray,
    delta: np.ndarray,
    profiling: bool
):
    """
    Incrementally adjust costs array after adding a string.

    If profiling, use pure-Python loops; otherwise call JIT-optimized.
    """
    if profiling:
        # Python version for profiling (slower but traceable)
        for p, d in zip(pix, delta):
            for k in range(pixel_ptr[p], pixel_ptr[p+1]):
                costs[flat_line_ids[k]] += d * flat_weights[k]
    else:
        update_costs_jit_mean(
            costs, pixel_ptr, flat_line_ids, flat_weights, pix, delta
        )




# def init_line_costs(lines: List[StringLine]) -> np.ndarray:
#     """
#     Compute initial cost for each line on a blank canvas:
#       cost[j] = -mean(base_pixel * importance).
#     """
#     n = len(lines)
#     costs = np.empty(n, dtype=np.float64)
#     for j, line in tqdm(
#         enumerate(lines), desc="Building initial line costs"
#     ):
#         weighted_sum = np.sum(line.base_image * line.importance_values)
#         costs[j] = -weighted_sum / line.base_image.size
#     return costs

@njit
def init_line_costs_csr_jit(
    line_ptr: np.ndarray,    # int32[L+1]
    line_pix: np.ndarray,    # int32[E]
    base_flat: np.ndarray,   # float64[P] flattened base image
    imp_flat: np.ndarray     # float64[P] flattened importance map
) -> np.ndarray:
    """
    JIT-compiled: Compute initial cost per line on an all-white canvas.

    Args:
      line_ptr: CSR pointer array of shape (L+1,)
      line_pix: Flattened pixel indices of shape (E,)
      base_flat: Flattened target image darkness, shape (P,)
      imp_flat:  Flattened importance weights, shape (P,)

    Returns:
      costs: 1D float64 array of length L, where
        costs[j] = -mean(base_flat[p] * imp_flat[p]) for p in line_ptr[j]:line_ptr[j+1]
    """
    L = line_ptr.shape[0] - 1
    costs = np.empty(L, dtype=np.float64)
    for j in range(L):
        s = line_ptr[j]
        e = line_ptr[j+1]
        total = 0.0
        for k in range(s, e):
            p = line_pix[k]
            total += base_flat[p] * imp_flat[p]
        costs[j] = -total / (e - s)
    return costs

# @njit
# def build_mapping_jit(
#     line_ptr: np.ndarray,    # int32[L+1]
#     line_pix: np.ndarray,    # int32[E]
#     line_wval: np.ndarray,   # float32[E]
#     H: int,
#     W: int
# ):
#     """
#     JIT-compose  pixel_ptr, flat_line_ids, flat_weights for fast updates.
#     Uses a two-pass CSR build (count+prefixsum, then fill).
#     """
#     P = H * W
#     E = line_pix.shape[0]

#     # 1) Count entries per pixel
#     counts = np.zeros(P, np.int32)
#     for k in range(E):
#         counts[line_pix[k]] += 1

#     # 2) Prefix-sum to build pixel_ptr
#     pixel_ptr = np.empty(P+1, np.int32)
#     pixel_ptr[0] = 0
#     for p in range(P):
#         pixel_ptr[p+1] = pixel_ptr[p] + counts[p]

#     # 3) Allocate destination arrays
#     flat_line_ids = np.empty(E, np.int32)
#     flat_weights = np.empty(E, np.float32)

#     # 4) Reset counts to use as write cursors
#     for p in range(P):
#         counts[p] = 0

#     # 5) Fill in the line indices & weights
#     L = line_ptr.shape[0] - 1
#     for j in range(L):
#         for k in range(line_ptr[j], line_ptr[j+1]):
#             p = line_pix[k]
#             dst = pixel_ptr[p] + counts[p]
#             flat_line_ids[dst] = j
#             flat_weights[dst] = line_wval[k]
#             counts[p] += 1

#     return pixel_ptr, flat_line_ids, flat_weights

@njit
def build_pixel_weights_direct_jit(
    line_ptr:             np.ndarray,  # int32[L+1]
    line_pix:             np.ndarray,  # int32[E]
    imp_flat:             np.ndarray,  # float32 or float64[P]
    pixel_to_line_ptr:    np.ndarray   # int32[P+1]
) -> np.ndarray:
    """
    For each line j and each k in line_ptr[j]:line_ptr[j+1],
    compute inv_len = 1/(line_length_j) and 
    weight = imp_flat[p] * inv_len,
    then scatter it into pixel_line_weights at the reverse CSR slots.

    Returns:
      pixel_line_weights: float32[E], aligned with pixel_line_indices.
    """
    P = pixel_to_line_ptr.shape[0] - 1
    E = line_pix.shape[0]
    L = line_ptr.shape[0] - 1

    # Output array
    pixel_line_weights = np.empty(E, dtype=np.float32)
    # Cursor per pixel for write offsets
    cursor = np.zeros(P, np.int32)

    # Single pass over every line→pixel entry
    for j in range(L):
        start = line_ptr[j]
        end   = line_ptr[j+1]
        inv_len = 1.0 / (end - start)
        for k in range(start, end):
            p = line_pix[k]
            w = imp_flat[p] * inv_len
            dst = pixel_to_line_ptr[p] + cursor[p]
            pixel_line_weights[dst] = w
            cursor[p] += 1

    return pixel_line_weights

# @njit
# def build_pixel_line_weights_jit(
#     line_ptr:             np.ndarray,  # int32[L+1]
#     line_pix:             np.ndarray,  # int32[E]
#     line_wval:            np.ndarray,  # float32[E]
#     pixel_to_line_ptr:    np.ndarray   # int32[P+1]
# ) -> np.ndarray:
#     """
#     Given:
#       - line_ptr/line_pix/line_wval: your forward CSR (line→pixels & weights)
#       - pixel_to_line_ptr:           your reverse CSR pointers (pixel→lines)
#     Build:
#       - pixel_line_weights: float32[E] aligned with pixel_line_indices
#     """
#     P1 = pixel_to_line_ptr.shape[0]
#     P  = P1 - 1
#     E  = line_pix.shape[0]

#     # allocate the reverse‐weights array
#     pixel_line_weights = np.empty(E, dtype=np.float32)

#     # write‐cursor per pixel
#     cursor = np.zeros(P, dtype=np.int32)

#     # walk every line‐entry once
#     L = line_ptr.shape[0] - 1
#     for j in range(L):
#         start = line_ptr[j]
#         end   = line_ptr[j+1]
#         for k in range(start, end):
#             p   = line_pix[k]
#             dst = pixel_to_line_ptr[p] + cursor[p]
#             pixel_line_weights[dst] = line_wval[k]
#             cursor[p] += 1

#     return pixel_line_weights


# def fill_line_weights(
#     line_ptr: np.ndarray,    # int32[L+1]
#     line_pix: np.ndarray,    # int32[E]
#     importance_map: np.ndarray  # float32 or float64 (H,W)
# ) -> np.ndarray:
#     """
#     Given CSR pointers 'line_ptr' and flat pixel indices 'line_pix',
#     and an importance map of shape (H,W), compute line_wval of shape (E,),
#     where for each line j:
    
#        inv_len = 1 / (line_ptr[j+1] - line_ptr[j])
#        for k in line_ptr[j]..line_ptr[j+1]-1:
#            line_wval[k] = importance_flat[line_pix[k]] * inv_len
#     """
#     H, W = importance_map.shape
#     P = H * W
    
#     # 1) flatten the importance map
#     importance_flat = importance_map.reshape(-1)
    
#     # 2) allocate output
#     E = line_pix.shape[0]
#     line_wval = np.empty(E, dtype=np.float32)
    
#     # 3) per-line fill
#     L = line_ptr.shape[0] - 1
#     for j in range(L):
#         start = line_ptr[j]
#         end   = line_ptr[j+1]
#         inv_len = 1.0 / (end - start)
#         # vectorized for this line
#         idxs = line_pix[start:end]         # p-values for this line
#         line_wval[start:end] = importance_flat[idxs] * inv_len
    
#     return line_wval

# def flatten_line_pixels(
#     lines: List[StringLine],
#     canvas_width: int
# ) -> Tuple[np.ndarray, np.ndarray]:
#     """
#     Flatten each line's pixel coordinates into CSR arrays:
#       line_ptr_pix and line_pix.
#     """
#     L = len(lines)
#     lengths = [ln.string_pixels.shape[0] for ln in lines]
#     total = sum(lengths)

#     line_ptr = np.zeros(L+1, dtype=np.int32)
#     line_pix = np.empty(total, dtype=np.int32)

#     idx = 0
#     for j, ln in tqdm(
#         enumerate(lines), desc="Flattening line pixels"
#     ):
#         line_ptr[j] = idx
#         xs = ln.string_pixels[:, 0]
#         ys = ln.string_pixels[:, 1]
#         flat = xs * canvas_width + ys
#         line_pix[idx:idx+lengths[j]] = flat
#         idx += lengths[j]
#     line_ptr[L] = idx
#     return line_ptr, line_pix

# def flatten_line_weights(lines: List[StringLine]) -> np.ndarray:
#     """
#     Flatten each line's per-pixel weight = importance / line_length.

#     Returns wflat, cooresponding to line_ptr similar to CSR pointers.
#     """
#     lengths = [ln.string_pixels.shape[0] for ln in lines]
#     total = sum(lengths)
#     line_wval = np.empty(total, dtype=np.float32)

#     idx = 0
#     for j, ln in tqdm(enumerate(lines), desc="Flattening line weights"):
#         inv_len = 1.0 / ln.string_pixels.shape[0]
#         line_wval[idx:idx+lengths[j]] = ln.importance_values * inv_len
#         idx += lengths[j]
#     return line_wval

# def flatten_line_darkness(lines: List[StringLine]) -> np.ndarray:
#     """
#     Turns each lines[j].string_darkness (length Lj)
#     CSR arrays:
#       line_ptr[j] .. line_ptr[j+1] slice into dark_flat
#     """
#     # Compute per‐line lengths
#     lengths = [ln.string_darkness.shape[0] for ln in lines]
#     total = sum(lengths)

#     line_dark = np.empty(total, dtype=np.float32)

#     idx = 0
#     for j, ln in enumerate(lines):
#         # copy the per-pixel darkness into one big array
#         line_dark[idx:idx+lengths[j]] = ln.string_darkness
#         idx += lengths[j]
#     return line_dark


# def build_pixel_line_mapping(
#     lines: List[StringLine],
#     canvas_shape: Tuple[int, int]
# ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
#     """
#     Original dict-based pixel→(line, weight) mapping (fallback / legacy).
#     """
#     H, W = canvas_shape
#     P = H * W

#     pixel_to_lines = defaultdict(list)
#     for j, line in tqdm(
#         enumerate(lines), desc="Building temporary lines"
#     ):
#         coords = line.string_pixels  # (Lj,2)
#         imp_vals = line.importance_values
#         Lj = coords.shape[0]
#         w_scale = 1.0 / Lj
#         for (x, y), imp in zip(coords, imp_vals):
#             p = x * W + y
#             w = imp * w_scale
#             pixel_to_lines[p].append((j, w))

#     E = sum(len(v) for v in pixel_to_lines.values())
#     pixel_ptr = np.zeros(P+1, dtype=np.int32)
#     flat_line_ids = np.empty(E, dtype=np.int32)
#     flat_weights  = np.empty(E, dtype=np.float32)

#     idx = 0
#     for p in tqdm(range(P), desc="Flattening line dict into arrays"):
#         pixel_ptr[p] = idx
#         for (j, w) in pixel_to_lines.get(p, []):
#             flat_line_ids[idx] = j
#             flat_weights[idx]  = w
#             idx += 1
#     pixel_ptr[P] = idx
#     return pixel_ptr, flat_line_ids, flat_weights


def build_anchor_to_lines(
    anchor_line_idx: Dict[Tuple[int,int], int]
) -> Dict[int, List[Tuple[int,int]]]:
    """
    Invert anchor-pair→line_idx mapping to anchor→list of lines.
    """
    d: Dict[int, List[Tuple[int,int]]] = defaultdict(list)
    for (a, b), j in tqdm(anchor_line_idx.items(), desc="Mapping anchors to lines"):
        d[a].append((j, b))
        d[b].append((j, a))
    return d


def get_neighbors(
    arr: list,
    idx: int,
    random_neighbor: bool=False
) -> List[int]:
    """
    Return the previous and next indices in a circular arrangement.
    If random_neighbor, returns a single random neighbor.
    """
    left_idx = (idx - 1) % len(arr)
    right_idx = (idx + 1) % len(arr)
    if random_neighbor:
        return [random.choice([left_idx, right_idx])]
    return [left_idx, right_idx]


def compare_images(
    input_img: np.ndarray,
    target_img: np.ndarray,
    method: str
) -> float:
    """
    Compute similarity between two images.
    Supports 'ssim' and 'psnr'.
    """
    if method == "ssim":
        return ssim(input_img, target_img, data_range=1)
    elif method == "psnr":
        mse = np.mean((input_img - target_img) ** 2)
        if mse == 0:
            return float('inf')
        return 20 * math.log10(255.0 / math.sqrt(mse))
    else:
        raise ValueError(f"Unknown comparison method: {method}")
