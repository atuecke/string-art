import numpy as np
import math
import random
from enum import Enum
from typing import List, Tuple, Dict
from collections import defaultdict

from tqdm import tqdm
from numba import njit
from skimage.metrics import structural_similarity as ssim

from stringart.core.stringimage import StringArtImage
from stringart.preprocessing.importancemaps import ImportanceMap
from stringart.algorithm.lines import StringLine
from stringart.algorithm.costmethod import CostMethod
from stringart.algorithm.cpu.eval import find_cost
from stringart.preprocessing.image import BaseImage


def create_string_art_cpu(
    first_anchor: int,
    base_img: BaseImage,
    string_art_img: StringArtImage,
    line_pixel_dict: dict,
    line_darkness_dict: dict,
    iterations: int,
    cost_method: CostMethod = CostMethod.MEAN,
    max_darkness: float = None,
    eval_interval: int = None,
    importance_map: ImportanceMap = None,
    use_prev_anchor: bool = True,
    random_neighbor: bool = False,
    profiling: bool = False
) -> StringArtImage:
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
        eval_interval: Interval (in iterations) to compute global similarity.
        importance_map: Per-pixel importance weighting; defaults to uniform.
        use_prev_anchor: If True, next line starts from last end anchor.
        random_neighbor: If True, randomly choose among two possible next anchors.
        profiling: If True, use pure-Python update for profiling (slower).

    Returns:
        string_art_img: The final StringArtImage with all added strings.
    """
    #--- Prepare importance map and line objects ---
    if not importance_map:
        print("Importance map not found, using uniform map.")
        importance_map = ImportanceMap(img=np.ones_like(base_img.img))

    anchor_line_idx: Dict[Tuple[int,int], int] = {}
    lines: List[StringLine] = []

    # Build StringLine objects for each anchor-pair
    for idx, anchors in tqdm(
        enumerate(line_pixel_dict.keys()), desc="Building line arrays"
    ):
        pixels = line_pixel_dict[anchors]
        x_coords, y_coords = pixels[:, 0], pixels[:, 1]

        anchor_line_idx[anchors] = idx
        lines.append(StringLine(
            base_image=base_img.img[x_coords, y_coords],
            importance_values=importance_map.img[x_coords, y_coords],
            string_pixels=pixels,
            string_darkness=line_darkness_dict[anchors]
        ))

    #--- Initialize core data structures ---
    H, W = string_art_img.img.shape  # Canvas height and width
    costs = init_line_costs(lines)

    # Flatten line pixels and weights for CSR-style mapping
    line_ptr_pix, line_pix = flatten_line_pixels(lines, W)
    line_ptr_w,   line_wval = flatten_line_weights(lines)

    # Map from each anchor to lines leaving it
    anchor_to_lines = build_anchor_to_lines(anchor_line_idx)

    # Build pixel -> (line, weight) mapping via JIT for speed
    print("Building pixel-to-line mapping (CSR)…")
    pixel_ptr, flat_line_ids, flat_weights = build_mapping_jit(
        line_ptr_pix, line_pix,
        line_wval, H, W
    )

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
        return best_anchors, best_cost, best_start, best_end

    #--- Begin iterative string placement ---
    canvas_flat = string_art_img.img.reshape(-1)
    previous_anchor_idx = (
        first_anchor if use_prev_anchor else random.randint(0, len(string_art_img.anchors)-1)
    )

    for iteration in tqdm(
        range(iterations), desc=f"Creating string art for {base_img.path}"
    ):
        # 1) Choose next string
        best_anchors, best_cost, best_start, best_end = find_best_line(previous_anchor_idx)
        best_j = anchor_line_idx[best_anchors]

        # 2) Extract pixels & darkness values for chosen line
        s_ptr, e_ptr = line_ptr_pix[best_j], line_ptr_pix[best_j+1]
        pix_flat = line_pix[s_ptr:e_ptr]
        darkness_vals = lines[best_j].string_darkness

        # 3) Update canvas and compute per-pixel deltas
        old_vals = canvas_flat[pix_flat]
        temp = old_vals + darkness_vals
        new_vals = np.clip(temp, 0, max_darkness) if max_darkness is not None else temp
        delta_vals = new_vals - old_vals
        canvas_flat[pix_flat] = new_vals

        # 4) Incrementally update line costs
        update_costs(
            costs, pixel_ptr, flat_line_ids, flat_weights,
            pix_flat, delta_vals, profiling=profiling
        )

        # 5) Record and advance anchor
        string_art_img.string_path.append((best_start, best_end))
        previous_anchor_idx = best_end
        string_art_img.cost_list.append((best_cost, iteration))
        string_art_img.best_anchors_list.append(best_anchors)

        # 6) Optional full-image similarity check
        if eval_interval and (iteration % eval_interval == 0):
            sim = compare_images(string_art_img.img, base_img.img, method="ssim")
            string_art_img.similarities.append((iteration, sim))

    return string_art_img


#------------------------------------------------------------------------------
# Helper routines
#------------------------------------------------------------------------------

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
        update_costs_jit(
            costs, pixel_ptr, flat_line_ids, flat_weights, pix, delta
        )


@njit
def update_costs_jit(
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


def init_line_costs(lines: List[StringLine]) -> np.ndarray:
    """
    Compute initial cost for each line on a blank canvas:
      cost[j] = -mean(base_pixel * importance).
    """
    n = len(lines)
    costs = np.empty(n, dtype=np.float64)
    for j, line in tqdm(
        enumerate(lines), desc="Building initial line costs"
    ):
        weighted_sum = np.sum(line.base_image * line.importance_values)
        costs[j] = -weighted_sum / line.base_image.size
    return costs


@njit
def build_mapping_jit(
    line_ptr: np.ndarray,    # int32[L+1]
    line_pix: np.ndarray,    # int32[E]
    line_wval: np.ndarray,   # float32[E]
    H: int,
    W: int
):
    """
    JIT-compose  pixel_ptr, flat_line_ids, flat_weights for fast updates.
    Uses a two-pass CSR build (count+prefixsum, then fill).
    """
    P = H * W
    E = line_pix.shape[0]

    # 1) Count entries per pixel
    counts = np.zeros(P, np.int32)
    for k in range(E):
        counts[line_pix[k]] += 1

    # 2) Prefix-sum to build pixel_ptr
    pixel_ptr = np.empty(P+1, np.int32)
    pixel_ptr[0] = 0
    for p in range(P):
        pixel_ptr[p+1] = pixel_ptr[p] + counts[p]

    # 3) Allocate destination arrays
    flat_line_ids = np.empty(E, np.int32)
    flat_weights = np.empty(E, np.float32)

    # 4) Reset counts to use as write cursors
    for p in range(P):
        counts[p] = 0

    # 5) Fill in the line indices & weights
    L = line_ptr.shape[0] - 1
    for j in range(L):
        for k in range(line_ptr[j], line_ptr[j+1]):
            p = line_pix[k]
            dst = pixel_ptr[p] + counts[p]
            flat_line_ids[dst] = j
            flat_weights[dst] = line_wval[k]
            counts[p] += 1

    return pixel_ptr, flat_line_ids, flat_weights


def flatten_line_weights(lines: List[StringLine]) -> Tuple[np.ndarray, np.ndarray]:
    """
    Flatten each line's per-pixel weight = importance / line_length.

    Returns (line_ptr_w, wflat) similar to CSR pointers.
    """
    lengths = [ln.string_pixels.shape[0] for ln in lines]
    L = len(lines)
    line_ptr = np.zeros(L+1, dtype=np.int32)
    total = sum(lengths)
    wflat = np.empty(total, dtype=np.float32)

    idx = 0
    for j, ln in tqdm(enumerate(lines), desc="Flattening line weights"):
        line_ptr[j] = idx
        inv_len = 1.0 / ln.string_pixels.shape[0]
        wflat[idx:idx+lengths[j]] = ln.importance_values * inv_len
        idx += lengths[j]
    line_ptr[L] = idx
    return line_ptr, wflat


def build_pixel_line_mapping(
    lines: List[StringLine],
    canvas_shape: Tuple[int, int]
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Original dict-based pixel→(line, weight) mapping (fallback / legacy).
    """
    H, W = canvas_shape
    P = H * W

    pixel_to_lines = defaultdict(list)
    for j, line in tqdm(
        enumerate(lines), desc="Building temporary lines"
    ):
        coords = line.string_pixels  # (Lj,2)
        imp_vals = line.importance_values
        Lj = coords.shape[0]
        w_scale = 1.0 / Lj
        for (x, y), imp in zip(coords, imp_vals):
            p = x * W + y
            w = imp * w_scale
            pixel_to_lines[p].append((j, w))

    E = sum(len(v) for v in pixel_to_lines.values())
    pixel_ptr = np.zeros(P+1, dtype=np.int32)
    flat_line_ids = np.empty(E, dtype=np.int32)
    flat_weights  = np.empty(E, dtype=np.float32)

    idx = 0
    for p in tqdm(range(P), desc="Flattening line dict into arrays"):
        pixel_ptr[p] = idx
        for (j, w) in pixel_to_lines.get(p, []):
            flat_line_ids[idx] = j
            flat_weights[idx]  = w
            idx += 1
    pixel_ptr[P] = idx
    return pixel_ptr, flat_line_ids, flat_weights


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


def flatten_line_pixels(
    lines: List[StringLine],
    canvas_width: int
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Flatten each line's pixel coordinates into CSR arrays:
      line_ptr_pix and line_pix.
    """
    L = len(lines)
    lengths = [ln.string_pixels.shape[0] for ln in lines]
    total = sum(lengths)

    line_ptr = np.zeros(L+1, dtype=np.int32)
    line_pix = np.empty(total, dtype=np.int32)

    idx = 0
    for j, ln in tqdm(
        enumerate(lines), desc="Flattening line pixels"
    ):
        line_ptr[j] = idx
        xs = ln.string_pixels[:, 0]
        ys = ln.string_pixels[:, 1]
        flat = xs * canvas_width + ys
        line_pix[idx:idx+lengths[j]] = flat
        idx += lengths[j]
    line_ptr[L] = idx
    return line_ptr, line_pix


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
