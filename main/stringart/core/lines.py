import numpy as np
import cupy as cp
from typing import Optional, Dict, List, Tuple, Any
import pickle
import json
from skimage.draw import line_aa
from typing import Tuple
import cv2

class StringLineCSRMapping():
    def __init__(self,
                 line_ptr: np.ndarray,  # int32[L+1]
                 line_pix: np.ndarray,  # int32[E]
                 line_dark: np.ndarray,  # float32[E]
                 anchor_to_lines:  Optional[Dict[int, List[Tuple[int,int]]]],
                 pixel_to_line_ptr,
                 pixel_line_indices,
                 pixel_line_weights_default,
                 folder_path
        ):
        self.line_ptr  = line_ptr   # int32[L+1]
        self.line_pix  = line_pix   # int32[E]
        self.line_dark = line_dark  # float32[E]
        self.pixel_to_line_ptr = pixel_to_line_ptr
        self.pixel_line_indices = pixel_line_indices
        self.pixel_line_weights_default = pixel_line_weights_default

        # build an empty anchor->lines map as well (to fill later)
        self.anchor_to_lines: Optional[Dict[int, List[Tuple[int,int]]]] = anchor_to_lines

        self.folder_path = folder_path
    
    @classmethod
    def load(cls, folder_path: str):
        # loads them back—just a few super-fast C-loops
        line_ptr = np.load(f"{folder_path}/line_ptr.npy", mmap_mode=None)
        line_pix = np.load(f"{folder_path}/line_pix.npy", mmap_mode=None)
        line_dark = np.load(f"{folder_path}/line_dark.npy", mmap_mode=None)
        pixel_to_line_ptr = np.load(f"{folder_path}/pixel_to_line_ptr.npy", mmap_mode=None)
        pixel_line_indices = np.load(f"{folder_path}/pixel_line_indices.npy", mmap_mode=None)

        
        with open(f"{folder_path}/anchor_to_lines.pkl", "rb") as f:
            anchor_to_lines = pickle.load(f)

        return cls(line_ptr=line_ptr, line_pix=line_pix, line_dark=line_dark, anchor_to_lines=anchor_to_lines, pixel_to_line_ptr=pixel_to_line_ptr, pixel_line_indices=pixel_line_indices, pixel_line_weights_default=None, folder_path=folder_path)
    
    def load_default_pixel_line_weights(self):
        pixel_line_weights_default = self.pixel_line_weights_default

        if not pixel_line_weights_default:
            pixel_line_weights_default = np.load(f"{self.folder_path}/pixel_line_weights_default.npy", mmap_mode=None)

        return pixel_line_weights_default
    
    def save(self, folder_path: str):
        # writes four very fast-to-load .npy files
        np.save(f"{folder_path}/line_ptr.npy", self.line_ptr)
        np.save(f"{folder_path}/line_pix.npy", self.line_pix)
        np.save(f"{folder_path}/line_dark.npy", self.line_dark)
        np.save(f"{folder_path}/pixel_to_line_ptr.npy", self.pixel_to_line_ptr)
        np.save(f"{folder_path}/pixel_line_indices.npy", self.pixel_line_indices)
        np.save(f"{folder_path}/pixel_line_weights_default.npy", self.pixel_line_weights_default)

        with open(f"{folder_path}/anchor_to_lines.pkl", "wb") as f:
            pickle.dump(self.anchor_to_lines, f)

    


class GpuStringLine():
    def __init__(
            self,
            string_darkness: cp.ndarray,
            string_pixels: cp.ndarray,
            line_slice: slice
    ) -> None:
        self.string_pixels = string_pixels
        self.string_darkness = string_darkness
        self.line_slice = line_slice
        self.string_pixel_map = []

class FlatGpuStringLines():
    def __init__(
            self,
            base_image: cp.ndarray,
            importance_values: cp.ndarray,
            string_pixels: cp.ndarray,
            string_darkness: cp.ndarray,
            string_art_values: cp.ndarray,
            pixelwise_cost: cp.ndarray,
            slices: list= []
    ) -> None:
        """
        """
        self.base_image = base_image
        self.importance_values = importance_values
        self.string_pixels = string_pixels
        self.string_darkness = string_darkness
        self.string_art_values = string_art_values
        self.pixelwise_cost = pixelwise_cost
        self.slices = slices


def draw_line(
    p0: Tuple[int,int],
    p1: Tuple[int,int],
    multiplier: float,
    mask: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Very fast Wu’s line from skimage, with final mask‐filter.
    """
    x0, y0 = p0
    x1, y1 = p1
    rr, cc, intensity = line_aa(x0, y0, x1, y1)  # note row=y, col=x

    # scale to your darkness multiplier
    vals = (intensity.astype(np.float32) * multiplier)

    # stack coords as (i,j) for mask indexing
    coords = np.stack([rr, cc], axis=1)  # shape (N,2)

    # vectorized masking
    keep = mask[rr, cc]
    return coords[keep], vals[keep]

def draw_line_old(p0: tuple, p1: tuple, multiplier: float, mask: np.ndarray):
    """
    Creates a dictionary of coordinates and their darkness values of a line between two points. Uses Xiaolin Wu's anti-aliasing algorithm

    Args: 
        p0: The first coordinate for the line
        p1: The second cordinate for the line
        multiplier: The darkness value that gets multiplied by each pixel
        mask: The boolean mask for a circle
    
    Returns:
        A dictionary of each pixel in the line and their darkness value
    """
    
    x0, y0 = p0
    x1, y1 = p1
    pixel_list = []
    darkness_list = []
    steep = abs(y1 - y0) > abs(x1 - x0)
    if steep:
        x0, y0 = y0, x0
        x1, y1 = y1, x1

    if x0 > x1:
        x0, x1 = x1, x0
        y0, y1 = y1, y0

    dx = x1 - x0
    dy = y1 - y0
    gradient = dy / dx if dx != 0 else 1

    # handle first endpoint (pixel) placement
    xend = round(x0)
    yend = y0 + gradient * (xend - x0)
    xgap = 1 - ((x0 + 0.5) % 1)
    xpxl1 = xend
    ypxl1 = int(yend)
    if steep:
        try:
            if mask[ypxl1, xpxl1]:
                pixel_list.append((ypxl1, xpxl1))
                darkness_list.append(xgap * (1 - (yend % 1)) * multiplier)
        except IndexError:
            pass
        try:
            if mask[ypxl1+1, xpxl1]:
                pixel_list.append((ypxl1+1, xpxl1))
                darkness_list.append(xgap * (yend % 1) * multiplier)
        except IndexError:
            pass
    else:
        try:
            if mask[xpxl1, ypxl1]:
                pixel_list.append((xpxl1, ypxl1))
                darkness_list.append(xgap * (1 - (yend % 1)) * multiplier)
        except IndexError:
            pass
        try:
            if mask[xpxl1, ypxl1+1]:
                pixel_list.append((xpxl1, ypxl1+1))
                darkness_list.append(xgap * (yend % 1) * multiplier)
        except IndexError:
            pass
    intery = yend + gradient

    # handle second endpoint
    xend = round(x1)
    yend = y1 + gradient * (xend - x1)
    xgap = (x1 + 0.5) % 1
    xpxl2 = xend
    ypxl2 = int(yend)
    if steep:
        try:
            if mask[ypxl2, xpxl2]:
                pixel_list.append((ypxl2, xpxl2))
                darkness_list.append(xgap * (1 - (yend % 1)) * multiplier)
        except IndexError:
            pass
        try:
            if mask[ypxl2+1, xpxl2]:
                pixel_list.append((ypxl2+1, xpxl2))
                darkness_list.append(xgap * (yend % 1) * multiplier)
        except IndexError:
            pass
    else:
        try:
            if mask[xpxl2, ypxl2]:
                pixel_list.append((xpxl2, ypxl2))
                darkness_list.append(xgap * (1 - (yend % 1)) * multiplier)
        except IndexError:
            pass
        try:
            if mask[xpxl2, ypxl2+1]:
                pixel_list.append((xpxl2, ypxl2+1))
                darkness_list.append(xgap * (yend % 1) * multiplier)
        except IndexError:
            pass

    # main loop
    for x in range(int(xpxl1 + 1), int(xpxl2)):
        if steep:
            try:
                if mask[int(intery), x]:
                    pixel_list.append((int(intery), x))
                    darkness_list.append((1 - (intery % 1)) * multiplier)
            except IndexError:
                pass
            try:
                if mask[int(intery)+1, x]:
                    pixel_list.append((int(intery)+1, x))
                    darkness_list.append((intery % 1) * multiplier)
            except IndexError:
                pass
        else:
            try:
                if mask[x, int(intery)]:
                    pixel_list.append((x, int(intery)))
                    darkness_list.append((1 - (intery % 1)) * multiplier)
            except IndexError:
                pass
            try:
                if mask[x, int(intery)+1]:
                    pixel_list.append((x, int(intery)+1))
                    darkness_list.append((intery % 1) * multiplier)
            except IndexError:
                pass
        intery += gradient
    return np.array(pixel_list), np.array(darkness_list)