import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import time
from tqdm.notebook import tqdm
from math import comb
import math
import os
import json
from pathlib import Path
import cProfile

from stringart.preprocessing.image import BaseImage, create_mask, apply_mask, create_anchors
from stringart.preprocessing.importancemaps import open_importance_maps, outline_importance_map, combine_importance_maps, background_importance_map
from stringart.core.stringimage import StringImage
from stringart.preprocessing.linedicts import make_line_dict
from stringart.algorithm.execute import create_string_art
from stringart.utils.io import save_string_art, load_string_art, save_instructions
from stringart.algorithm.costmethod import CostMethod

THREAD_WIDTH = 0.2
PLATE_DIAMETER = 300
NUM_ANCHORS = 150
LINE_WIDTH = 1

IMG_SIZE = int(PLATE_DIAMETER/THREAD_WIDTH)
IMG_SIZE = 800
print(f"Image Size: {IMG_SIZE}")

base_img = BaseImage(path="./data/tom.jpg", resize_to=IMG_SIZE)
mask = create_mask(base_img.img)
base_img.img = apply_mask(base_img.img, mask)
anchors = create_anchors(img=base_img.img, num_anchors=NUM_ANCHORS)

string_art_img = StringImage(base_image=base_img, anchors=anchors, line_darkness=0.2, mask=mask)

importance_maps = open_importance_maps(folder_path="./data/importance_maps/tom/", string_art_img_shape=string_art_img.img.shape)
# TODO: make a way to increase contrast

outline_importance_maps = []
for map in importance_maps:
    outline_map = outline_importance_map(importance_map=map, edge_thickness=3)
    outline_importance_maps.append(outline_map)

background_map = background_importance_map(importance_map=importance_maps[2], cutoff=0.1)

for map in importance_maps:
    map.img *= 2
for map in outline_importance_maps:
    map.img *= 4

background_map.img *= (-1)

all_importance_maps = importance_maps + outline_importance_maps + [background_map]
main_importance_map = combine_importance_maps(all_importance_maps)
main_importance_map.apply_gaussian_blur(35)
main_importance_map.apply_dynamic_sigmoid(exponent=1, std_exponent=1)

line_pixel_dict, line_darkness_dict = make_line_dict(
    data_folder="./data",
    string_art_img=string_art_img,
    closest_neighbors=10,
)

def main():
    final_img = create_string_art(
        first_anchor=0,
        base_img=base_img,
        string_art_img=string_art_img,
        line_pixel_dict=line_pixel_dict,
        line_darkness_dict=line_darkness_dict,
        iterations=1000,
        cost_method=CostMethod.MEAN,
        max_darkness=1,
        importance_map=main_importance_map,
        use_prev_anchor=True,
        random_neighbor=False,
        profiling=True
    )

main()

# line to run:
# python -m cProfile -o profile.prof benchmark.py && snakeviz profile.prof