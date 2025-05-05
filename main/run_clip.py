import numpy as np
from tqdm import tqdm
import json
from pathlib import Path
import random
import argparse

from stringart.preprocessing.image import BaseImage, apply_mask
from stringart.preprocessing.importancemaps import outline_importance_map, combine_importance_maps, background_importance_map, below_threshold, above_threshold
from stringart.core.stringimage import StringImage
from stringart.algorithm.execute import create_string_art
from stringart.algorithm.costmethod import CostMethod
from stringart.utils.io import save_image
from stringart.bad_apple import create_anchors_rectangle, make_line_dict_rectangle

INPUT_FRAMES = "./data/case/frames"
OUTPUT_FRAMES = "./outputs/case/frames"
NUM_ANCHORS = 400
NUM_LINES = 3000

def run_range(folder_path: str, start: int, end: int):

    parser = create_parser()
    args = parser.parse_args()
    start = args.start
    end = args.end

    init_base_img = BaseImage(path=f"{folder_path}/frame_{start}.jpg")
    mask = np.ones_like(init_base_img.img)
    anchors = create_anchors_rectangle(img=init_base_img.img, num_anchors=NUM_ANCHORS)
    init_string_art_img = StringImage(base_image=init_base_img, anchors=anchors, line_darkness=0.2, mask=mask)

    line_pixel_dict, line_darkness_dict = make_line_dict_rectangle(
        data_folder="./data/case/",
        string_art_img=init_string_art_img
    )

    for idx in tqdm(range(start, end+1), desc=f"Creating frames"):
        img_path = f"{INPUT_FRAMES}/frame_{idx}.jpg"
        string_frame = create_frame(img_path=img_path, mask=mask, anchors=anchors, line_pixel_dict=line_pixel_dict, line_darkness_dict=line_darkness_dict)
        save_image(image=string_frame.img, directory=OUTPUT_FRAMES, name=f"frame_{idx}.jpg")


def create_frame(img_path:str, mask: np.ndarray, anchors: list, line_pixel_dict: dict, line_darkness_dict: dict):
    ""
    base_img = BaseImage(path=img_path)
    base_img.img = apply_mask(base_img.img, mask)
    mask = np.ones_like(base_img)
    anchors = create_anchors_rectangle(img=base_img.img, num_anchors=NUM_ANCHORS)

    string_art_img = StringImage(base_image=base_img, anchors=anchors, line_darkness=0.2, mask=mask)

    string_frame = create_string_art(
        first_anchor= random.randint(0, NUM_ANCHORS-1),
        base_img=base_img,
        string_art_img=string_art_img,
        line_pixel_dict=line_pixel_dict,
        line_darkness_dict=line_darkness_dict,
        iterations=NUM_LINES,
        cost_method=CostMethod.MEAN,
        max_darkness=1,
        use_prev_anchor=False,
        random_neighbor=True
    )

    return string_frame


def create_parser():
    parser = argparse.ArgumentParser(description="Process some integers.")
    parser.add_argument('start', type=int, help='Index to start at')
    parser.add_argument('end', type=int, help='Index to end at')
    return parser


if __name__ == "__main__":
    run_range(folder_path=INPUT_FRAMES, start=0, end=3)