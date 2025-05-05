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

PATH_TO_FRAMES = "./data/bad_apple/frames"
CONFIG_PATH = "./data/bad_apple/config.json"
NUM_ANCHORS = 300
NUM_LINES_CONSTANT = 8000
NUM_LINES_MIN = {
    "black": 600,
    "white": 850
}

def run_range(folder_path: str, config_path: str, start: int, end: int):

    parser = create_parser()
    args = parser.parse_args()
    start = args.start
    end = args.end

    with open(config_path) as config_file:
        config = json.load(config_file)

    init_base_img = BaseImage(path=f"{folder_path}/frame_{start}.jpg")
    mask = np.ones_like(init_base_img.img)
    anchors = create_anchors_rectangle(img=init_base_img.img, num_anchors=NUM_ANCHORS)
    init_string_art_img = StringImage(base_image=init_base_img, anchors=anchors, line_darkness=0.2, mask=mask)

    line_pixel_dict, line_darkness_dict = make_line_dict_rectangle(
        data_folder="./data/bad_apple/",
        string_art_img=init_string_art_img
    )

    for idx in tqdm(range(start, end+1), desc=f"Creating frames"):
        img_path, focus_color = process_index(index=idx, config=config)
        string_frame = create_frame(img_path=img_path, focus_color=focus_color, mask=mask, anchors=anchors, line_pixel_dict=line_pixel_dict, line_darkness_dict=line_darkness_dict)
        if focus_color == "white": string_frame.img = 1-string_frame.img
        save_image(image=string_frame.img, directory="./outputs/bad_apple/frames", name=f"frame_{idx}.jpg")


def create_frame(img_path:str, focus_color: str, mask: np.ndarray, anchors: list, line_pixel_dict: dict, line_darkness_dict: dict):
    ""
    base_img = BaseImage(path=img_path)
    base_img.img = apply_mask(base_img.img, mask)
    mask = np.ones_like(base_img)
    anchors = create_anchors_rectangle(img=base_img.img, num_anchors=NUM_ANCHORS)


    if focus_color == "white": base_img.img = 1-base_img.img

    black_ratio = np.sum(base_img.img >= 0.5)/base_img.img.size
    num_lines = int(black_ratio*NUM_LINES_CONSTANT) + NUM_LINES_MIN[focus_color]
    #print(black_ratio, num_lines)
    
    color_importance_map = above_threshold(img=base_img.img, threshold=0.1)
    outline_map = outline_importance_map(importance_map=color_importance_map, edge_thickness=3)
    background_map = background_importance_map(importance_map=color_importance_map, cutoff=0.1)

    color_importance_map.img *= (2)
    outline_map.img *= 2
    background_map.img *= (-1)

    all_importance_maps = [color_importance_map] + [outline_map] + [background_map]
    main_importance_map = combine_importance_maps(all_importance_maps)
    main_importance_map.apply_gaussian_blur(15)
    main_importance_map.apply_dynamic_sigmoid(exponent=1, std_exponent=1)

    base_img.img = np.clip(base_img.img + 0.2, 0, 1)

    string_art_img = StringImage(base_image=base_img, anchors=anchors, line_darkness=0.2, mask=mask)

    string_frame = create_string_art(
        first_anchor= random.randint(0, NUM_ANCHORS-1),
        base_img=base_img,
        string_art_img=string_art_img,
        line_pixel_dict=line_pixel_dict,
        line_darkness_dict=line_darkness_dict,
        iterations=num_lines,
        cost_method=CostMethod.MEAN,
        max_darkness=1,
        importance_map=main_importance_map,
        use_prev_anchor=False,
        random_neighbor=True
    )

    return string_frame

    

def process_index(index, config):
    img_path = f"{PATH_TO_FRAMES}/frame_{index}.jpg"
    for range_conf in config["ranges"]:
        if range_conf["start"] <= index <= range_conf["end"]:
            # Call the function based on the action name
            focus_color=range_conf["focus_color"]
            return img_path, focus_color
    
    print(f"no slice found for frame {index}")
    focus_color=range_conf["focus_color"]
    return img_path, focus_color

def create_parser():
    parser = argparse.ArgumentParser(description="Process some integers.")
    parser.add_argument('start', type=int, help='Index to start at')
    parser.add_argument('end', type=int, help='Index to end at')
    return parser


if __name__ == "__main__":
    run_range(folder_path=PATH_TO_FRAMES, config_path=CONFIG_PATH, start=0, end=3)