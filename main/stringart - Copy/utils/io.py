import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import time
from tqdm import tqdm
from math import comb
import os
import pickle

from stringart.core.stringimage import StringArtImage

def save_string_art(string_art_img: StringArtImage, directory: str):
    # Check if directory exists; if not, create it
    if not os.path.exists(directory):
        os.makedirs(directory)

    save_image(string_art_img.img, directory, "string_art.jpg")

    with open(f"{directory}/config.pkl", 'wb') as file:
        pickle.dump(string_art_img, file)

def save_image(image, directory, name):
            image_data = (1-np.transpose(image))*255
            image = Image.fromarray(image_data.astype(np.uint8))
            image.save(f"{directory}/{name}")

def load_string_art(data_dir: str):
    with open(f"{data_dir}/config.pkl", "rb") as file:
        string_art_img = pickle.load(file)

    return string_art_img
    
def save_instructions(string_path: str, num_anchors: int, path: str):
    instructions_list = ["0", "I"]
    for anchors in string_path:
        instructions_list.append(str(anchors[0]))
        instructions_list.append("O") #move servo out
        instructions_list.append(str(anchors[1]))
        instructions_list.append("I") #move servo in
    
    instructions_list = ",".join(instructions_list)

    with open(path, "w") as file:
        file.write(instructions_list)