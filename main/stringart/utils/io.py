import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import time
from tqdm import tqdm
from math import comb
import os
import pickle

def save_image(image, directory, name):
            image_data = (1-np.transpose(image))*255
            image = Image.fromarray(image_data.astype(np.uint8))
            image.save(f"{directory}/{name}")

    
def save_instructions(string_path: str, directory: str):
    instructions_list = ["0", "I"]
    for anchors in string_path:
        instructions_list.append(str(anchors[0]))
        instructions_list.append("O") #move servo out
        instructions_list.append(str(anchors[1]))
        instructions_list.append("I") #move servo in

    with open(directory, "w") as file:
        file.write('\n'.join(instructions_list))