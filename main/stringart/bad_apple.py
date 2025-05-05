import cv2
import os
import numpy as np
from stringart.preprocessing.image import Anchor
from tqdm import tqdm
from pathlib import Path
import pickle
import re

from stringart.core.stringimage import StringImage
from main.stringart.core.lines import draw_line

def save_frames(video_path, target_fps, output_folder, width=None, height=None):
    # Make sure the output folder exists
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Capture video
    video = cv2.VideoCapture(video_path)
    if not video.isOpened():
        print("Error: Could not open video.")
        return

    # Get original FPS of the video
    fps = video.get(cv2.CAP_PROP_FPS)
    frame_interval = int(round(fps / target_fps))

    frame_count = 0
    extracted_count = 0

    while True:
        success, frame = video.read()
        if not success:
            break

        # Resize the frame if width and height are provided
        if width is not None and height is not None:
            frame = cv2.resize(frame, (width, height))

        # Check if this frame needs to be saved
        if frame_count % frame_interval == 0:
            frame_file = os.path.join(output_folder, f"frame_{extracted_count}.jpg")
            cv2.imwrite(frame_file, frame)
            extracted_count += 1

        frame_count += 1

    video.release()
    print(f"Extracted {extracted_count} frames to {output_folder}")

def images_to_video(image_folder, output_video_file, fps):
    # List and sort the images based on the frame number
    images = [img for img in os.listdir(image_folder) if img.endswith(".jpg") or img.endswith(".png")]
    images.sort(key=lambda x: int(re.findall("frame_(\d+)", x)[0]))

    # Determine the width and height from the first image
    frame = cv2.imread(os.path.join(image_folder, images[0]))
    height, width, layers = frame.shape

    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 'mp4v' or 'x264' might also work
    video = cv2.VideoWriter(output_video_file, fourcc, fps, (width, height))

    for image in images:
        video.write(cv2.imread(os.path.join(image_folder, image)))

    video.release()

def create_anchors_rectangle(img: np.ndarray, num_anchors: int):
    height, width = img.shape[1], img.shape[0]
    perimeter = 2 * (width + height)

    # Calculate the distance between anchors based on the perimeter
    distance_between_anchors = perimeter / num_anchors

    anchors = []
    current_distance = 0

    # Loop through the entire perimeter and add anchors
    for _ in range(num_anchors):
        # Top edge
        if current_distance < width:
            x = int(current_distance)
            y = 0
        # Right edge
        elif current_distance < width + height:
            x = width - 1
            y = int(current_distance - width)
        # Bottom edge
        elif current_distance < width * 2 + height:
            x = int(width - 1 - (current_distance - width - height))
            y = height - 1
        # Left edge
        else:
            x = 0
            y = int(height - 1 - (current_distance - width * 2 - height))

        anchors.append(Anchor(None, (x, y)))
        current_distance += distance_between_anchors

    return anchors


def make_line_dict_rectangle(data_folder:str, string_art_img: StringImage):
    """
    Makes a dictionary of every pixel and its darkness value for each line for every possible combination of anchors

    Args:
        data_folder: The path to the folder where the line dictionary is saved to/loaded from. This makes quickly iterating over different line dictionaries faster, as you don't need to remate the entire dictionary every time.
        anchors: The list of anchors around the circle
        shape: the shape of the base image
        mask: The circular mask applied to all relevant images
        closest_neighbors: Creating a string between two anchors very close to one another doesn't do much, so we bother to generate or check for them
        line_darkness: The maximum darkness for each anti-aliased line

        Returns:
            line_pixel_dict: A ditionary of line pixels for each anchor combination
            line_darkness_dict: A dictionary of line darkness values for each anchor combenation, cooresponds to line_pixel_dict
    """
    anchors = string_art_img.anchors
    shape = string_art_img.img.shape
    line_darkness = string_art_img.line_darkness
    mask = string_art_img.mask

    height, width = string_art_img.img.shape[1], string_art_img.img.shape[0]
    
    #Create a new line darkness dict or load an existing on if it already exists
    pkl_path = f"{data_folder}/line_dicts/{shape[0]}x{shape[1]}-{len(anchors)}.pkl"
    if Path(pkl_path).exists():
        tqdm.write("Opening existing line dictionary")
        with open(pkl_path, "rb") as file:
            line_dicts = pickle.load(file)
            return line_dicts["line_pixel_dict"], line_dicts["line_darkness_dict"]
    else:
        tqdm.write("Creating new line dictionary")
        line_pixel_dict = {}
        line_darkness_dict = {}
        for start_index in tqdm(range(len(anchors)), desc="Creating Lines"):
            for end_index in range(len(anchors)):
                if start_index == end_index: continue
                if are_on_same_edge(anchors[start_index], anchors[end_index], width, height): continue
                both_anchors = tuple(sorted((start_index, end_index))) #Sorts the indices for the lines, this prevents the same lines being made for two anchors in reverse order
                if both_anchors not in line_pixel_dict: #Makes sure that the anchors aren't already in the dictionary, only in a different order. This makes the number of lines needed n choose 2.
                    pixel_list, darkness_list = draw_line(p0=anchors[both_anchors[0]].coordinates, p1=anchors[both_anchors[1]].coordinates, multiplier=line_darkness, mask=mask)
                    if len(pixel_list) == 0: continue
                    line_pixel_dict[both_anchors], line_darkness_dict[both_anchors] = pixel_list, darkness_list
        tqdm.write("Saving new line dictionary")

        #Saves it for future use
        with open(pkl_path, 'wb') as file:
            pickle.dump({"line_pixel_dict": line_pixel_dict, "line_darkness_dict": line_darkness_dict}, file)
        return line_pixel_dict, line_darkness_dict
    

def are_on_same_edge(anchor1, anchor2, img_width, img_height):
    x1, y1 = anchor1.coordinates
    x2, y2 = anchor2.coordinates

    # Top edge: y coordinate is 0
    if y1 == 0 and y2 == 0 and x1 != x2:
        return True

    # Bottom edge: y coordinate is img_height - 1
    elif y1 == img_height - 1 and y2 == img_height - 1 and x1 != x2:
        return True

    # Left edge: x coordinate is 0
    elif x1 == 0 and x2 == 0 and y1 != y2:
        return True

    # Right edge: x coordinate is img_width - 1
    elif x1 == img_width - 1 and x2 == img_width - 1 and y1 != y2:
        return True

    return False
