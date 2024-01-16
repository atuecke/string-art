import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import time
from tqdm import tqdm
from math import comb
import os
import json
from pathlib import Path
import pickle
import cv2
from enum import Enum
from skimage.metrics import structural_similarity as ssim
import math
import multiprocessing
from itertools import product
import random
import scipy.stats as stats
import cupy as cp

class BaseImage():
    """
    The origonal image to create the string art off of

    Attributes:
        path: The path to the origonal image
        color_img: The resized and copped image
        img: The reseized, greyscaled, and cropped image
    """
    def __init__(
            self,
            path: str = None,
            img: np.ndarray = None,
            resize_to: int = None
    ) -> None:
        """
        Args:
            path: The path to the image
            img: If an image is already loaded, you can diretly set the image instead of path
            resize_to: Resize the image to scale of (x,x)
        """
        if(path):
            img = np.array(Image.open(path))
            # Transpose only the first two dimensions to switch x and y coordinates
            img = img.transpose(1, 0, 2)
        
        if(resize_to):
            img = resize_img(img=img, radius=resize_to)

        self.color_img = img

        img = make_greyscale(img=img)

        self.img = img
        self.path = path

class StringArtImage():
    """
    The string art image created from the base image

    Attributes:
        img: The rendered version of the string art imgage
        anchors: The list of anchors around the image
        string_path: A list of anchor sets that the string takes
    """
    def __init__(
            self,
            base_image: BaseImage,
            anchors: list,
            line_darkness: float,
            mask: np.ndarray
    ) -> None:
        """
        Args:
            base_img: The origonal image after preprocessing to be used as the scale
            anchors: The list of anchors around the image
        """
        self.img = np.zeros(base_image.img.shape)
        self.anchors = anchors
        self.string_path = []
        self.similarities = []
        self.line_darkness = line_darkness
        self.mask = mask
        self.loss_list = []

class Anchor():
    """
    An individual anchor placed around the string art
    Attributes:
        angle: The angle of the anchor from 0 to 2pi (think unit circle)
        coordinates: The coordinates on the string art image where the anchor is placed
    """
    def __init__(
            self,
            angle: float = None,
            coordinates: tuple = None
    ) -> None:
        """
        Args:
            angle: The angle of the anchor from 0 to 2pi (think unit circle)
            coordinates: The coordinates on the string art image where the anchor is placed
        """
        self.angle = angle
        self.coordinates = coordinates


class ImportanceMap():
    """
    A map of importance of different pixels across the string art, higher value means more importance

    Attributes:
        img: The map image
        blur: The (optional) gaussian blur applied to the importance map
    """
    def __init__(
            self,
            img: np.ndarray = None
    ) -> None:
        """
        Args:
            map: The map image
        """
        self.img = img
        self.blur: tuple = None

    def apply_gaussian_blur(self, blur_size: int):
        """
        Applies a gaussian blur to the importance map
        Args:
            blur_size: The diameter (in pixels) of the gaussian blur, must be an odd integer
        """
        self.img = cv2.GaussianBlur(self.img, (blur_size, blur_size), 0)
        self.blur = blur_size

    def apply_dynamic_sigmoid(self, exponent=1, std_exponent=1):
        """
        Applies a dynamic sigmoid function to the importance map, normalizing the map.
        """
        sigma = np.std(self.img)
        if sigma == 0: sigma = 1
        transformed = 1 / (1 + np.exp(-self.img/(sigma**std_exponent)))**exponent
        self.img = transformed

class StringLine():
    def __init__(
            self,
            base_image,
            importance_values,
            string_pixels,
            string_darkness
    ) -> None:
        """
        """
        self.base_image = base_image
        self.importance_values = importance_values
        self.string_pixels = string_pixels
        self.string_darkness = string_darkness

class CostMethods(Enum):
    DIFFERENCE = 1
    MEAN = 2
    SMSE = 3
    SMRSE = 4
    MEDIAN = 5


def make_line_dict(data_folder:str, string_art_img: StringArtImage, closest_neighbors: int = 10):
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
    
    #Create a new line darkness dict or load an existing on if it already exists
    pkl_path = f"{data_folder}/line_dicts/{shape[0]}x{shape[1]}-{len(anchors)}.pkl"
    if Path(pkl_path).exists():
        print("Opening existing line dictionary")
        with open(pkl_path, "rb") as file:
            line_dicts = pickle.load(file)
            print("Done!")
            return line_dicts["line_pixel_dict"], line_dicts["line_darkness_dict"]
    else:
        print("Creating new line dictionary")
        line_pixel_dict = {}
        line_darkness_dict = {}
        for start_index in tqdm(range(anchors), desc="Creating Lines"):
            for end_index in range(anchors):
                if is_within_range(anchors, start_index, end_index): continue #Adding lines to anchors close to one another doesn't realistically add much to the image, so we don't generate or consider them
                both_anchors = tuple(sorted((start_index, end_index))) #Sorts the indices for the lines, this prevents the same lines being made for two anchors in reverse order
                if both_anchors not in line_pixel_dict: #Makes sure that the anchors aren't already in the dictionary, only in a different order. This makes the number of lines needed n choose 2.
                    pixel_list, darkness_list = draw_line(p0=anchors[both_anchors[0]].coordinates, p1=anchors[both_anchors[1]].coordinates, multiplier=line_darkness, mask=mask)
                    line_pixel_dict[both_anchors], line_darkness_dict[both_anchors] = pixel_list, darkness_list
        print("Saving new line dictionary")

        #Saves it for future use
        with open(pkl_path, 'wb') as file:
            pickle.dump({"line_pixel_dict": line_pixel_dict, "line_darkness_dict": line_darkness_dict}, file)
        print("Done!")
        return line_pixel_dict, line_darkness_dict

def difference(a, b):
        """
        Calculates the absolute difference between two pixel values
        Args:
            a: The first value/values
            b: The second value/values
        """
        return np.abs(a - b)

def create_string_art(first_anchor: int, base_img: np.ndarray, string_art_img: StringArtImage, line_pixel_dict: dict, line_darkness_dict: dict, iterations: int, loss_method: str = "mean_error", max_darkness: float = None, eval_interval: int = None, importance_map: ImportanceMap = None):
    """
    Creates the completed string art

    Args:
        first_anchor: The starting anchor
        base_img: The origonal image, used for finding the best line
        string_art_img: The string art image that is supposed to be made TODO: Create the string art image in in the function
        line_pixel_dict: The ditionary of line pixels for each anchor combination
        line_darkness_dict: The dictionary of line darkness values for each anchor combenation, cooresponds to line_pixel_dict
        iterations: The number of strings drawn. Range vaires with anchor count and image size
        loss_method: The method that the function uses to determine the loss of a line
        max_darkness: The maximum darkness of a line in the string art image. If set to none, there is no limit
        eval_interval: Evalutate the entire image on this interval, saved in the returned string_art_img
        importance_map: The image map of importance values for each pixel

    Returns:
        string_art_img: The final string art image
        difference_img: The difference image used in this creation
    """
    anchor_line_idx = {}
    lines = []
    if not importance_map:
        importance_map = ImportanceMap(img=np.ones_like(base_img))
    
    print("Building line arrays")
    for idx, anchors in enumerate(line_pixel_dict.keys()):
        pixels = line_pixel_dict[anchors]
        x_coords = pixels[:, 0]
        y_coords = pixels[:, 1]

        anchor_line_idx[anchors] = idx
        lines.append(StringLine(
            base_image=base_img[x_coords, y_coords],
            importance_values=importance_map.img[x_coords, y_coords],
            string_pixels=pixels,
            string_darkness=line_darkness_dict[anchors]
        ))
    print("Done!")
   
    if not importance_map:
        importance_map = np.ones(string_art_img.img.shape)
    importance_map = importance_map.img
                    
    def find_best_line(previous_anchor_idx: int):
        """
        Starts at a specified anchor and find the loss for every string leading to every other anchor, updating the best loss whenever a better one is found

        Args:
            previous_anchor_idx: The previous anchor, which was the end anchor for the last line

        Returns:
            best_anchors: The tuple of the two best anchors
            best_loss: The best line loss found
        """
        best_loss = np.inf #TODO set this to the starting loss and make the algorithm terminate when there isn't a possible improvement
        best_anchors = None
        best_end_anchor = None
        best_start_anchor = None
        for start_anchor_idx in get_neighbors(string_art_img.anchors, previous_anchor_idx): #Finds the neighbors of the start anchor
            for end_anchor_idx in range(len(string_art_img.anchors)):
                both_anchors = tuple(sorted((start_anchor_idx, end_anchor_idx))) #Makes sure to get the right order for the indices, set in make_line_dict().
                if both_anchors not in line_pixel_dict: continue
                line_idx = anchor_line_idx[both_anchors]
                temp_loss = find_loss(line_idx)
                if(temp_loss < best_loss): #Updates best loss if temp loss is better
                    best_loss = temp_loss
                    best_anchors = both_anchors
                    best_end_anchor = end_anchor_idx
                    best_start_anchor = start_anchor_idx
        return best_anchors, best_loss, best_start_anchor, best_end_anchor
    
    def find_loss(line_idx):
        """
        """
        line: StringLine = lines[line_idx]
        
        x_coords = line.string_pixels[:, 0]
        y_coords = line.string_pixels[:, 1]

        string_art_values = string_art_img.img[x_coords, y_coords]
        total_darkness_values = string_art_values + line.string_darkness
        match loss_method:
            case "differences":
                diff_values = difference(line.base_image, string_art_values)
                weighted_difference = line.importance_values * (difference(line.base_image, total_darkness_values) - diff_values)
                loss = np.sum(weighted_difference)/len(x_coords)
            case "mean": 
                weighted_difference = (total_darkness_values - line.base_image)*line.importance_values
                loss = np.sum(weighted_difference)/len(x_coords)
            case "SMSE": #signed mean squared error
                weighted_difference = (total_darkness_values - line.base_image)*line.importance_values
                signed_squared_difference = np.sign(weighted_difference) * weighted_difference**2
                loss = np.sum(signed_squared_difference)/len(x_coords)
            case "SRMSE": #signed root mean squared error THIS GIVES SAME RESULT AS MEAN ERROR AND IDK WHY
                weighted_difference = (total_darkness_values - line.base_image)*line.importance_values
                signed_squared_difference = np.sign(weighted_difference) * (weighted_difference**2)
                mean_difference = np.sum(signed_squared_difference)/len(x_coords)
                loss = np.sqrt(abs(mean_difference))*np.sign(mean_difference)
            case "median":
                weighted_difference = (total_darkness_values - line.base_image)*line.importance_values
                loss = np.median(weighted_difference)
        return loss

    previous_anchor_idx = first_anchor
    for iter in tqdm(range(iterations)):
        best_anchors, best_loss, best_start_anchor, best_end_anchor = find_best_line(previous_anchor_idx=previous_anchor_idx)

        #Load the best line and add it to the string art image
        best_string_pixels, best_string_darkness_values = line_pixel_dict[best_anchors], line_darkness_dict[best_anchors]
        x_coords = best_string_pixels[:, 0]
        y_coords = best_string_pixels[:, 1]
        
        if max_darkness:
            new_values = np.clip(string_art_img.img[x_coords, y_coords] + best_string_darkness_values, 0, max_darkness)
            string_art_img.img[x_coords, y_coords] = new_values
        else:
            new_values = string_art_img.img[x_coords, y_coords] + best_string_darkness_values
            string_art_img.img[x_coords, y_coords] = new_values
        string_art_img.string_path.append((best_start_anchor, best_end_anchor))
        previous_anchor_idx = best_end_anchor
        string_art_img.loss_list.append(best_loss)

        #Evaluate the string art
        if eval_interval:
            if iter%eval_interval == 0:
                similarity = compare_images(string_art_img.img, base_img, method="ssim")
                string_art_img.similarities.append((iter, similarity))
    
    return string_art_img

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
    
def save_instructions(string_path: str, path: str):
    instructions_list = ["0", "L"]
    for anchors in string_path:
        instructions_list.append(str(anchors[0]))
        instructions_list.append("R")
        instructions_list.append(str(anchors[1]))
        instructions_list.append("L")
    
    instructions_list = ",".join(instructions_list)

    with open(path, "w") as file:
        file.write(instructions_list)

def draw_line(p0: tuple, p1: tuple, multiplier: float, mask: np.ndarray):
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


def resize_img(img: np.ndarray, radius: int):
    if len(img.shape) == 2:
        width, height = img.shape
    else:
        width, height, _ = img.shape
    center = (int(width / 2), int(height / 2))
    min_side_legth = min(center[0], center[1])

    if len(img.shape) == 2:
        img = img[center[0]-min_side_legth:center[0]+min_side_legth, center[1]-min_side_legth:center[1]+min_side_legth]
    else:
        img = img[center[0]-min_side_legth:center[0]+min_side_legth, center[1]-min_side_legth:center[1]+min_side_legth, :]

    img = Image.fromarray(img.astype(np.uint8))
    img = img.resize((radius, radius), Image.LANCZOS)
    img = np.array(img)
    
    return img

def make_greyscale(img: np.ndarray):
    img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # Normalize the image to [0, 1] and invert so that 1 is dark and 0 is bright
    img = 1 - np.array(img) / 255.0
    return img

def create_mask(img: np.ndarray):
    width, height = img.shape
    center = (int(width / 2), int(height / 2))
    radius = min(img.shape)/2

    # Create the circular mask
    x, y = np.ogrid[-center[0]:width-center[0], -center[1]:height-center[1]]
    print(x)
    mask = x*x + y*y <= radius*radius

    return mask

def apply_mask(img, mask):
    return np.multiply(img, mask)

def create_anchors(img: np.ndarray, num_anchors: int):
    width, height = img.shape
    center = (int(width / 2), int(height / 2))
    radius = min(img.shape)/2
    """Creates a list of tuples, each being the cordinates of an anchor
    Args:
        center: The center of the image
        radius: The radius of the image (1/2 of the width)
        num_anchors: The number of the anchors around the image
    """
    # Calculate the coordinates of the anchor points
    angles = np.linspace(0, 2*np.pi, num_anchors, endpoint=False)
    anchor_x = np.round(center[0] + (radius - 1) * np.cos(angles)).astype(int)
    anchor_y = np.round(center[1] + (radius - 1) * np.sin(angles)).astype(int)
    #anchors = list(zip(anchor_x, anchor_y))
    anchors = []
    for i, angle in enumerate(angles):
        anchor = Anchor(angle=angle, coordinates=(anchor_x[i], anchor_y[i]))
        anchors.append(anchor)

    return anchors


def get_neighbors(arr: list, idx):
    # Calculate indices for neighbors with wrapping
    left_idx = (idx - 1) % len(arr)
    right_idx = (idx + 1) % len(arr)

    return left_idx, right_idx

def compare_images(input_img, target_img, method):
    """
    """
    if method == "ssim":
        similarity = ssim(input_img, target_img, data_range=1)
    elif method == "psnr":
        mse = np.mean((input_img - target_img) ** 2)
        if mse == 0:
            similarity = float('inf')
        max_pixel = 255.0
        psnr = 20 * math.log10(max_pixel / math.sqrt(mse))
        similarity = psnr

    
    return similarity

def open_importance_maps(folder_path: str, string_art_img_shape: tuple):
    importance_map_list = []
    files = os.listdir(folder_path)
    
    for file in files:
        file_path = os.path.join(folder_path, file)

        # Check if the file is an image (e.g., jpg, png, etc.)
        if file_path.lower().endswith(('.png', '.jpg', '.jpeg')):
            img = np.array(Image.open(file_path))
            # Transpose only the first two dimensions to switch x and y coordinates
            img = img.transpose(1, 0, 2)
            img = resize_img(img=img, radius=string_art_img_shape[0])
            img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            img = img / 255
            importance_map = ImportanceMap(img=img)
            importance_map_list.append(importance_map)

    return importance_map_list

def outline_importance_map(importance_map: ImportanceMap, edge_thickness = 3):
    """
    """
    edges = detect_edges_color(img_array=importance_map.img, gaussian_blur_size=1, dilate_iterations=edge_thickness)
    edge_importance_map = ImportanceMap(img=edges)
    return edge_importance_map

def background_importance_map(importance_map: ImportanceMap, cutoff: 0):
    img = np.where(importance_map.img <= cutoff, 1, 0)
    background_map = ImportanceMap(img=img)
    return background_map

def combine_importance_maps(all_maps: list):
    shape = all_maps[0].img.shape
    sum_maps = np.zeros(shape)
    for map in all_maps:
        sum_maps += map.img
    
    overall_map = ImportanceMap(img=sum_maps)
    return overall_map

    
def detect_edges_grayscale(img_array, low_threshold=50, high_threshold=150, gaussian_blur_size=5, dilate_iterations=1):
    """
    Detects edges in a grayscale image using the Canny edge detector.
    
    Parameters:
        - img_array: 2D numpy array representing the grayscale image
        - low_threshold, high_threshold: thresholds for the Canny edge detector. Edges with intensity gradient more than 'high_threshold' 
          are sure to be edges and those below 'low_threshold' are sure to be non-edges.
        - gaussian_blur_size: kernel size for Gaussian blur pre-processing.
        - dilate_iterations: number of dilation iterations to make the edges thicker.

    Returns:
        normalized_edges: 2D numpy array representing the normalized edges in the image.
    """
    img_array = img_array*255
    # Ensure the image is a grayscale image
    if len(img_array.shape) != 2:
        raise ValueError("The input image must be a grayscale image")
    
    # Apply Gaussian blur
    img_blurred = cv2.GaussianBlur(img_array, (gaussian_blur_size, gaussian_blur_size), 0)
    
    # Detect edges using Canny
    edges = cv2.Canny(img_blurred, low_threshold, high_threshold)
    
    # Dilation makes edges thicker
    if dilate_iterations > 0:
        kernel = np.ones((3,3), np.uint8)
        edges = cv2.dilate(edges, kernel, iterations=dilate_iterations)
    
    # Normalize the edge image
    normalized_edges = edges / 255.0

    return normalized_edges


def detect_edges_color(img_array, low_threshold=50, high_threshold=150, gaussian_blur_size=5, dilate_iterations=1):
    """
    Detects edges in an image using the Canny edge detector, but processes RGB images for better precision.
    
    Parameters:
        - img_array: 3D numpy array representing the image
        - low_threshold, high_threshold: thresholds for the Canny edge detector. Edges with intensity gradient more than 'high_threshold' 
          are sure to be edges and those below 'low_threshold' are sure to be non-edges.
        - gaussian_blur_size: kernel size for Gaussian blur pre-processing.
        - dilate_iterations: number of dilation iterations to make the edges thicker.

    Returns:
        edges_combined: 2D numpy array representing the edges in the image.
    """
    
    # Ensure the image is in RGB format
    if len(img_array.shape) == 2:
        img_array = np.interp(img_array, (img_array.min(), img_array.max()), (0, 255)).astype(np.uint8)
        img_array = cv2.cvtColor(img_array, cv2.COLOR_GRAY2RGB)
    
    # Apply Gaussian blur
    img_blurred = cv2.GaussianBlur(img_array, (gaussian_blur_size, gaussian_blur_size), 0)
    
    # Detect edges using Canny for each channel and combine
    edges_r = cv2.Canny(img_blurred[:,:,0], low_threshold, high_threshold)
    edges_g = cv2.Canny(img_blurred[:,:,1], low_threshold, high_threshold)
    edges_b = cv2.Canny(img_blurred[:,:,2], low_threshold, high_threshold)
    edges_combined = edges_r | edges_g | edges_b
    
    # Dilation makes edges thicker
    if dilate_iterations > 0:
        kernel = np.ones((3,3), np.uint8)
        edges_combined = cv2.dilate(edges_combined, kernel, iterations=dilate_iterations)
    
    normalized_edges = edges_combined / 255.0

    return normalized_edges