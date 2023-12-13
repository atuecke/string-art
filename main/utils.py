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
        map: The map image
        blur: The (optional) gaussian blur applied to the importance map
    """
    def __init__(
            self,
            map: np.ndarray = None
    ) -> None:
        """
        Args:
            map: The map image
        """
        self.map = map
        self.blur: tuple = None

    def apply_gaussian_blur(self, blur_size: int):
        """
        Applies a gaussian blur to the importance map
        Args:
            blur_size: The diameter (in pixels) of the gaussian blur, must be an odd integer.
        """
        self.map = cv2.GaussianBlur(self.img, (blur_size, blur_size), 0)
        self.blur = blur_size

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
    pkl_path = f"{data_folder}/line_dicts/{shape[0]}-{shape[1]}-{len(anchors)}.pkl"
    if Path(pkl_path).exists():
        print("Opening existing line dictionary")
        with open(pkl_path, "rb") as file:
            line_dicts = pickle.load(file)
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
        print("Done!")

        #Saves it for future use
        with open(pkl_path, 'wb') as file:
            pickle.dump({"line_pixel_dict": line_pixel_dict, "line_darkness_dict": line_darkness_dict}, file)
        return line_pixel_dict, line_darkness_dict

def difference(a, b):
        """
        Calculate the absolute difference between two pixel values

        Args:
            a: The first value/values
            b: The second value/values
        """
        return np.abs(a - b)

def create_string_art(first_anchor: int, base_img: np.ndarray, string_art_img: StringArtImage, line_pixel_dict: dict, line_darkness_dict: dict, iterations: int, loss_method: str = "difference_img", max_darkness: float = None, eval_interval: int = 100):
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

    Returns:
        string_art_img: The final string art image
        difference_img: The difference image used in this creation
    """
    difference_img = np.zeros_like(base_img)
                    
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
        for start_anchor_idx in get_neighbors(string_art_img.anchors, previous_anchor_idx): #Finds the neighbors of the start anchor
            for end_anchor_idx in range(len(string_art_img.anchors)):
                both_anchors = tuple(sorted((start_anchor_idx, end_anchor_idx))) #Makes sure to get the right order for the indices, set in make_line_dict().
                if both_anchors not in line_pixel_dict: continue
                temp_loss = find_loss(line_pixel_dict[both_anchors], line_darkness_dict[both_anchors]) #Finds the loss of that string
                if(temp_loss < best_loss): #Updates best loss if temp loss is better
                    best_loss = temp_loss
                    best_anchors = both_anchors
        return best_anchors, best_loss
    
    def find_loss(string_pixels: list, string_darkness_values: list):
        """
        Finds the loss for one string

        Args:
            string_pixels: The list of the string pixel coordinates
            string_darkness_values: The list of the darkness values for the string

        Returns:
            loss: The loss value of that string
        """

        x_coords = string_pixels[:, 0]
        y_coords = string_pixels[:, 1]

        target_darkness_values = base_img[x_coords, y_coords]
        total_darkness_values = string_art_img.img[x_coords, y_coords] + string_darkness_values

        #Use the loss method set in parameters
        match loss_method:
            case "difference_img":
                diff_values = difference_img[x_coords, y_coords]
                weighted_difference = difference(target_darkness_values, total_darkness_values) - diff_values
                loss = np.sum(weighted_difference)/len(string_pixels)
            case "euclidean_distance":
                loss = np.linalg.norm(target_darkness_values - total_darkness_values)/len(string_pixels)
            case "MAE:":
                loss = np.mean(np.abs(target_darkness_values - total_darkness_values))
            case "MSE":
                loss = np.mean((target_darkness_values - total_darkness_values) ** 2)
            case "RMSE":
                loss = np.sqrt(np.mean((target_darkness_values - total_darkness_values) ** 2))
            case "MSLE":
                loss = np.mean((np.log1p(target_darkness_values) - np.log1p(total_darkness_values)) ** 2)
            case "cosine_proximity":
                loss = -np.dot(target_darkness_values, total_darkness_values) / (np.linalg.norm(target_darkness_values) * np.linalg.norm(total_darkness_values))/len(string_pixels)
        
        return loss
    
    #If the loss method is difference img, create the difference image
    if loss_method == "difference_img":
        difference_img[string_art_img.mask] = difference(0, base_img[string_art_img.mask])
    
    
    previous_anchor_idx = first_anchor
    for iter in tqdm(range(iterations)):
        best_anchors, best_loss = find_best_line(previous_anchor_idx=previous_anchor_idx)

        best_string_pixels, best_string_darkness_values = line_pixel_dict[best_anchors], line_darkness_dict[best_anchors]
        x_coords = best_string_pixels[:, 0]
        y_coords = best_string_pixels[:, 1]
        if max_darkness:
            new_values = np.clip(string_art_img.img[x_coords, y_coords] + best_string_darkness_values, 0, max_darkness)
            difference_img[x_coords, y_coords] = difference(base_img[x_coords, y_coords], new_values)
            string_art_img.img[x_coords, y_coords] = new_values
        else:
            new_values = string_art_img.img[x_coords, y_coords] + best_string_darkness_values
            difference_img[x_coords, y_coords] = difference(base_img[x_coords, y_coords], new_values)
            string_art_img.img[x_coords, y_coords] = new_values
        previous_anchor_idx = best_anchors[0]
        string_art_img.string_path.append(best_anchors)
        string_art_img.loss_list.append(best_loss)
        if eval_interval:
            if iter%eval_interval == 0:
                similarity = compare_images(string_art_img.img, base_img, method="ssim")
                string_art_img.similarities.append((iter, similarity))
    
    return string_art_img, difference_img

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