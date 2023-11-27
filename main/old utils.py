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
import dlib



class StringImage():
    def __init__(self, img: np.ndarray) -> None:
        self.img = img
        self.center: tuple
        self.radius: int
        self.width: int
        self.height: int
        
        self.width, self.height, self.center = self.get_width_height_center_radius()
    
    def update(self, new_img):
        self.img = new_img
        self.width, self.height, self.center, self.radius = self.get_width_height_center_radius()
    
    def get_width_height_center_radius(self):
        img = self.img
        if len(img.shape) == 2:
            height, width = img.shape
        else:
            height, width, _ = img.shape
        center = (int(height / 2), int(width / 2))
        return width, height, center
    
    def get_min_side_length(self):
        min_side_length = min(self.center[0], self.center[1])
        return min_side_length
    
    def crop(self, radius):
        center = self.center
        img = self.img
        if len(img.shape) == 2:
            self.img = img[center[0]-radius:center[0]+radius, center[1]-radius:center[1]+radius]
        else:
            self.img = img[center[0]-radius:center[0]+radius, center[1]-radius:center[1]+radius, :]

    def resize(self, img_size: int):
        img = resize_img(self.img, img_size)
        self.update(img)


class Anchor():
    def __init__(self, number: int, angle: float, pixels: tuple) -> None:
        self.num = number
        self.ang = angle
        self.pixels = pixels

class ImportanceMap(StringImage):
    def __init__(self, img: np.ndarray, importance_value: int = 1, gaussian_blur = 0) -> None:
        super().__init__(img = img)
        self.importance = importance_value
    
    def apply_gaussian_blur(self, blur_size: int):
        self.img = cv2.GaussianBlur(self.img, (blur_size, blur_size), 0)

    def normalize(self, max):
        self.img = dynamic_sigmoid(image=self.img, max=max)




class StringArt():
    def __init__(self, image_path: str, num_anchors: int, line_darkness: float, num_lines: int, ) -> None:
        self.image_path = image_path
        self.num_anchors = num_anchors
        self.line_darkness = line_darkness
        self.num_lines = num_lines
        self.img: np.ndarray = None
        self.img_color: np.ndarray = None
        self.mask: np.ndarray = None
        self.center: tuple = None
        self.radius: int = None
        self.anchors: list = None
        self.line_pixel_dict: dict = None
        self.line_darkness_dict: dict = None
        self.difference_img: np.ndarray = None
        self.string_art_img: np.ndarray = None
        self.importance_map: np.ndarray = None
        self.anchor_list: list = []
        self.img_size: int = None
        self.importance_map_multiplier = 10
        self.data_folder = "data"
        self.closest_neighbors = 10

    def load_img(self):
        # Load the image
        img = np.array(Image.open(self.image_path))

        # Transpose only the first two dimensions to switch x and y coordinates
        img = img.transpose(1, 0, 2)

        # Determine the center and radius
        center, radius = get_center_and_radius(img)

        self.img = img
        self.center, self.radius = center, radius

    def crop_img(self):
        # Determine the center and radius
        img = self.img
        center, radius = self.center, self.radius
        
        # Crop the image
        self.img = img[center[0]-radius:center[0]+radius, center[1]-radius:center[1]+radius, :]

    def make_greyscale(self):
        img = self.img
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        # Normalize the image to [0, 1] and invert so that 1 is dark and 0 is bright
        img = 1 - np.array(img) / 255.0
        self.img = img

    def apply_mask(self):
        img = self.img
        height, width = img.shape
        center = self.center
        radius = self.radius

        # Create the circular mask
        x, y = np.ogrid[-center[0]:width-center[0], -center[1]:height-center[1]]
        self.mask = x*x + y*y <= radius*radius

        # Apply the mask
        self.img = np.multiply(img, self.mask)

    def resize_img(self):
        self.img = resize_img(self.img, self.img_size)
        self.center, self.radius = get_center_and_radius(self.img)


    def add_to_importance_map(self, new_map):
        self.importance_map = np.add(self.importance_map, new_map)
    
    def finalize_importance_map(self):
        self.importance_map = np.multiply(self.importance_map, self.mask)
        self.importance_map *= self.importance_map_multiplier
        self.importance_map += 1

    def create_anchors(self):
        """Creates a list of tuples, each being the cordinates of an anchor
        Args:
            center: The center of the image
            radius: The radius of the image (1/2 of the width)
            num_anchors: The number of the anchors around the image
        """
        # Calculate the coordinates of the anchor points
        angles = np.linspace(0, 2*np.pi, self.num_anchors, endpoint=False)
        anchor_x = np.round(self.center[0] + (self.radius - 1) * np.cos(angles)).astype(int)
        anchor_y = np.round(self.center[1] + (self.radius - 1) * np.sin(angles)).astype(int)
        anchors = list(zip(anchor_x, anchor_y))
        
        self.anchors = anchors
    
    def create_circle_mask(self):
        """Create a boolean mask for a circle to make sure that we don't add pixels outside of the circle boundaries when drawing the lines"""
        x = np.arange(0, self.img.shape[1])[None, :]
        y = np.arange(0, self.img.shape[0])[:, None]
        mask = (x - self.center[0]) ** 2 + (y - self.center[1]) ** 2 <= self.radius ** 2

        self.mask = mask
    
    def benchmark_line_dict(self):

        def find_all_lines(start_anchor):
            """Generates all lines between one anchor and all other anchors. Only used for benchmarking"""
            line_pixel_list = {}
            line_darkness_list = {}
            for end_anchor in self.anchors:
                if end_anchor is not start_anchor:
                    both_anchors = tuple(sorted((start_anchor, end_anchor)))
                    if both_anchors not in line_pixel_list:
                        pixel_list, darkness_list = draw_line(both_anchors[0], both_anchors[1], self.line_darkness, self.mask)
                        line_pixel_list[both_anchors] = pixel_list
                        line_darkness_list[both_anchors] = darkness_list
            return line_pixel_list, line_darkness_list
        
        #benchmark
        start_time = time.perf_counter()
        benchmark = find_all_lines(self.anchors[0])
        print(f"Starting benchmark for line dict")
        run_time = (time.perf_counter() - start_time)*1000
        estimated_total_time = run_time*(comb(self.num_anchors, 2)/self.num_anchors)
        print(f"Done! Generated {self.num_anchors} anchor lines on a canvas of size of {self.radius*2} in {round(run_time, 1)} milliseconds. Estimated time for generating line dict is about {round(estimated_total_time/1000, 1)} seconds")

    def is_within_range(self, arr, idx1, idx2, x):
        # Calculate the distance between indices in both directions considering wrapped boundaries
        forward_distance = (idx2 - idx1) % len(arr)
        backward_distance = (idx1 - idx2) % len(arr)

        # Check if either forward or backward distance is less than or equal to x
        return forward_distance <= x or backward_distance <= x or idx1 == idx2
    
    def make_line_dict(self):
        """Makes a dictionary of every pixel and its darkness value for each line for every possible combination of anchors"""
        pkl_path = f"{self.data_folder}/line_dicts/{self.img.shape[0]}-{self.img.shape[1]}-{self.num_anchors}.pkl"
        if Path(pkl_path).exists():
            with open(pkl_path, "rb") as file:
                line_dicts = pickle.load(file)
                self.line_pixel_dict, self.line_darkness_dict = line_dicts["line_pixel_dict"], line_dicts["line_darkness_dict"]
        else:
            line_pixel_dict = {}
            line_darkness_dict = {}
            for start_index, start_anchor in enumerate(tqdm(self.anchors, desc="Creating Lines")):
                for end_index, end_anchor in enumerate(self.anchors):
                    if self.is_within_range(self.anchors, start_index, end_index, self.closest_neighbors): continue #You cant make a line between two of the same anchors
                    both_anchors = tuple(sorted((start_anchor, end_anchor))) #Sorts the indices for the lines.
                    if both_anchors not in line_pixel_dict: #Makes sure that the anchors aren't already in the dictionary, only in a different order. This makes the number of lines needed n choose 2.
                        pixel_list, darkness_list = draw_line(both_anchors[0], both_anchors[1], self.line_darkness, self.mask) #Draws the line
                        line_pixel_dict[both_anchors] = pixel_list
                        line_darkness_dict[both_anchors] = darkness_list
            print("Done!")
            self.line_pixel_dict, self.line_darkness_dict = line_pixel_dict, line_darkness_dict
            with open(pkl_path, 'wb') as file:
                pickle.dump({"line_pixel_dict": line_pixel_dict, "line_darkness_dict": line_darkness_dict}, file)

    def difference(self, pixel1, pixel2):
        """Calculate the absolute difference between two pixel values."""
        return np.abs(np.subtract(pixel1, pixel2))
    
    def create_string_art_img(self):
        """Creates a dictionary of pixels inside of the circle, and sets them equal to 0. This will be updated for every added string."""
        string_art_img = np.zeros(self.img.shape)
        self.string_art_img = string_art_img

    def create_difference_img(self):
        """Creates a dictionary of pixels inside of the circle, and sets them equal to the values of the original image. This difference dictionary is used to save the differences between the string_art_img and the  original image"""
        height, width = self.img.shape
        difference_img = np.zeros((height, width))
        for y in range(height):
            for x in range(width):
                if self.mask[x, y]:
                    difference_img[x, y] = self.difference(0, self.img[x, y])
                    
        self.difference_img = difference_img

    def find_new_loss(self, string_pixels, string_values):
        """
        Finds the loss of a single string using numpy arrays.

        string_pixels: A numpy array where each row represents the (x, y) coordinates of a pixel.
        string_values: A numpy array where each entry represents the darkness of the corresponding pixel in string_pixels.
        """
        # Split the string_pixels into separate x and y arrays
        x_coords = string_pixels[:, 0]
        y_coords = string_pixels[:, 1]

        string_art_values = self.string_art_img[x_coords, y_coords]
        img_values = self.img[x_coords, y_coords]
        diff_values = self.difference_img[x_coords, y_coords]
        importance_values = self.importance_map[x_coords, y_coords]

        total_values = string_art_values + string_values

        weighted_difference = importance_values * (self.difference(img_values, total_values) - diff_values)
        
        loss = np.sum(weighted_difference)/len(string_pixels)

        return loss
    
    def find_best_line(self, start_anchor):
        """Starts at a specified anchor and find the loss for every string leading to every other anchor, updating the best loss whenever a better one is found"""
        best_loss = np.inf #TODO set this to the starting loss and make the algorithm terminate when there isn't a possible improvement
        best_anchors = None #Also saves the set of anchors
        best_end_anchor = None
        for neighbor in get_neighbors(self.anchors, start_anchor): #Finds the neighbors of the start anchor
            for end_anchor in range(len(self.anchors)):
                both_anchors = tuple(sorted((neighbor, self.anchors[end_anchor]))) #Makes sure to get the right order for the indices, set in make_line_dict().
                if both_anchors not in self.line_pixel_dict: continue
                temp_loss = self.find_new_loss(self.line_pixel_dict[both_anchors], self.line_darkness_dict[both_anchors]) #Finds the loss of that string
                if(temp_loss < best_loss): #Check if the loss is better
                    best_loss = temp_loss
                    best_anchors = both_anchors
                    best_end_anchor = end_anchor
        return best_anchors, best_end_anchor, best_loss
    
    def benchmark_string_art(self):
        #benchmark
        start_time = time.perf_counter()
        self.find_best_line(start_anchor=0)
        print(f"Starting benchmark for string art")
        run_time = (time.perf_counter() - start_time)*1000
        estimated_total_time = run_time*self.num_lines
        print(f"Done! Found the best line out of {len(self.anchors)-1} anchors lines on a canvas of size of {self.img.shape[0]}x{self.img.shape[0]} in {round(run_time, 1)} milliseconds. Estimated time for finding all lines is {round(estimated_total_time/1000, 1)} seconds")

    def update_with_best_line(self, best_anchors):
        """Updates the overall loss value, string_art_img pixels, and difference_img pixels after finding the best line"""
        
        # Draw the line and get the pixel locations and values
        line_pixels, line_values = draw_line(best_anchors[0], best_anchors[1], self.line_darkness, self.mask)
        
        # Split the line_pixels into separate x and y arrays
        x_coords = line_pixels[:, 0]
        y_coords = line_pixels[:, 1]
        
        # Update the string_art_img using the pixel locations and values
        new_values = np.clip(self.string_art_img[x_coords, y_coords] + line_values, 0, 1)
        self.string_art_img[x_coords, y_coords] = new_values
        
        # Update the difference_img with the new differences
        self.difference_img[x_coords, y_coords] = self.difference(self.img[x_coords, y_coords], new_values)
    
    def create_string_art(self, starting_anchor: int = 0):
        current_anchor = starting_anchor
        for i in tqdm(range(self.num_lines), desc="Drawing Line"):
            best_anchors, current_anchor, best_loss = self.find_best_line(current_anchor)
            self.update_with_best_line(best_anchors)
            self.anchor_list.append(best_anchors)

    def ooga_booga_mode(self):
        for i in tqdm(range(self.num_lines)):
            best_loss = np.inf
            for start_anchor in range(len(self.anchors)):
                anchors, current_anchor, loss = self.find_best_line(start_anchor)
                if(loss < best_loss):
                    best_loss, best_anchors = loss, anchors
            self.update_with_best_line(best_anchors)


    def save_data(self, directory):
        """Saves the given numpy arrays as JPG images in the specified directory."""
        
        # Check if directory exists; if not, create it
        if not os.path.exists(directory):
            os.makedirs(directory)
        
        save_image(self.difference_img, directory, "difference_image.jpg")
        save_image(self.string_art_img, directory, "string_art.jpg")
        save_image(self.img, directory, "original.jpg")

        config_pkl = {
            "num_anchors": self.num_anchors,
            "num_lines": self.num_lines,
            "line_darkness": self.line_darkness,
            "img_shape": self.img.shape,
            "all_strings": self.anchor_list,
            "all_anchors": self.anchors,
            "mask": self.mask
        }
        with open(f"{directory}/config.pkl", 'wb') as file:
            pickle.dump(config_pkl, file)


def save_image(image, directory, name):
            image_data = (1-np.transpose(image))*255
            image = Image.fromarray(image_data.astype(np.uint8))
            image.save(f"{directory}/{name}")

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

def draw_line(p0: tuple, p1: tuple, multiplier: float, mask: np.ndarray):
    """Creates a dictionary of coordinates and their darkness values of a line between two points. Uses Xiaolin Wu's anti-aliasing algorithm
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

def create_img_from_anchors(data_dir, percents):
    with open(f"{data_dir}/config.pkl", "rb") as file:
        config = pickle.load(file)
    
    img_shape = config["img_shape"]
    all_strings = config["all_strings"]
    multiplier = config["line_darkness"]
    mask = config["mask"]
    
    img = np.zeros(img_shape)
    max_num_lines = int(len(all_strings) * (max(percents) / 100))
    num_lines = []

    for percent in percents:
        num_lines.append(int(len(all_strings) * (percent / 100)))
    
    for line in range(max_num_lines):
            if all_strings[line] == 0: continue
            p0 = all_strings[line][0]
            p1 = all_strings[line][1]
            line_pixels, line_values = draw_line(p0=p0, p1=p1, multiplier=multiplier, mask=mask)
            
            x_coords, y_coords = line_pixels.T  # Transpose and unpack
            
            # Clip and update directly using numpy operations
            img[x_coords, y_coords] = np.clip(img[x_coords, y_coords] + line_values, 0, 1)

            if line+1 in num_lines:
                save_image(img, data_dir, f"{line+1}_lines.jpg")

def outline_facial_features(image_array, 
                            predictor_path="shape_predictor_68_face_landmarks.dat",
                            feature_thickness=5,
                            blur_size=7):
    
    image_array = image_array.transpose(1, 0, 2)

    # Convert the 3D numpy array to grayscale
    gray = cv2.cvtColor(image_array, cv2.COLOR_BGR2GRAY)

    # Load the facial detector and landmark predictor
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(predictor_path)

    # Detect faces in the grayscale image
    faces = detector(gray)

    # Initialize heatmap canvas to white
    heatmap = np.ones_like(gray) * 255

    for face in faces:
        # Predict landmarks
        landmarks = predictor(gray, face)

        # Define facial features
        features_to_fill = [
            list(range(17, 22)),    # Right eyebrow
            list(range(22, 27)),    # Left eyebrow
            list(range(36, 42)),    # Right eye
            list(range(42, 48)),    # Left eye
            list(range(48, 61)),    # Outer lips
            [27, 31, 32, 33, 34, 35] # Nose
        ]

        features_to_draw = [
            list(range(0, 17)),     # Jaw
        ]

        # Draw polygons around facial features that need filling in black
        for feature in features_to_fill:
            points = [(landmarks.part(n).x, landmarks.part(n).y) for n in feature]
            points = np.array(points, dtype=np.int32)
            cv2.fillPoly(heatmap, [points], color=0)

        # Draw straight lines for features
        for feature in features_to_draw:
            points = [(landmarks.part(n).x, landmarks.part(n).y) for n in feature]
            points = np.array(points, dtype=np.int32)
            cv2.polylines(heatmap, [points], isClosed=False, color=0, thickness=feature_thickness)

        # Erode the features to increase their size
        kernel = np.ones((feature_thickness, feature_thickness), np.uint8)
        heatmap = cv2.erode(heatmap, kernel, iterations=1)

    # Blur the heatmap for smoother transition
    heatmap = cv2.GaussianBlur(heatmap, (blur_size, blur_size), 0)
    heatmap = 1 - (heatmap / 255)
    heatmap = heatmap.T

    return heatmap

def resize_img(img: np.ndarray, img_size: int):
    img = Image.fromarray((img * 255).astype(np.uint8))
    img = img.resize((img_size, img_size), Image.LANCZOS)
    img = np.array(img) / 255.0
    
    return img

def load_img_as_texture(img_path: str):
    image = np.array(Image.open(img_path))
    height, width, _ = image.shape
    texture_data = image.flatten().tolist()
    return width, height, texture_data

def get_neighbors(arr, idx):
    # Calculate indices for neighbors with wrapping
    left_idx = (idx - 1) % len(arr)
    right_idx = (idx + 1) % len(arr)

    return arr[left_idx], arr[right_idx]

def get_center_and_radius(image):
        if len(image.shape) == 2:
            height, width = image.shape
        else:
            height, width, _ = image.shape
        center = (int(height / 2), int(width / 2))
        radius = min(center[0], center[1])
        return center, radius

def combine_importance_maps(maps: list[ImportanceMap], importance_value = 1, gaussian_blur = 0):
    combined_map: np.ndarray
    for map in maps:
        combined_map += map
    combined_map = ImportanceMap(combined_map, importance_value, gaussian_blur)
    
def dynamic_sigmoid(image, max=1):
    sigma = np.std(image)
    if sigma == 0: sigma = 1
    #sigma=1
    transformed = max / (1 + np.exp(-image/sigma))
    return transformed