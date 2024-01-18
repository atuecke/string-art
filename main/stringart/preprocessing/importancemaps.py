import numpy as np
from PIL import Image
import os
import cv2

from stringart.preprocessing.image import resize_img

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