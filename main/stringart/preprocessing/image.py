import numpy as np
from PIL import Image
import cv2

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