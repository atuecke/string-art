import numpy as np
from tqdm import tqdm
import math
import pickle
import os

from stringart.preprocessing.stringartblueprint import StringArtBlueprint
from stringart.core.lines import draw_line
from stringart.utils.io import save_image, save_instructions

class StringImage():
    """
    The string art image created from the base image

    Attributes:
        img: The rendered version of the string art imgage
        anchors: The list of anchors around the image
        string_path: A list of anchor sets that the string takes
    """
    def __init__(
            self,
            blueprint: StringArtBlueprint,
            anchors: list,
            line_darkness: float,
            mask: np.ndarray
    ) -> None:
        """
        Args:
            base_img: The origonal image after preprocessing to be used as the scale
            anchors: The list of anchors around the image
        """
        self.blueprint = blueprint
        self.img = np.zeros(blueprint.img.shape)
        self.anchors = anchors
        self.string_path = []
        self.line_darkness = line_darkness
        self.mask = mask
        self.best_anchors_list = []
        self.accuracy_list = []

    def render_img_from_anchors(self, from_line: int = 0, to_line: int = math.inf):
        to_line = int(min(to_line, len(self.best_anchors_list)))
        img = np.zeros(self.blueprint.img.shape)
        for idx in tqdm(range(from_line, to_line, 1)):
            start, end = self.best_anchors_list[idx]
            p0 = self.blueprint.anchors[start].coordinates
            p1 = self.blueprint.anchors[end].coordinates
            line_pix, line_darkness = draw_line(p0, p1, 0.2, self.blueprint.mask)
            
            rows = line_pix[:, 0]
            cols = line_pix[:, 1]

            img[rows, cols] += line_darkness

        return img
    
    # def __getstate__(self):
    #     state = self.__dict__.copy()
    #     return state
    
    # def __setstate__(self, state):
    #     self.__dict__.update(state)
    
    def save(self, folder_path: str, name: str):
        # Check if directory exists; if not, create it
        path = f"{folder_path}/{name}"
        final_path = ""
        i = 1
        if not os.path.exists(f"{folder_path}/{name}"):
            final_path = path
        else: 
            while final_path == "":
                if os.path.exists(f"{folder_path}/{name}-{i}"):
                    i += 1
                else:
                    final_path = f"{path}-{i}"
                    break

        os.makedirs(final_path)

        # save the image
        save_image(self.img, final_path, "output.jpg")

        # save the instructions
        save_instructions(string_path=self.string_path, directory=f"{final_path}/instructions.txt")

        # save instance of class
        with open(f"{final_path}/save.pkl", "wb") as f:
            pickle.dump(self, f)
    
    @classmethod
    def from_save(cls, folder_path: str):
        with open(f"{folder_path}/save.pkl", "rb") as f:
            instance: StringImage = pickle.load(f)
            return instance
        
        