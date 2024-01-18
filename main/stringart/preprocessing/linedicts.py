from tqdm import tqdm
from pathlib import Path
import pickle

from stringart.core.stringimage import StringArtImage
from stringart.algorithm.lines import draw_line

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