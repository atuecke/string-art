import numpy as np

from stringart.preprocessing.image import BaseImage

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