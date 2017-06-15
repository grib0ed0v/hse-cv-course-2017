import cv2
import numpy as np
from preprocessor.processor.abstractprocessor import AbstractProcessor


class TonalProcessor(AbstractProcessor):
    def __init__(self, gamma=1):
        self.gamma = gamma

    # Method receives image, modifies it in order to fix tones and propagates it to the next processor.
    def process(self, image):
        # build  table with their adjusted gamma values
        newGamma = 1.0 / self.gamma
        adj_gamma = np.array([((i / 255.0) ** newGamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
        image = cv2.LUT(image, adj_gamma)  # function LUT fills the output array with values from the adjusted gamma
        return image
