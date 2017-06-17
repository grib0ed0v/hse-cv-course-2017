import cv2
from preprocessor.processor.abstractprocessor import AbstractProcessor


class ContrastProcessor(AbstractProcessor):
    def __init__(self, clipLimit=2.0, tileGridSize=(8, 8)):
        self.clipLimit = clipLimit
        self.tileGridSize = tileGridSize

    # Method receives image, modifies it in order to rebalance contrasts and propagates it to the next processor.
    def process(self, image):
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        h, s, v = cv2.split(hsv)
        clahe = cv2.createCLAHE(self.clipLimit, self.tileGridSize)
        v = clahe.apply(v)
        hsv = cv2.merge((h, s, v))
        cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB, image)
