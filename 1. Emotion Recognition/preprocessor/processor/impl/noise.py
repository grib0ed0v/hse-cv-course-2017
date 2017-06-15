import cv2
from preprocessor.processor.abstractprocessor import AbstractProcessor


class NoiseProcessor(AbstractProcessor):
    def __init__(self, h=10, hColor=10, templateWindowSize=7, searchWindowSize=21):
        self.h = h
        self.hColor = hColor
        self.templateWindowSize = templateWindowSize
        self.searchWindowSize = searchWindowSize

    # Method receives image, modifies it in order to denoise image and propagates it to the next processor.
    def process(self, image):
        image = cv2.fastNlMeansDenoisingColored(image, None, self.h, self.hColor, self.templateWindowSize, self.searchWindowSize)
        return image
