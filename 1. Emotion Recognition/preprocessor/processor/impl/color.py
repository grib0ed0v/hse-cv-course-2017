import cv2
from preprocessor.processor.abstractprocessor import AbstractProcessor


class ColorProcessor(AbstractProcessor):
    def __init__(self):
        self.whiteBalancer = cv2.xphoto.createGrayworldWB()

    # Method receives image, modifies it in order to rebalance colours and propagates it to the next processor.
    def process(self, image):
        self.whiteBalancer.balanceWhite(image, image)
