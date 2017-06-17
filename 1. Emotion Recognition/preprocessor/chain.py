# Class describes sequence of processor, which should be applied to the image
#import cv2
import logging


class ProcessorChain:
    # Method sets sequence of processors
    def __init__(self, chain):
        self.chain = chain

    # main method for running pre process actions
    def run(self, image):
        for p in self.chain:
            logging.info('Start: %s', p.__class__.__name__)
            p.process(image)
            logging.info('Finished: %s', p.__class__.__name__)
            # cv2.imshow(p.__class__.__name__, image)  # debug output
        return image
