# Class describes sequence of processor, which should be applied to the image
from preprocessor.processor.impl.noise import NoiseProcessor
import cv2


class ProcessorChain:
    # Method sets sequence of processors
    def __init__(self):
        self.chain = list()
        self.chain.append(NoiseProcessor())
        # add more processors here

    # main method for running pre process actions
    def run(self, image):
        i = 0
        for p in self.chain:
            image = p.process(image)
            cv2.imshow(str(i), image)
            i +=1
        cv2.waitKey(0)  # debug output
        return image
