import cv2


# Class describes sequence of processor, which should be applied to the image
class ProcessorChain:
    # Method sets sequence of processors
    def __init__(self, chain):
        self.chain = chain

    # main method for running pre process actions
    def run(self, image):
        i = 0
        for p in self.chain:
            image = p.process(image)
            cv2.imshow(str(i), image)  # debug output
            i += 1
        #cv2.waitKey(0)
        return image
