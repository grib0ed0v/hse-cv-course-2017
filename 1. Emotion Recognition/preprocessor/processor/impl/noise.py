from preprocessor.processor.abstractprocessor import AbstractProcessor


class NoiseProcessor(AbstractProcessor):
    def __init__(self):
        pass

    # Method receives image, modify it in order to denoise image and propagate it to the next processor.
    def process(self, image):
        return image
