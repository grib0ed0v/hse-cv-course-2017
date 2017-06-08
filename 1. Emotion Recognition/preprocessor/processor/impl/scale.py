from preprocessor.processor.abstractprocessor import AbstractProcessor


class ScaleProcessor(AbstractProcessor):
    def __init__(self):
        pass

    # Method receives image, modify its size and propagate it to the next processor.
    def process(self, image):
        return image
