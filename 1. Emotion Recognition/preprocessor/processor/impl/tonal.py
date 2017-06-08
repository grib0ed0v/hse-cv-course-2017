from preprocessor.processor.abstractprocessor import AbstractProcessor


class TonalProcessor(AbstractProcessor):
    def __init__(self):
        pass

    # Method receives image, modify it in order to fix tones and propagate it to the next processor.
    def process(self, image):
        return image
