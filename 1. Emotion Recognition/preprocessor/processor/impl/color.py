from preprocessor.processor.abstractprocessor import AbstractProcessor


class ColorProcessor(AbstractProcessor):
    def __init__(self):
        pass

    # Method receives image, modifies it in order to rebalance colours and propagates it to the next processor.
    def process(self, image):
        return image
