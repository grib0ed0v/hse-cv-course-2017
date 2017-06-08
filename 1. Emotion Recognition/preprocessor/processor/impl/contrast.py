from preprocessor.processor.abstractprocessor import AbstractProcessor


class ContrastProcessor(AbstractProcessor):
    def __init__(self):
        pass

    # Method receives image, modify it in order to rebalance contrasts and propagate it to the next processor.
    def process(self, image):
        return image
