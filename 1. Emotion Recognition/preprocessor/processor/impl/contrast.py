from preprocessor.processor.abstractprocessor import AbstractProcessor


class ContrastProcessor(AbstractProcessor):
    def __init__(self):
        pass

    # Method receives image, modifies it in order to rebalance contrasts and propagates it to the next processor.
    def process(self, image):
        return image
