from preprocessor.processor.abstractprocessor import AbstractProcessor


class LightProcessor(AbstractProcessor):
    def __init__(self):
        pass

    # Method receives image, modifies it in order to improve light and propagates it to the next processor.
    def process(self, image):
        return image
