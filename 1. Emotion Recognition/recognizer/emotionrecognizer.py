class EmotionRecognizer:
    def __init__(self):
        # list with emotion labels
        self.emotions = []
        self.scale_size = 224

    # implement scaling
    def scale(self, image):
        return image

    # Method starts CNN and returns String with predicted Emotion
    # add logging!
    def recognize(self, image):
        scaled_image = self.scale(image)
        return 'my emotion'
