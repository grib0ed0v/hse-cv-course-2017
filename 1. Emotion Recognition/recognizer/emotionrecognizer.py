import cv2


class EmotionRecognizer:
    def __init__(self):
        # list with emotion labels
        self.emotions = []
        self.scale_size = (224, 224)

    # implement scaling
    def scale(self, image):
        image = cv2.resize(image, self.scale_size)
        return image

    # Method starts CNN and returns String with predicted Emotion
    # add logging!
    def recognize(self, image):
        cv2.imshow('111', image)
        scaled_image = self.scale(image)
        cv2.imshow('scale', scaled_image)
        return 'my emotion'
