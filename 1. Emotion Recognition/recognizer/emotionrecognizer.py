import cv2
import numpy as np


class EmotionRecognizer:
    def __init__(self):
        self.scale_size = (224, 224)
        self.emotions = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']  # list with emotion labels

        net_model_file = "./resources/deploy.txt"
        net_pretrained = "./resources/EmotiW_VGG_S.caffemodel"
        self.Emotion_Recognition_CNN = cv2.dnn.readNetFromCaffe(net_model_file,
                                                                net_pretrained)  # Read caffe model with cv2.dnn module

    def scale(self, image):
        image = cv2.resize(image, self.scale_size)
        return image

    # Method starts CNN and returns String with predicted Emotion
    def recognize(self, image):
        scaled_image = self.scale(image)

        blob = np.moveaxis(scaled_image, 2, 0)
        blob = np.reshape(blob.astype(np.float32), (-1, 3, 224, 224))

        # Setting input & processing network
        self.Emotion_Recognition_CNN.setBlob('.data', blob)
        self.Emotion_Recognition_CNN.forward('prob')
        predictions = self.Emotion_Recognition_CNN.getBlob("prob")
        emotion = self.emotions[predictions.argmax()]
        return emotion
