import cv2
import logging
import numpy as np


class EmotionRecognizer:
    def __init__(self):
        # list with emotion labels
        self.emotions = ['Angry' , 'Disgust' , 'Fear' , 'Happy'  , 'Neutral' ,  'Sad' , 'Surprise']
        self.scale_size = (224, 224)

    # implement scaling
    def scale(self, image):
        image = cv2.resize(image, self.scale_size)
        return image

    # Method starts CNN and returns String with predicted Emotion
    # add logging!
    def recognize(self, image):
        scaled_image = self.scale(image)
        logging.info(cv2.__version__)
        model_dir = "../starter/resources/"
        net_model_file = model_dir + "deploy.txt"
        net_pretrained = model_dir + "EmotiW_VGG_S.caffemodel"
        # Read caffe model with cv2.dnn module
        Emotion_Recognition_CNN = cv2.dnn.readNetFromCaffe(net_model_file, net_pretrained)
        # Additional image processing for input format
        blob = np.moveaxis(scaled_image, 2, 0)
        blob = np.reshape(blob.astype(np.float32), (-1, 3, 224, 224))
        # Setting input & processing network
        Emotion_Recognition_CNN.setBlob('.data', blob)
        Emotion_Recognition_CNN.forward('prob')
        predictions = Emotion_Recognition_CNN.getBlob("prob")
        emotion = (self.emotions[predictions.argmax()]))
        return emotion
