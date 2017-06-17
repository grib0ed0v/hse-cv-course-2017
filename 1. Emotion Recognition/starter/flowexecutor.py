import logging
import cv2

from preprocessor.chain import ProcessorChain
from recognizer.emotionrecognizer import EmotionRecognizer
from preprocessor.facedetector import FaceDetector
from starter.configresolver import ConfigResolver


class FlowExecutor:
    def __init__(self):
        self.config_resolver = ConfigResolver()

        self.chain = ProcessorChain(self.config_resolver.get_processor_chain())
        self.face_detector = FaceDetector()
        self.emotion_recognizer = EmotionRecognizer()

    # crop image of size
    def __crop(self, image, p1, p2):
        return image[p1[0]:p2[0], p1[1]:p2[1], :]

    # add bounding box of appropriate color with emotions label
    def __add_labeled_bounding_box(self, image, predicted_emotion, pt1, pt2):
        cv2.rectangle(image, pt1, pt2, (0, 255, 0))
        cv2.putText(image, predicted_emotion, (pt1[0], pt1[1] + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0))

    def execute(self, image):
        image_copy = image.copy()
        image_copy = self.chain.run(image_copy)  # image will be processed in chain methods
        faces = self.face_detector.detect_faces(image_copy)  # faces coordinates

        if faces is not None and len(faces) > 0:
            for (x, y, w, h) in faces:
                face = self.__crop(image_copy, (x, y), (x + w, y + h))
                predicted_emotion = self.emotion_recognizer.recognize(face)
                self.__add_labeled_bounding_box(image, predicted_emotion, (x, y), (x + w, y + h))  # pass original image for bounding
        else:
            logging.warning('No face was found!')

        return image
