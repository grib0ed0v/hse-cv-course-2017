import logging

from preprocessor.chain import ProcessorChain
from recognizer.emotionrecognizer import EmotionRecognizer
from preprocessor.facedetector import FaceDetector

class FlowExecutor:

    def __init__(self):
        self.chain = ProcessorChain()
        self.face_detector = FaceDetector()
        self.emotion_recognizer = EmotionRecognizer()

    # crop image of size
    def __crop(self, image, x0, y0, x1, y1):
        return image[x0:x1, y0:y1, :]

    # add bounding box of appropriate color with emotions label
    def __add_labeled_bounding_box(self):
        pass

    def execute(self, image):
        self.chain.run(image)  # original image will be processed in chain methods
        faces = self.face_detector.detect_faces(image)  # faces coordinates

        if faces is not None and len(faces) > 0:
            for (x0, y0, x1, y1) in faces:
                face = self.__crop(image, x0, y0, x1, y1)
                predicted_emotion = self.emotion_recognizer.recognize(face)
                self.__add_labeled_bounding_box()
        else:
            logging.warning('No face was found!')
