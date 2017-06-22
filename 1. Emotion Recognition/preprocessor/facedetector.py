import cv2
import logging


class FaceDetector:
    def __init__(self, cascade_classifier_path, scaleFactor, minNeighbors, minSize):
        self.frontal_face_cascade = cv2.CascadeClassifier(cascade_classifier_path)
        self.scaleFactor = scaleFactor
        self.minNeighbors = minNeighbors
        self.minSize = minSize

    def detect_faces(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = self.frontal_face_cascade.detectMultiScale(gray, scaleFactor=self.scaleFactor,
                                                           minNeighbors=self.minNeighbors,
                                                           minSize=self.minSize,
                                                           flags=cv2.CASCADE_SCALE_IMAGE)
        logging.info('Found %s face(s)', len(faces))
        return faces
