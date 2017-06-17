import cv2
import logging


class FaceDetector:
    def __init__(self):
        self.frontal_face_cascade = cv2.CascadeClassifier('./resources/haarcascade_frontalface_default.xml')

    def detect_faces(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = self.frontal_face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=4, minSize=(40, 40),
                                                           flags=cv2.CASCADE_SCALE_IMAGE)
        logging.info('Found %s face(s)', len(faces))
        return faces
