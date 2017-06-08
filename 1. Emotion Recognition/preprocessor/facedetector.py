# Class for Face detection.
# Use two types of Haar cascades - for frontal and profile face detection.
# Potential problems: - Faces might overlap on one image.
#                     - Cascades might found the same face and we will duplicate operations.


class FaceDetector:
    # Load resources here
    # Constructor receive parameter bound, which describes
    # how many faces might be found in order to save RT processing.
    def __init__(self, bound):
        self.bound = bound

    # Method detects faces on the image and returns list of tuples with 4 ROI coordinates:
    # x0, y0, x1,y1 - which bound face on the image.
    # NOTE: Method does not modify image.
    def detect_faces(self, image):
        faces = list()

        # use bound here
        faces.append(self.detect_front_faces(image))
        faces.append(self.detect_profile_faces(image))

        return faces

    # Method for detecting front faces on the image.
    # Returns list of tuples with 4 ROI coordinates:
    # x0, y0, x1,y1 - which bound face on the image.
    # NOTE: Method does not modify image.
    def detect_front_faces(self, image):
        front_faces = list()
        height, weight = image.shape[:2]
        front_faces.append((0, 0, height, weight))
        return front_faces

    # Method for detecting profile faces on the image.
    # Returns list of tuples with 4 ROI coordinates:
    # x0, y0, x1,y1 - which bound face on the image.
    # NOTE: Method does not modify image.
    #
    # TODO: might be useless. Test it later
    def detect_profile_faces(self, image):
        front_faces = list()
        height, weight = image.shape[:2]
        front_faces.append((0, 0, height, weight))
        return front_faces
