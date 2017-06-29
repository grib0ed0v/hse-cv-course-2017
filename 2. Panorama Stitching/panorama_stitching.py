import numpy as np
import imutils
import cv2
import argparse


class TwoImagesStitcher:
    def __init__(self, ratio=0.75, reproj_thresh=4.0, show_matches=False):
        self.is_opencv_3 = imutils.is_cv3()
        self.ratio = ratio
        self.reproj_thresh = reproj_thresh
        self.show_matches = show_matches

    def stitch(self, images, iteration=1):
        (left_image, right_image) = images
        (left_image_keypoints, left_image_features) = self.detect_and_describe(left_image)
        (right_image_keypoints, right_image_features) = self.detect_and_describe(right_image)

        matched_keypoints = self.match_keypoints(right_image_keypoints, left_image_keypoints,
                                right_image_features, left_image_features)
        if matched_keypoints is None:
            return None
        (matches, homography, status) = matched_keypoints
        result = cv2.warpPerspective(right_image, homography,
                                     (right_image.shape[1] + left_image.shape[1], right_image.shape[0]))
        result[0:left_image.shape[0], 0:left_image.shape[1]] = left_image
        if self.show_matches:
            self.draw_matches(right_image, left_image, right_image_keypoints, left_image_keypoints, matches,
                              status, iteration)
        result = self.crop_border(result, right_image_keypoints, left_image_keypoints, left_image.shape[1],
                                  matches, status)
        return result

    def crop_border(self, result, right_image_keypoints, left_image_keypoints, left_image_width, matches, status):
        border_width = 0
        for ((trainIdx, queryIdx), s) in zip(matches, status):
            if s == 1:
                current_border_width = int(right_image_keypoints[queryIdx][0]) + left_image_width\
                                       - int(left_image_keypoints[trainIdx][0])
                if current_border_width > border_width:
                    border_width = current_border_width
        return result[:, :-border_width]

    def detect_and_describe(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        if self.is_opencv_3:
            descriptor = cv2.xfeatures2d.SIFT_create()
            (keypoints, features) = descriptor.detectAndCompute(image, None)

        else:
            detector = cv2.FeatureDetector_create("SIFT")
            keypoints = detector.detect(gray)

            extractor = cv2.DescriptorExtractor_create("SIFT")
            (keypoints, features) = extractor.compute(gray, keypoints)

        keypoints = np.float32([keypoint.pt for keypoint in keypoints])
        return (keypoints, features)

    def match_keypoints(self, right_image_keypoints, left_image_keypoints, right_image_features, left_image_features):
        matcher = cv2.DescriptorMatcher_create("BruteForce")
        raw_matches = matcher.knnMatch(right_image_features, left_image_features, 2)
        matches = []

        for m in raw_matches:
            if len(m) == 2 and m[0].distance < m[1].distance * self.ratio:
                matches.append((m[0].trainIdx, m[0].queryIdx))
        if len(matches) > 4:
            left_image_points = np.float32([left_image_keypoints[i] for (i, _) in matches])
            right_image_points = np.float32([right_image_keypoints[i] for (_, i) in matches])

            (homography, status) = cv2.findHomography(right_image_points, left_image_points, cv2.RANSAC,
                                             self.reproj_thresh)
            return (matches, homography, status)
        return None

    def draw_matches(self, right_image, left_image, right_image_keypoints, left_image_keypoints, matches, status,
                     iteration):
        (left_image_height, left_image_width) = left_image.shape[:2]
        (right_image_height, right_image_width) = right_image.shape[:2]
        image_with_matches = np.zeros((max(left_image_height, right_image_height),
                                      left_image_width + right_image_width, 3), dtype="uint8")
        image_with_matches[0:left_image_height, 0:left_image_width] = left_image
        image_with_matches[0:right_image_height, left_image_width:] = right_image


        for ((trainIdx, queryIdx), s) in zip(matches, status):
            if s == 1:
                first_point = (int(left_image_keypoints[trainIdx][0]),
                               int(left_image_keypoints[trainIdx][1]))
                second_point = (int(right_image_keypoints[queryIdx][0]) + left_image_width,
                                int(right_image_keypoints[queryIdx][1]))
                cv2.line(image_with_matches, first_point, second_point, (0, 255, 0), 1)
        cv2.imwrite('Matched_points_iteration_' + str(iteration) + '.jpg', image_with_matches)


class PanoramaStitcher:
    def __init__(self, image_paths, ratio=0.75, reproj_thresh=4.0, show_matches=False):
        self.ratio = ratio
        self.reproj_thresh = reproj_thresh
        self.show_matches = show_matches
        self.images = []
        for image_path in image_paths:
            self.images.append(cv2.imread(image_path))

    def stitch(self):
        stitcher = TwoImagesStitcher(ratio=self.ratio, reproj_thresh=self.reproj_thresh, show_matches=self.show_matches)
        result = stitcher.stitch(self.images[:2])
        for i in range(2,len(self.images)):
            result = stitcher.stitch([result,self.images[i]], i)
        return result


if __name__ == '__main__':
    args_parser = argparse.ArgumentParser()
    args_parser.add_argument('images', metavar='images', type=str, nargs='+',
                             help='images to be stitched')
    args_parser.add_argument('--show_matches', dest='show_matches', action='store_true',
                             help='show matched keypoints')

    args = vars(args_parser.parse_args())

    panorama_stitcher = PanoramaStitcher(args['images'], show_matches=args['show_matches'])
    result = panorama_stitcher.stitch()

    cv2.imwrite('Result.jpg', result)
    cv2.waitKey(0)
