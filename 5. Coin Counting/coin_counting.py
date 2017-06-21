import numpy as np
import cv2
from sklearn.externals import joblib


def get_circles(img):
    final_rects = []
    cascade = cv2.CascadeClassifier("cascade.xml")
    sf = min(640. / img.shape[1], 480. / img.shape[0])
    gray = cv2.resize(img, (0, 0), None, sf, sf)
    rects = cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=4, minSize=(40, 40))

    for x, y, w, h in rects:
        rect = img[int(y / sf): int(y / sf) + int(h / sf), int(x / sf): int(x / sf) + int(w / sf)]
        cv2.imshow("edges+face {}".format(x), rect)
        final_rects.append(rect)
        cv2.rectangle(img, (int(x / sf), int(y / sf)), (int(x / sf) + int(w / sf), int(y / sf) + int(h / sf)),
                      (0, 0, 255), 2)
    cv2.imshow("edges+face", img)
    return final_rects


def find_circles(img):
    # Use gaussian filter to delete noise
    grey_blur = cv2.GaussianBlur(img, (15, 15), 0)

    # adaptive tresholding - threshold is local, parameters: adaptive method - gaussian/mean; block_size - the neighbourhood; ...
    thresh = cv2.adaptiveThreshold(grey_blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY_INV, 11, 1)

    # morfological operations
    kernel = np.ones((3, 3), np.uint8)
    closing = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE,
                               kernel, iterations=4)

    # find contours
    cont_img = closing.copy()
    # contours = cv2.findContours(cont_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    im2, contours, hierarchy = cv2.findContours(cont_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    circles = []
    img_2 = img.copy()
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 5000:  # or area > 50000:
            continue
        if len(cnt) < 5:
            continue
        ellipse = cv2.fitEllipse(cnt)
        circles.append(ellipse)
        cv2.ellipse(img_2, ellipse, (0, 255, 0), 2)

    # cv2.imshow("Adaptive Thresholding", thresh)
    # cv2.imshow("Morphological Closing", closing)
    cv2.imshow('Contours', img_2)
    cv2.waitKey()
    return circles


def form_rectangles(circle, size):
    coord = []
    m = 0.5 * max(circle[1][0], circle[1][1])
    coord.append(max(circle[0][0] - m, 0))
    coord.append(max(circle[0][1] - m, 0))
    coord.append(min(circle[0][0] + m, size[1]))
    coord.append(min(circle[0][1] + m, size[0]))
    coord.append(circle[0][0] - coord[0])
    coord.append(circle[0][1] - coord[1])
    return coord


def rect_img(circles, grey_img):
    rectangles = []
    size = grey_img.shape
    for circle in circles:
        coord = form_rectangles(circle, size)
        ior = grey_img.copy()
        curr_img = ior[int(coord[1]):int(coord[3]), int(coord[0]):int(coord[2])]
        curr_size = curr_img.shape
        for i in range(curr_size[0]):
            for j in range(curr_size[1]):
                m = max(0.5 * circle[1][0], 0.5 * circle[1][1])
                x = i - coord[5]
                y = j - coord[4]
                a = 0.5 * circle[1][0]
                b = 0.5 * circle[1][1]
                phi = np.math.radians(circle[2])
                threshold = x * x * (
                    np.math.pow(np.math.sin(phi), 2) / (a * a) + np.math.pow(np.math.cos(phi), 2) / (
                        b * b)) + y * y * (
                    np.math.pow(np.math.cos(phi), 2) / (a * a) + np.math.pow(np.math.sin(phi), 2) / (
                        b * b)) + x * y * np.math.sin(2 * phi) * (1 / (a * a) - 1 / (b * b))
                if (threshold > 1):
                    curr_img[i][j] = 255
        rectangles.append(curr_img)
    return rectangles


def adaptive_thresholding(img):
    # find adaptive threshold value with OpenCV function
    res = cv2.GaussianBlur(img, (15, 15), 0)
    res = cv2.adaptiveThreshold(res, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                cv2.THRESH_BINARY_INV, 11, 1)
    return res

def morphology(img, parameter):
    kernel = np.ones((3, 3), np.uint8)
    res = cv2.morphologyEx(img, parameter, kernel, iterations=1)
    return res

def delete_components(curr_img, color, part_1, part_2):
    img = curr_img.copy()
    curr_size = img.shape
    min_w = curr_size[1]
    max_w = -1
    min_h = curr_size[0]
    max_h = -1
    i, j = 0, 0
    while (img[i][j] != 0) and (i < curr_size[0] - 1):
        if (j < curr_size[1] - 1):
            j = j + 1
        else:
            if (i < curr_size[0] - 1):
                i = i + 1
                j = 0
    if (i == curr_size[0] - 1):
        return [];
    stack = [(i, j)]
    list = [(i, j)]
    min_w = max_w = j
    min_h = max_h = i
    while (len(stack) > 0):
        el = stack[len(stack) - 1]
        i, j = el[0], el[1]
        img[i, j] = color
        stack.pop()
        for a in range(3):
            for b in range(3):
                if (((a == 1) | (b == 1)) & (a != b)):
                    if (0 <= i + a - 1) & (i + a - 1 < curr_size[0]) & (0 <= j + b - 1) & (j + b - 1 < curr_size[1]):
                        if (img[i + a - 1][j + b - 1] == 0):
                            min_w = min(min_w, j + b - 1)
                            max_w = max(max_w, j + b - 1)
                            min_h = min(min_h, i + a - 1)
                            max_h = max(max_h, i + a - 1)
                            stack.append((i + a - 1, j + b - 1))
                            list.append((i + a - 1, j + b - 1))
    size1 = max(max_h - min_h, 0)
    size2 = max(max_w - min_w, 0)
    S = size1 * size2
    S0 = curr_size[0] * curr_size[1]
    if ((S < part_1 * S0) | (S > part_2 * S0) | (min_w < 3) | (max_w > curr_size[0] - 3) | (min_h < 3) | (
                max_h > curr_size[1] - 3)):
        for l in list:
            img[l[0]][l[1]] = 255
    return img

def diagonal_feature(img):
    new_img = img.copy()
    new_img = cv2.resize(new_img, (60, 90), interpolation=cv2.INTER_LINEAR)
    # cv2.imshow('Inter', new_img)
    step = 10
    diag_features = []
    averages_features = []
    for y in range(0, 90, step):
        for x in range(0, 60, step):
            square = new_img[y:y + step, x:x + step]
            averages_features.append(np.average(square))
            list_diag = []
            for i in range(19): list_diag.append([])
            for i in range(10):
                for j in range(10):
                    list_diag[i + j].append(square[i][j])
            list_average = []
            for i in range(19):
                list_average.append(np.average(list_diag[i]))
            diag_features.append(np.average(list_average))
    return np.concatenate((diag_features, averages_features), axis=0)


def haar_features(img):
    new_img = img.copy()
    new_img = cv2.resize(new_img, (60, 90), interpolation=cv2.INTER_LINEAR)
    haar1 = np.average(new_img[0:45, 30:60] - new_img[45:90, 30:60])
    haar2 = np.average(new_img[0:45, 0:30] - new_img[45:90, 0:30])
    return haar1, haar2


def hist_features(img):
    new_img = img.copy()
    new_img = cv2.resize(new_img, (60, 90), interpolation=cv2.INTER_LINEAR)
    step = 10
    histogramm_features = []
    for y in range(0, 90, step):
        for x in range(0, 60, step):
            square = new_img[y:y + step, x:x + step]
            square_features = cv2.calcHist([square], [0], None, [9], [0, 256])
            for i in range(9):
                histogramm_features.append(square_features[i][0])
    return histogramm_features

def get_features(img):
    diag_features = diagonal_feature(img)
    haar1, haar2 = haar_features(img)
    histogramm_features = hist_features(img)
    features = [haar1, haar2]
    features = np.concatenate((features, np.concatenate((diag_features, histogramm_features), axis=0)), axis=0)
    return features

def test_preprocessing(original_img):
    grey_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2GRAY)
    rectangles = get_circles(grey_img)
    thresholds = []
    for i in range(len(rectangles)):
        rectangles[i] = cv2.Canny(rectangles[i], 100, 200)
        # cv2.imshow('Canny: {}'.format(i), rectangles[i])
        ret, rectangles[i] = cv2.threshold(rectangles[i], 127, 255, cv2.THRESH_BINARY_INV)
        # cv2.imshow('Thresh1: {}'.format(i), rectangles[i])
        # rectangles[i] = morphology(rectangles[i], cv2.MORPH_CLOSE)
        # cv2.imshow('Morph: {}'.format(i), rectangles[i])
        thresholds.append(adaptive_thresholding(rectangles[i]))
        cv2.imshow('Threshold: {}'.format(i), thresholds[i])

    res = []
    for i in range(len(thresholds)):
        curr_img = thresholds[i].copy()
        # delete contour
        part_1 = 0.025
        part_2 = 0.3
        res1 = curr_img.copy()
        res2 = curr_img.copy()
        new_img1 = res1
        new_img2 = res2
        # Delete components
        while (len(res1) != 0):
            res1 = delete_components(res1, 128, part_1, part_2)
            if (len(res1) != 0):
                new_img1 = res1.copy()
                # if (new_img1 != []):
                # cv2.imshow('Image1: {}'.format(i), new_img1)
        # Inverse the image and delete components
        ret, res2 = cv2.threshold(res2, 127, 255, cv2.THRESH_BINARY_INV)
        while (len(res2) != 0):
            res2 = delete_components(res2, 128, part_1, part_2)
            if (len(res2) != 0):
                new_img2 = res2.copy()
                # if (new_img2 != []):
                # cv2.imshow('Image2: {}'.format(i), new_img2)
        # Result
        res.append(2 * (cv2.threshold(new_img1, 192, 255, cv2.THRESH_BINARY_INV)[1] +
                        cv2.threshold(new_img2, 192, 255, cv2.THRESH_BINARY_INV)[1]))

    X = []
    for i in range(len(res)):
        X.append(get_features(res[i]))
    return X


def main():
    img = cv2.imread("Pic4.jpg")
    clf = joblib.load('classificator.pkl')
    x_test = test_preprocessing(img)
    y_test = []
    for i in range(len(x_test)):
        y_test.append(clf.predict(x_test[i]))

    y_sum=0
    for k in range(y_test.__len__()):
        y_sum=y_sum+int(y_test[k][0])
    print(y_sum)

if __name__ == '__main__':
    main()