import numpy as np
import cv2

def houg():
    img_path = 'train_img_0.12/5/DSC00804.JPG'
    img = cv2.imread(img_path, 0)
    img = cv2.medianBlur(img, 5)
    cimg = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    circles = cv2.HoughCircles(img, cv2.HOUGH_GRADIENT, 1, 20,
                               param1=50, param2=30, minRadius=50, maxRadius=0)

    circles = np.uint16(np.around(circles))
    for i in circles[0, :]:
        # draw the outer circle
        cv2.circle(cimg, (i[0], i[1]), i[2], (0, 255, 0), 2)
        # draw the center of the circle
        cv2.circle(cimg, (i[0], i[1]), 2, (0, 0, 255), 3)

    cv2.imshow('detected circles', cimg)
    cv2.waitKey(0) & 0xFF
    cv2.destroyAllWindows()

def circle_test():
    #img_path = 'train_img_0.12/1/DSC01279.JPG'
    img_path = 'train_img_0.12/5/DSC00804.JPG'
    # img_path = 'test_images/Picture1.jpg'
    roi = cv2.imread(img_path)
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

    gray_blur = cv2.GaussianBlur(gray, (35, 35), 0)
    thresh = cv2.adaptiveThreshold(gray_blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY_INV, 11, 1)

    kernel = np.ones((3, 3), np.uint8)
    closing = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE,
                               kernel, iterations=4)

    cont_img = closing.copy()
    im2, contours, hierarchy = cv2.findContours(cont_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    max_cnt = max(contours, key=lambda cnt: cv2.contourArea(cnt))

    #for cnt in contours:
        #area = cv2.contourArea(cnt)
        #if area < 2000 or area > 4000:
        #    continue

        #if len(cnt) < 5:
        #    continue

    ellipse = cv2.fitEllipse(max_cnt)
    x, y, w, h = cv2.boundingRect(max_cnt)

    mask = np.zeros(gray.shape, np.uint8)
    cv2.ellipse(mask, ellipse, (255), cv2.FILLED)
    # cv2.imshow('Circle', mask)
    # print(roi.shape, circle_img.shape)
    # mask_inv = 255 - mask
    res = mask*gray
    # res = res[y:y+h,x:x+h]
    res = res[~np.all(res == 0, axis=1)]
    res = res[:, ~np.all(res == 0, axis=0)]
    res = cv2.equalizeHist(res)
    edges = cv2.Canny(res,240, 600)
    edges = cv2.dilate(edges, (3, 3))
    cv2.imshow("Res", res)
    cv2.imshow("Edges", edges)
    # cv2.imshow("Morphological Closing", closing)
    # cv2.imshow("Adaptive Thresholding", thresh)
    # cv2.imshow('Contours', roi)
    cv2.waitKey(0) & 0xFF
    cv2.destroyAllWindows()

if __name__ == '__main__':
    # houg()
    circle_test()

