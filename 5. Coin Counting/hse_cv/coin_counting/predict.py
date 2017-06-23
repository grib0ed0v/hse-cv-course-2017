import cv2
from sklearn.externals import joblib
from .image_preprocessing import get_features
from ..config import config


def predict_sum(image_path):
    print('Read image...')
    img = cv2.imread(image_path)
    print('Image read. Loading classifier...')
    clf = joblib.load(config['CLASSIFIER']['TrainedModelPath'])
    print('Getting features...')
    x_test, img_with_coin = get_features(img)
    y_sum = 0
    for i in range(len(x_test)):
        coin_value = int(clf.predict([x_test[i]])[0])
        y_sum = y_sum + coin_value
    print('Sum = {}'.format(y_sum))
    color = (255, 0, 0)
    font = cv2.FONT_HERSHEY_DUPLEX
    cv2.putText(img_with_coin, str(y_sum), (150, 200), font, 6, color, 2)

    cv2.imshow("Result", img_with_coin)
    cv2.waitKey()
