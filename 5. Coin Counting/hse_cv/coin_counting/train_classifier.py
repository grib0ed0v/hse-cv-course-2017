import cv2
import os
from sklearn import svm
from sklearn.externals import joblib
from .image_preprocessing import get_features
from ..config import config


def train_samples(path):
    labels = []
    x = []
    for folder_name in os.listdir(path):
        # Process only folders
        if os.path.isdir(os.path.join(path, folder_name)):
            for img_filename in os.listdir(os.path.join(path, folder_name)):
                img = cv2.imread(os.path.join(path, folder_name, img_filename))
                img_x = get_features(img, train_mode=True)
                if img_x is not None:
                    x.append(img_x)
                    labels.append(folder_name)
    return x, labels


def train_model(train_images_path):
    print('Getting features from training images...')
    x, y = train_samples(train_images_path)
    print('Finished getting features. Creating and training model...')
    clf = svm.SVC(kernel=config['CLASSIFIER']['kernel'], C=int(config['CLASSIFIER']['C']),
                  gamma=float(config['CLASSIFIER']['Gamma']))
    clf.fit(x, y)
    print('Finished training model. Dump it to file...')
    joblib.dump(clf, config['CLASSIFIER']['TrainedModelPath'])
    print('Dump trained model to file. Finish.')
