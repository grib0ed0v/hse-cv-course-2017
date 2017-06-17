import numpy as np
import cv2
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.datasets.samples_generator import make_blobs
from sklearn.preprocessing import StandardScaler

def clusterGoodPoints(img1, kp1, img2, kp2, matches):
    rows2 = img2.shape[0]
    cols2 = img2.shape[1]
    points = []
    for mat in kp2:
        # Get the matching keypoints for each of the images
        #img2_idx = mat.trainIdx

        # x - columns
        # y - rows
        #(x2,y2) = kp2[img2_idx].pt
        (x2, y2) = mat.pt
        points.append((x2,y2))
    db = DBSCAN(eps=0.3, min_samples=3).fit(points)
    core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
    core_samples_mask[db.core_sample_indices_] = True
    labels = db.labels_

    # Number of clusters in labels, ignoring noise if present.
    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)

    print('Estimated number of clusters: %d' % n_clusters_)

    #drawing clusters
    unique_labels = set(labels)
    colors = plt.cm.Spectral(np.linspace(0, 1, len(unique_labels)))
    for k, col in zip(unique_labels, colors):
        if k == -1:
            # Black used for noise.
            col = 'k'

        class_member_mask = (labels == k)

        #print int(col[0]*255)
        #print type(col[0])

        #xy = points[class_member_mask & core_samples_mask]
        #print len(labels)
        xy = [points[I] for I in range(len(labels)) if class_member_mask[I] == True]
        #xy = [points[I] for I in range(len(labels)) if class_member_mask[I] == True and core_samples_mask[I] == True]
        print str(k) + str(xy)
        xs = [x[0] for x in xy]
        ys = [x[1] for x in xy]
        #plt.plot(xs, ys, 'o', markerfacecolor=col,
        #         markeredgecolor='k', markersize=14)

        for i in range(len(xs)):
            cv2.circle(img1, (int(xs[i]), int(ys[i])), 10, (int(k*10), 0, 0), 1)
    cv2.imshow('ololo', img1)
    cv2.waitKey(0)
    return points