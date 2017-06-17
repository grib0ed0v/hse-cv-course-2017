import numpy as np
import cv2
from sklearn.cluster import MeanShift, estimate_bandwidth
import matplotlib.pyplot as plt
import random

def random_color():
    levels = range(100,256,100)
    return tuple(random.choice(levels) for _ in range(3))

def clusterGoodPoints(img, kp):
    points = []
    for mat in kp:
        (x, y) = mat.pt
        points.append((x,y))
    point1 = np.array(points)
    bandwidth = estimate_bandwidth(point1, quantile=0.1, n_samples=500)
    ms = MeanShift(bandwidth=bandwidth, bin_seeding=True)
    ms.fit(points)
    labels = ms.labels_
    cluster_centers = ms.cluster_centers_

    labels_unique = np.unique(labels)
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

        xy = [points[I] for I in range(len(labels)) if class_member_mask[I] == True]
        print str(k) + str(xy)
        xs = [x[0] for x in xy]
        ys = [x[1] for x in xy]

        color = random_color()
        print color
        for i in range(len(xs)):
            cv2.circle(img, (int(xs[i]), int(ys[i])), 10, color, 1)
    cv2.imshow('ololo', img)
    cv2.waitKey(0)
    return points