import numpy as np
import cv2
from sklearn.cluster import MeanShift, estimate_bandwidth
from sklearn.cluster import DBSCAN
from sklearn.cluster import AffinityPropagation
from sklearn.cluster import Birch
import matplotlib.pyplot as plt
import random

def random_color():
    levels = range(100,256,100)
    return tuple(random.choice(levels) for _ in range(3))



def dbscan_clustering(points):
    db = DBSCAN(eps=0.6, min_samples=3, algorithm='brute').fit(points)
    core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
    core_samples_mask[db.core_sample_indices_] = True
    labels = db.labels_
    # Number of clusters in labels, ignoring noise if present.
    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
    print('Estimated number of clusters: %d' % n_clusters_)
    return labels, n_clusters_

def meanshift_clustering(points):
    point1 = np.array(points)
    bandwidth = estimate_bandwidth(point1, quantile=0.1, n_samples=1000)
    if bandwidth <= 0.0 or bandwidth is None:
        return None, None
    ms = MeanShift(bandwidth=bandwidth, bin_seeding=True)
    ms.fit(points)
    labels = ms.labels_

    # Number of clusters in labels, ignoring noise if present.
    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
    print('Estimated number of clusters: %d' % n_clusters_)
    return labels, n_clusters_

def affpropagation_clustering(points):
    af = AffinityPropagation(preference=-50000).fit(points)
    cluster_centers_indices = af.cluster_centers_indices_
    labels = af.labels_
    n_clusters_ = len(cluster_centers_indices)
    print('Estimated number of clusters: %d' % n_clusters_)
    return points, labels

def birch_clustering(points, clust_num):
    br = Birch(n_clusters=clust_num).fit(points)
    labels = br.labels_
    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
    print('Estimated number of clusters: %d' % n_clusters_)
    return points, labels


def draw_clusters(img, points, labels):
    unique_labels = set(labels)
    colors = plt.cm.Spectral(np.linspace(0, 1, len(unique_labels)))
    for k, col in zip(unique_labels, colors):
        if k == -1:
            # Black used for noise.
            col = 'k'
        class_member_mask = (labels == k)
        xy = [points[I] for I in range(len(labels)) if class_member_mask[I] == True]
        xs = [x[0] for x in xy]
        ys = [x[1] for x in xy]
        color = random_color()
        for i in range(len(xs)):
            cv2.circle(img, (int(xs[i]), int(ys[i])), 5, color, 3)
    cv2.imshow('ololo', img)
    cv2.waitKey(0)