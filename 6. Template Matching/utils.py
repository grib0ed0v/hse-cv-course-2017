import cv2
import numpy as np
from copy import deepcopy
from sklearn.cluster import DBSCAN


def check_match(matches, threshold, txt):
    count = 0
    if (matches[0].distance < threshold):
        for i in range(0, len(matches)):
            if (int(matches[i].distance) <= threshold):
                count += 1
        return count
    else:
        print str(txt) + " not found"


def load_etalon_query(_etalon_path, _query_path):
    _etalon = cv2.imread(_etalon_path)
    _query = cv2.imread(_query_path)

    _getalon = cv2.cvtColor(_etalon, cv2.COLOR_BGR2GRAY)
    _gquery = cv2.cvtColor(_query, cv2.COLOR_BGR2GRAY)
    return _etalon, _query, _getalon, _gquery


def matching(descr_template, descr_source):
    FLANN_INDEX_KDTREE = 0
    _index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=10)
    _search_params = dict(checks=500)
    _flann = cv2.FlannBasedMatcher(_index_params, _search_params)
    _matches = _flann.knnMatch(descr_template, descr_source, 2)
    # matches = []
    # for m, n in _matches:
    #     matches.append(m)
    return get_good_matches(_matches)


def keypoints_match(_etalon, _query):
    sift = cv2.SIFT()
    _kpc, _desc = sift.detectAndCompute(_etalon, None)
    _kpq, _desq = sift.detectAndCompute(_query, None)

    matches = matching(_desc, _desq)

    _src_pts = np.float32([_kpc[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    _dst_pts = np.float32([_kpq[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

    return _kpc, _desc, _kpq, _desq, _src_pts, _dst_pts, matches


def get_good_matches(matches):
    _good_matches = []
    for m, n in matches:
        if m.distance < 0.9 * n.distance:
            _good_matches.append(m)
    return  _good_matches


def get_mached_kpoints(src_kpoints, dst_kpoints, matches):
    src_kp = np.array([src_kpoints[m.queryIdx] for m in matches])
    dst_kp = np.array([dst_kpoints[m.trainIdx] for m in matches])
    return src_kp, dst_kp


def get_points_from_matches(kp, matches):
    points = []
    for mat in matches:
        # Get the matching keypoints for each of the images
        img2_idx = mat.trainIdx

        # x - columns
        # y - rows
        (x2, y2) = kp[img2_idx].pt
        points.append((x2, y2))
    return points


def get_points(kp):
    points = []
    for mat in kp:
        (x, y) = mat.pt
        points.append((x, y))
    return points


def remove_cluster(kpoints, descriptors, labels, clremoved):
    """
    Remove key points which belongs to clremoved cluster
    
    :param kpoints(np.array): key points  
    :param labels(np.array): labels of clusters for each key point 
    :param clremoved(int): number of cluster that will be removed 
    :return: 
    """

    removed = list()
    for i in range(len(labels)):
        if labels[i] == clremoved:
            removed.append(i)
    return np.delete(kpoints, removed), np.delete(descriptors, removed), np.delete(labels, removed)


def clsplite(kpoints, labels, nclusters):
    result = [[] for _ in range(nclusters)]
    for i in range(nclusters):
        result[i] = [kpoints[j] for j in range(len(labels)) if labels[j] == i]

    return np.array(result)


def max_cluster(labels, nclusters):
    result = -1
    maximum = -1
    labels_list = list(labels)
    for i in range(nclusters):
        if maximum < labels_list.count(i):
            maximum = labels_list.count(i)
            result = i

    return result, maximum


def delete_key_points(key_points, descriptors, matched_kp, labels, clremoved):
    removed = list()
    for i in range(len(labels)):
        if labels[i] == clremoved:
            removed.append(matched_kp[i])

    remove_indexes = list()

    for item in removed:
        remove_indexes.append(list(key_points).index(item))
    remove_indexes.sort()

    return np.delete(key_points, remove_indexes), np.delete(descriptors, remove_indexes, axis=0)
