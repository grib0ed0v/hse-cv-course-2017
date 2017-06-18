import cv2
import numpy as np
import drawMatches as dm
import anotherCluster as acl
import copy


def get_points_from_matches(kp, matches):
    points = []
    for mat in matches:
        # Get the matching keypoints for each of the images
        img2_idx = mat.trainIdx

        # x - columns
        # y - rows
        (x2,y2) = kp[img2_idx].pt
        points.append((x2, y2))
    return points


def get_points(kp):
    points = []
    for mat in kp:
        (x, y) = mat.pt
        points.append((x, y))
    return points


def check_match(matches, threshold, txt):
    count = 0
    if (matches[0].distance < threshold):
        for i in range(0, len(matches)):
            if (int(matches[i].distance) <= threshold):
                count += 1
        return count
    else:
        print str(txt) + " not found"


def load_etalon_query(_etalon_path,_query_path):
    _etalon = cv2.imread(_etalon_path)
    _query = cv2.imread(_query_path)

    _getalon = cv2.cvtColor(_etalon, cv2.COLOR_BGR2GRAY)
    _gquery = cv2.cvtColor(_query, cv2.COLOR_BGR2GRAY)
    return _etalon,_query,_getalon,_gquery


def keypoints_match(_etalon,_query):
    sift = cv2.SIFT()
    _kpc, _desc = sift.detectAndCompute(_etalon, None)
    _kpq, _desq = sift.detectAndCompute(_query, None)

    FLANN_INDEX_KDTREE = 0
    _index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=10)
    _search_params = dict(checks=500)
    _flann = cv2.FlannBasedMatcher(_index_params, _search_params)
    _matches = _flann.knnMatch(_desc, _desq, 2)

    _good_matches = []
    for m, n in _matches:
        if m.distance < 0.9 * n.distance:
            _good_matches.append(m)

    _src_pts = np.float32([_kpc[m.queryIdx].pt for m in _good_matches]).reshape(-1, 1, 2)
    _dst_pts = np.float32([_kpq[m.trainIdx].pt for m in _good_matches]).reshape(-1, 1, 2)

    return _kpc,_desc,_kpq,_desq,_src_pts,_dst_pts,_good_matches

coke,query,C,gquery = load_etalon_query('pictures/C.jpg','pictures/C8.jpg')
query2 = copy.copy(query)
query3 = copy.copy(query)
kpc,desc,kpq,desq,src_pts,dst_pts,good = keypoints_match(C,gquery)

img1 = cv2.drawKeypoints(C,kpc,C)
img2 = cv2.drawKeypoints(query,kpq,query)




M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,3.0)
# print 'm',M
# print 'mask',mask
h,w = gquery.shape


pts = np.float32([[0, 0], [0, h], [w, h], [w, 0]]).reshape(-1, 1, 2)
print pts
dst = cv2.perspectiveTransform(pts, M)
cv2.polylines(gquery, [np.int32(dst)], True, 0, 2)

cv2.imshow("etalon", gquery)
cv2.imshow("query", img2)
cv2.waitKey()

dm.drawMatches(C, kpc, gquery, kpq, good)

goodpoints = get_points_from_matches(kpq, good)
points = get_points(kpq)
pointsm, labelsm, n = acl.meanshift_clustering(goodpoints)
pointsq, labelsq = acl.birch_clustering(points, n)
acl.draw_clusters(query2, pointsq, labelsq)
acl.draw_clusters(query3, pointsm, labelsm)
