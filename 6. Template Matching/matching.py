import cv2
import numpy as np
import drawMatches as dm
import anotherCluster as acl
import copy
from utils import *

coke, query, C, gquery = load_etalon_query('pictures/P.jpg', 'pictures/P9.jpg')
query2 = copy.copy(query)
query3 = copy.copy(query)
kpc, desc, kpq, desq, src_pts, dst_pts, matches = keypoints_match(C, gquery)

img1 = cv2.drawKeypoints(C, kpc, C)
img2 = cv2.drawKeypoints(query, kpq, query)

M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 3.0)
# print 'm',M
# print 'mask',mask
h, w = gquery.shape

pts = np.float32([[0, 0], [0, h], [w, h], [w, 0]]).reshape(-1, 1, 2)
dst = cv2.perspectiveTransform(pts, M)


cv2.polylines(gquery, [np.int32(dst)], True, 0, 2)

#cv2.imshow("etalon", gquery)
#cv2.imshow("query", img2)
#cv2.waitKey()


def iterativ_mathcing(template, source):

    kp_template, descr_template, kp_source, descr_source, src_pts, dst_pts, matches = keypoints_match(template, source)

    while True:
        # matching key points

        matches = matching(descr_template, descr_source)

        mkpc, mkpq = get_mached_kpoints(kp_template, kp_source, matches)
        mpointsq = get_points(mkpq)

        # clustering matched key points
        labelsq, nclq = acl.meanshift_clustering(mpointsq)
        if labelsq is None:
            return
        # find cluster with maximum elements
        maxcluster, maximum = max_cluster(labelsq, nclq)
        if maximum < 3:
            return

        # drawing clusters
        iiiimg = deepcopy(query2)
        acl.draw_clusters(iiiimg, mpointsq, labelsq)

        # delete cluster from source key points

        kp_source, descr_source = \
            delete_key_points(kp_source, descr_source, mkpq, labelsq, maxcluster)



iterativ_mathcing(coke, query)

# good = get_good_matches(matches)
# dm.drawMatches(C, kpc, gquery, kpq, good)
#
# goodpoints = get_points_from_matches(kpq, good)
# points = get_points(kpq)
# pointsm, labelsm, n = acl.meanshift_clustering(goodpoints)
# pointsq, labelsq = acl.birch_clustering(points, n)
# acl.draw_clusters(query2, pointsq, labelsq)
# acl.draw_clusters(query3, pointsm, labelsm)
