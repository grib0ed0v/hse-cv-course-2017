import argparse
import cv2
import numpy as np
import drawMatches as dm
import anotherCluster as acl
import copy
from utils import *

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--template', help="template file")
    parser.add_argument('--file', help="target file")
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()
    if args.template == None or args.file == None:
        raise IOError("Set please template and target files") 
    coke, query, C, gquery = load_etalon_query(args.template, args.file)
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

    list_keypoints = get_points(kpq)
    list_keypoints_int = cast_list_to_int(list_keypoints)
    iiiimg1 = deepcopy(query2)
    labels_cluster_keypoints, number_cluster_keypoints = acl.meanshift_clustering(list_keypoints, quantile=0.05, n_samples=1000)


    matches = matching(desc, desq)
    mkpc, mkpq = get_mached_kpoints(kpc, kpq, matches)
    list_match = get_points(mkpq)
    list_match_int = cast_list_to_int(list_match)
    labelsq, nclq = acl.meanshift_clustering(list_match, quantile=0.15, n_samples=500)
    iiiimg = deepcopy(query2)

    iiiim3 = deepcopy(query2)
    epsilon = 1
    clusters_of_matching = {} # keys - cluster label in matching clusters_of_matching,
							  # values - cluster labels of global clasters
    for i in range(nclq):
        clusters_of_matching[i] = []
        for j in [k for k in range(len(labelsq)) if labelsq[k] == i]:
            tmp = list_match_int[j]
            for f in range(tmp[0]-epsilon,tmp[0]+epsilon):
                find = False
                for ff in range(tmp[1]-epsilon,tmp[1]+epsilon):
                    if (f,ff) in list_keypoints_int:
                        tmp = (f,ff)
                        find = True
                        break
                if find:
                    break
            if tmp in list_keypoints_int:
                currentCluster = labels_cluster_keypoints[list_keypoints_int.index(tmp)]
                for n in [k for k in range(len(labels_cluster_keypoints)) if labels_cluster_keypoints[k] == currentCluster]:
                    cv2.circle(iiiim3, list_keypoints_int[n], 5, (255,255,255), 3)
                clusters_of_matching[i].append(currentCluster)
        clusters_of_matching[i] = set(clusters_of_matching[i])
    # cv2.imshow('ololoq', iiiim3)
    # cv2.waitKey()
    acl.draw_clusters(iiiimg1, list_keypoints_int, labels_cluster_keypoints,"olololololo")
    acl.draw_clusters(iiiimg, list_match, labelsq)
    print clusters_of_matching

    for i in clusters_of_matching:
        tmp = desq[np.array([k for j in clusters_of_matching[i] for k in range(len(labels_cluster_keypoints)) if j == labels_cluster_keypoints[k]])]
        voop = []
        for z in [k for j in clusters_of_matching[i] for k in range(len(labels_cluster_keypoints)) if j == labels_cluster_keypoints[k]]:
            voop.append(kpq[z])
        matches = matching(desc, tmp)
        mkpc, mkpq = get_mached_kpoints(kpc, voop, matches)
        list_match = get_points(mkpq)
        iiiimg4 = deepcopy(query2)
        for n in list_match:
            cv2.circle(iiiimg4, (int(n[0]),int(n[1])), 5, (255,255,255), 3)
        cv2.imshow('olasoloq', iiiimg4)
        cv2.waitKey()
