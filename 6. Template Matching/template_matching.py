import argparse
import cv2
import numpy as np
import drawMatches as dm
import anotherCluster as acl
import copy
from utils import *

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--template_file','-t', default='pictures/C.png',help="template file")
    parser.add_argument('--target_file', '-f', default = 'pictures/C1.jpg',help= "target file")
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()
    if args.template == None or args.file == None:
        raise IOError("Set please template and target files")
    coke, query, C, gquery = load_etalon_query(args.template_file, args.target_file)

    kpc, desc, kpq, desq, src_pts, dst_pts, matches = keypoints_match(C, gquery)
    list_keypoints = get_points(kpq)
    list_keypoints_int = cast_list_to_int(list_keypoints)


    matches = matching(desc, desq)
    mkpc, mkpq = get_mached_kpoints(kpc, kpq, matches)
    list_match = get_points(mkpq)
    list_match_int = cast_list_to_int(list_match)
    labelsq, nclq = acl.meanshift_clustering(list_match, quantile=0.15, n_samples=500)
    lines_list = []



    gTempl = deepcopy(C)
    queryImg = deepcopy(gquery)
    for i in xrange(nclq):
        kpc, desc, kpq, desq, src_pts, dst_pts, matches = keypoints_match(gTempl, queryImg)
        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        h,w = C.shape
        pts = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)
        dst = cv2.perspectiveTransform(pts,M)
        check = np.array(dst).flatten()
        check = check + abs(min(check))
        if len(matches) > (len(kpc)*1/4):
            lines_list.append(dst)
            cv2.fillPoly(queryImg,[np.int32(dst)],color = (255,255,255))


    for i in lines_list:
        cv2.polylines(query, [np.int32(i)], True, thickness=4, color=(0, 255, 0))
    cv2.imshow('result', query)
    cv2.waitKey()



