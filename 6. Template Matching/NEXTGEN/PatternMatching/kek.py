import cv2
import numpy as np
import drawMatches as dm

def check_match(matches, threshold, txt):
    count = 0
    if (matches[0].distance < threshold):
        for i in range(0, len(matches)):
            if (int(matches[i].distance) <= threshold):
                count += 1
        return count


#########################################
#########################################
#########################################
# TODO put all that jazz inside function
# Ideal Logo
def load_etalon_query(_etalon_path,_query_path):
    _etalon = cv2.imread(_etalon_path)
    _query = cv2.imread(_query_path)

    _getalon = cv2.cvtColor(_etalon, cv2.COLOR_BGR2GRAY)
    _gquery = cv2.cvtColor(_query, cv2.COLOR_BGR2GRAY)
    # _getalon=eliminate_Noises(_getalon)
    # _gquery=eliminate_Noises(_gquery)

    # _getalon = cv2.Canny(_getalon,100,200)
    # _gquery = cv2.Canny(_gquery,100,200)
    return _etalon,_query,_getalon,_gquery


def keypoints_match(_etalon,_query):
    sift = cv2.SIFT()
    cv2.imshow(' ', _query)
    cv2.waitKey()

    _kpc, _desc = sift.detectAndCompute(_etalon, None)
    _kpq, _desq = sift.detectAndCompute(_query, None)

    FLANN_INDEX_KDTREE = 0
    _index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    _search_params = dict(checks=50)
    _flann = cv2.FlannBasedMatcher(_index_params, _search_params)
    _matches = _flann.knnMatch(_desc, _desq, 2)

    _good_matches = []
    for m, n in _matches:
        if m.distance <= 0.9 * n.distance:
            _good_matches.append(m)

    _src_pts = np.float32([_kpc[m.queryIdx].pt for m in _good_matches]).reshape(-1, 1, 2)
    _dst_pts = np.float32([_kpq[m.trainIdx].pt for m in _good_matches]).reshape(-1, 1, 2)

    return _kpc,_desc,_kpq,_desq,_src_pts,_dst_pts,_good_matches

def eliminate_Noises(image):
    # Construct threshhold image for coloritem, then perform
    # a series of dilation and erosions to remove any small
    # blobs left in the threshold image.

    ret, thresh = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)  # (thresh, 127,255, 0)

    thresh = cv2.erode(thresh, None, iterations=5)
    thresh = cv2.erode(thresh, None, iterations=5)
    thresh = cv2.dilate(thresh, None, iterations=5)
    return thresh

coke,query,C,gquery = load_etalon_query('pictures/C.jpg','pictures/C5.jpg')
kpc,desc,kpq,desq,src_pts,dst_pts,good = keypoints_match(C,gquery)
#
# img1 = cv2.drawKeypoints(C,kpc,C)
# img2 = cv2.drawKeypoints(query,kpq,query)

dm.drawMatches(C, kpc, gquery, kpq, good)


M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC)
# print 'm',M
# print 'mask',mask
h,w = C.shape
# print src_pts


pts = np.float32( [[0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1, 1, 2)
dst = cv2.perspectiveTransform(pts, M)
print dst.shape
print np.array(dst).flatten()[0]
dm.drawMatches(C, kpc, gquery, kpq, good)
# cv2.rectangle(gquery,[np.int32(dst[0][0])],[np.int32(dst[3][0])],thickness=-1,color = (255,0,0))
cv2.polylines(gquery, [np.int32(dst)], True,thickness= 4 ,color = (0,0,0))
#

# # #################
# # #################
# # #################
# kpc,desc,kpq,desq,src_pts,dst_pts,good = keypoints_match(C,gquery)
# M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,3.0)
# # print 'm',M
# # print 'mask',mask
# h,w = C.shape
# # print src_pts
#
#
# pts = np.float32([[0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1, 1, 2)
# dst = cv2.perspectiveTransform(pts, M)
# print dst
#
# # cv2.rectangle(gquery,[np.int32(dst[0][0])],[np.int32(dst[3][0])],thickness=-1,color = (255,0,0))
# cv2.polylines(gquery, [np.int32(dst)], True, 0, thickness=4)
#
# cv2.imshow("etalon", gquery)
# cv2.waitKey()
# #######
# #######
# #######
# kpc,desc,kpq,desq,src_pts,dst_pts,good = keypoints_match(C,gquery)
# M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,3.0)
# # print 'm',M
# # print 'mask',mask
# h,w = C.shape
# # print src_pts
#
#
# pts = np.float32([[0, 0], [0, h], [w, h], [w, 0]]).reshape(-1, 1, 2)
# dst = cv2.perspectiveTransform(pts, M)
# print dst
#
# # cv2.rectangle(gquery,[np.int32(dst[0][0])],[np.int32(dst[3][0])],thickness=-1,color = (255,0,0))
# cv2.polylines(gquery, [np.int32(dst)], True, 0, thickness=50)
#
# cv2.imshow("etalon", gquery)
# cv2.waitKey()
# #################
# #################
# while True:
#     kpc,desc,kpq,desq,src_pts,dst_pts,good = keypoints_match(C,gquery)
#     M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,3.0)
#     # print 'm',M
#     # print 'mask',mask
#     h,w = C.shape
#     # print src_pts
#
#
#     pts = np.float32([[0, 0], [0, h], [w, h], [w, 0]]).reshape(-1, 1, 2)
#     dst = cv2.perspectiveTransform(pts, M)
#     print dst
#
#     # cv2.rectangle(gquery,[np.int32(dst[0][0])],[np.int32(dst[3][0])],thickness=-1,color = (255,0,0))
#     cv2.fillPoly(gquery, [np.int32(dst)],color = (255,255,255))
#
#     cv2.imshow("etalon", gquery)
#     cv2.waitKey()
# #######
# #######
# #######
#
#
# dm.drawMatches(C, kpc, gquery, kpq, good)
