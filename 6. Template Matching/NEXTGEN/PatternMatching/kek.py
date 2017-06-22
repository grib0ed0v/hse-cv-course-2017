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
        if m.distance < 1 * n.distance:
            _good_matches.append(m)

    _src_pts = np.float32([_kpc[m.queryIdx].pt for m in _good_matches]).reshape(-1, 1, 2)
    _dst_pts = np.float32([_kpq[m.trainIdx].pt for m in _good_matches]).reshape(-1, 1, 2)

    return _kpc,_desc,_kpq,_desq,_src_pts,_dst_pts,_good_matches


coke,query,C,gquery = load_etalon_query('pictures/C.png','pictures/C7.jpg')
kpc,desc,kpq,desq,src_pts,dst_pts,good = keypoints_match(C,gquery)

img1 = cv2.drawKeypoints(C,kpc,C)
img2 = cv2.drawKeypoints(query,kpq,query)



M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,3.0)
# print 'm',M
# print 'mask',mask
h,w = C.shape
# print src_pts


pts = np.float32([[0, 0], [0, h], [w, h], [w, 0]]).reshape(-1, 1, 2)
dst = cv2.perspectiveTransform(pts, M)
print dst
dm.drawMatches(C, kpc, gquery, kpq, good)
# cv2.rectangle(gquery,[np.int32(dst[0][0])],[np.int32(dst[3][0])],thickness=-1,color = (255,0,0))
cv2.fillPoly(gquery, [np.int32(dst)], color = (255,255,255))

cv2.imshow("etalon", gquery)
cv2.waitKey()
kpc,desc,kpq,desq,src_pts,dst_pts,good = keypoints_match(C,gquery)
M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,3.0)
# print 'm',M
# print 'mask',mask
h,w = C.shape
# print src_pts
dm.drawMatches(C, kpc, gquery, kpq, good)


pts = np.float32([[0, 0], [0, h], [w, h], [w, 0]]).reshape(-1, 1, 2)
dst = cv2.perspectiveTransform(pts, M)
print dst

# cv2.rectangle(gquery,[np.int32(dst[0][0])],[np.int32(dst[3][0])],thickness=-1,color = (255,0,0))
cv2.polylines(gquery, [np.int32(dst)], True, 0, thickness=20)

#################
#################
#################
kpc,desc,kpq,desq,src_pts,dst_pts,good = keypoints_match(C,gquery)
M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,3.0)
# print 'm',M
# print 'mask',mask
h,w = C.shape
# print src_pts


pts = np.float32([[0, 0], [0, h], [w, h], [w, 0]]).reshape(-1, 1, 2)
dst = cv2.perspectiveTransform(pts, M)
print dst

# cv2.rectangle(gquery,[np.int32(dst[0][0])],[np.int32(dst[3][0])],thickness=-1,color = (255,0,0))
cv2.polylines(gquery, [np.int32(dst)], True, 0, thickness=40)

cv2.imshow("etalon", gquery)
cv2.waitKey()
#######
#######
#######
kpc,desc,kpq,desq,src_pts,dst_pts,good = keypoints_match(C,gquery)
M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,3.0)
# print 'm',M
# print 'mask',mask
h,w = C.shape
# print src_pts


pts = np.float32([[0, 0], [0, h], [w, h], [w, 0]]).reshape(-1, 1, 2)
dst = cv2.perspectiveTransform(pts, M)
print dst

# cv2.rectangle(gquery,[np.int32(dst[0][0])],[np.int32(dst[3][0])],thickness=-1,color = (255,0,0))
cv2.polylines(gquery, [np.int32(dst)], True, 0, thickness=50)

cv2.imshow("etalon", gquery)
cv2.waitKey()
#################
#################
kpc,desc,kpq,desq,src_pts,dst_pts,good = keypoints_match(C,gquery)
M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,3.0)
# print 'm',M
# print 'mask',mask
h,w = C.shape
# print src_pts


pts = np.float32([[0, 0], [0, h], [w, h], [w, 0]]).reshape(-1, 1, 2)
dst = cv2.perspectiveTransform(pts, M)
print dst

# cv2.rectangle(gquery,[np.int32(dst[0][0])],[np.int32(dst[3][0])],thickness=-1,color = (255,0,0))
cv2.polylines(gquery, [np.int32(dst)], True, 0, thickness=2)

cv2.imshow("etalon", gquery)
cv2.waitKey()
#######
#######
#######


dm.drawMatches(C, kpc, gquery, kpq, good)
