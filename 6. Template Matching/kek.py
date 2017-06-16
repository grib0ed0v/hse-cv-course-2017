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
    else:
        print str(txt) + " not found"

#########################################
#########################################
#########################################
# TODO put all that jazz inside function
# Ideal Logo
coke = cv2.imread('pictures/C.jpg')

# query image
query = cv2.imread('pictures/C2.jpg')
gquery = cv2.cvtColor(query, cv2.COLOR_BGR2GRAY)
C = cv2.cvtColor(coke, cv2.COLOR_BGR2GRAY)

sift = cv2.SIFT()
kpc, desc = sift.detectAndCompute(C, None)
kpq, desq = sift.detectAndCompute(gquery, None)

img1 = cv2.drawKeypoints(C,kpc,C)
img2 = cv2.drawKeypoints(query,kpq,query)

#########################################
#########################################
#########################################
# TODO put all that jazz inside get_matches(desc,desq) function

FLANN_INDEX_KDTREE = 0
index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 10)
search_params = dict(checks = 500)
flann = cv2.FlannBasedMatcher(index_params, search_params)
matches = flann.knnMatch(desc,desq,2)

print type(matches)
good = []
for m,n in matches:
    if m.distance < 0.9*n.distance:
        good.append(m)
#########################################
#########################################
#########################################
src_pts = np.float32([kpc[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
dst_pts = np.float32([kpq[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

print type(good)


M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,1.0)
print 'm',M
print 'mask',mask
h,w = gquery.shape


pts = np.float32([[0, 0], [0, h], [w, h], [w, 0]]).reshape(-1, 1, 2)
dst = cv2.perspectiveTransform(pts, M)
cv2.polylines(gquery, [np.int32(dst)], True, 0, 2)

cv2.imshow("etalon", gquery)
cv2.imshow("query", img2)
cv2.waitKey()

dm.drawMatches(C, kpc, gquery, kpq, good)
