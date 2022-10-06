import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt

# img1 = cv.imread('kpick/apps/fpcb/test_images/box.png', cv.IMREAD_GRAYSCALE)  # queryImage
# img2 = cv.imread('kpick/apps/fpcb/test_images/box_in_scene.png', cv.IMREAD_GRAYSCALE)  # trainImage
img1 = cv.imread('data/apps/fpcb/testset2/000.png', cv.IMREAD_GRAYSCALE)  # queryImage
img2 = cv.imread('data/apps/fpcb/testset2/003.png', cv.IMREAD_GRAYSCALE)  # trainImage


# Initiate SIFT detector
sift = cv.SIFT_create()
# find the keypoints and descriptors with SIFT
kp1, des1 = sift.detectAndCompute(img1, None)
kp2, des2 = sift.detectAndCompute(img2, None)




# FLANN parameters
FLANN_INDEX_KDTREE = 1
index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
search_params = dict(checks=20)  # or pass empty dictionary
flann = cv.FlannBasedMatcher(index_params, search_params)
matches = flann.knnMatch(des1, des2, k=2)
# Need to draw only good matches, so create a mask
matchesMask = [[0, 0] for i in range(len(matches))]
# ratio test as per Lowe's paper
for i, (m, n) in enumerate(matches):
    if m.distance < 0.7 * n.distance:
        matchesMask[i] = [1, 0]
draw_params = dict(matchColor=(0, 255, 0),
                   singlePointColor=(255, 0, 0),
                   matchesMask=matchesMask,
                   flags=cv.DrawMatchesFlags_DEFAULT)
img3 = cv.drawMatchesKnn(img1, kp1, img2, kp2, matches, None, **draw_params)


# # BFMatcher with default params
# bf = cv.BFMatcher()
# matches = bf.knnMatch(des1,des2,k=2)
# # Apply ratio test
# good = []
# for m,n in matches:
#     if m.distance < 0.75*n.distance:
#         good.append([m])
# # cv.drawMatchesKnn expects list of lists as matches.
# img3 = cv.drawMatchesKnn(img1,kp1,img2,kp2,good,None,flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)



# plt.imshow(img3, ), plt.show()
cv.imshow('im', img3[:,:,::-1])
cv.waitKey()