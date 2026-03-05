import cv2
import numpy as np 

width , height = 300, 400

img = cv2.imread("resources/docs.jpg")

pts1 = np.float32([[723,25],[1109,18],[748,557],[1228,545]])
pts2 = np.float32([[0,0], [width, 0], [0,height], [width, height]])

metrix = cv2.getPerspectiveTransform(pts1, pts2)
img_out = cv2.warpPerspective(img, metrix, (width, height))

cv2.imshow('docs', img)
cv2.imshow('docs_warp', img_out)


cv2.imshow('docs', img)
cv2.waitKey(0)