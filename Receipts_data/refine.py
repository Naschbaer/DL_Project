import cv2
import os, fnmatch

listOfFiles = os.listdir('D:/hack/')
pattern = "test*"

for entry in listOfFiles:
    if fnmatch.fnmatch(entry, pattern):
        im = cv2.imread('D:/hack/' + entry)
        #im = cv2.GaussianBlur(im, (3, 3), 0)
        #h, w = im.shape[:2]
        print(entry)
        gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        t, ret = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        cv2.imwrite(entry, ret)

cv2.imshow('img', ret)
cv2.waitKey(0)
