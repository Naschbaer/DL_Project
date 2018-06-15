import cv2
import os, fnmatch

listOfFiles = os.listdir('D:/hack/Receipts_data')
pattern = "test_2*"

for entry in listOfFiles:
    if fnmatch.fnmatch(entry, pattern):
        im = cv2.imread('D:/hack/Receipts_data/' + entry)
        #h, w, c = im.shape[:3]
        #print(h,w,c)
        gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        t, ret = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        ret = cv2.GaussianBlur(ret, (3, 3), 0)
        cv2.imwrite('Receipts_data/Bahaar/'+entry, ret)

cv2.imshow('img', ret)
cv2.waitKey(0)
