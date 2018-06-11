import numpy as np
import cv2

im = cv2.imread('D:/hack/img2.jpg')

height, width = im.shape[:2]
img = cv2.resize(im, (1000, 1500), interpolation=cv2.INTER_CUBIC)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

#gray = cv2.GaussianBlur(gray, (5, 5), 0)
edges = cv2.Canny(gray, 100, 250)

krnl = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (1, 2))
#t, ret = cv2.threshold(gray, 0, 200, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, krnl)

im, contours, hierarchy = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

idx = 0
x = np.zeros(len(contours)).astype(int)
y = np.zeros(len(contours)).astype(int)
w = np.zeros(len(contours)).astype(int)
h = np.zeros(len(contours)).astype(int)
points = np.zeros((len(contours), 3))
op = np.zeros((28, 28, len(contours)))

for cnt in contours:
    x[idx], y[idx], w[idx], h[idx] = cv2.boundingRect(cnt)
    #cv2.drawContours(edges, contours, idx, [255, 255, 255], 2)

    if (x[idx] > 0 and y[idx] > 0 and x[idx] < width and y[idx] < height and w[idx] > 3 and h[idx] > 15 and w[idx] < 100 and h[idx] < 50):
        #print(x[idx]-5, y[idx]-10, w[idx], h[idx], idx)
        #cv2.rectangle(gray, (x[idx]-5, y[idx]-10), (x[idx]+w[idx]+5, y[idx]+h[idx]+10), (200, 100, 50), 1)
        points[idx] = [x[idx]-5, y[idx]-10, idx]
        roi = gray[y[idx]-10:y[idx]+h[idx]+10, x[idx]-5:x[idx]+w[idx]+5]
        op[:, :, idx] = cv2.resize(roi, (28, 28), interpolation=cv2.INTER_CUBIC)

    idx += 1

points = np.delete(points, np.where(points[:, 0:1] == 0), axis=0)
points = points[points[:, 1].argsort()]
ind = np.zeros(len(points))

for i in range(1, len(points)):
    if abs(points[i-1, 1] - points[i, 1]) < 25:
        points[i, 1] = points[i-1, 1]
    else:
        ind[i] = i

ind = np.delete(ind, np.where(ind == 0), axis=0)
points_rearranged = np.split(points[:, 0:3], ind.astype(int))

ordered_points = [[0, 0, 0]]
for i in range(0, len(points_rearranged)):
    points_rearranged[i] = points_rearranged[i][points_rearranged[i][:, 0].argsort()]
    ordered_points = np.append(ordered_points, points_rearranged[i], axis=0)

ordered_points = np.delete(ordered_points, np.where(ordered_points[:, 0:1] == 0), axis=0)

for i in range(0,len(ordered_points)):
    cv2.imwrite("test"+str(i)+".jpg", op[:, :, ordered_points[i, 2].astype(int)])

cv2.imshow('img', gray)
cv2.waitKey(0)
