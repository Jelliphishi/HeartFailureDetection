import cv2
import csv
import numpy as np
import matplotlib.pyplot as plt

csvfile = open('EchoNet-Dynamic\\VolumeTracings.csv')
labels = list(csv.reader(csvfile))

plt.figure()
pointList = []
pointList.append([float(labels[1][1]), float(labels[1][2])])
for row in range(2,22):
    pointList.append([float(labels[row][1]), float(labels[row][2])])
pointList.append([float(labels[1][3]), float(labels[1][4])])
for row in range(0,20):
    pointList.append([float(labels[22-row][3]), float(labels[22-row][4])])
pointList = np.array(pointList)

vid = cv2.VideoCapture('EchoNet-Dynamic\Videos\\0X1A0A263B22CCD966.avi')
vid.set(cv2.CAP_PROP_POS_FRAMES, 30)
ret, img = vid.read()
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
mask = np.zeros_like(img)
print(mask.shape)
print(type(mask))
cv2.fillPoly(mask, np.int32([pointList]), [255,255,255])

for row in range(1,22):
    x = [float(labels[row][1]), float(labels[row][3])]
    y = [float(labels[row][2]), float(labels[row][4])]
    plt.plot(x, y, color="maroon", linewidth=2)
plt.imshow(mask)

plt.waitforbuttonpress()
plt.close('all')