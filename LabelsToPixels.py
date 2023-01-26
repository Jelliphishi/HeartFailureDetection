import cv2
import csv
import numpy as np

csvfile = open('EchoNet-Dynamic\\VolumeTracings.csv')
labels = list(csv.reader(csvfile))

prevx = 0
prevy = 0
x = 0
y = 0
lowerSegmentList = []
upperSegmentList = []

#sorting through line segments
for i in range(0,1):
    for row in range(2, 22): #first loop goes through first coordinates of the lines, second goes through second coords
        x = float(labels[row][2*i+1])
        y = float(labels[row][2*i+2])
        if prevx == 0: #skips to next point for first point
            prevx = x
            prevy = y
            continue
        if x < prevx:
            upperSegmentList.append([prevx, prevy, x, y])
        elif x > prevx:
            lowerSegmentList.append([prevx, prevy, x, y])
        else:
            upperSegmentList.append([prevx, prevy, x, y])
            lowerSegmentList.append([prevx, prevy, x, y])
        prevx = x
        prevy = y

print(upperSegmentList)

image = np.empty((0,112,3))

for row in range(112):
    tempRow = np.empty((0,3))
    for col in range(112):
        inShape = False
        for lSeg in lowerSegmentList:
            if col >= lSeg[0] and col <= lSeg[2]:
                for uSeg in upperSegmentList:
                    if col >= uSeg[0] and col <= uSeg[2]:
                        aboveL = row >= lSeg[1]+(lSeg[3]-lSeg[1])*(row-lSeg[0])/(lSeg[2]-lSeg[0])
                        belowU = row <= uSeg[1]+(uSeg[3]-uSeg[1])*(row-uSeg[0])/(uSeg[2]-uSeg[0])
                        if aboveL and belowU:
                            inShape = True
                        break
                break
        if inShape:
            tempRow = np.append(tempRow, [[255,255,255]], axis=0)
        else:
            tempRow = np.append(tempRow, [[0,0,0]], axis=0)
    image = np.append(image, [tempRow], axis=0)

import matplotlib.pyplot as plt

plt.figure()
plt.imshow(image)

plt.waitforbuttonpress()
plt.close('all')

        




