import cv2
import csv
import numpy as np

csvfile = open('EchoNet-Dynamic\\VolumeTracings.csv')
labels = list(csv.reader(csvfile))

pointList = []
for row in range(2,22):
    pointList.append([float(labels[row][1]), float(labels[row][2])])
    pointList.append([float(labels[row][3]), float(labels[row][4])])

pointList = sorted(pointList, key=lambda point: point[0])

#lowerSegmentList = [] #for the side of the hull facing down
#upperSegmentList = [] #for the side of teh hull facing up
indexP = 0
indexR = 0

hull = [pointList[0]]

def orientation(p, q, r): #checks if points p, q, and r are counterclockwise
                  #0: collinear, 1: clockwise, 2: counterclockwise
    val = (q[1] - p[1]) * (r[0] - q[0]) - \
          (q[0] - p[0]) * (r[1] - q[1])
    if val == 0: 
        return 0
    elif val > 0:
        return 1
    else:
        return 2

while indexP < len(pointList): #loop for each segment
    for indexQ in range(len(pointList)): #loop for each point in pointList
        if indexQ == indexP:
            continue
        elif indexR == indexP:
            indexR = indexQ
            continue
        elif indexQ == indexR:
            continue
        if orientation(pointList[indexP], pointList[indexQ], pointList[indexR]) == 1:
            indexR = indexQ
    deltaX = pointList[indexR][0]-pointList[indexP][0]
    hull.append(pointList[indexR])
    #if deltaX > 0:
    #    print(indexR)
    #    upperSegmentList.append([pointList[indexP], pointList[indexR]])
    #elif deltaX < 0:
    #    print(indexR)
    #    lowerSegmentList.append([pointList[indexP], pointList[indexR]])
    if indexR == 0:
        break
    if not indexP == 0:
        pointList.pop(indexP)
        if indexR > indexP:
            indexR -= 1
    indexP = indexR

print(hull)
