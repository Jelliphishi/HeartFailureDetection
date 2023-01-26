import cv2
import csv
import numpy as np

csvfile = open('EchoNet-Dynamic\\VolumeTracings.csv')
labels = list(csv.reader(csvfile))

pointList = []
for row in range(2,22):
    pointList.append([labels[row][1], labels[row][2]])
    pointList.append([labels[row][3], labels[row][4]])

pointList = sorted(pointList, key=lambda point: point[0])
print(pointList)