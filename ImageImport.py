import cv2
import os
import csv
import numpy as np

#iterates through labeled data and adds the labelled frames to the dataset
csvfile = open('EchoNet-Dynamic\\VolumeTracings.csv')
labels = csv.reader(csvfile)
vidNames = []       #names of videos used
frameNums = []      #frame numbers used
data = []           #list of labeled frames
for row in labels:
    if row[0] == 'FileName':
        continue
    if not(row[0] in vidNames and row[5] in frameNums): #if it is a new video or frame
        if not row[0] in vidNames: #if it is a new video
            vidNames.append(row[0])
            frameNums = []
        frameNums.append(row[5])
        vid = cv2.VideoCapture('EchoNet-Dynamic\\Videos\\' + row[0])
        vid.set(cv2.CAP_PROP_POS_FRAMES, float(row[5]))
        ret, frame = vid.read()
        data.append(frame)
        print('Video: ' + row[0] + ', Frame#: ' + row[5] + ', Read: ' + str(ret))


