import cv2
import csv              #for accessing CSV file
import numpy as np      #for arrays
from PIL import Image   #for exporting images

#iterates through labeled data and adds the labelled frames to the dataset
csvfile = open('EchoNet-Dynamic\\VolumeTracings.csv')
labels = csv.reader(csvfile)
vidNames = []       #names of videos used
frameNums = []      #frame numbers used
data = []           #list of labeled frames
count = 0
imgPath = 'Labeled-Data'
for row in labels:
    if row[0] == 'FileName':
        continue
    if not(row[0] in vidNames and row[5] in frameNums): #if it is a new video or frame
        if not row[0] in vidNames: #if it is a new video
            vidNames.append(row[0])
            frameNums = []
        frameNums.append(row[5])

        #extracts the frame from the video
        vid = cv2.VideoCapture('EchoNet-Dynamic\\Videos\\' + row[0])
        vid.set(cv2.CAP_PROP_POS_FRAMES, float(row[5]))
        ret, frame = vid.read()
        if frame is None:
            continue
        im = Image.fromarray(frame)
        im.save(f"{imgPath}\\{row[0]}frame{row[5]}.jpeg")
        count += 1
        if count%100 == 0:
            print(f'Frame #{count}')


