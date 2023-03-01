import cv2
import csv
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import math

csvfile = open('EchoNet-Dynamic\\VolumeTracings.csv')
labels = list(csv.reader(csvfile))
index = 1
labelCount = 0

vid = cv2.VideoCapture('EchoNet-Dynamic\Videos\\0X1A0A263B22CCD966.avi')
vid.set(cv2.CAP_PROP_POS_FRAMES, 30)
ret, img = vid.read()
errors = []


while index < len(labels)-1:
    count = 0
    fileName = labels[index][0]
    frame = labels[index][5]
    while True:
        if index > len(labels)-1:
            print(index)
            break
        if not fileName == labels[index][0] or not frame == labels[index][5]:
            break
        index += 1
        count += 1
    if count != 21:
        print("ERROR!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        if fileName not in errors:
            errors.append(fileName)
    pointList = []

    deltaX1 = float(labels[index-count][1]) - float(labels[index-count+1][1])
    deltaY1 = float(labels[index-count][2]) - float(labels[index-count+1][2])
    deltaX2 = float(labels[index-count][3]) - float(labels[index-count+1][3])
    deltaY2 = float(labels[index-count][4]) - float(labels[index-count+1][4])
    dist1 = math.sqrt((deltaX1*deltaX1) + (deltaY1*deltaY1))
    dist2 = math.sqrt((deltaX2*deltaX2) + (deltaY2*deltaY2))

    if dist1 < dist2:
        pointList.append([float(labels[index-count][1]), float(labels[index-count][2])])
    else:
        pointList.append([float(labels[index-count][3]), float(labels[index-count][4])])
    
    for row in range(index-count+1,index):
        pointList.append([float(labels[row][1]), float(labels[row][2])])
    
    if dist1 > dist2:
        pointList.append([float(labels[index-count][1]), float(labels[index-count][2])])
    else:
        pointList.append([float(labels[index-count][3]), float(labels[index-count][4])])

    for row in range(0, count-1):
        pointList.append([float(labels[index-1-row][3]), float(labels[index-1-row][4])])

    pointList = np.array(pointList)
    
    mask = np.zeros_like(img)
    cv2.fillPoly(mask, np.int32([pointList]), [255,255,255])
    if mask is None:
        print(str(count) + str(frame) + "ALKDJSLKJDALKSDJLASKJDLAKSJDKLAWJSD")
        break
    im = Image.fromarray(mask)
    im.save(f"{'Labels'}\\{labels[index-1][0]}frame{labels[index-1][5]}.jpeg")
    labelCount += 1
    if labelCount%100 == 0:
        print(f'Processed labelCount: {labelCount}')

print(len(errors))
with open(r'Errors.txt', 'w') as fp:
    for entry in errors:
        fp.write("%s\n" % entry)
    print('Done')
    

