import cv2
import os
import numpy as np
import matplotlib.pyplot as plt


vid = cv2.VideoCapture(r'C:\Users\jelli\Desktop\Research Project 2023\EchoNet-Dynamic\Videos\0X1A0A263B22CCD966.avi')
fps = vid.get(cv2.CAP_PROP_FPS)
print('fps = ' + str(fps))

vid.set(cv2.CAP_PROP_POS_FRAMES, 30)
ret, frame = vid.read()
print(ret)
print(frame.shape)

plt.figure()
plt.imshow(frame)

plt.waitforbuttonpress()
plt.close('all')
