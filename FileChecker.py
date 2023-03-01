import glob
import numpy as np
import cv2
import os

allImages = []
for directory_path in glob.glob('Labeled-Data'):
    for img_path in glob.glob(os.path.join(directory_path, '*.jpeg')):
        allImages.append(os.path.basename(img_path))

for directory_path in glob.glob('Labels'):
    for label_path in glob.glob(os.path.join(directory_path, '*.jpeg')):
        if os.path.basename(label_path) not in allImages:
            print(os.path.basename(label_path))