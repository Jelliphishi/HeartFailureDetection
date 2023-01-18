from segmentation_models import Unet
from tensorflow import keras
import numpy as np
np.random.seed(1122)

import cv2 #for csv files and VideoCapture
import os
from PIL import Image #for resizing images (pretty sure not needed)

os.environ['KERAS_BACKEND'] = 'tensorflow' #defining and environment

#########################################################################

dataset = []
labels = []

model = Unet('resnet34', encoder_weights='imagenet')
#variables: optimizer and loss function
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['mse'])

print(model.summary())