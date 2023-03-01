from segmentation_models import Unet
from segmentation_models import get_preprocessing
from segmentation_models.losses import bce_jaccard_loss
from segmentation_models.metrics import iou_score
import tensorflow as tf 
#tf.compat.v1.disable_v2_behavior()
from tensorflow import keras
import numpy as np
np.random.seed(1122)

import cv2 #for csv files and VideoCapture
import os
from PIL import Image #for resizing images (pretty sure not needed)
import glob #for looking through directory
import matplotlib.pyplot as plt

os.environ['KERAS_BACKEND'] = 'tensorflow' #defining and environment

#########################################################################

dataset = []
labels = []

print("loading model")
model = Unet(backbone_name='resnet34', classes = 3, activation = 'sigmoid', encoder_weights='imagenet')
#model = Linknet()
#variables: optimizer and loss function
print("compiling model")
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['mse'])

print("importing images")
allImages = []
for directory_path in glob.glob('Labeled-Data'):
    for img_path in glob.glob(os.path.join(directory_path, '*.jpeg')):
        img = cv2.imread(img_path)
        allImages.append(cv2.resize(img, dsize=(128, 128)))
allImages = np.array(allImages)

print("importing labels")
allLabels = []
for directory_path in glob.glob('Labels'):
    for label_path in glob.glob(os.path.join(directory_path, '*.jpeg')):
        label = cv2.imread(label_path)
        allLabels.append(cv2.resize(label, dsize=(128, 128)))
allLabels = np.array(allLabels).astype(float)

print("splitting data")
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(allImages, allLabels, test_size = 0.1, random_state=88)
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size = 1/9, random_state=88)

print("preprocessing inputs")
preprocess_input = get_preprocessing('resnet34')
x_train = preprocess_input(x_train)
x_val = preprocess_input(x_val)
x_test = preprocess_input(x_test)

print("fitting model")
history = model.fit(
                x = x_train, 
                y = y_train,
                batch_size = 16,
                epochs = 100,
                verbose = 1,
                validation_data = (x_val, y_val))

print("plotting")
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(loss) + 1)
plt.plot(epochs, loss, 'y', label='Training loss')
plt.plot(epochs, val_loss, 'r', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()