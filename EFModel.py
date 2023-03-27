#pip install -r requirements.txt
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

errors = open("Errors.txt").read()

print("loading model")
model = Unet(backbone_name='resnet34', classes = 3, activation = 'relu', encoder_weights='imagenet')
print("compiling model")
model.compile(optimizer='adam', loss="binary_crossentropy", metrics=['accuracy'])

print("importing images")
imgDict = []
for directory_path in glob.glob('Labeled-Data'):
    for img_path in glob.glob(os.path.join(directory_path, '*.jpeg')):
        if img_path[13:len(img_path)-13] in errors:
            print(img_path)
            continue
        img = cv2.imread(img_path)
        imgDict.append((img_path, cv2.resize(img, dsize=(128, 128))))

print("importing labels")
labelDict = []
for directory_path in glob.glob('Labels'):
    for label_path in glob.glob(os.path.join(directory_path, '*.jpeg')):
        if label_path[7:len(label_path)-13] in errors:
            print(label_path)
            continue
        label = cv2.resize(cv2.imread(label_path), dsize=(128, 128))
        for i in range(128):
            for j in range(128):
                if label[i][j][0] > 127:
                    label[i][j] = 1
                else:
                    label[i][j] = 0
        labelDict.append((label_path, label))

print("splitting data")
from sklearn.model_selection import train_test_split
x_train_dict, x_test_dict, y_train_dict, y_test_dict = train_test_split(imgDict, labelDict, test_size = 0.1, random_state=88)
x_train_dict, x_val_dict, y_train_dict, y_val_dict = train_test_split(x_train_dict, y_train_dict, test_size = 1/9, random_state=88)

def dict_to_array(dict):
    list = []
    for tuple in dict:
        list.append(tuple[1])
    return list

x_train = np.array(dict_to_array(x_train_dict)).astype(float)
y_train = np.array(dict_to_array(y_train_dict)).astype(float)
x_val = np.array(dict_to_array(x_val_dict)).astype(float)
y_val = np.array(dict_to_array(y_val_dict)).astype(float)
x_test = np.array(dict_to_array(x_test_dict)).astype(float)
y_test = np.array(dict_to_array(y_test_dict)).astype(float)

with open(r'x_test.txt', 'w') as fp:
    for entry in x_test_dict:
        fp.write("%s\n" % entry[0])

print("preprocessing inputs")
preprocess_input = get_preprocessing('resnet34')
x_train = preprocess_input(x_train)
x_val = preprocess_input(x_val)
x_test = preprocess_input(x_test)

print("fitting model")
history = model.fit(
                x = x_train, 
                y = y_train,
                batch_size = 32,
                epochs = 128,
                verbose = 1,
                validation_data = (x_val, y_val))

print('saving')
model.save('model')

print("plotting")
loss = history.history['loss']
val_loss = history.history['val_loss']
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
epochs = range(1, len(loss) + 1)

#plotting loss
plt.ylim(0,0.25)
plt.plot(epochs, loss, 'y', label='Training Loss')
plt.plot(epochs, val_loss, 'r', label='Validation Loss')
plt.title('Training and Validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.savefig(r'Figures\loss.png')

plt.clf()

#plotting accuracy
plt.ylim(0,1)
plt.plot(epochs, acc, 'y', label='Training Accuracy')
plt.plot(epochs, val_acc, 'r', label='Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.savefig(r'Figures\accuracy.png')

plt.clf()

