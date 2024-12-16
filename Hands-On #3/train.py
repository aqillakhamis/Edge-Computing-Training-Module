"""
This code is use for Image Classification multi class problem using RESNET50
"""

# import libraries or dependencies 
import os
import time

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import random
import sys
import gc

import matplotlib.pyplot as plt
import matplotlib.image as mpimg

import cv2
import seaborn as sns
from sklearn.model_selection import train_test_split 
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import confusion_matrix, multilabel_confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report

import keras
from tensorflow.keras.applications import ResNet50
from tensorflow.keras import layers
from tensorflow.keras.layers import Dense, Flatten, Conv2D, Dropout, MaxPooling2D, GlobalAveragePooling2D, Dropout
from tensorflow.keras import optimizers
from tensorflow.keras import models
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras import preprocessing
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from tensorflow.keras.utils import plot_model

# global variables
img_size = 224 #image use in this model is 224 x 224 
batch_size = 32
epochs = 10 #model train for 10 epoch can be customized
train_size = 0.7
val_size = 0.2
test_size = 0.1
seed = 4321
channels = 3
learning_rate = 0.00001 #tune learning rate 

# get path directories 
d = 'dataset'

# get information of each class
classes = [item for item in os.listdir(d) if not item.startswith('.')]

paths = [os.path.join(d, o) for o in os.listdir(d) if os.path.isdir(os.path.join(d,o))]

nbEntries = []

for i in range(len(classes)):
    nbEntries.append(len(os.listdir(paths[i])))

print('Number of Classes: {}'.format(len(classes)))
print('Type of Categories :', classes)
print('Number of Entries in Each Categories:', nbEntries)

## Plot Classes Data Distribution 
df = pd.DataFrame({'classes':classes, 'entries':nbEntries})
ax = df.sort_values(by='entries', ascending=True).plot.bar(x='classes', y='entries', color='cornflowerblue',legend=False, figsize=(12,8))
ax.set_title('Utilities Class Distribution')
ax.set_ylabel("# entries")
for p in ax.patches:
    ax.annotate(str(p.get_height()), xy=(p.get_x(), p.get_height()-30))

