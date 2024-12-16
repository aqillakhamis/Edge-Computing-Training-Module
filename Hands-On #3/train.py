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

# display samples distribution
plt.show()

## Get All Images 
total_set = []
total_labels = []

file_format = [".JPEG", ".jpeg", ".TIF", ",tif", ".JPG", ".jpg", ".PNG", ".png", ".pdf", ".PDF"]
count_file_format = len(file_format)
print('# of File Format: ', count_file_format)

for i in range(count_file_format):
    for root, dirs, files in os.walk(d): 
        for file in files: 
            if file.endswith(file_format[i]):
                path = os.path.join(root, file)
                total_set.append(path)
                total_labels.append(root.split(os.path.sep)[-1])

# Return image class based on list entry (path)         
def getClass(img):
    return img.split(os.path.sep)[-2]

## Shuffle the Dataset 
random.Random(seed).shuffle(total_set)

# Get data and separate it in sets
total_len = len(total_set)

index = 0

train_set = []
train_label = []

val_set = []
val_label = []

test_set = []
test_label = []

for i in total_set[0: int(total_len*train_size)] :
    train_set.append(i)
    train_label.append(getClass(i))
    
index = int(total_len*train_size)+1

for i in total_set[index: int(index + total_len*val_size)] :
    val_set.append(i)
    val_label.append(getClass(i))
    
index = int(index + total_len*val_size)+1 

for i in total_set[index: total_len] :
    test_set.append(i)
    test_label.append(getClass(i))

print('Total Sample: ', len(train_set) + len(test_set) + len(val_set))

print('Train Set:', len(train_set))
print('Test Set:', len(test_set))
print('Val Set:', len(val_set))

#TRAIN SET
instances = [0] * len(classes)
for index, val in enumerate(classes) :
    for e in train_set :
        if(val == getClass(e)) :
            instances[index] += 1
          
df = pd.DataFrame({'classes':classes, 'entries':instances})
ax = df.sort_values(by='entries', ascending=True).plot.bar(x='classes', y='entries', color='cornflowerblue',legend=False, figsize=(12,8))
ax.set_title('Documents Train Set Distribution')
ax.set_ylabel("# entries")
for p in ax.patches:
    ax.annotate(str(p.get_height()), xy=(p.get_x(), p.get_height()-20))

################################################
#VAL SET
instances = [0] * len(classes)
for index, val in enumerate(classes) :
    for e in val_set :
        if(val == getClass(e)) :
            instances[index] += 1

df = pd.DataFrame({'classes':classes, 'entries':instances})
ax = df.sort_values(by='entries', ascending=True).plot.bar(x='classes', y='entries', color='cornflowerblue',legend=False, figsize=(12,8))
ax.set_title('Documents Val Set Distribution')
ax.set_ylabel("# entries")
for p in ax.patches:
    ax.annotate(str(p.get_height()), xy=(p.get_x(), p.get_height()-3))
    
################################################
#TEST SET
instances = [0] * len(classes)
for index, val in enumerate(classes) :
    for e in test_set :
        if(val == getClass(e)) :
            instances[index] += 1

df = pd.DataFrame({'classes':classes, 'entries':instances})
ax = df.sort_values(by='entries', ascending=True).plot.bar(x='classes', y='entries', color='cornflowerblue',legend=False, figsize=(12,8))
ax.set_title('Documents Test Set Distribution')
ax.set_ylabel("# entries")
for p in ax.patches:
    ax.annotate(str(p.get_height()), xy=(p.get_x(), p.get_height()-8))

plt.show()

# pre-process the image (resize the image)

def process_images(img_set) : 
    processed_img = []

    for i in range(len(img_set)) :
        try:
            processed_img.append(cv2.resize(cv2.imread(img_set[i], cv2.IMREAD_COLOR), (img_size, img_size)))
        except Exception as e: 
            print(str(e))

    return processed_img

print('Start the Pre-Processing Session...\n')
tic = time.perf_counter()
x_train = process_images(train_set)
x_test = process_images(test_set)
x_val = process_images(val_set)
toc = time.perf_counter()
print(f'Finish Pre-Processing in {toc-tic:0.4f} seconds ...\n')
print('End of the Pre-Processing Session...\n')

lb = LabelBinarizer()
lb.fit(list(classes))

x_train = np.array(x_train)
y_train =lb.transform(np.array(train_label))

x_test = np.array(x_test)
y_test = lb.transform(np.array(test_label))

x_val = np.array(x_val)
y_val = lb.transform(np.array(val_label))

# using pre-trained RESNET50 model

base_model = ResNet50(weights = "imagenet", include_top=False, input_shape = (img_size, img_size, channels))
    
base_model.summary()

model = models.Sequential()

model.add(base_model)
model.add(layers.Flatten())
model.add(layers.Dense(128, activation='relu',  name='dense'))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(len(classes), activation='softmax',  name='predictions'))

model.summary()

print('Number of trainable weights : ', len(model.trainable_weights))

# trained the model
model.compile(optimizer=optimizers.Adam(lr=learning_rate), loss='categorical_crossentropy', metrics=['accuracy'])

tic_train = time.perf_counter()
train_model = model.fit(x_train, y_train,
                        batch_size=batch_size,
                        epochs=epochs,
                        verbose=1,
                        validation_data=(x_val, y_val)) 
toc_train = time.perf_counter()
print(f'Finish Train Model in {toc_train-tic_train:0.4f} seconds ... ')

# plot model accuracy and loss
fig = plt.figure(figsize =  (10,7))

#Plot loss functions
plt.subplot(221)
plt.plot(train_model.history['loss'], label='training loss')
plt.plot(train_model.history['val_loss'], label='validation loss')
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.grid()
plt.legend()

#Plot accuracy 
plt.subplot(222)
plt.plot(train_model.history['accuracy'], label='training accuracy')
plt.plot(train_model.history['val_accuracy'], label='validation accuracy')
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.grid()
plt.legend()

plt.show()

# evaluate on test dataset
test_loss, test_accuracy = model.evaluate(x_test, y_test, verbose=1)
print('Test Loss: ', test_loss)
print('Test Accuracy: ', test_accuracy)

# model prediction and confusion matrix
y_pred = np.argmax(model.predict(x_test),axis=1)
y_test = np.argmax(y_test, axis=1)

cfn_mat = confusion_matrix(y_test, y_pred)

class_names = classes 
fig, ax = plt.subplots()
tick_marks = np.arange(len(class_names))
plt.xticks(tick_marks, class_names)
plt.yticks(tick_marks, class_names)
sns.heatmap(pd.DataFrame(cfn_mat), annot = True, cmap = "RdBu_r", fmt = 'g')
ax.xaxis.set_label_position("top")
plt.tight_layout()
plt.title('Confusion matrix', y=1.1)
plt.ylabel('Actual label')
plt.xlabel('Predicted label')
plt.show()

count_misclassified = (y_test != y_pred).sum()
print('Misclassified samples: {}'.format(count_misclassified))
accuracy = accuracy_score(y_test, y_pred)
print('Accuracy: {:.2f}'.format(accuracy))
print(classification_report(y_test, y_pred))

# save trained model
model.save('resnet50.h5')