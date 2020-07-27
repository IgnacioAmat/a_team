"""
Face mask detection model using MobileNetV2 model
"""

#Imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import cv2
import random
import shutil

import plotly.express as px
import plotly.graph_objects as go
import plotly.figure_factory as ff
from plotly.subplots import make_subplots

import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, MaxPooling2D, Flatten, Dropout, Conv2D, GlobalAveragePooling2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint

# Fetch data
DATAPATH = 'files\\images\\face_mask_data'
MASKPATH = 'files\\images\\face_mask_data\\with_mask'
NOMASKPATH = 'files\\images\\face_mask_data\\without_mask'
TESTPATH = 'files\\images\\test_face_mask_data'

# Visualize data
def view(pth):
    images = list()
    for img in random.sample(os.listdir(pth),9):
        images.append(img)
    i = 0
    fig,ax = plt.subplots(nrows=3, ncols=3, figsize=(30,20))
    for row in range(3):
        for col in range(3):
            ax[row,col].imshow(cv2.imread(os.path.join(pth,images[i])))
            i+=1

view(MASKPATH)

view(NOMASKPATH)

# Analyse data
fig = go.Figure(
    data=[go.Pie(labels=['WITHMASK','WITHOUTMASK'], 
        values=[len(os.listdir(MASKPATH)),len(os.listdir(NOMASKPATH))])
    ])
fig.show()

# Splitting data
os.mkdir(TESTPATH)
os.mkdir(os.path.join(TESTPATH,'with_mask'))
os.mkdir(os.path.join(TESTPATH,'without_mask'))

def getTest(pth):
    dataSplit = int(np.ceil(len(os.listdir(pth))*0.02))
    for img in os.listdir(pth)[-dataSplit:]:
        shutil.move(os.path.join(pth,img), os.path.join('files\\images\\test_face_mask_datadata'.join(pth.split('files\\images\\face_mask_data')),img))
getTest(MASKPATH)
getTest(NOMASKPATH)

len(os.listdir(MASKPATH)),len(os.listdir(NOMASKPATH))

len(os.listdir(os.path.join(TESTPATH,'with_mask'))),len(os.listdir(os.path.join(TESTPATH,'without_mask')))

# Prepare data input pipeline
BATCH_SIZE = 32

trainGen = ImageDataGenerator(
    rescale= 1/255.,
    horizontal_flip=True,
    validation_split = 0.1
)

testGen = ImageDataGenerator(
    rescale= 1/255.,
)

train = trainGen.flow_from_directory(
    DATAPATH, 
    target_size=(224, 224),
    classes=['with_mask','without_mask'],
    class_mode='categorical', 
    batch_size=BATCH_SIZE, 
    shuffle=True,
    subset='training'
)

validation = trainGen.flow_from_directory(
    DATAPATH, 
    target_size=(224, 224),
    classes=['with_mask','without_mask'],
    class_mode='categorical', 
    batch_size=BATCH_SIZE, 
    shuffle=True,
    subset='validation'
)

test = testGen.flow_from_directory(
    TESTPATH, 
    target_size=(224, 224), 
    classes=['with_mask','without_mask'],
    class_mode='categorical', 
    batch_size=BATCH_SIZE, 
    shuffle=True,
)

# Model building
mob = MobileNetV2(
    input_shape = (224,224,3),
    include_top = False,
    weights = 'imagenet',
)
mob.trainable = False

model = Sequential()
model.add(mob)
model.add(GlobalAveragePooling2D())
model.add(Dense(64,activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(2,activation='softmax'))
model.summary()


# serialize model to JSON
model_json = model.to_json()
with open("files\\model\\model_mask.json", "w") as json_file:
    json_file.write(model_json)

model.compile(optimizer=Adam(),loss='categorical_crossentropy',metrics=['acc'])

#Add checkpoints
checkpoint = ModelCheckpoint(
    'files\\weights\\model_mask.h5',
    monitor='val_loss',
    verbose=1,
    save_best_only=True,
    save_weights_only=True,
    mode='min'
)

#Fit model
hist = model.fit(
    train,
    epochs = 15,
    validation_data = validation,
    callbacks = [checkpoint]
)

model.evaluate(test)

