"""
    Face detection methods for both image and video
    Code based in https://towardsdatascience.com/face-detection-in-2-minutes-using-opencv-python-90f89d7c0f81
"""

import cv2

from tensorflow.keras.models import model_from_json
import matplotlib.pyplot as plt
import numpy as np
import cv2

from tensorflow.keras.models import Sequential
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, MaxPooling2D, Flatten, Dropout, Conv2D, GlobalAveragePooling2D
from tensorflow.keras.optimizers import Adam
    
with open('files\\model\\model_mask.json', 'r') as f:
    loaded_model_json = f.read()
model = model_from_json(loaded_model_json)

model.load_weights("files\\weights\\model_mask.h5")
print("Loaded model from disk")

resMap = {
        0 : 'Mask On',
        1 : 'No Mask'
    }

colorMap = {
        0 : (0,255,0),
        1 : (0,0,255)
    }

def prepImg(pth):
    return cv2.resize(pth,(224,224)).reshape(1,224,224,3)/255.0

def faceDetectorImg(path):
    
    # Load the cascade
    face_cascade = cv2.CascadeClassifier('files\\weights\\haarcascade_frontalface_default.xml')
    # Read the input image
    img = cv2.imread(path)
    # Convert into grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Detect faces
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
    # Draw rectangle around the faces
    for (x, y, w, h) in faces:
        #slicedImg = img[y:y+h,x:x+w]
        pred = model.predict(prepImg(img))
        pred = np.argmax(pred)
        cv2.rectangle(img, (x, y), (x+w, y+h), colorMap[pred], 2)
        cv2.putText(img, resMap[pred],(x,y-10),cv2.FONT_HERSHEY_SIMPLEX,0.8,(255,255,255),2) 
    # Display the output
    cv2.imshow('img', img)
    cv2.waitKey()
    cv2.destroyAllWindows()
    
def faceDetectorVideo(path = ""):
    # Load the cascade
    face_cascade = cv2.CascadeClassifier('files\\weights\\haarcascade_frontalface_default.xml')
    
    if path == "":
        # To capture video from webcam. 
        cap = cv2.VideoCapture(0)
    else:
        # To use a video file as input 
        cap = cv2.VideoCapture(path)
    
    while True:
        # Read the frame
        _, img = cap.read()
        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # Detect the faces
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)
        # Draw the rectangle around each face
        for (x, y, w, h) in faces:
            #slicedImg = img[y:y+h,x:x+w]
            pred = model.predict(prepImg(img))
            pred = np.argmax(pred)
            cv2.rectangle(img, (x, y), (x+w, y+h), colorMap[pred], 2)
            cv2.putText(img, resMap[pred],(x,y-10),cv2.FONT_HERSHEY_SIMPLEX,0.8,(255,255,255),2) 
        # Display
        cv2.imshow('img', img)
        # Stop if escape key is pressed
        k = cv2.waitKey(30) & 0xff
        if k==27:
            break
    # Release the VideoCapture object
    cap.release()
    cv2.destroyAllWindows()
      
def main():
    faceDetectorImg('files\\images\\test.jpg')
    faceDetectorVideo()

if __name__ == "__main__":
    main()