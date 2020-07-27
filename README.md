# Safer Wherever

The aim of this project is to develop a solution based on the use of AI for a post CoVid-19 world as part of the **#BuildwithAI : Emergence!** Hackathon.
The whole project can be found in our team [repo](https://github.com/Build-with-AI-a-team).

## Face Mask Training

Face mask detection model based on the MobileNetV2 architecture.
The training and test images used to train the model can be found on [prajnasb](https://github.com/prajnasb/observations) github.

## Face Detection

Face detection implementation that applies face mask detection model for each face detected. The face detection algorithm is able to detect face on both image or video (file or webcam) by using either Haar classifier or MTCNN classifier.

Result of the mask detection model applied to detected faces.

![Face mask detection image](https://github.com/IgnacioAmat/a_team/blob/master/files/images/mask_detection.PNG)

## Social Distancing

In order to be able to run this model it will be necesary to first download [YOLO weights and cfg files](https://pjreddie.com/darknet/yolo/) and also create a file for storing the name of the classes the algorithm was trained to detect with the extension **.names**. In this project this file contains the COCO class labels that YOLO model was trained on.

![CCTV social distancing image](https://github.com/IgnacioAmat/a_team/blob/master/files/images/cctv_video_frame_detected.jpg)
