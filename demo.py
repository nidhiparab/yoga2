import math
import cv2
import numpy as np
from time import time
import mediapipe as mp
import matplotlib.pyplot as plt
import csv
import os
import sys
from sklearn import svm
from sklearn.preprocessing import StandardScaler
import pandas as pd

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

array = [0,0,0,0,0]
dataset = pd.read_csv('C:/Nidhi/vscode/yoga/Chakra_yoga/yoga.csv')
dataset1=dataset.fillna(0)

X = dataset1.drop(columns= 'Label', axis=1)
Y = dataset1['Label']

classifier = svm.SVC(kernel='linear',probability=True)

classifier.fit(X,Y)

# cap = cv2.VideoCapture(0) # change the argument to use a different webcam (e.g. 1 for a USB camera)

# # Initialize MediaPipe Pose model
# with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
#     while cap.isOpened():
#         ret, frame = cap.read()
#         x_row=[]
#         # Flip the frame horizontally for a mirror effect
#         frame = cv2.flip(frame, 1)
        
#         # Convert the image to grayscale for processing
#         gray = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
#         # Use MediaPipe to detect body landmarks
#         results = pose.process(gray)
        
#         if results.pose_landmarks:
#             for i in range(33):
#                 p = results.pose_landmarks.landmark[mp_pose.PoseLandmark(i).value]
#                 landmarks = [mp_pose.PoseLandmark(i) for i in range(mp_pose.PoseLandmark(i).value)]
#             print(landmarks)
#             print("+++++++++++++++++++++++++++")
#         for landmark in landmarks:
#             x = results.pose_landmarks.landmark[landmark.value].x
#             x_row.append(x)
#             y = results.pose_landmarks.landmark[landmark.value].y
#             x_row.append(y)
#             z = results.pose_landmarks.landmark[landmark.value].z
#             x_row.append(z)
#         x_row.append(0)
#         x_row.append(0)
#         x_row.append(0)    
        
        
#         input = x_row
#         input_as_np = np.asarray(input)
#         input_reshaped = input_as_np.reshape(1,-1)
#         a = classifier.predict_proba(input_reshaped)
#         print(int(a[0][0]*100))
#         color = (0, 255, 0)
#         cv2.putText(frame, str(int(a[0][0]*100)), (10, 30),cv2.FONT_HERSHEY_PLAIN, 2, color, 2)
#         # Display the resulting frame
#         cv2.imshow('frame', frame)
#         if cv2.waitKey(1) & 0xFF == ord('q'):
#             break

# cap.release()
# cv2.destroyAllWindows()

def detectPose(image, pose, display=True):
    
    # Create a copy of the input image.
    output_image = image.copy()
    
    # Convert the image from BGR into RGB format.
    imageRGB = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Perform the Pose Detection.
    results = pose.process(imageRGB)
    
    # Retrieve the height and width of the input image.
    height, width, _ = image.shape
    
    # Initialize a list to store the detected landmarks.
    landmarks = []
    
    # Check if any landmarks are detected.
    if results.pose_landmarks:
    
        # Draw Pose landmarks on the output image.
        mp_drawing.draw_landmarks(image=output_image, landmark_list=results.pose_landmarks,
                                  connections=mp_pose.POSE_CONNECTIONS)
        
        # Iterate over the detected landmarks.
        for landmark in results.pose_landmarks.landmark:
            
            # Append the landmark into the list.
            landmarks.append((landmark.x))
            landmarks.append((landmark.y))
            landmarks.append((landmark.z))
        
        return output_image, landmarks

pose_video = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5,model_complexity=1)

#Initialize the videocapture to read from web cam
video = cv2.VideoCapture(0)

#resizing purpose
cv2.namedWindow('Pose Detection', cv2.WINDOW_NORMAL)

#initialize videocapture to read from video
#set video size
video.set(3,1280)
video.set(4,960)

time1=0

#Initialize until the video is accepted succesfully
while video.isOpened():
    ok, frame=video.read()
    if not ok:
        break
    
    #flip horizontally
    frame = cv2.flip(frame,1)
    
    frame_height, frame_width,_=frame.shape
    
    #resize
    frame=cv2.resize(frame,(int(frame_width*(640/frame_height)),640))
    
    #landmark detection
    frame,landmarks=detectPose(frame,pose_video,display=False)
    print(landmarks)
    print(len(landmarks))
    
    input = landmarks
    input_as_np = np.asarray(input)
    input_reshaped = input_as_np.reshape(1,-1)
    a = classifier.predict_proba(input_reshaped)
    
    array.pop(0)
    array.append(int(a[0][2]*100))
    print(a)
    avg_percent = sum(array) / len(array)
    color = (0, 255, 0)
    cv2.putText(frame, str(int(a[0][2]*100)), (10, 30),cv2.FONT_HERSHEY_PLAIN, 2, color, 2)
    # cv2.putText(frame, str(classifier.predict(input_reshaped)), (10, 30),cv2.FONT_HERSHEY_PLAIN, 2, color, 2)
        
    # time2=time()
    # if(time2-time1)>0:
    #     frame_per_second=1.0/(time2-time1)
    #     cv2.putText(frame,'FPS:{}'.format(int(frame_per_second)),(10,30),cv2.FONT_HERSHEY_PLAIN,2,(0,255,0),3)
    # time1=time2
    cv2.imshow('Pose Detection',frame)
    
    # Display the resulting frame
        #wait until a key is pressed
    k=cv2.waitKey(1) & 0xFF
    #check if esc is pressed
    if(k==27):
        break
video.release()
cv2.destroyAllWindows()


