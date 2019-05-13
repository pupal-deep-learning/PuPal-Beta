import cv2
import numpy as np
import argparse
from PIL import Image
from prediction import *
from circler import *
import tensorflow as tf
import os
from datetime import datetime

sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))

print("\nThis are your available webcams:")
os.system("ls -ltrh /dev/video*")

webcam = input("\nPlease choose the device number you want to use (0 / 1): ")

createFile = input("Do you wish to save a log file of the results? (y / n)): ")

if createFile == 'y':
    fileName = input('Please write the file name: ')

# OpenCV webcam vars
eye_cascade = cv2.CascadeClassifier('./haar/haarcascade_eye_tree_eyeglasses.xml')

# Open webcam
cap = cv2.VideoCapture(int(webcam))

# Set the width and height of the frame
cap.set(3,1280) # Width
cap.set(4,720) # Height

frame_index = 0

success,image = cap.read()

# Vars
zoom_delta = 10
zoomlevel = 2
count = 0
#res = (256,256)
rgbcode = [0,0,0]
brighter = 100
timestamps = []
log = []

# shapes
img_y, img_x = image.shape[0:2]
label_offset = 15
vizbox = (512,256)
vizbox_x = [(img_x - vizbox[0], img_x - vizbox[1]),
            (img_x - vizbox[1], img_x)]

time_list = []
ratios_list = []

while success:
    # read image
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # timestamp from video
    timestamps.append(cap.get(cv2.CAP_PROP_POS_MSEC))
    #print(timestamps)

    # Detect eyes in the image
    eyes = eye_cascade.detectMultiScale(gray, 1.1, 5)

    # indices
    zoom = zoomlevel * zoom_delta
    ratios = []

    # Check if only two detected
    if len(eyes) == 2:

        # Next image if not centered
        if (eyes[:][:2] - zoom < 0).any():
            success,image = cap.read()
            continue

        # Side
        side_index = [i for i in np.argsort([ eyes[0][0],eyes[1][0] ])]

        # Loop trough eyes
        for index, (x, y, w, h) in enumerate(eyes):
            #Define ROI square
            roi = gray[(y-zoom):(y+h+zoom), (x-zoom):(x+w+zoom)]

            # Shape up for input to model
            img = cv2.resize(roi,(vizbox[1],vizbox[1]))

            # Predictions
            seg = predict(img)

            # Reshape and add RGB channels
            seg = (np.squeeze(seg) * 255).astype('uint8')
            _,seg = cv2.threshold(seg,127,255,cv2.THRESH_BINARY)

            # Fit circles
            circle_params = circler(seg)

            # # save segment
            seg = np.array(Image.fromarray(seg).convert("RGB"))

            # Overlay
            background = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
            background = np.where((255 - background) < brighter,
                                    255,background+brighter)

            overlay = seg.copy()
            rows,cols,channels = overlay.shape
            overlay=cv2.addWeighted(background[0:rows, 0:0+cols],0.5,overlay,0.5,0)
            background[0:rows, 0:0+cols] = overlay

            if circle_params is not None:
                ratio, radius_i, center_i, radius_p, center_p = circle_params

                ratios.append(ratio)

                now = datetime.now()
                time_list.append(str(now))
                ratios_list.append(ratio)

                # Draw circles
                background = cv2.circle(background, center_i, radius_i, (255, 0, 0), 2)
                background = cv2.circle(background, center_p, radius_p, (0, 255, 0), 2)

            # mask
            image[(img_y-vizbox[1]):img_y,
            vizbox_x[side_index[index]][0]:vizbox_x[side_index[index]][1],:] = background


            # label text

            if index == 1:
                # bounding box
                cv2.rectangle(image, ((img_x-vizbox[0]), (img_y-vizbox[1])),
                (img_x, img_y), (0,0,0), 2)

                if circle_params is not None:
                    label = "avg. ratio: " + str(round(sum(ratios)/len(ratios),3)+1e-6)
                    frame = "frame: " + str(count)
                    lab_x = (img_x-vizbox[0]) + label_offset
                    lab_y = (img_y-vizbox[1]) + label_offset + 5
                    cv2.putText(image, label, (lab_x, lab_y),
    			            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 2)
                    cv2.putText(image, frame, (lab_x, lab_y + 15),
    			            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 2)

    count += 1

    # Display the images
    cv2.imshow('image', image)
    
    # Press Q on keyboard to exit 
    if cv2.waitKey(1) & 0xFF== ord ('q'):
        break


    #count += 1
    success,image = cap.read()

def genFile(createFile, time_list, ratios_list, fileName = ""):
    
    if createFile == 'y':
        newFile = open(fileName + ".txt", 'w')
        newFile.write("DateTime,Ratio" + "\n")
        
        for i in range(len(ratios_list)):
            newFile.write(str(time_list[i]) + ',' + str(round(ratios_list[i], 3))+"\n")
            
    else:
        return

logFile = genFile(createFile, time_list, ratios_list, fileName)
