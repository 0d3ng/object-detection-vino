#  Created by od3ng on 23/06/2019 09:53:08 PM.
#  Project: object-detection-vino
#  File: object-detection-vino.py
#  Email: lepengdados@gmail.com
#  Telegram: @nopriant0
#  Telfon: 085878554150
#  Website: sinaungoding.com

import cv2 as cv
import os

# Load class txt
LABELS = open("classes.txt").read().strip().split("\n")

# Load the model.
net = cv.dnn.readNet('frozen_inference_graph.xml',
                     'frozen_inference_graph.bin')
# Specify target device.
net.setPreferableTarget(cv.dnn.DNN_TARGET_MYRIAD)

font_scale = 1
font = cv.FONT_HERSHEY_PLAIN
rectangle_bgr = (255, 255, 255)

# Read an image.
frame = cv.imread('objects.jpg')

rows = frame.shape[0]
cols = frame.shape[1]

# Prepare input blob and perform an inference.
blob = cv.dnn.blobFromImage(frame, size=(672, 384), ddepth=cv.CV_8U)
net.setInput(blob)
out = net.forward()
# Draw detected faces on the frame.
for detection in out.reshape(-1, 7):
    confidence = float(detection[2])
    class_id = int(detection[1])
    left = detection[3] * cols
    top = detection[4] * rows
    right = detection[5] * cols
    bottom = detection[6] * rows
    if confidence > 0.8:
        left = detection[3] * cols
        top = detection[4] * rows
        right = detection[5] * cols
        bottom = detection[6] * rows
        text = "{} {:.2f}".format(LABELS[class_id], confidence)

        (text_width, text_height) = cv.getTextSize(text, font, fontScale=font_scale, thickness=1)[0]
        text_offset_x = int(left)
        text_offset_y = int(top) - 2
        box_coord = ((text_offset_x, text_offset_y), (text_offset_x + text_width-2, text_offset_y - text_height - 2))

        cv.rectangle(frame, (int(left), int(top)), (int(right), int(bottom)), (255, 255, 255), thickness=2)
        cv.rectangle(frame, box_coord[0], box_coord[1], rectangle_bgr, cv.FILLED)
        cv.putText(frame, text, (text_offset_x, text_offset_y), font, fontScale=font_scale, color=(0, 0, 0),thickness=1)

# Save the frame to an image file.
cv.imwrite('out.png', frame)
