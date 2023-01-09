import cv2
import numpy as np
import streamlit as st
import av

from streamlit_webrtc import webrtc_streamer, RTCConfiguration

st.title("Real Time Object Detection")

# Download model if not present in the folder. This all file path not run when you are using local machine
modelFile = "ssd_mobilenet_frozen_inference_graph.pb"
configFile = "ssd_mobilenet_v2_coco_2018_03_29.pbtxt"
classFile = "coco_class_labels.txt"

import requests
from os import path

# Read tensorflow neural network
net = cv2.dnn.readNetFromTensorflow(modelFile, configFile)

# Check Class Labels.

with open(classFile, 'r') as fp:
    labels = fp.read().split("\n")


# Detect Objects.
def detect_objects(net, img):
    """Run object detection over the input image."""
    # Blob dimension (dim x dim)
    dim = 300

    mean = (0, 0, 0)

    # Create a blob from the image
    blob = cv2.dnn.blobFromImage(img, 1.0, (dim, dim), mean, True)
    # Pass blob to the network
    net.setInput(blob)
    # Perform Prediction
    objects = net.forward()
    return objects


# Display single prediction.
def draw_text(im, text, x, y):
    """Draws text label at a given x-y position with a black background."""
    fontface = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.7
    thickness = 1

    # Get text size
    textSize = cv2.getTextSize(text, fontface, font_scale, thickness)
    dim = textSize[0]
    baseline = textSize[1]

    # Use text size to create a black rectangle.
    cv2.rectangle(im, (x, y), (x + dim[0], y + dim[1] + baseline), (0, 0, 0), cv2.FILLED);
    # Display text inside the rectangle.
    cv2.putText(im, text, (x, y + dim[1]), fontface, font_scale, (0, 255, 255), thickness, cv2.LINE_AA)


# Display all predictions.
def draw_objects(im, objects, threshold=0.25):
    """Displays a box and text for each detected object exceeding the confidence threshold."""
    rows = im.shape[0]
    cols = im.shape[1]

    # For every detected object.
    for i in range(objects.shape[2]):
        # Find the class and confidence.
        classId = int(objects[0, 0, i, 1])
        score = float(objects[0, 0, i, 2])

        # Recover original cordinates from normalized coordinates
        x = int(objects[0, 0, i, 3] * cols)
        y = int(objects[0, 0, i, 4] * rows)
        w = int(objects[0, 0, i, 5] * cols - x)
        h = int(objects[0, 0, i, 6] * rows - y)
        # Check if the detection is of good quality
        if score > threshold:
            draw_text(im, "{}".format(labels[classId]), x, y)
            cv2.rectangle(im, (x, y), (x + w, y + h), (255, 255, 255), 2)

    return im


# every time image provide video frame
def callback(img):
    frame = img.to_ndarray(format="bgr24")
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    det_object = detect_objects(net, img_rgb)
    result = draw_objects(img_rgb, det_object, 0.6)

    img_bgr = cv2.cvtColor(result, cv2.COLOR_RGB2BGR)

    return av.VideoFrame.from_ndarray(img_bgr, format="bgr24")

webrtc_streamer(key="Real Time", video_frame_callback=callback, media_stream_constraints={
    "video": True,
    "audio": False},
                rtc_configuration=RTCConfiguration(
                    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
                ))
