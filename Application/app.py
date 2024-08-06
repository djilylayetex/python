import cv2
import streamlit as st


# Correct path to the Haar cascade file
cascade_path = "C:/Users/Lenovo/anaconda3/Lib/site-packages/cv2/data/haarcascade_righteye_2splits.xml"

# Initialize the face cascade
face_cascade = cv2.CascadeClassifier(cascade_path)

if face_cascade.empty():
    raise ValueError("The cascade file was not loaded correctly. Please check the path.")

import cv2
import os
import numpy as np
from PIL import Image


def detect_faces():
    # Initialize the webcam
    cap = cv2.VideoCapture(0)
    while True:
        # Read the frames from the webcam
        ret, frame = cap.read()
        # Convert the frames to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # Detect the faces using the face cascade classifier
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)
        # Draw rectangles around the detected faces
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        # Display the frames
        cv2.imshow('Face Detection using Viola-Jones Algorithm', frame)
        # Exit the loop when 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        # Release the webcam and close all windows
    cap.release()
    cv2.destroyAllWindows()
def app():
    st.title("Face Detection using Viola-Jones Algorithm")
    st.write("Press the button below to start detecting faces from your webcam")
    # Add a button to start detecting faces
    if st.button("Detect Faces"):
        # Call the detect_faces function
        detect_faces()
    # Call the detect_faces function
if __name__ == "__main__":
    app()