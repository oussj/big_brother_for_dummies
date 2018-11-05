import cv2
import sys
import pandas as pd
from helpers import load_database
import PIL
import numpy as np
import face_recognition

names, face_descriptors = load_database()

opencv_path = 'big_brother/lib/python3.7/site-packages/cv2/data/'
face_cascade = cv2.CascadeClassifier(opencv_path + 'haarcascade_frontalface_default.xml')

video_capture = cv2.VideoCapture(0)

while True:
    # Capture frame-by-frame
    ret, frame = video_capture.read()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)
    
    # Draw a rectangle around the faces
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

        pil_im = PIL.Image.fromarray(frame[y:y+h, x:x+w])
        face = np.array(pil_im.convert('RGB'))
        try:
            face_descriptor = face_recognition.face_encodings(face)[0]
        except Exception:
            continue
        distances = np.linalg.norm(face_descriptors - face_descriptor, axis=1)
        if(np.min(distances) < 20):
            found_name = names[np.argmin(distances)]
            print(found_name)
            #y = top - 15 if top - 15 > 15 else top + 15
            cv2.putText(frame, found_name, (x, y-15), cv2.FONT_HERSHEY_SIMPLEX,
                        0.75, (0, 255, 0), 2)

    # Display the resulting frame
    cv2.imshow('Video', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything is done, release the capture
video_capture.release()
cv2.destroyAllWindows()