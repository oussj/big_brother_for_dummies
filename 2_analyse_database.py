import face_recognition
import PIL
import numpy as np
import os
import pandas as pd
import json
from helpers import load_image_file

pictures_dir = 'database/'

pictures = os.listdir(pictures_dir)
face_features = []
names = []

for counter, name in enumerate(pictures):
    if '.jpg' not in name:
        continue
    image = load_image_file(pictures_dir + name)
    try:
        face_features.append(face_recognition.face_encodings(image)[0])
        name = name.replace('_', ' ')
        names.append(name.replace('.jpg', ''))
    except IndexError:
        print(name + ' no face in picture')
        continue
    except RuntimeError:
        print(name + 'unsupported image')

features_df = pd.DataFrame(face_features, names)
features_df.to_csv('face_descriptors.csv')
