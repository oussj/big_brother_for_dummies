import PIL
import numpy as np
import pandas as pd

def load_image_file(file, mode='RGB'):
    try:
        im = PIL.Image.open(file)
        if mode:
            im = im.convert(mode)
        return np.array(im)
    except OSError:
        print(file + ' has a problem')
        return np.array(np.ones(300))

def load_database(filepath='face_descriptors.csv'):
    database = pd.read_csv(filepath, index_col=0)
    names = []
    face_descriptors = []
    for index, row in database.iterrows():
        names.append(index)
        face_descriptors.append([row[i] for i in range(128)])
    return names, face_descriptors