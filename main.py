import pickle
import cv2
import os

import face_encoding
import face_recognition_images
import face_recognition_videos
import utils

# load the encodings + names dictionary
with open("encodings.pickle", "rb") as f:
    name_encodings_dict = pickle.load(f)


# load the input image
image = cv2.imread("examples/ts1.jpeg")
# get the 128-d face embeddings for each face in the input image
encodings = utils.face_encodings(image)
