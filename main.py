import pickle
import cv2
import os

import face_encoding
import face_recognition_images
import face_recognition_active
import utils


def main():
    image = input("image name: ")
    file = cv2.imread(f'examples/{image}.jpeg')
    test1_encoding = utils.face_encodings(file)
    print(test1_encoding)
    # load the encodings + names dictionary
    # with open("encodings.pickle", "rb") as f:
    #     name_encodings_dict = pickle.load(f)


    # load the input image
    # face_recognition_images.face_recognition_images(file, test1_encoding)`



if __name__ == '__main__':
    main()
