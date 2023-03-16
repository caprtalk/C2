import pickle
import cv2
import os
import numpy as np


import face_encoding
import face_recognition_images
import face_recognition_active
import utils


def main():
    option = input("1. Face detection active\n2. Face recognition active\n3. Face recognition images\n4. Process Dataset\n5. Face recognition images with input\n")
    if option == '3':
        image = input("image name: ")
        face_recognition_images.face_recognition_images(image, 'encodings.pickle')
    if option == '4':
        face_encoding.face_encoding()


if __name__ == '__main__':
    main()
