import pickle
import cv2
import os

from utils import get_image_paths
from utils import face_encodings

def face_encoding():
    root_dir = "dataset"
    class_names = os.listdir(root_dir)

    # get the paths to the images
    image_paths = get_image_paths(root_dir, class_names)
    # initialize a dictionary to store the name of each person and the corresponding encodings
    name_encondings_dict = {}


    # initialize the number of images processed
    nb_current_image = 1
    # now we can loop over the image paths, locate the faces, and encode them
    for image_path in image_paths:
        print(f"Image processed {nb_current_image}/{len(image_paths)}")
        image = cv2.imread(image_path)
        encodings = face_encodings(image)
        name = image_path.split(os.path.sep)[-2]
        e = name_encondings_dict.get(name, [])
        e.extend(encodings)
        name_encondings_dict[name] = e
        nb_current_image += 1

    # save the name encodings dictionary to disk
    with open("encodings.pickle", "wb") as f:
        pickle.dump(name_encondings_dict, f)

