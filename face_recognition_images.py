import pickle
import cv2

from utils import face_rects
from utils import face_encodings
from utils import nb_of_matches

def face_recognition_images(image, name_encodings_dict):
    # load the encodings + names dictionary
    with open("encodings.pickle", "rb") as f:
        name_encodings_dict = pickle.load(f)

    # load the input image
    image = cv2.imread(f'examples/{image}.jpeg')
    encodings = face_encodings(image)
    names = []


    # loop over the encodings
    for encoding in encodings:
        # initialize a dictionary to store the name of the
        counts = {}
        # loop over the known encodings
        for (name, encodings) in name_encodings_dict.items():
            counts[name] = nb_of_matches(encodings, encoding)
        if all(count == 0 for count in counts.values()):
            name = "Unknown"

        else:
            name = max(counts, key=counts.get)

        names.append(name)


    for rect, name in zip(face_rects(image), names):
        # get the bounding box for each face using the `rect` variable
        x1, y1, x2, y2 = rect.left(), rect.top(), rect.right(), rect.bottom()
        # draw the bounding box of the face along with the name of the person
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(image, name, (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)

    cv2.imshow("image", image)
    cv2.waitKey(0)

