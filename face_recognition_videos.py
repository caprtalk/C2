import pickle
import cv2
import dlib

from utils import face_rects
from utils import face_encodings
from utils import nb_of_matches

class face_detection_videos:
    # initialize the video stream
    def __init__(self):
        self.video_capture = cv2.VideoCapture(0)
        self.frame_nb = 0

    # display rectangle over detected faces
    def detect_faces(self, frame):
        frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rects = face_rects(rgb)
        # input image using the `face_rects` function
        for rect in rects:
            x1, y1, x2, y2 = rect.left(), rect.top(), rect.right(), rect.bottom()
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        return frame

    # run the video stream
    def run(self):
        # loop over frames from the video file stream
        while True:
            ret, frame = self.video_capture.read()
            frame = self.detect_faces(frame)
            cv2.imshow("frame", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        cv2.destroyAllWindows()
        self.video_capture.release()



class face_recognition_videos:
    def __init__(self):
        # load the encodings + names dictionary
        with open("encodings.pickle", "rb") as f:
            self.name_encodings_dict = pickle.load(f)
        # initialize the video stream
        self.video_capture = cv2.VideoCapture(0)
        # initialize the frame number
        self.frame_nb = 0

    def run(self):
        # loop over frames from the video file stream
        while True:
            ret, frame = self.video_capture.read()
            # resize the frame to have a width of 750px (to speedup processing)
            frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
            # convert the input frame from BGR to RGB
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            encodings = face_encodings(rgb)
            # names of detections
            names = []
            # loop over the encodings
            for encoding in encodings:
                counts = {}
                for (name, encodings) in self.name_encodings_dict.items():
                    counts[name] = nb_of_matches(encodings, encoding)
                # check if all the number of matches are equal to 0
                # if there is no match for any name, then we set the name to "Unknown"
                if all(count == 0 for count in counts.values()):
                    name = "Unknown"
                # otherwise, we get the name with the highest number of matches
                else:
                    name = max(counts, key=counts.get)
                # add the name to the list of names
                names.append(name)
            for rect, name in zip(face_rects(rgb), names):
                x1, y1, x2, y2 = rect.left(), rect.top(), rect.right(), rect.bottom()
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

                # draw the predicted name on the rectangle
                y = y1 - 15 if y1 - 15 > 15 else y1 + 15
                cv2.putText(frame, name, (x1, y), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)
            # show the output frame
            cv2.imshow("frame", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

