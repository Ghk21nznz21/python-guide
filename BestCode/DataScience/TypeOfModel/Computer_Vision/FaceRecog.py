import cv2
import dlib
import numpy as np
from deepface import DeepFace
import face_recognition


class FaceRecognition:
    def __init__(self):
        self.targets_encodings = []
        self.target_names = []

    def add_target(self, path: str, name: str):
        img = face_recognition.load_image_file(path)
        self.target_names.append(name)
        self.targets_encodings.append(face_recognition.face_encodings(img)[0])

    def face_recognition(self, image):
        small_frame = cv2.resize(image, (0, 0), fx=0.25, fy=0.25)
        rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)
        face_names = []
        for face_encoding in face_encodings:
            matches = face_recognition.compare_faces(self.targets_encodings, face_encoding)
            name = "Unknown"

            face_distances = face_recognition.face_distance(self.targets_encodings, face_encoding)
            best_match_index = np.argmin(face_distances)

            if matches[best_match_index]:
                name = self.target_names[best_match_index]

            top, right, bottom, left = face_locations[0]
            face_img = image[top:bottom, left:right]

            age, gender = self.face_age_gender(face_img)
            face_names.append(name)

            print(f"Name: {name}, Age: {age}, Gender: {gender}")

        return face_locations, face_names

    def face_age_gender(self, img):
        cv2.imwrite("temp.jpg", img)
        demography = DeepFace.analyze(face_path="temp.jpg", actions=['age', 'gender'])
        age = demography['age']
        gender = demography['gender']
        return age, gender
    
# Testing 
face_recog = FaceRecognition()
face_recog.add_target("images/elon_musk.jpg", "Elon Musk")
face_recog.face_recognition(cv2.imread("images/elon_musk.jpg"))

face_recog.face_recognition(cv2.imread("images/other_people.jpg"))

face_recog.face_age_gender(cv2.imread("images/other_people.jpg"))

                            