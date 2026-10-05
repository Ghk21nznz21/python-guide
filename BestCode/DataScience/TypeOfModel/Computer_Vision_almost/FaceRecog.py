import cv2
import dlib
import numpy as np
from deepface import DeepFace
import face_recognition
import matplotlib.pyplot as plt
import matplotlib.patches as patches


class FaceRecognition:
    def __init__(self):
        self.targets_encodings = []
        self.target_names = []

    def add_target(self, path: str, name: str):
        img = face_recognition.load_image_file(path)
        self.target_names.append(name)
        self.targets_encodings.append(face_recognition.face_encodings(img)[0])
        
    def draw_faces(self, img, faces, face_names):
        ''' Helps to visialize the faces detected in the image '''
        #for i, (x, y, w, h) in enumerate(faces):
        #    color = (255, 0, 0) if face_names[i] != 'Unknown' else (0, 255, 0)
        #    cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
        #cv2.imshow("Result", img)
        #cv2.waitKey(0)
        #cv2.destroyAllWindows()
        
        fig, ax = plt.subplots(1)
        ax.imshow(img)
        for i, (x, y, w, h) in enumerate(faces):
            color = 'r' if face_names[i] != 'Unknown' else 'g'
            rect = patches.Rectangle((x, y), w, h, linewidth=2, edgecolor=color, facecolor='none')
            ax.add_patch(rect)
        plt.show()

    def face_recognition(self, image):
        img = image.copy()
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

            stats = self.face_age_gender(face_img)
            face_names.append(name)

            print(f"Name: {name} Stats: {stats}")
        
        # FAILING TO RECOGNIZE 
        self.draw_faces(img, face_locations, face_names)  
        return face_locations, face_names

    def face_age_gender(self, img):
        ''' Saves section of image to temp file and uses DeepFace to analyze it 
        Returns Predictions for that face ("emotion", "age", "gender", "race")
        However due to unbalance of the dataset used to train the model, the predictions are not very accurate in regards 
        to some ages and races 
        '''
        cv2.imwrite("temp.jpg", img)
        return DeepFace.analyze(img_path="temp.jpg", enforce_detection=False)
    

face_recog = FaceRecognition()
face_recog.add_target("images/elon_musk1.jpg", "Elon Musk")
face_recog.face_recognition(cv2.imread("images/elon_musk1.jpg"))

face_recog.face_recognition(cv2.imread("images/other_people.jpg"))
