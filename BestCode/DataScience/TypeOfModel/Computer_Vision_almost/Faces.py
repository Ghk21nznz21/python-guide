'''
Detect Faces
Recognize age, gender, race or a specific face 
'''


import cv2
import dlib
import numpy as np
from deepface import DeepFace #https://pypi.org/project/deepface/
#import face_recognition



class FaceDetector:
    def __init__(self, model_weights_path: str):
        # Load possible face detection models
        self.face_cascade = cv2.CascadeClassifier(model_weights_path)
        self.detector = dlib.get_frontal_face_detector()

    def draw_faces(self, img, faces):
        for (x, y, w, h) in faces:
            cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
        cv2.imshow("Result", img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def rotate_image(self, image, angle):
        center = tuple(np.array(image.shape[1::-1]) / 2)
        rotated_image = cv2.getRotationMatrix2D(center, angle, 1.0)
        #rotated_image = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)
        return rotated_image

    def rotate_face(self, image, face, angle):
        x, y, w, h = face
        rotated_image = self.rotate_image(image, angle)
        rotated_point = cv2.transform(np.array([[(x + w/2, y + h/2)]], dtype='float32'), rotated_image)[0][0]
        x, y = int(rotated_point[0] - w/2), int(rotated_point[1] - h/2)
        return (x, y, w, h)

    def face_detection_CascadeClassifier(self, img_path: str):
        # Load the image and convert it to grayscale
        img = cv2.imread(img_path)
        imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Detect faces using the Cascade Classifier
        faces = self.face_cascade.detectMultiScale(imgGray, scaleFactor=1.1, minNeighbors=4, minSize=(30, 30))
        self.draw(img, faces)
        return faces

    def face_detection_dLib(self, img_path: str, angle=0):
        # Load the image and convert it to grayscale
        img = cv2.imread(img_path)
        imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Rotate the image if needed
        if angle != 0:
            rotated_image = self.rotate_image(imgGray, angle)
        else:
            rotated_image = imgGray

        # Detect faces using dLib
        faces = self.detector(rotated_image)
        if angle != 0:
            detected_faces = [self.rotate_face(rotated_image, face, angle) for face in faces]
            self.draw_faces(img, detected_faces)
        else:
            self.draw_faces(img, faces)
        return faces
        

# Test
face_detect = FaceDetector("haarcascade_frontalface_default.xml") # missing due to size
face_detect.face_detection_CascadeClassifier("images/other_people.jpg")