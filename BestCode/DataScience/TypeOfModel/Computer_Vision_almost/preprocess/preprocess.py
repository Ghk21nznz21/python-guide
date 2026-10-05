import cv2 
import numpy as np


class Preproces:
    '''
    (optinal) Invert -> cv2.bitwise_not
    1- Grayscale
    (optinal) Blurr - cv2.GaussianBlur
    2- Binary -> cv2.threshold
    3- Noise Removal 
    Optionals:
        Dilation
        Erosion
    
    Those you dont have functions means that it really depends on arguments used 
    
    '''

    @staticmethod
    def invert(image):
        '''
        Some techniques require a black background, like:
            Erode
            Dilation
        '''
        inverted_image = cv2.bitwise_not(image)
        return cv2.imwrite("temp/inverted.jpg", inverted_image)
    
    @staticmethod
    def grayscale(image):
        ''' Dimension Reduction'''
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


    @staticmethod
    def noise_removal(image):
        ''' Noise: Random variations in pixel values 
        Process:
           Dilation: fill in small gaps or holes in the objects
           Erosion: shrinks back objects to normal size 
           Morphological Closing: closing small holes or gaps in the foreground, 
            like objects countors 
           Blurr: removing salt-and-pepper (bright-and-black pixels) noise. 
        '''
        kernel = np.ones((1, 1), np.uint8)
        image = cv2.dilate(image, kernel, iterations=1)
        kernel = np.ones((1, 1), np.uint8)
        image = cv2.erode(image, kernel, iterations=1)
        image = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)
        image = cv2.medianBlur(image, 3)
        return (image)


    def erosion(self, image, white_background: bool):
        ''' Shrinks foreground objects '''
        if white_background:
            self.invert(image = image)
        kernel = np.ones((2,2),np.uint8)
        image = cv2.erode(image, kernel, iterations=1)
        if white_background:
            # revert
            self.invert(image = image)
        return (image)

    def dilation(self, image, white_background: bool):
        ''' Expand foreground objects '''
        if white_background:
            # revert
            self.invert(image = image)
        kernel = np.ones((2,2),np.uint8)
        image = cv2.dilate(image, kernel, iterations=1)
        if white_background:
            # revert
            self.invert(image = image)
        return (image)

  
