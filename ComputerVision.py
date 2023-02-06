import cv2
import numpy as np
import matplotlib.pyplot as plt

class CameraClassifier():
    def __init__(self):
        cam_port = 0
        self.cam = cv2.VideoCapture(cam_port)

    def ReadImage(self):
        result = False
        while not result:
            result, image = self.cam.read()

        self.capturedImage = image

    def ShowImage(self):
        cv2.imshow('Captured Image', self.capturedImage)

    def ProcessImage(self):
        """
        Process image into the 32x32 format and RGB
        """
        rezied_image = cv2.resize(self.capturedImage, (32,32))
        self.processedImage = np.flip(np.swapaxes(rezied_image.transpose(),1,2), axis=0)

    
if __name__ == '__main__':
    myCV = CameraClassifier()

    myCV.ReadImage()
    myCV.ProcessImage()
