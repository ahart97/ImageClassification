import os
import torch

from CNN_model import MyCNN, trainCNN
from signal_utils import CustomDataset, PickleLoad, PickleDump
from load_data import Load_CIFAR10_Data
from ComputerVision import CameraClassifier


if __name__ == '__main__':

    try:
        Trained_CNN = PickleLoad('CNN_model.pickle')
    except:
        X, y, X_test, y_test, classes = Load_CIFAR10_Data(os.path.join(os.getcwd(), 'ImageClassification', 'data', 'cifar-10'))
        train_dataset = CustomDataset(X, y)
        myCNN = MyCNN(classes)
        Trained_CNN = trainCNN(myCNN, train_dataset)
        PickleDump(Trained_CNN, 'CNN_model.pickle')


    webcam = CameraClassifier()

    #TODO: Update this to be in a tkinter GUI to have a preview of the image and better UI
    try:
        while True:
            action = input("Classify Image (y or n)?")
            if action == 'y':
                #Get a process image
                webcam.ReadImage()
                webcam.ProcessImage()

                #Run through the algorithm
                prediction = torch.argmax(Trained_CNN(webcam.processedImage)).detach().numpy().astype(int)
                print('The image is: {}'.format(Trained_CNN.classes[prediction]))


    except KeyboardInterrupt:
        print('Program Shutdown')
