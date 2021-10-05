import os
import numpy as np
import cv2
import cv2 as cv
from PIL import Image


def getImageAndLabels(p_Path):
    # store facial feature 2D MATRIX
    l_facialFeature = []
    # store name id data
    l_idLst = []
    # store images data
    l_imagePaths = [os.path.join(p_Path, f) for f in os.listdir(p_Path)]
    # setup classifier
    l_faceDetector = cv2.CascadeClassifier("trained_model/haarcascade_frontalface_alt2.xml")
    # open each image
    for l_eachImagePath in l_imagePaths:
        # PIL L function, L is grey image
        PIL_img = Image.open(l_eachImagePath).convert('L')
        # convert img to array
        img_numpy = np.array(PIL_img, 'uint8')
        # find the face feature using recognizer
        faces = l_faceDetector.detectMultiScale(img_numpy)
        # get each photo id and name
        id = int(os.path.split(l_eachImagePath)[1].split('_')[0])
        # prevent no face photo
        for x, y, width, height in faces:
            l_idLst.append(id)
            l_facialFeature.append(img_numpy[y:y + height, x:x + width])
    # print facial feature and ids
    print('id', l_idLst)
    print('fs:', l_facialFeature)
    return l_facialFeature, l_idLst


if __name__ == "__main__":
    # local photo
    l_imgPath = 'img_folder/SWY'
    faces, ids = getImageAndLabels(l_imgPath)
    # load name recognizer
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    # trainning
    recognizer.train(faces, np.array(ids))
    # save the model
    recognizer.write('trained_model/SWY_model.yml')
