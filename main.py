import cv2 as cv


def face_detector(p_image):
    # convert img to grey
    l_grayImg = cv.cvtColor(p_image, cv.COLOR_BGR2GRAY)
    # setup classifier
    l_faceDector = cv.CascadeClassifier('haarcascade_frontalface_default.xml')
    # detect the face from img
    l_face = l_faceDector.detectMultiScale(l_grayImg, 1.1, 5, 0, (150, 150), (1500, 1500))
    for x,y,width,height in l_face:
        cv.rectangle(p_image,(x,y),(x+width,y+height),color=(0,0,255),thickness=3)
    cv.imshow('Facial Detector',p_image)


