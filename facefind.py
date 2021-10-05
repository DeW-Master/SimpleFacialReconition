import cv2 as cv

# setup recognizer
recognizer = cv.face.LBPHFaceRecognizer_create()
# load trained model
recognizer.read('trained_model/SWY_model.yml')
# name list
names = []
# global warning counter
WARNINGCOUNTER = 0


def warning():
    print("face found " + str(WARNINGCOUNTER))


def face_detector(p_image):
    # convert img to grey
    l_grayImg = cv.cvtColor(p_image, cv.COLOR_BGR2GRAY)
    # setup classifier
    l_faceDector = cv.CascadeClassifier('trained_model/haarcascade_frontalface_alt2.xml')
    # detect the face from img
    l_face = l_faceDector.detectMultiScale(l_grayImg, 1.1, 5, 0, (150, 150), (1500, 1500))
    for x, y, width, height in l_face:
        cv.rectangle(p_image, (x, y), (x + width, y + height), color=(0, 0, 255), thickness=3)
        # facial recognition
        ids, confidence = recognizer.predict(l_grayImg[y:y + height, x:x + width])
        if confidence > 80:
            global WARNINGCOUNTER
            WARNINGCOUNTER += 1
            if WARNINGCOUNTER > 100:
                warning()
                WARNINGCOUNTER = 0
            cv.putText(p_image, 'unknown', (x + 10, y - 10), cv.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 1)
        else:
            cv.putText(p_image, 'SWY', (x + 10, y - 10), cv.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 1)
    cv.imshow('result', p_image)


cam = cv.VideoCapture(0)

while True:
    l_flag, l_frame = cam.read()
    if not l_flag:
        break
    face_detector(l_frame)
    if ord('q') == cv.waitKey(100):
        break

# free the memory
cam.release()
cv.destroyAllWindows()
