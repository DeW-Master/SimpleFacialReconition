import cv2 as cv

# set up camera
cam = cv.VideoCapture(0)

FLAG = 1
PHOTONUM = 1

# make sure that camera is open
while cam.isOpened():
    # get each frame that taken from camera
    ret_flag, l_eachFrame = cam.read()
    # show each photo
    cv.imshow("photo taken", l_eachFrame)
    l_keyInput = cv.waitKey(1)
    # press S to save photo
    if l_keyInput == ord('s'):
        cv.imwrite("img_folder/SWY/" + str(PHOTONUM) + "_SWY" + ".jpg", l_eachFrame)
        print('success to save ' + str(PHOTONUM) + ".jpg\n-------------------------------")
        PHOTONUM += 1
    # quit the photo taking
    elif l_keyInput == ord('q'):
        break

# free the memory
cam.release()
cv.destroyAllWindows()
