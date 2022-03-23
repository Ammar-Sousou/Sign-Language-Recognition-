import cv2
import numpy as np

cap = cv2.VideoCapture(1)

while(1):
    
    _, frame = cap.read()

    # Convert BGR to HSV
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # define range of skin color in HSV
    lower_skin = np.array([0, 20, 60])
    upper_skin = np.array([20, 180, 255])


    mask = cv2.inRange(hsv, lower_skin, upper_skin)
    
    
    res = cv2.bitwise_and(frame,frame, mask= mask)
    
    kernel = np.ones((5,5), np.uint8)
    res = cv2.dilate(res, kernel)    
    
    cv2.imshow('frame',frame)
    #cv2.imshow('mask',mask)
    cv2.imshow('res',res)

    k = cv2.waitKey(5) & 0xFF
    if k == 27:
        break

cv2.destroyAllWindows()