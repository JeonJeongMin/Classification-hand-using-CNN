import cv2
import numpy as np

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)


while True:
    
    ret, frame = cap.read()
    
    if not ret:
        break

    cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    #YCrCb 변환
    hsv = cv2.cvtColor(frame,cv2.COLOR_BGR2YCrCb)

    #Cr:133~173, Cb:77~127
    mask_hand = cv2.inRange(hsv,np.array([0,133,77]),np.array([255,173,127]))
    
    cv2.imshow("Hands",mask_hand)
    cv2.imshow("origin",frame)

    if cv2.waitKey(1)>0:
        break

cap.release()
cv2.destroyAllWindows()

