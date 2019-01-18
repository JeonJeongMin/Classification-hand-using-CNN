import cv2
import numpy as np

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

face_cascade = cv2.CascadeClassifier("./haarcascade/haarcascade_frontalface_default.xml")


while True:
    
    ret, frame = cap.read()
    
    #카메라오류
    if not ret:
        break

    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray,1.08,5)
    for x,y,w,h in faces:
        cv2.rectangle(frame, (x,y), (x+w, y+h), (0,255,0), 2)
    
    cv2.imshow("Test",frame)
    if cv2.waitKey(1)>0:
        break

cap.release()
cv2.destroyAllWindows()
