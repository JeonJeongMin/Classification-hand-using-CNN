import cv2
import numpy as np

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

face_cascade = cv2.CascadeClassifier("./haarcascade/haarcascade_frontalface_default.xml")

def DrawCircle(img, rects, color):
    for x, y, w, h in rects:
        cv2.circle(img, (x+int(w/2),y+int(h/2)), int(h/2), (255,255,255), -1)
        
    
while True:
    
    ret, frame = cap.read()
    
    if not ret:
        break

    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray,1.3,5)
    for x,y,w,h in faces:
        #cv2.circle(frame, (x+int(w/2),y+int(h/2)), int(h/2), (255,255,255), -1)
        DrawCircle(frame, faces, (255,255,255))
        
        
    cv2.imshow("Face",frame)
    if cv2.waitKey(1)>0:
        break

cap.release()
cv2.destroyAllWindows()
