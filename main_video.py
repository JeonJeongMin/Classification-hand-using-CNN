import cv2
import numpy as np
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

face_cascade = cv2.CascadeClassifier("./haarcascade/haarcascade_frontalface_default.xml")

#얼굴을 가리기 위해 원 그리기
def DrawCircle(img, rects, color):
    for x, y, w, h in rects:
        cv2.circle(img, (x+int(w/2),y+int(h/2)), int((h+100)/2), (0,0,0), -1)
        
#YCrCb색상을 이용하여 피부색검출
def maskedByYCrCb(img):
    #YCrCb 변환
    ycrcb = cv2.cvtColor(img,cv2.COLOR_BGR2YCrCb)
    #Cr:133~173, Cb:77~127
    masked = cv2.inRange(ycrcb,np.array([0,133,77]),np.array([255,173,127]))
    return masked

#얼굴에 사각형 그리기
def DrawRect(img, rects, color):
    for x,y,w,h in rects:
        cv2.rectangle(img, (x, y), (x+w, y+h), color, 2)

def findHand(img):
    ret, thr = cv2.threshold(img,127,255,0)
    contours, _ = cv2.findContours(thr, cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(img, contours, -1, (0,255,0),1)

    
while True:
    
    ret, frame = cap.read()
    
    if not ret:
        print('No captured video!')
        break

    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray,1.3,5)

    #얼굴에 초록 사각형 그리기
    DrawRect(frame, faces, (0,255,0))

    #원본 영상에 살색검출
    masked = maskedByYCrCb(frame)

    #마스크 이미지에 얼굴 가리기
    DrawCircle(masked, faces, (255,255,255))

    #손 Contour그리기
    #findhandContour(masked)


        
    cv2.imshow("Face",frame)
    cv2.imshow("masked",masked)
    if cv2.waitKey(1)>0:
        break

cap.release()
cv2.destroyAllWindows()
