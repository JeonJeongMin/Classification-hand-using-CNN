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

#HSV색상을 이용하여 피부색검출
def maskedByHSV(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h_high = cv2.getTrackbarPos('H_high', 'color_HSV')
    h_low = cv2.getTrackbarPos('H_low', 'color_HSV')
    s_high = cv2.getTrackbarPos('S_high', 'color_HSV')
    s_low = cv2.getTrackbarPos('S_low', 'color_HSV')
    v_high = cv2.getTrackbarPos('V_high', 'color_HSV')
    v_low = cv2.getTrackbarPos('V_low', 'color_HSV')

    lower_thr = np.array([h_low, s_low, v_low])
    upper_thr = np.array([h_high,s_high,v_high])

    mask = cv2.inRange(hsv, lower_thr, upper_thr)
    return mask

#얼굴에 사각형 그리기
def DrawRect(img, rects, color):
    for x,y,w,h in rects:
        cv2.rectangle(img, (x, y), (x+w, y+h), color, 2)

def DrawContourHand(img,mask):
    ret, thr = cv2.threshold(mask,127,255,0)
    contours, _ = cv2.findContours(thr, cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    print(len(contours))
    cv2.drawContours(img, contours[len(contours)-1], -1, (0,255,0),2)

def onChange(x):
    pass

def createTrackBarHSV():
    cv2.namedWindow('color_HSV')
    #트랙바 생성
    cv2.createTrackbar('H_low', 'color_HSV', 0,179,onChange)
    cv2.createTrackbar('H_high', 'color_HSV', 21,179,onChange)
    cv2.createTrackbar('V_low', 'color_HSV', 105,255,onChange)
    cv2.createTrackbar('V_high', 'color_HSV', 241,255,onChange)
    cv2.createTrackbar('S_low', 'color_HSV', 42,255,onChange)
    cv2.createTrackbar('S_high', 'color_HSV', 255,255,onChange)
    
def startVideo():
    HSV_first_flag = True
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
        if(HSV_first_flag):
            createTrackBarHSV()
            HSV_first_flag=False
        masked = maskedByHSV(frame)    
        #masked = maskedByYCrCb(frame)
        
        #마스크 이미지에 얼굴 가리기
        DrawCircle(masked, faces, (255,255,255))

        #손 Contour그리기
        DrawContourHand(frame,masked)
            
        cv2.imshow("Face",frame)
        cv2.imshow("masked",masked)
        
        if cv2.waitKey(1)>0:
            break

    cap.release()
    cv2.destroyAllWindows()


#카메라 작동
def capture():
  cap = cv2.VideoCapture(0)
  cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
  cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 300)

  while True:

    _, frame = cap.read()
    resized = cv2.resize(frame,(64,64))
    resized = np.reshape(resized,[-1,64,64,3])
    predicted = sess.run(prediction,feed_dict={X:resized})
    print('prediction:',predicted)

    #글씨 양식
    red = (0,0,255)
    green = (0,255,0)
    thickness = 2
    frame_h = frame.shape[0]
    frame_w = frame.shape[1]
    org = ((int)(frame_w / 2) - 200, (int)(frame_h / 2) - 100)
    font = cv2.FONT_ITALIC
    fontScale = 1.2

    #글씨 출력
    if predicted == 0:
      cv2.putText(frame, 'paper', org, font, fontScale, red, thickness, cv2.LINE_AA)
    else:
      cv2.putText(frame, 'rock', org, font, fontScale, green, thickness, cv2.LINE_AA)
      
    cv2.imshow('origin',frame)

    
    k = cv2.waitKey(1) & 0xFF#esc key
    if k == 27:
        break    
  cv2.destroyAllWindows()
