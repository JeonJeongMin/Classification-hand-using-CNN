import numpy as np
import cv2
#0 64 195 255 27 255 day
#0 14 11 159 117 255 night

def onChange(x):
    pass
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 300)

cv2.namedWindow('color_HSV')
#트랙바 생성
cv2.createTrackbar('H_low', 'color_HSV', 0,179,onChange)
cv2.createTrackbar('H_high', 'color_HSV', 14,179,onChange)
cv2.createTrackbar('V_low', 'color_HSV', 11,255,onChange)
cv2.createTrackbar('V_high', 'color_HSV', 159,255,onChange)
cv2.createTrackbar('S_low', 'color_HSV', 117,255,onChange)
cv2.createTrackbar('S_high', 'color_HSV', 255,255,onChange)

while True:

    _, frame = cap.read()
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    #트랙바로부터 값받아오기
    h_high = cv2.getTrackbarPos('H_high', 'color_HSV')
    h_low = cv2.getTrackbarPos('H_low', 'color_HSV')
    s_high = cv2.getTrackbarPos('S_high', 'color_HSV')
    s_low = cv2.getTrackbarPos('S_low', 'color_HSV')
    v_high = cv2.getTrackbarPos('V_high', 'color_HSV')
    v_low = cv2.getTrackbarPos('V_low', 'color_HSV')
    
    lower_thr = np.array([h_low, s_low, v_low])
    upper_thr = np.array([h_high,s_high,v_high])

    mask = cv2.inRange(hsv, lower_thr, upper_thr)

    ret, thr = cv2.threshold(mask,127,255,0)
    contours, _ = cv2.findContours(thr, cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(frame, contours, -1, (0,255,0),1)

    
    cv2.imshow('color_HSV',mask)
    cv2.imshow('origin',frame)
    
    k = cv2.waitKey(1) & 0xFF#esc key
    if k == 27:
        break

    
    
cv2.destroyAllWindows()
