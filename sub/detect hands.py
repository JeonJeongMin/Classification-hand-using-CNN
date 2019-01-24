'''
import cv2
import numpy as np

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)


while True:
    
    ret, frame = cap.read()
    
    if not ret:
        break

    #YCrCb 변환
    ycrcb = cv2.cvtColor(frame,cv2.COLOR_BGR2YCrCb)

    #Cr:133~173, Cb:77~127
    mask_hand = cv2.inRange(ycrcb,np.array([0,133,77]),np.array([255,173,127]))

    #contours 찾기
    ret, thr = cv2.threshold(mask_hand,127,255,0)
    contours, _ = cv2.findContours(thr, cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

    #contours 그리기
    cv2.drawContours(frame, contours, -1, (0,255,0),1)


    cv2.imshow("Hands",mask_hand)
    cv2.imshow("origin",frame)

    if cv2.waitKey(1)>0:
        break

cap.release()
cv2.destroyAllWindows()

#이미지를 불러와서 테스트
'''
import cv2
import numpy as np
img = cv2.imread('./hand_sample.jpg')

#YCrCb 변환
ycrcb = cv2.cvtColor(img,cv2.COLOR_BGR2YCrCb)
#Cr:133~173, Cb:77~127
mask_hand = cv2.inRange(ycrcb,np.array([0,133,77]),np.array([255,173,127]))

#thr값을 통과한 이미지 받아오기
ret, thr = cv2.threshold(mask_hand,127,255,0)

#thr에서 contour찾기
contours, _ = cv2.findContours(thr, cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

#0번째 contour 사각형 그리기
cnt = contours[0]
x,y,w,h = cv2.boundingRect(cnt)

#가장작은 사각형 만들기(불필요)
rect = cv2.minAreaRect(cnt)
box = cv2.boxPoints(rect)
box = np.int0(box)

#사각형 그리기
cv2.rectangle(img, (x-50,y-50),(x+w+50,y+h+50),(0,0,255),1)

#손모양 자르기
hands = img[y-50+1:y+h+50,x-50+1:x+w+50]
hands = cv2.resize(hands, (64,64))

#contour그리기
#cv2.drawContours(img, [box], 0, (0,255,0),1)

#center찾고 점그리기
center_x, center_y = int(x+w/2), int(y+h/2)
cv2.circle(img, (center_x,center_y),1, (0,0,255),-1)


#cv2.imshow("thr",~thr)
#cv2.imshow("mask_hand",mask_hand)
cv2.imshow("Origin",img)
cv2.imshow("hands",hands)


cv2.waitKey(0)

cv2.destroyAllWindows()

