#data이미지 중 사이즈가 200x200x3 다른 것 찾기

import cv2
import os

os.chdir('../database/')
dir_list = os.listdir()
cnt = 0
for data in dir_list:
    for i in range(1,101):
        img = cv2.imread('./'+data+'/'+data+' ('+str(i)+').jpg')
                
        h, w, ch_n = img.shape
        print(h,w,ch_n)
        if h==200 and w==200 and ch_n==3:
            cnt+=1
            
        #cv2.imshow("paper",img)
        #cv2.waitKey(0)  
if cnt==200:
    print('No Error')

cv2.destroyAllWindows()
