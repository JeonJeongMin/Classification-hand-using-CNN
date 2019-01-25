import os
import cv2

class_list = os.listdir('../database')

for class_name in class_list:
    limit=0
    data_list = os.listdir('../database/'+class_name)
    for i, data in enumerate(data_list):
        #if limit>5:
        #    break
        img = cv2.imread('../database/'+class_name+'/'+data)
        new_img = cv2.resize(img,dsize=(64,64))#원하는 사이즈 입
        
        tmp_str=''
        
        if i<10:
            tmp_str='00'
        elif i<100:
            tmp_str='0'
        else:
            tmp_str=''
        
            
        cv2.imwrite('./data_resized/'+class_name+'/'+class_name+tmp_str+str(i)+'.jpg',new_img)
        #print('./data_resized/'+class_name+'/'+class_name+str(i)+'.jpg')
        #limit+=1
        
