import matplotlib.pyplot as plt
import numpy as np
import os
import csv
import pandas as pd

class_list = os.listdir('../database')

first_flag=True

for class_name in class_list:
    limit=0
    data_list = os.listdir('../database/'+class_name)
    for data in data_list:
        if limit>99:
            break
        img = plt.imread('../database/'+class_name+'/'+data)
        img = np.reshape(img,[1,120000])
        if first_flag:
            csv_data = img
            first_flag=False
        else:
            csv_data = np.append(csv_data,img,axis=0)
        limit+=1
        
'''
f= open('img.csv','w', encoding = 'utf-8', newline='')
wr = csv.writer(f)
for i in range(1,10):
    wr.writerow([1,'jm'])
f.close
'''
dataframe = pd.DataFrame(csv_data)
dataframe.to_csv("./img.csv",header=False,index=False)
