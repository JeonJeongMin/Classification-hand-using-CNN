import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import random
import os
import cv2
tf.set_random_seed(777)

learning_rate = 0.001
training_epoch=10
batch_size=100
n_classes=1
img_size=[1,64,64,3]
classes=['paper','rock']
global_step = tf.Variable(0, trainable=True, name='global_step')#초기값0 trainable-카운트

#일정 비율로 training/test나누기
def MakeData(x,y,prop):
  
  train_prop = prop
  num_data = int(x.shape[0]/2)
  num_train = int(num_data*train_prop)
  
  x_train = x[:num_train]
  x_train = np.append(x_train, x[num_data:num_data+num_train],axis=0)
  
  x_test = x[num_train:num_data]
  x_test = np.append(x_test, x[num_data+num_train:],axis=0)
  
  y_train = y[:num_train]
  y_train = np.append(y_train, y[num_data:num_data+num_train],axis=0)
  
  y_test = y[num_train:num_data]
  y_test = np.append(y_test, y[num_data+num_train:],axis=0)
  
  print('X data:',x.shape, ',Y data:',y.shape)
  print('X train:',x_train.shape, ',Y train:',y_train.shape)
  print('X test:',x_test.shape, ',Y test:',y_test.shape)
  
  return (x_train,y_train),(x_test,y_test)

#랜덤 배치 생성 
def next_batch(num, data, labels):
    idx = np.arange(0,len(data))
    np.random.shuffle(idx)
    idx=idx[:num]
    data_shuffle=[data[i] for i in idx]
    labels_shuffle=[labels[i] for i in idx]
    
    return np.asarray(data_shuffle), np.asarray(labels_shuffle)
  
  
#plt를 이용해서 사진 뽑아보기
def Showimg(type, num):
  plt.grid(None) 
  if type=='train':
    img = x_train[num]
    plt.imshow(img)
    plt.show()
    print(int(y_train[num][0]),classes[int(y_train[num][0])])
    
  elif type=='test':
    img = x_test[num]
    plt.imshow(img)
    plt.show()
    print(int(y_test[num][0]),classes[int(y_test[num][0])])
    
  elif type=='data':
    img = x_data[num]
    plt.imshow(img)
    plt.show()
    print(int(y_data[num][0]),classes[int(y_data[num][0])])

#데이터 읽어오기 0:paper 1:rock
data_dir = './database/paper and rock_resied/'
class_list = os.listdir(data_dir)
first_flag = True

data_num=900

for data_class in class_list:
    data_class_list = os.listdir(data_dir+data_class)
    limit_num=data_num
    for data_name in data_class_list:
        if limit_num==0:
          break
                
        img = plt.imread(data_dir+data_class+'/'+data_name)
        img = np.reshape(img,img_size)
        if first_flag:
            x_data=img
            first_flag = False
        else:
            x_data=np.append(x_data,img,axis=0)

        limit_num-=1

y_data = np.zeros([data_num])
y_data = np.append(y_data, np.ones([data_num]))
y_data = np.reshape(y_data,[data_num*2,-1])

(x_train, y_train), (x_test, y_test) = MakeData(x_data,y_data,0.9)

print('Reading Data Success!')

X = tf.placeholder(tf.float32,[None,64,64,3])
Y = tf.placeholder(tf.float32,[None,1])

L1 = tf.layers.conv2d(X,64,[3,3],padding='SAME',activation=tf.nn.relu)
L1 = tf.layers.max_pooling2d(L1,[2,2],[2,2],padding='SAME')

L2 = tf.layers.conv2d(L1,128,[3,3],padding='SAME',activation=tf.nn.relu)
L2 = tf.layers.max_pooling2d(L2,[2,2],[2,2],padding='SAME')

L3 = tf.layers.conv2d(L2,256,[3,3],padding='SAME',activation=tf.nn.relu)
L3 = tf.layers.max_pooling2d(L3,[2,2],[2,2],padding='SAME')

L4 = tf.layers.flatten(L3)
L4 = tf.layers.dense(L4,256,activation=tf.nn.relu)

logits = tf.layers.dense(L4,n_classes,activation=None)

hypothesis = tf.nn.sigmoid(logits)

cost = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=logits,labels=Y))
optimizer = tf.train.AdamOptimizer(0.001).minimize(cost)

prediction = tf.round(hypothesis)
is_correct=tf.equal(prediction,Y)
accuracy = tf.reduce_mean(tf.cast(is_correct,tf.float32))

sess=tf.Session()
saver = tf.train.Saver(tf.global_variables())#앞서 정의한 변수를 가져오는 함수

ckpt = tf.train.get_checkpoint_state('./tf_saver')
if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
    saver.restore(sess, ckpt.model_checkpoint_path)
    print('Saver Load Success')
else:
    sess.run(tf.global_variables_initializer())
    print('Saver Load Failed')

#test데이터 정확도 출력 및 랜덤사진 결과
def test():
  xs, ys = next_batch(x_test.shape[0],x_test,y_test)
  acc=sess.run(accuracy, feed_dict={X:xs, Y:ys})
  print('Accuracy:','%5f'%(100*acc),'%')

  test_num=random.randint(0,x_data.shape[0])
  Showimg('data',test_num)
  print('prediction:',sess.run(prediction,feed_dict={X:np.reshape(x_data[test_num],[-1,64,64,3])}))

def capture():
  cap = cv2.VideoCapture(0)
  cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
  cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 300)

  while True:

    _, frame = cap.read()
    resized = cv2.resize(frame,(64,64))
    resized = np.reshape(resized,[-1,64,64,3])
    print('prediction:',sess.run(prediction,feed_dict={X:resized}))

    cv2.imshow('origin',frame)

    
    k = cv2.waitKey(1) & 0xFF#esc key
    if k == 27:
        break    
  cv2.destroyAllWindows()
