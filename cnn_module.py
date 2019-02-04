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



#랜덤 배치 생성 
def next_batch(num, data, labels):
    idx = np.arange(0,len(data))
    np.random.shuffle(idx)
    idx=idx[:num]
    data_shuffle=[data[i] for i in idx]
    labels_shuffle=[labels[i] for i in idx]
    
    return np.asarray(data_shuffle), np.asarray(labels_shuffle)
  


class CNN:
    def __init__(self,sess):
        self.sess = sess
        self.build_net()

    def build_net(self):
        self.X = tf.placeholder(tf.float32,[None,64,64,3])
        self.Y = tf.placeholder(tf.float32,[None,1])

        L1 = tf.layers.conv2d(self.X,64,[3,3],padding='SAME',activation=tf.nn.relu)
        L1 = tf.layers.max_pooling2d(L1,[2,2],[2,2],padding='SAME')

        L2 = tf.layers.conv2d(L1,128,[3,3],padding='SAME',activation=tf.nn.relu)
        L2 = tf.layers.max_pooling2d(L2,[2,2],[2,2],padding='SAME')

        L3 = tf.layers.conv2d(L2,256,[3,3],padding='SAME',activation=tf.nn.relu)
        L3 = tf.layers.max_pooling2d(L3,[2,2],[2,2],padding='SAME')

        L4 = tf.layers.flatten(L3)
        L4 = tf.layers.dense(L4,256,activation=tf.nn.relu)

        self.logits = tf.layers.dense(L4,n_classes,activation=None)

        self.hypothesis = tf.nn.sigmoid(self.logits)

        self.cost = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.logits,labels=self.Y))
        self.optimizer = tf.train.AdamOptimizer(0.001).minimize(self.cost)

        self.prediction = tf.round(self.hypothesis)
        is_correct=tf.equal(self.prediction,self.Y)
        self.accuracy = tf.reduce_mean(tf.cast(is_correct,tf.float32))

    #데이터 읽어오기 0:paper 1:rock
    def get_data(self,directory):
        data_dir = './database/paper and rock_resied/'#dir로 변경
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

        print('Reading Data Success!')

        return self.MakeData(x_data,y_data,0.9)

    #일정 비율로 training/test나누기
    def MakeData(self,x,y,prop):
    
        train_prop = prop
        num_data = int(x.shape[0]/2)
        num_train = int(num_data*train_prop)
        
        x_train = x[:num_train]
        self.x_train = np.append(x_train, x[num_data:num_data+num_train],axis=0)
        
        x_test = x[num_train:num_data]
        self.x_test = np.append(x_test, x[num_data+num_train:],axis=0)
        
        y_train = y[:num_train]
        self.y_train = np.append(y_train, y[num_data:num_data+num_train],axis=0)
        
        y_test = y[num_train:num_data]
        self.y_test = np.append(y_test, y[num_data+num_train:],axis=0)
        
        print('X data:',x.shape, ',Y data:',y.shape)
        print('X train:',x_train.shape, ',Y train:',y_train.shape)
        print('X test:',x_test.shape, ',Y test:',y_test.shape)
        
        return (self.x_train,self.y_train),(self.x_test,self.y_test)

    def load_saver(self,dir):
        saver = tf.train.Saver(tf.global_variables())#앞서 정의한 변수를 가져오는 함수

        ckpt = tf.train.get_checkpoint_state('./tf_saver')##dir로 변경
        if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
            saver.restore(self.sess, ckpt.model_checkpoint_path)
            print('Saver Load Success')
        else:
            self.sess.run(tf.global_variables_initializer())
            print('Saver Load Failed')

    #test데이터 정확도 출력 및 랜덤사진 결과
    def test(self):
        xs, ys = next_batch(self.x_test.shape[0],self.x_test,self.y_test)
        acc=self.sess.run(self.accuracy, feed_dict={self.X:xs, self.Y:ys})
        print('Accuracy:','%5f'%(100*acc),'%')

        test_num=random.randint(0,self.x_test.shape[0])
        #self.Showimg('test',test_num)
        print('real:',int(self.y_test[test_num][0]),classes[int(self.y_test[test_num][0])])
        print('prediction:',self.sess.run(self.prediction,feed_dict={self.X:np.reshape(self.x_test[test_num],[-1,64,64,3])}))

    #plt를 이용해서 사진 뽑아보기
    def Showimg(self, type, num):
        plt.grid(None) 
        if type=='train':
            img = self.x_train[num]
            plt.imshow(img)
            plt.show()
            
        elif type=='test':
            img = self.x_test[num]
            plt.imshow(img)
            plt.show()
            
        elif type=='data':
            img = self.x_data[num]
            plt.imshow(img)
            plt.show()

if __name__=='__main__':
    sess = tf.Session()
    model = CNN(sess)
    model.get_data('')
    model.load_saver('')
    model.test()
