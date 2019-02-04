import video_module as vd
import cnn_module as cnn
import tensorflow as tf
import matplotlib.pyplot as plt

sess = tf.Session()
model = cnn.CNN(sess)#CNN network생성
model.load_saver('dir')
model.get_data('dir')
model.test()

vd.capture(sess,model.prediction,model.X)
