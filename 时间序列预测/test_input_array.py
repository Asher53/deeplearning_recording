# coding: utf-8
from __future__ import print_function
import numpy as np
import matplotlib

matplotlib.use('agg')
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.contrib.timeseries.python.timeseries import NumpyReader

'''
利用np.sin，生成一个实验用的时间序列数据，这个时间序列数据实际上就是在正弦曲线上加上了上升的趋势和一些随机的噪声
'''
x = np.array(range(1000))
# 上界，下界，输出数目
noise = np.random.uniform(-0.2, 0.2, 1000)
y = np.sin(np.pi * x / 100) + x / 200. + noise
# plt.plot(x, y)
# plt.savefig('timeseries_y.jpg')

'''
TFTS读入x和y的方式非常简单,我们首先把x和y变成python中的词典（变量data）。
变量data中的键值tf.contrib.timeseries.TrainEvalFeatures.TIMES实际就是一个字符串“times”，
而tf.contrib.timeseries.TrainEvalFeatures.VALUES就是字符串”values”。
所以上面的定义直接写成“data = {‘times’:x, ‘values’:y}”也是可以的。写成比较复杂的形式是为了和源码中的写法保持一致。
'''
data = {
    tf.contrib.timeseries.TrainEvalFeatures.TIMES: x,
    tf.contrib.timeseries.TrainEvalFeatures.VALUES: y,
}


reader = NumpyReader(data)

'''
得到的reader有一个read_full()方法，它的返回值就是时间序列对应的Tensor，我们可以用下面的代码试验一下：
不能直接使用sess.run(reader.read_full())来从reader中取出所有数据。原因在于read_full()方法会产生读取队列，而队列的线程此时还没启动，
我们需要使用tf.train.start_queue_runners启动队列，才能使用sess.run()来获取值。
'''

with tf.Session() as sess:
    full_data = reader.read_full()
    # 使用 tf.train.Coordinator()来创建一个线程管理器（协调器）对象。
    coord = tf.train.Coordinator()
    # 只有调用tf.train.start_queue_runners才会让系统中的队列真正的运行
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)
    print(sess.run(full_data))
    coord.request_stop()
#

'''
tf.contrib.timeseries.RandomWindowInputFn会在reader的所有数据中，随机选取窗口长度为window_size的序列，
并包装成batch_size大小的batch数据。换句话说，一个batch内共有batch_size个序列，每个序列的长度为window_size。
'''
train_input_fn = tf.contrib.timeseries.RandomWindowInputFn(
    reader, batch_size=2, window_size=10)
'''
以batch_size=2, window_size=10为例，我们可以打出一个batch内的数据：
'''


with tf.Session() as sess:
    batch_data = train_input_fn.create_batch()
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)
    one_batch = sess.run(batch_data[0])
    coord.request_stop()

print('one_batch_data:', one_batch)
