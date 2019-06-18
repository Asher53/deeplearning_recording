'''
定义一个基本RNN单元和一个LSTM基本单元
RNNCell的类属性state_size和output_size分别规定了隐层的大小和输出向量的大小。
通常是以batch形式输入数据，input的形状为（batch_size，input_size），
调用call函数时对应的隐层的形状是（batch_size，state_size），
输出的形状是（batch_size，output_size）。
'''

# import tensorflow as tf
#
# rnn_cell = tf.keras.layers.SimpleRNNCell(128)
# print(rnn_cell.output_size)  # 128
# print(rnn_cell.state_size)  # 128


# import tensorflow as tf
# lstm_cell = tf.nn.rnn_cell.LSTMCell(num_units=128)
# print(lstm_cell.state_size)   # LSTMStateTuple(c=128, h=128)


'''
单层RNN能力有限，需要多层RNN。
将x输入到第一层RNN后得到隐层状态h，这个隐层状态相当于第二层RNN的输入，第二层RNN的隐层状态又相当于第三层RNN的输入，以此类推。三层RNN串联
在TensorFlow中，使用tf.nn.rnn_cell.MultiRNNCell函数对RNN进行堆叠
'''

# import tensorflow as tf
# import numpy as np
#
#
# # 每次调用这个函数返回一个BasicRNNCell
# def get_a_cell():
#     return tf.nn.rnn_cell.BasicRNNCell(num_units=128)
#
#
# # 用tf.nn.rnn _cell_MultiRNNCell创建三层RNN
# cell = tf.nn.rnn_cell.MultiRNNCell([get_a_cell() for _ in range(3)])
#
# # 得到的cell实际也是RNNCell的子类
# # 它的state_size是（128，128,128）代表3个隐层状态，每个隐层状态是128
# print(cell.state_size)  # (128, 128, 128)
#
# # 使用对应的call函数
# inputs = tf.placeholder(np.float32, shape=(32, 100))  # 32是batch size
# # 通过zero_state方法得到一个全0的初始状态
# hO = cell.zero_state(32, np.float32)
#
# output, hl = cell.call(inputs, hO)
# print(hl)  # tuple 中含有3个32x128的向量



