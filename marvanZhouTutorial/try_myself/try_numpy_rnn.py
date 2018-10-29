import tensorflow as tf
import numpy as np

X = [1, 2]
state = [0.0, 0.0]

#  分开定义不通输入部分的权重
wCellState = np.asarray([[0.1, 0.2], [0.3, 0.4]])
wCellInput = np.asarray([0.5, 0.6])
bCell = np.asarray([0.1, -0.1])

# 定义用于输出的全连接层参数
w_output = np.asarray([[1.0], [2.0]])
b_output = 0.1

for i in range(len(X)):
    before_activation = np.dot(state, wCellState) + X[i] * wCellInput + bCell
    state = np.tanh(before_activation)

    final_output = np.dot(state, w_output) + b_output
    print("before activation: ", before_activation)
    print("state:", state)
    print("output:", final_output)
