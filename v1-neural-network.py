import pickle

import numpy as np
import matplotlib.pylab as plt
from dataset.mnist import load_mnist

def step_function(x):
    return np.array(x > 0, dtype=int)


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


# 实现softmax，但是不能处理数值过大的数据。
def softmax_v1(a):
    exp_a = np.exp(a)
    sum_exp_a = np.sum(exp_a)
    return exp_a / sum_exp_a


# 优化版，可以处理数据比较大的数据。
# 优化依据是分子分母同时乘上一个系数c，原有等式不变。
# 又可以化简为，将exp外的这个系数c提取到等式里面，则变成了 exp(a + c)，这时c为负数也不影响原有等式。
def softmax_v2(a):
    c = np.max(a)
    exp_a = np.exp(a - c)
    sum_exp_a = np.sum(exp_a)
    return exp_a / sum_exp_a


def identity_function(a):
    return a


def init_network():
    with open("sample_weight.pkl", 'rb') as f:
        network = pickle.load(f)
        return network


def predict(X, network):
    W1, W2, W3 = network['W1'], network['W2'], network['W3']
    b1, b2, b3 = network['b1'], network['b2'], network['b3']
    a1 = np.dot(X, W1) + b1
    print(a1)
    z1 = sigmoid(a1)
    a2 = np.dot(z1, W2) + b2
    z2 = sigmoid(a2)
    a3 = np.dot(z2, W3) + b3
    y = softmax_v2(a3)

    return y

#当前使用测试数据
def get_data():
    (x_train, t_train), (x_test, t_test) = load_mnist(flatten=True, normalize=False)
    return x_test, t_test


x, t = get_data()
network = init_network()
accuracy_cnt = 0
for i in range(len(x)):
    y = predict(network, x[i])
    p = np.argmax(y) # 获取概率最高的元素的索引
    if p == t[i]:
        accuracy_cnt += 1

print("Accuracy:" + str(float(accuracy_cnt) / len(x)))