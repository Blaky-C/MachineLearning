import pandas as pd
import numpy as np


# 数据预处理
def dataProcess(df):
    x_list, y_list = [], []

    # 将空数据填充为0
    df = df.replace(['NR'], [0.0])
    # 转换array中元素的数据类型
    array = np.array(df).astype(float)

    for i in range(0, 18 * 240, 18):
        for j in range(24 - 9):
            mat = array[i:i + 18, j:j + 9]
            label = array[i + 9, j + 9]  # 第10行是PM2.5
            x_list.append(mat)
            y_list.append(label)

    x = np.array(x_list)
    y = np.array(y_list)

    return x, y, array


# 更新参数，训练模型
def train(x_train, y_train, epoch):
    # 初始化偏置项和权重
    bias = 0
    weights = np.ones(9)

    # 初始化学习率和正则项系数
    learning_rate = 1
    reg_rate = 0.001
    bg2_sum = 0  # 存放偏置项的梯度平方和
    wg2_sum = np.zeros(9)  # 存放权重的梯度平方和

    for i in range(epoch):
        b_g = 0
        w_g = np.zeros(9)

        for j in range(20 * 12 * 15-400):
            b_g += (y_train[j] - weights.dot(x_train[j, 9, :]) - bias) * (-1)

            for k in range(9):
                w_g[k] += (y_train[j] - weights.dot(
                    x_train[j, 9, :]) - bias) * (-x_train[j, 9, k])

        # 求平均
        b_g /= 3200
        w_g /= 3200

        # 加上正则化项
        for m in range(9):
            w_g[m] += reg_rate * weights[m]

        # adagrad
        bg2_sum += b_g ** 2
        wg2_sum += w_g ** 2

        # 更新权重和偏置
        bias -= learning_rate / bg2_sum ** 0.5 * b_g
        weights -= learning_rate / wg2_sum ** 0.5 * w_g

    return weights, bias


# 验证模型效果
def validate(x_val, y_val, weights, bias):
    loss = 0
    for i in range(400):
        loss += (y_val[i] - weights.dot(x_val[i, 9, :]) - bias) ** 2
    return loss / 400


def main():
    # 从csv中读取有用的信息
    # 若读取失败，可在参数栏中加入encoding='gb18030'
    df = pd.read_csv('train.csv', usecols=range(3, 27))
    x, y, _ = dataProcess(df)

    # 划分训练集与验证集
    x_train, y_train = x[0:3200], y[0:3200]
    x_val, y_val = x[3200:3600], y[3200:3600]
    epoch = 2000  # 训练轮数

    # 开始训练
    w, b = train(x_train, y_train, epoch)

    # 在验证集上看效果
    loss = validate(x_val, y_val, w, b)
    print("The loss on val data is: ", loss)


if __name__ == '__main__':
    main()
