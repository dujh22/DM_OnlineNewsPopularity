# -*- coding:utf-8 -*-

import numpy as np


class knn:
    def __init__(self, k):
        """
        初始化函数
        :param k: K Near Neighbor 邻居个数
        """
        self.k = k

    def fit(self, X, y):
        """
        KNN 训练函数：只需要保存起来带标签数据集
        :param X: 带标签数据集的特征矩阵，每一行为一个特征向量，每一列表示一个特征维度取值，共样本数量行
        :param y: 带标签数据集的结果标签列向量，每一行为对应特征矩阵中同行特征向量的结果标签值
        """
        self.X = np.asarray(X)
        self.y = np.asarray(y)

    def predict(self, X):
        """
        KNN 预测函数：对输入的无标签数据集的特征矩阵，输出对其结果标签列向量的预测
        :param X: 无标签数据集的特征矩阵，每一行为一个特征向量，每一列表示一个特征维度取值，共样本数量行
        :return:  无标签数据集的预测结果标签列向量，每一行为对应特征矩阵中同行特征向量的预测结果标签值
        """
        X = np.asarray(X)
        prey = []

        # 对无标签数据集的特征矩阵的每一行，也就是每一个特征向量样本
        for x in X:
            # 计算与训练集中所有样本的距离
            dis = np.sqrt(np.sum((x - self.X) ** 2, axis=1))  # axis = 1 按行求和
            # 获得dis排序后每个元素在原数组中的索引
            index = dis.argsort()
            # 截取前k个元素
            index = index[:self.k]
            # 统计前k个元素对应各种标签的数量
            count = np.bincount(self.y[index])  # np.bincount是统计从0到array数组中最大数字出现的个数的函数，并同样以array数组输出显示。
            # 返回数量最多的标签
            prey.append(count.argmax())

        return prey
