import numpy as np
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import NearestNeighbors, KernelDensity
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier


class Probs(object):
    def __init__(self,X, y, minority_points,unique_classes_):
        self.X = X
        self.y = y
        self.minority_points = minority_points
        self.unique_classes_ = unique_classes_

    def knn_probs(self):
        # 寻找最近邻（不包括自身）
        probs = []
        IR = int((len(self.X) - len(self.minority_points)) / len(self.minority_points))
        if IR <= 5:
            self.k = 5
        else:
            self.k = IR
        nbrs = NearestNeighbors(n_neighbors=self.k + 1).fit(self.X)
        distances, indices = nbrs.kneighbors(self.minority_points)
        for i in range(len(self.minority_points)):
            neighbor_indices = indices[i][1:]  # 排除自身
            neighbor_labels = self.y[neighbor_indices]
            counts = np.bincount(neighbor_labels, minlength=2)
            probs.append(counts / counts.sum())
        return np.array(probs)

    def logistic_probs(self):
        #随机森林
        Logistic = LogisticRegression()
        Logistic.fit(self.X, self.y)
        probs = Logistic.predict_proba(self.minority_points)  # 每个样本的类别概率分布
        return probs

    def svm_probs(self):
        #支持向量基
        svm = SVC(probability=True)  # 开启 Platt 缩放
        svm.fit(self.X, self.y)
        probs = svm.predict_proba(self.minority_points)  # 每个样本的类别概率分布
        return probs

    def bagging_probs(self):
        # 定义 BaggingClassifier（基分类器支持概率输出）
        bagging = BaggingClassifier(estimator=DecisionTreeClassifier(), n_estimators=20)
        bagging.fit(self.X, self.y)
        probs = bagging.predict_proba(self.minority_points)  # 每个样本的类别概率分布
        return probs

    def kde_probs(self):
        probs = []
        kde_models = {}
        # 1. 为每个类别拟合一个 KDE 模型
        for cls in self.unique_classes_:
            X_cls = self.X[self.y == cls]
            kde = KernelDensity(bandwidth=1.0)
            kde.fit(X_cls)
            kde_models[cls] = kde
        # 2. 对每个样本，计算它在各类别 KDE 中的概率密度
        for i in range(len(self.minority_points)):
            log_probs = []
            for cls in self.unique_classes_:
                log_prob = kde_models[cls].score_samples(self.minority_points[i].reshape(1, -1))[0]  # log density
                log_probs.append(log_prob)
            prob_arr = np.array(log_probs)
            prob_arr = np.exp(prob_arr)  # 转换为密度
            prob_arr /= prob_arr.sum()  # 归一化为“概率分布”
            probs.append(prob_arr)
        probs = np.array(probs)
        return probs