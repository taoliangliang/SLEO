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
        probs = []
        IR = int((len(self.X) - len(self.minority_points)) / len(self.minority_points))
        if IR <= 5:
            self.k = 5
        else:
            self.k = IR
        nbrs = NearestNeighbors(n_neighbors=self.k + 1).fit(self.X)
        distances, indices = nbrs.kneighbors(self.minority_points)
        for i in range(len(self.minority_points)):
            neighbor_indices = indices[i][1:]  
            neighbor_labels = self.y[neighbor_indices]
            counts = np.bincount(neighbor_labels, minlength=2)
            probs.append(counts / counts.sum())
        return np.array(probs)

    def logistic_probs(self):
        Logistic = LogisticRegression()
        Logistic.fit(self.X, self.y)
        probs = Logistic.predict_proba(self.minority_points)
        return probs

    def svm_probs(self):
        svm = SVC(probability=True) 
        svm.fit(self.X, self.y)
        probs = svm.predict_proba(self.minority_points) 
        return probs

    def bagging_probs(self):
        bagging = BaggingClassifier(estimator=DecisionTreeClassifier(), n_estimators=20)
        bagging.fit(self.X, self.y)
        probs = bagging.predict_proba(self.minority_points) 
        return probs

    def kde_probs(self):
        probs = []
        kde_models = {}
        for cls in self.unique_classes_:
            X_cls = self.X[self.y == cls]
            kde = KernelDensity(bandwidth=1.0)
            kde.fit(X_cls)
            kde_models[cls] = kde
        for i in range(len(self.minority_points)):
            log_probs = []
            for cls in self.unique_classes_:
                log_prob = kde_models[cls].score_samples(self.minority_points[i].reshape(1, -1))[0]  # log density
                log_probs.append(log_prob)
            prob_arr = np.array(log_probs)
            prob_arr = np.exp(prob_arr)  
            prob_arr /= prob_arr.sum()  
            probs.append(prob_arr)
        probs = np.array(probs)

        return probs
