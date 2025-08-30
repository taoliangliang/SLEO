import math
import numpy as np
from numpy.linalg import norm
from scipy.spatial.distance import minkowski
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import check_array
from scipy.spatial import distance_matrix

__author__ = 'taoll'

from LazyDistanceMatrix import LazyDistanceMatrix
from LazyMajDistance import LazyMajDistance
from prob import Probs



class SLEO(object):

    def __init__(self,
                 alpha=20,
                 prob_type='bagging',
                 p_norm=2):
        '''
        :param prob_type: 计算概率的方式
        :param alpha:  熵衰减因子
        :param p_norm: 距离计算方式
        '''
        self.prob_type = prob_type
        self.alpha = alpha
        self.p_norm = p_norm

    def get_dataInfo(self, X, y):
        '''
         todo：获取数据集的基本信息
        :param X:
        :param y:
        :return:
        '''

        self.X = check_array(X)
        self.y = np.array(y)

        classes = np.unique(y)
        sizes = np.array([sum(y == c) for c in classes])
        indices = np.argsort(sizes)[::-1]
        self.sort_size = sizes[indices]
        self.unique_classes_ = classes[indices]
        self.maj_class_ = self.unique_classes_[0]


    def fit_sample(self, X, y):
        '''
        todo：采样方法
        :param X: 原始样本特征
        :param y: 样本标签
        :return: 合并后的采样生成样本与原始样本
        '''
        self.get_dataInfo(X, y)
        self.balanced_points = X[y == self.maj_class_]
        self.balanced_labels = y[y == self.maj_class_]
        for i in range(1, len(self.unique_classes_)):
            current_class = self.unique_classes_[i]
            self.n = self.sort_size[0]-self.sort_size[i]
            current_minority_points = X[y == current_class]
            current_minority_labels = y[y == current_class]
            appended_minority_points= self.generate_samples(current_minority_points)
            if len(appended_minority_points) > 0:
                self.balanced_points = np.concatenate([self.balanced_points, current_minority_points, appended_minority_points])
                self.balanced_labels = np.concatenate([self.balanced_labels, current_minority_labels, np.tile([current_class], len(appended_minority_points))])
            else:
                self.balanced_points = np.concatenate([self.balanced_points, current_minority_points])
                self.balanced_labels = np.concatenate([self.balanced_labels, current_minority_labels])

        return self.balanced_points, self.balanced_labels

    def sample_inside_sphere(self, dimensionality, radius):
        """
        超球体内生成随机方向向量
        :param dimensionality: 维度
        :param radius: 半径
        :return:
        """
        direction_unit_vector = (2 * np.random.rand(dimensionality) - 1)
        direction_unit_vector = direction_unit_vector / norm(direction_unit_vector)
        return direction_unit_vector * np.random.rand() * radius

    def generate_samples(self, current_minority_points):
        '''
        todo：入口调用函数  计算每个少数样本权重
        :param X:
        :param y:
        :param minority_class:
        :return:
        '''
        minority_points = current_minority_points
        clusters, labels_, clusters_center = self.semi_cluster(minority_points, self.balanced_points)
        self.compute_entropy(minority_points, clusters)
        self.balanced_points = self.entropy_decay(self.balanced_points, clusters, clusters_center)
        self.compute_weight(clusters)
        appended_minority_points = []
        for i in range(len(clusters)):
            for _ in range(int(self.gi[i])):
                new_data = clusters_center[i] + self.sample_inside_sphere(len(clusters_center[i]), self.radii[i])
                appended_minority_points.append(new_data)
        return appended_minority_points

    def compute_weight(self, clusters):
        min_weight = np.zeros(len(clusters))
        weight_sum = np.sum(self.min_cls_entropy)
        for i in range(len(clusters)):
            min_weight[i] = self.min_cls_entropy[i] / weight_sum
        self.gi = np.rint(min_weight * self.n).astype(np.int32)

    def entropy_decay(self, majority_points, clusters, clusters_center):
        """
        熵衰减模型
        :param majority_points: 多数类别
        :param clusters: 类簇
        :param clusters_center: 类簇中心
        :return:
        """
        num_clusters = len(clusters)
        self.radii = np.zeros(num_clusters)
        translations = np.zeros_like(majority_points, dtype=np.float64)

        for i in range(num_clusters):
            center = clusters_center[i]
            maj_dists = np.array([minkowski(center, pt, self.p_norm) for pt in majority_points])
            num_maj_in_radius = 0
            asc_sort_indexs = np.argsort(maj_dists)
            self.radii[i] = maj_dists[asc_sort_indexs[0]]
            if self.min_cls_entropy[i] > 0.01:
                for j in range(1, len(asc_sort_indexs)):
                    remain_entropy = self.min_cls_entropy[i] * math.exp(-self.alpha * (maj_dists[asc_sort_indexs[j]] - self.radii[i]))
                    num_maj_in_radius += 1
                    if remain_entropy <= 0.01:
                        self.radii[i] = maj_dists[asc_sort_indexs[j]]
                        break
            if num_maj_in_radius > 0:
                for j in range(num_maj_in_radius):
                    majority_point = majority_points[asc_sort_indexs[j]]
                    d = np.sum(np.abs(majority_point - clusters_center[i]) ** self.p_norm) ** (1 / self.p_norm)

                    if d != 0.0:
                        translation = (self.radii[i] - d) / d * (majority_point - clusters_center[i])
                    else:

                        translation = self.radii[i] * (majority_point) / np.linalg.norm(majority_point)
                    translations[asc_sort_indexs[j]] = translations[asc_sort_indexs[j]] + translation

        majority_points = majority_points.astype(np.float64)
        majority_points += translations

        return majority_points

    def compute_entropy(self, minority_points, clusters):
        '''
            计算初类簇熵
        :param majority_points: 多数类别
        :param minority_points: 少数类别
        :param clusters: 类簇数量
        :return:
        '''

        self.min_sample_entropy = np.zeros(len(minority_points))
        self.min_cls_entropy = np.zeros(len(clusters))

        if self.prob_type == 'knn':
            probs = Probs(self.X, self.y, minority_points, self.unique_classes_).knn_probs()
        elif self.prob_type == 'kde':
            probs = Probs(self.X, self.y, minority_points, self.unique_classes_).kde_probs()
        elif self.prob_type == 'logistic':
            probs = Probs(self.X, self.y, minority_points, self.unique_classes_).logistic_probs()
        elif self.prob_type == 'svm':
            probs = Probs(self.X, self.y, minority_points, self.unique_classes_).svm_probs()
        elif self.prob_type == 'bagging':
            probs = Probs(self.X, self.y, minority_points, self.unique_classes_).bagging_probs()
        else:
            probs = Probs(self.X, self.y, minority_points, self.unique_classes_).bagging_probs()
        probs = probs+ 0.001
        self.min_sample_entropy = -np.sum(probs * np.log2(probs), axis=1)
        for i in range(len(clusters)):
            for j in range(len(clusters[i])):
                self.min_cls_entropy[i] = self.min_cls_entropy[i] + self.min_sample_entropy[clusters[i][j]]

    def semi_cluster(self, minority_points, majority_points):
        """
        半监督层次聚类
        :param minority_points: 少数类别
        :param majority_points: 多数类别
        :return:
        """
        dm_min = LazyDistanceMatrix(minority_points)
        dm_maj = LazyMajDistance(minority_points, majority_points)

        while len(dm_min.clusters) > 1:
            min_dist = math.inf
            merge_idx = (-1, -1)
            for i in range(len(dm_min.clusters)):
                for j in range(i + 1, len(dm_min.clusters)):
                    dist = dm_min.get(i, j)
                    if dist < min_dist:
                        min_dist = dist
                        merge_idx = (i, j)

            i, j = merge_idx
            if i == -1 or j == -1:
                break
            if dm_maj.get_min_distance(dm_min.clusters[i]) < min_dist or \
                    dm_maj.get_min_distance(dm_min.clusters[j]) < min_dist:
                dm_min.cache[(min(i, j), max(i, j))] = np.inf
                continue
            clusters_i = dm_min.clusters[i]
            clusters_j = dm_min.clusters[j]
            dm_min.update_merge(i, j)
            dm_maj.update_merge(i, j, clusters_i, clusters_j)
        final_clusters = dm_min.clusters
        labels_ = np.zeros(len(minority_points), dtype=int)
        for cluster_idx, cluster in enumerate(final_clusters):
            for idx in cluster:
                labels_[idx] = cluster_idx

        cluster_centers = np.zeros((len(final_clusters), minority_points.shape[1]))
        for i, cluster in enumerate(final_clusters):
            cluster_centers[i] = np.mean(minority_points[cluster], axis=0)

        return final_clusters, labels_, cluster_centers








