import numpy as np
import math
from scipy.spatial.distance import minkowski

class LazyDistanceMatrix:
    def __init__(self, data):
        self.data = data  # 原始数据点
        self.clusters = [[i] for i in range(len(data))]  # 每个样本初始为一个簇
        self.cache = {}  # 距离缓存，避免重复计算

    def get(self, i, j):
        if i == j:
            return np.inf  # 防止合并自己
        key = (min(i, j), max(i, j))
        if key in self.cache:
            return self.cache[key]

        # 计算簇 i 与簇 j 中最小距离（Single-Linkage）
        min_dist = math.inf
        for idx1 in self.clusters[i]:
            for idx2 in self.clusters[j]:
                dist = minkowski(self.data[idx1], self.data[idx2])
                if dist < min_dist:
                    min_dist = dist
        self.cache[key] = min_dist
        return min_dist

    def update_merge(self, i, j):
        # 合并簇 j 到 i
        self.clusters[i].extend(self.clusters[j])
        del self.clusters[j]

        # 更新缓存：簇 j 被移除，i 保留，更新与 i 的最小距离
        new_cache = {}
        for key in self.cache:
            a, b = key
            if j in key:
                continue  # 删除与 j 相关的项
            if a > j: a -= 1
            if b > j: b -= 1
            new_cache[(a, b) if a < b else (b, a)] = self.cache[key]
        self.cache = new_cache

        # 重建 i 的距离
        for k in range(len(self.clusters)):
            if k == i:
                continue
            dist = self._min_dist_between_clusters(i, k)
            self.cache[(min(i, k), max(i, k))] = dist

    def _min_dist_between_clusters(self, i, j):
        # 提供内部更新最小距离函数
        min_dist = math.inf
        for idx1 in self.clusters[i]:
            for idx2 in self.clusters[j]:
                dist = minkowski(self.data[idx1], self.data[idx2])
                if dist < min_dist:
                    min_dist = dist
        return min_dist