import numpy as np
import math
from scipy.spatial.distance import minkowski

class LazyDistanceMatrix:
    def __init__(self, data):
        self.data = data 
        self.clusters = [[i] for i in range(len(data))] 
        self.cache = {} 
        
    def get(self, i, j):
        if i == j:
            return np.inf  
        key = (min(i, j), max(i, j))
        if key in self.cache:
            return self.cache[key]

      
        min_dist = math.inf
        for idx1 in self.clusters[i]:
            for idx2 in self.clusters[j]:
                dist = minkowski(self.data[idx1], self.data[idx2])
                if dist < min_dist:
                    min_dist = dist
        self.cache[key] = min_dist
        return min_dist

    def update_merge(self, i, j):
        self.clusters[i].extend(self.clusters[j])
        del self.clusters[j]

       
        new_cache = {}
        for key in self.cache:
            a, b = key
            if j in key:
                continue  
            if a > j: a -= 1
            if b > j: b -= 1
            new_cache[(a, b) if a < b else (b, a)] = self.cache[key]
        self.cache = new_cache

      
        for k in range(len(self.clusters)):
            if k == i:
                continue
            dist = self._min_dist_between_clusters(i, k)
            self.cache[(min(i, k), max(i, k))] = dist

    def _min_dist_between_clusters(self, i, j):
        min_dist = math.inf
        for idx1 in self.clusters[i]:
            for idx2 in self.clusters[j]:
                dist = minkowski(self.data[idx1], self.data[idx2])
                if dist < min_dist:
                    min_dist = dist

        return min_dist
