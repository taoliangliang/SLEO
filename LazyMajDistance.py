import numpy as np
from scipy.spatial.distance import euclidean, minkowski


class LazyMajDistance:
    def __init__(self, minority, majority):
        self.minority = minority
        self.majority = majority
        self.cache = {}

    def get_min_distance(self, cluster):
        key = tuple(sorted(cluster))
        if key in self.cache:
            return self.cache[key]
        min_dist = min(
            minkowski(self.minority[i], m) for i in cluster for m in self.majority
        )
        self.cache[key] = min_dist
        return min_dist

    def update_merge(self, i, j, clusters_i, clusters_j):
        merged = clusters_i
        other = clusters_j
        new_key = tuple(sorted(merged + other))
        self.cache[new_key] = min(
            self.get_min_distance(merged),
            self.get_min_distance(other)
        )

        self.cache.pop(tuple(sorted(merged)), None)

        self.cache.pop(tuple(sorted(other)), None)
