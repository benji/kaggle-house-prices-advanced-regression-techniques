import numpy as np
from sklearn.neighbors import NearestNeighbors
samples = [[0, 0, 4], [1, 0, 0], [0, 0, 2]]

neigh = NearestNeighbors(2, 0.4)
print neigh.fit(samples)  

print neigh.kneighbors([[0, 0, 1.3]], 2, return_distance=True)
