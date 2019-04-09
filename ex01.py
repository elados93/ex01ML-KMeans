import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt

from scipy.misc import imread
from k_means import KMeans

# data preperation (loading, normalizing, reshaping)

path = 'dog.jpeg'
A = imread(path)
A = A.astype(float) / 255.
img_size = A.shape
X = A.reshape(img_size[0] * img_size[1], img_size[2])

for k in [2,
          4,
          8,
          16
          ]:
    algorithm = KMeans(k=k, picture=X)
    algorithm.run_k_means(max_iterations=10)
