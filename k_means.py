import numpy as np

from init_centroids import init_centroids


class KMeans:

    def __init__(self, k, picture):
        self.k = k
        self.picture = picture
        self.centroids = init_centroids(X=picture, K=k)
        # self.related_centroids_vector = [0 for _ in range(len(picture))]
        self.centroid_to_pixels = {centroid: [] for centroid in self.centroids}

    def run_k_means(self, max_iterations):
        for iteration in range(max_iterations):
            print('iter {}: {}'.format(iteration, self.centroids))

            # find the closest centroid for each pixel
            for pixel_index, pixel in enumerate(self.picture):
                min_centroid, min_dist = None, float('inf')
                for centroid in self.centroids:
                    dist = np.linalg.norm(pixel - centroid)
                    if dist < min_dist:
                        min_centroid, min_dist = centroid, dist

                self.centroid_to_pixels[centroid].append(pixel)  # assign the pixel to the closest centroid

            for centroid in self.centroids:
                total_sum = 0
