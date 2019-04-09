import numpy as np

from init_centroids import init_centroids


class KMeans:

    def __init__(self, k, picture):
        self.k = k
        self.picture = picture
        self.centroids = init_centroids(X=picture, K=k)
        self.centroid_to_pixels = {centroid_index: [] for centroid_index in range(len(self.centroids))}

    def run_k_means(self, max_iterations):
        print('k={}:'.format(self.k))
        for iteration in range(max_iterations):
            print('iter {}: {}'.format(iteration, np.floor(self.centroids * 100) / 100))

            # find the closest centroid for each pixel
            for pixel_index, pixel in enumerate(self.picture):
                min_centroid_index, min_dist = None, float('inf')
                for index, centroid in enumerate(self.centroids):
                    dist = np.linalg.norm(pixel - centroid)
                    if dist < min_dist:
                        min_centroid_index, min_dist = index, dist

                # assign the pixel to the closest centroid
                self.centroid_to_pixels[min_centroid_index].append(pixel)

            for centroid_index, list_of_pixels in self.centroid_to_pixels.items():
                # update the new centroid as the mean of all pixels
                self.centroids[centroid_index] = sum(list_of_pixels) / len(list_of_pixels)
            