"""
Name: Elad Aharon
ID: 311200786
"""
import math
import numpy as np

from init_centroids import init_centroids
import matplotlib.pyplot as plt


class KMeans:

    def __init__(self, k, picture):
        self.k = k
        self.picture = picture
        self.centroids = init_centroids(X=picture, K=k)
        self.centroid_to_pixels = self.reset_centroids_dict()
        self.pixels_to_centroids = {}

    def reset_centroids_dict(self):
        return {centroid_index: [] for centroid_index in range(self.k)}

    def run_k_means(self, max_iterations):
        print('k={}:'.format(self.k))
        for iteration in range(max_iterations + 1):
            print('iter {}: {}'.format(iteration, self.print_cent()))

            # find the closest centroid for each pixel
            for pixel_index, pixel in enumerate(self.picture):
                min_centroid_index, min_dist = None, float('inf')
                for index, centroid in enumerate(self.centroids):
                    dist = np.linalg.norm(pixel - centroid)

                    if dist < min_dist:
                        min_centroid_index, min_dist = index, dist

                # assign the pixel to the closest centroid
                self.centroid_to_pixels[min_centroid_index].append(pixel)
                self.pixels_to_centroids[tuple(pixel)] = min_centroid_index

            # update all centroids as the mean of their pixels
            for centroid_index, list_of_pixels in self.centroid_to_pixels.items():
                if len(list_of_pixels) != 0:
                    self.centroids[centroid_index] = sum(list_of_pixels) / len(list_of_pixels)

            # reset the centroids dictionary
            self.centroid_to_pixels = self.reset_centroids_dict()

        # copy the result into a new vector
        img_size = self.picture.shape
        result_vec = np.empty([img_size[0], img_size[1]])

        for index, pixel in enumerate(self.picture):
            centroid_index = self.pixels_to_centroids[tuple(pixel)]
            result_vec[index] = self.centroids[centroid_index]

        # showing the image
        picture_len = int(math.sqrt(img_size[0]))
        picture_after_k_means = result_vec.reshape(picture_len, picture_len, img_size[1])
        plt.imshow(picture_after_k_means)
        plt.grid(False)
        plt.show()


    def print_cent(self):
        """
        Function given by instructors in order to print the centroids.
        :return: string representation for the centroids.
        """
        if type(self.centroids) == list:
            self.centroids = np.asarray(self.centroids)
        if len(self.centroids.shape) == 1:
            return ' '.join(str(np.floor(100 * self.centroids) / 100).split()).replace('[ ', '[').replace('\n',
                                                                                                          ' ').replace(
                ' ]',
                ']').replace(
                ' ', ', ')
        else:
            return ' '.join(str(np.floor(100 * self.centroids) / 100).split()).replace('[ ', '[').replace('\n',
                                                                                                          ' ').replace(
                ' ]',
                ']').replace(
                ' ', ', ')[1:-1]
