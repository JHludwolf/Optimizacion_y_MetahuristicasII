import numpy as np
import matplotlib.pyplot as plt
import cv2

class Fitness:
    def __init__(self, target):
        self.target = target
        self.max_error = (np.square((1 - (self.target >= 127)) * 255 - self.target)).mean(axis=None)

    def score(self, specie):
        specie = cv2.resize(src=specie, dsize=self.target.shape[:-1], interpolation=cv2.INTER_AREA)
        fit = (np.square(specie - self.target)).mean(axis=None)
        fit = (self.max_error - fit) / self.max_error
        return fit

    def show_fitness(self, specie):
        plt.figure(figsize=(10,5))
        plt.subplot(1,2,1)
        plt.title("Target Image")
        plt.imshow(self.target / 255)
        plt.subplot(1,2,2)
        plt.title("Circle Approximation")
        plt.imshow(specie / 255)
        plt.show()