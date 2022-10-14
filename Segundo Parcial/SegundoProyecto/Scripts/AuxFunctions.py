import cv2
import numpy as np
import matplotlib.pyplot as plt
from IPython.display import clear_output
from differential_evolution import differential_evolution

def load_image(image_path, size=None):
    target = cv2.imread(image_path, cv2.IMREAD_COLOR)
    target = cv2.cvtColor(target, cv2.COLOR_BGR2RGB)

    if size:
        target = cv2.resize(src=target, dsize=size, interpolation=cv2.INTER_AREA)
    return target

def show_image(img_arr):
    clear_output(wait=True)
    plt.figure()
    plt.axis("off")
    plt.imshow(img_arr / 255)
    plt.show()

def show_fitness(target, species, path=None, save=False):
    
    subplots = len(species) + 1
    plt.figure(figsize=(subplots*4,5))
    plt.axis("off")

    plt.subplot(1, subplots, 1)
    plt.title("Target Image")
    plt.imshow(target / 255)

    for i, specie in enumerate(species):
        plt.subplot(1, subplots, i+2)
        plt.title(str(specie[1]) + " Gen Circle Approx")
        plt.imshow(specie[0] / 255)

    if save:
        plt.savefig('../Images/Outputs/' + filename_from_path(path) + '_circle_approx.png', facecolor='white')
    plt.show()

def add_circle(phenotype, gene):
    size = phenotype.shape
    radius_avg = (size[0] + size[1]) / 2 / 6

    overlay = phenotype.copy()
    color = (gene[3:-1] * 255).astype(np.uint8).tolist()
    cv2.circle(
        overlay,
        center = (int(gene[1] * size[1]), int(gene[0] * size[0])),
        radius = int(gene[2] * radius_avg),
        color = color,
        thickness = -1,
    )

    alpha = gene[-1]
    phenotype = cv2.addWeighted(overlay, alpha, phenotype, 1 - alpha, 0)
    
    return phenotype

def filename_from_path(path):
    return path.split('/')[-1].split('.')[0]