import numpy as np
import cv2

class Specie:
    def __init__(self, size, genes=128, genotype=None):
        self.size = size
        self.genotype_width = 5 if len(size) < 3 else 7
        self.genotype = genotype if genotype is not None else np.random.rand(genes, self.genotype_width)
        self.phenotype = np.zeros(size)

    def genes(self):
        return self.genotype.shape[0]

    def genotype_to_image(self):
        self.phenotype.fill(255)
        radius_avg = (self.size[0] + self.size[1]) / 2 / 6
        
        for row in self.genotype:
            overlay = self.phenotype.copy()
            color = (row[3:-1] * 255).astype(np.uint8).tolist()
            cv2.circle(
                overlay,
                center = (int(row[1] * self.size[1]), int(row[0] * self.size[0])),
                radius = int(row[2] * radius_avg),
                color = color,
                thickness = -1,
            )

            alpha = row[-1]
            self.phenotype = cv2.addWeighted(overlay, alpha, self.phenotype, 1 - alpha, 0)