import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#from numpy.random import choice
from random import choice
from Specie import Specie

class DifferentialEvolution:
    def __init__(self, target, fitness, genes=128, niter=1000):
        self.target = target
        self.size = target.shape        # np.shape of target image
        self.fitness = fitness(target)
        self.genes = genes
        self.niter = niter
        self.curFit = None
        self.specie = Specie(size=self.size, genes=genes)   # Init population

    def print_progress(self, g):
        print("GEN {}, FIT {:.8f}".format(g, self.curFit))

    def crossover(self, X, Y):
        choice = np.random.randint(2, size = X.size).reshape(X.shape).astype(bool)
        return np.where(choice, X, Y)

    def selection(self, i, new_gene):
        mutated = Specie(size=self.size, genotype=np.array(self.specie.genotype))
        mutated.genotype[i] = new_gene

        mutated.genotype_to_image()
        newFit = self.fitness.score(mutated.phenotype)

        if newFit > self.curFit:
            self.curFit = newFit
            self.specie = mutated

    def solve(self):
        self.specie.genotype_to_image()
        self.curFit = self.fitness.score(self.specie.phenotype)

        for g in range(self.niter):
            for i, gene in enumerate(self.specie.genotype):
                # MUTATION
                F = np.random.uniform(0,2)
                v = choice(self.specie.genotype) + F * np.absolute(choice(self.specie.genotype) - choice(self.specie.genotype))

                # CROSSOVER
                new_gene = self.crossover(gene, v)

                # SELECTION
                self.selection(i, new_gene)

            self.print_progress(g)
        
        return self.specie

    def solve_faster(self):
        self.specie.genotype_to_image()
        self.curFit = self.fitness.score(self.specie.phenotype)

        for g in range(self.niter):
            for i, gene in enumerate(self.specie.genotype):
                # MUTATION
                F = np.random.uniform(0,2)
                v = choice(self.specie.genotype) + F * np.absolute(choice(self.specie.genotype) - choice(self.specie.genotype))

                # CROSSOVER
                choices = np.random.randint(2, size = gene.size).reshape(gene.shape).astype(bool)
                new_gene = np.where(choices, gene,v)

                # SELECTION
                mutated = Specie(size=self.size, genotype=np.array(self.specie.genotype))
                mutated.genotype[i] = new_gene

                mutated.genotype_to_image()
                newFit = self.fitness.score(mutated.phenotype)

                if newFit > self.curFit:
                    self.curFit = newFit
                    self.specie = mutated

            self.print_progress(g)
        
        return self.specie

def differential_evolution(target, fitness, genes=128, niter=1000):
    return DifferentialEvolution(target, fitness, genes=genes, niter=niter).solve()