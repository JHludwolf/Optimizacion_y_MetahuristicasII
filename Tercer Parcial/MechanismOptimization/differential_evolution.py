import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#from numpy.random import choice
from IPython.display import clear_output
from random import choice

class DifferentialEvolution:
    def __init__(self, func, bounds, args=(), popsize=30, niter=100, callback=None):
        self.func = func
        self.bounds = np.array(bounds)
        self.genes = len(bounds)
        self.args = args
        self.popsize = popsize
        self.niter = niter
        self.population = None              # <np.array> len = popsize
        self.fitness = None                 # <np.array> len = len(self.population)
        self.callback = callback
        self.best = [-1,-1]

    def init_population(self):
        var_no = len(self.bounds)
        self.population = np.zeros((self.popsize, var_no))
        self.fitness = np.zeros((self.popsize))

        for val in range(var_no):
            min_val, max_val = self.bounds[val, 0], self.bounds[val, 1]
            self.population[:, val] = np.random.uniform(min_val, max_val, (self.popsize))

        for i in range(self.popsize):
            P = self.population[i,:]
            self.fitness[i] = self.func(P, *self.args)

        self.best = 0, self.fitness[0]
        #print(self.population)

    def print_progress(self, g):
        clear_output(wait=True)
        print("DIFFERENTIAL EVOLUTION, {} GENS, {} POPULATION, {} GENES".format(self.niter, self.popsize, self.genes))
        print("GEN {}, FIT {:.8f}".format(g, np.mean(self.fitness)))

    def mutation(self):
        F = np.random.uniform(0,2)
        v = choice(self.population) + F * (choice(self.population) - choice(self.population))
        #v = self.population[self.best[0]] + F * (choice(self.population) - choice(self.population))
        return np.clip(v, self.bounds[:,0], self.bounds[:,1]) # bounds[:, 0], bounds[:, 1]

    def crossover(self, X, Y):
        choice = np.random.randint(2, size = X.size).reshape(X.shape).astype(bool)
        return np.where(choice, X, Y)

    def selection(self, X, Y, y_fitness):
        x_fitness = self.func(X, *self.args)
        return (X, x_fitness) if x_fitness < y_fitness else (Y, y_fitness)

    def solve(self):
        self.init_population()

        for g in range(self.niter):
            for i, chromosome in enumerate(self.population):
                # MUTATION
                mutated = self.mutation()
                
                # CROSSOVER
                new_chrom = self.crossover(chromosome, mutated)

                # SELECTION
                self.population[i], self.fitness[i] = self.selection(new_chrom, chromosome, self.fitness[i])
                if self.fitness[i] < self.best[1]:
                    self.best = i, self.fitness[i]
            
            self.print_progress(g)
            
            if self.callback:
                best_value_idx = np.argmin(self.fitness)
                self.callback(self.population[best_value_idx], self.fitness[best_value_idx])
        
        best_value_idx = np.argmin(self.fitness)
        return self.population[best_value_idx], self.fitness[best_value_idx]

def differential_evolution(func, bounds, args=(), popsize=100, niter=1000, callback=None):
    return DifferentialEvolution(func, bounds, args, popsize, niter, callback=callback).solve()