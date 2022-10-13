import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#from numpy.random import choice
from random import choice

class DifferentialEvolution:
    def __init__(self, objective_function, bounds, args=(), popsize=30, niter=100, callback= None):
        self.objective_function = objective_function
        self.bounds = np.array(bounds)
        self.args = args
        self.popsize = popsize
        self.niter = niter
        self.population = None              # <np.array> len = popsize
        self.fitness = None                 # <np.array> len = len(self.population)
        self.callback = callback

    def init_population(self):
        var_no = len(self.bounds)
        self.population = np.zeros((self.popsize, var_no))
        self.fitness = np.zeros((self.popsize))

        for val in range(var_no):
            min_val, max_val = self.bounds[val, 0], self.bounds[val, 1]
            self.population[:, val] = np.random.uniform(min_val, max_val, (self.popsize))

        #print(self.population)

        for i in range(self.popsize):
            P = self.population[i,:]
            self.fitness[i] = self.objective_function(P, *self.args)

    def print_progress(self, g):
        print("GEN {}, FIT {:.8f}".format(g, np.mean(self.fitness)))

    def crossover(self, X, Y):
        unique_choice_values = 1
        while unique_choice_values == 1:
            choice = np.random.randint(2, size = X.size).reshape(X.shape).astype(bool)
            unique_choice_values = len(np.unique(choice))
        return np.where(choice, X, Y)

    def selection(self, X, Y, y_fitness):
        x_fitness = self.objective_function(X, *self.args)
        return (X, x_fitness) if x_fitness > y_fitness else (Y, y_fitness)

    def solve(self):
        self.init_population()

        for g in range(self.niter):
            for i, chromosome in enumerate(self.population):
                # MUTATION
                F = np.random.uniform(0,2)
                v = choice(self.population) + F * np.absolute(choice(self.population) - choice(self.population))
                '''
                r1, r2, r3 = np.random.choice(self.popsize,3)
                xr1,xr2,xr3 = self.population[r1,:], self.population[r2,:], self.population[r3,:]
                v = xr1 + F * (xr2 - xr3)
                '''
                # CROSSOVER
                new_chrom = self.crossover(chromosome, v)

                # SELECTION
                self.population[i], self.fitness[i] = self.selection(new_chrom, chromosome, self.fitness[i])
                if self.callback: self.callback(self.population[i])
                
            self.print_progress(g)
        best_value_idx = np.argmin(self.fitness)
        return self.population[best_value_idx], self.fitness[best_value_idx]

def differential_evolution(objective_function, bounds, args=(), popsize=30, niter=100, callback=None):
    return DifferentialEvolution(objective_function, bounds, args, popsize, niter).solve()