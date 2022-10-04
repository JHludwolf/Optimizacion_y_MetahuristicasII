import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class EvolutionaryProgramming:

    # Constructor: Initialize attributes with input parameters
    def __init__(self, objective_function, bounds, args=(), popsize=30, niter=100):
        self.objective_function = objective_function
        self.bounds = np.array(bounds)
        self.args = args
        self.popsize = popsize
        self.niter = niter
        self.population = None
        self.fitness = None

    # Create the initial population
    def init_population(self):
        varNo = len(self.bounds)
        self.population = np.zeros((self.popsize, varNo*2))
        self.fitness = np.zeros((self.popsize))

        for val in range(varNo):
            minVal, maxVal = self.bounds[val, 0], self.bounds[val, 1]
            mutVal = np.abs(maxVal - minVal)/10
            self.population[:, val] = np.random.uniform(minVal, maxVal, (self.popsize))
            self.population[:, val+varNo] = np.abs(np.random.normal(loc=0, scale=mutVal, size=(self.popsize))) + 0.001

        #print(self.population)

        for i in range(self.popsize):
            P = self.population[i, :varNo]
            self.fitness[i] = self.objective_function(P, *self.args)

            #print(self.population[i, varNo], self.fitness[i])

    def mutation(self, alpha=0.2):
        varNo = len(self.bounds)
        mut_population = np.zeros((self.popsize, varNo*2))

        for i, chromosome in enumerate(self.population):
            new_chrom = np.zeros(len(self.bounds)*2)
            
            for j, gene in enumerate(chromosome[:varNo]):
                new_chrom[j] = gene + np.random.normal(0, chromosome[j+varNo])

            for j, gene in enumerate(chromosome[varNo:]):
                new_chrom[j+varNo] = gene * (1 + np.random.normal(0, alpha))
            
            mut_population[i] = new_chrom
                
        return mut_population

    def survivor_selection(self, childPopulation, popSize):
        survivor_pop = np.concatenate((self.population, childPopulation), axis=0)
        scores = np.array([self.objective_function(chrom, *self.args) for chrom in survivor_pop]) # MODIFICAR EN CASO DE MAS DE 2 ARGUMENTOS
        #pop_with_scores = np.array(list(zip(survivor_pop, scores)))

        self.population = survivor_pop[scores.argsort()][ : popSize]
        self.fitness = scores.argsort()

    def solve(self):
        self.init_population()

        for _ in range(self.niter):
            mut_population = self.mutation()
            self.survivor_selection(mut_population, self.popsize)
            
        return self.population[0], self.fitness[0]
        

def evolutionary_programming(objective_function, bounds, args=(), popsize=30, niter=100):
    return EvolutionaryProgramming(objective_function, bounds, args, popsize, niter).solve()