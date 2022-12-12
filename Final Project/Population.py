import random
import pprint
import numpy as np
import pandas as pd
from IPython.display import clear_output
from copy import copy, deepcopy

from Individual import Individual

class Population:
    def __init__(self, pop_size, wildcards=None, tolerance = 5):
        self.pop_size = pop_size
        self.tolerance = tolerance
        self.wildcards = wildcards
        self.individuals = [Individual(wildcards=self.wildcards) for _ in range(self.pop_size)]
        
        self.survivors, self.local_fitness = [], []
        self.elite = [None, -1] # Individual, fitness
        self.bests_per_cicle = []
        self.generations_without_survivors = 0

    def get_survivors(self):
        new_survivors = []
        new_local_fitness = []

        for individual in self.individuals:
            if individual.get_fitness():
                new_survivors.append(individual)
                new_local_fitness.append(individual.get_local_fitness())
        
        if not new_survivors:
            print('No change 1')
            self.generations_without_survivors += 1
            return deepcopy(self.survivors), deepcopy(self.local_fitness)
        else:
            self.generations_without_survivors = 0
            return new_survivors, np.array(new_local_fitness, dtype=int)
        

    def new_population(self, pop_size = None, crossover_size=1):
        if self.generations_without_survivors >= self.tolerance: return True
        if pop_size: self.pop_size = pop_size
        
        
        self.survivors, self.local_fitness = self.get_survivors()
        
        best_fit_idx = np.argmax(self.local_fitness)
        if self.local_fitness[best_fit_idx] > self.elite[1]:
            self.elite = [self.survivors[best_fit_idx], self.local_fitness[best_fit_idx]]
        
        self.bests_per_cicle.append(self.local_fitness[best_fit_idx])

        self.individuals = []
        for _ in range(self.pop_size):
            A, B = random.choices(self.survivors, weights=self.local_fitness, k=2)
            self.individuals.append(self.crossover(A, B, crossover_size))
        self.mutation()
        
        return False

    
    def crossover(self, A, B, sample_size=1):
        sample_genotype = B.genotype
        C = deepcopy(A)

        for i in range(3):
            rnd_genes = np.random.choice(list(sample_genotype[i]), size=sample_size, replace=True)
            C.add_genes(i, rnd_genes)

        return C

    def mutation(self):
        for individual in self.individuals:
            individual.mutate()

    def get_results(self):
        print("ELITE INDIVIDUAL: \n\t{}".format(self.elite[0])) # , self.elite[0].get_fitness()
        return self.elite[0]

    
    def __str__(self):
        s = ''
        for i, individual in enumerate(self.individuals):
            s += str(i+1) + '. ' + individual.__str__()
        return s