import random
import numpy as np
from copy import copy, deepcopy
from IPython.display import clear_output

from Wildcards import Wildcards
from ObjectiveFunctions import count_nodes_objective_func

class WildcardPopulation:
    def __init__(self, pop_size, phenotype):
        self.pop_size = pop_size
        self.phenotype = phenotype
        self.individuals = [Wildcards(phenotype) for _ in range(self.pop_size)]
        self.elite = [None, -1]
        self.fitnesses = self.update_fitnesses()

    def evolve(self, n_gen):    
        g = 0
        result = None

        while g < n_gen:
            clear_output(wait=True)
            result = self.new_population()
            
            print("WILDCARDS GENETIC ALGORITHM | THE MERRY MOVIE MONTAGE ")
            print(f'Generation: {g}/{n_gen}'.ljust(len(str(g)) + len(str(n_gen)) + 20) + f'Population: {self.pop_size}'.ljust(20) + f'Elite: {max(self.fitnesses)}'.ljust(20))

            g += 1
        return self.get_results()


    def update_fitnesses(self):
        self.fitnesses = [count_nodes_objective_func(individual.get_phenotype()) for individual in self.individuals]
        

    def new_population(self):

        new_population = []
        weights = self.fitnesses
        for _ in range(self.pop_size):
            A, B = random.choices(self.individuals, weights=weights, k=2)
            new_population.append(self.crossover(A, B))

        self.individuals = new_population
        self.mutation()
        self.get_elite()

    def get_elite(self):
        self.update_fitnesses()
        best_idx = np.argmax(self.fitnesses)
        if(self.fitnesses[best_idx] > self.elite[1]):
            self.elite = [self.individuals[best_idx], self.fitnesses[best_idx]]
    
    def crossover(self, A, B):
        sample_genotype = B.wildcards
        C = deepcopy(A)

        rdm_x_idx = random.randint(0,2)
        rdm_y_idx = random.randint(0,1)

        C.update_gene(rdm_x_idx, rdm_y_idx, sample_genotype[rdm_x_idx][rdm_y_idx])

        return C

    def mutation(self):
        for individual in self.individuals:
            individual.mutate()

    def get_results(self):
        print("ELITE INDIVIDUAL: \n\t{0} {1}".format(self.elite[0], self.elite[1])) # , self.elite[0].get_fitness()
        return self.elite

    
    def __str__(self):
        s = ''
        for i, individual in enumerate(self.individuals):
            s += str(i+1) + '. ' + individual.__str__() + '\n'
        return s