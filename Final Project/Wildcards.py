import random
import pandas as pd
import numpy as np
import pprint

from ObjectiveFunctions import count_nodes_objective_func

class Wildcards:
    def __init__(self, phenotype, wildcard_symbol = '🌟'):
        self.wildcards = []
        self.wildcard_symbol = wildcard_symbol
        self.fitness = -1
        self.phenotype = phenotype

        for i in range(3):
            self.wildcards.append(list(np.random.randint(0, len(self.phenotype['schedule'][i]), 2)))

    def update_gene(self, str_idx=None, wldcrd_idx=None, value=None):
        if str_idx is None and wldcrd_idx is None and value is None:
            str_idx = random.randint(0,2)
            wldcrd_idx = random.randint(0,1)
            value_limit = len(self.phenotype['schedule'][str_idx])
            value = random.randint(0, value_limit)

        elif str_idx is not None and wldcrd_idx is not None and value is None:
            value_limit = len(self.phenotype['schedule'][str_idx])
            value = random.randint(0, value_limit)
        
        self.wildcards[str_idx][wldcrd_idx] = value

    def get_phenotype(self):
        phenome_lsts = [list(p) for p in self.phenotype['schedule']]
        for i in range(3):
            for j in range(2):
                phenome_lsts[i][self.wildcards[i][j]] = self.wildcard_symbol
        phenome = [''.join(phenome_lst) for phenome_lst in phenome_lsts]
        return pd.DataFrame(phenome, columns=['schedule'])

    def mutate(self):
        self.update_gene()
    
    def get_fitness(self):
        self.fitness = count_nodes_objective_func(self.get_phenotype())
        return self.fitness

    def __str__(self):
        return f'{pprint.pformat(self.wildcards)} {self.get_fitness()}\n'