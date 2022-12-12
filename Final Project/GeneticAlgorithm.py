import numpy as np

from Population import Population
from IPython.display import clear_output

def genetic_algorithm(pop_size=100, n_gen = 1000, crossover_size = 7, wildcards=None):
    population = Population(pop_size, wildcards=wildcards)               # Init population
    
    g = 0
    result = None

    while g < n_gen:
        clear_output(wait=True)
        result = population.new_population(pop_size, np.random.randint(1, crossover_size))
        
        print("GENETIC ALGORITHM | THE MERRY MOVIE MONTAGE ")
        print(f'Generation: {g}/{n_gen}'.ljust(len(str(g)) + len(str(n_gen)) + 20) + f'Population: {pop_size}'.ljust(20) + f'Elite: {population.elite[1]}'.ljust(20) + f'Survivors: {len(population.survivors)}'.ljust(20) + f'Generations Without Survivors: {population.generations_without_survivors}'.ljust(20))

        if(result): break
        g += 1

    return population.get_results()