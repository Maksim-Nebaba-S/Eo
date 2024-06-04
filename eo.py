import numpy as np


population_size = 800
num_generations = 50000
mutation_rate = 0.1
crossover_rate = 0.8
lower_bound = -8
upper_bound = 55


def objective_function(x):
    return -39 - 2 * x - 7 * x**2 + x**3


def initialize_population(size, lower_bound, upper_bound):
    return np.random.uniform(lower_bound, upper_bound, size)


def calculate_fitness(population):
    return objective_function(population)


def select_parents(population, fitness_scores):
    min_fitness = np.min(fitness_scores)
    adjusted_fitness = fitness_scores - min_fitness + 1e-5  
    probabilities = adjusted_fitness / adjusted_fitness.sum()
    parents_indices = np.random.choice(np.arange(len(population)), size=len(population), p=probabilities)
    return population[parents_indices]


def perform_crossover(parents, crossover_rate):
    offspring = np.empty(parents.shape)
    for i in range(0, len(parents), 2):
        if np.random.rand() < crossover_rate and i + 1 < len(parents):
            crossover_point = np.random.randint(1)
            offspring[i] = crossover_point * parents[i] + (1 - crossover_point) * parents[i + 1]
            offspring[i + 1] = crossover_point * parents[i + 1] + (1 - crossover_point) * parents[i]
        else:
            offspring[i] = parents[i]
            if i + 1 < len(parents):
                offspring[i + 1] = parents[i + 1]
    return offspring


def apply_mutation(offspring, mutation_rate, lower_bound, upper_bound):
    for i in range(len(offspring)):
        if np.random.rand() < mutation_rate:
            offspring[i] = np.random.uniform(lower_bound, upper_bound)
    return offspring


population = initialize_population(population_size, lower_bound, upper_bound)

for generation in range(num_generations):
    fitness_scores = calculate_fitness(population)
    parents = select_parents(population, fitness_scores)
    offspring = perform_crossover(parents, crossover_rate)
    population = apply_mutation(offspring, mutation_rate, lower_bound, upper_bound)

  
    max_fitness = np.max(fitness_scores)
    mean_fitness = np.mean(fitness_scores)
    best_individual = population[np.argmax(fitness_scores)]
    
    print(f"Generation  {generation+1}\t: Max Fitness = {max_fitness}, Mean Fitness = {mean_fitness}, Best Individual = {best_individual}")


final_fitness_scores = calculate_fitness(population)
best_solution = population[np.argmax(final_fitness_scores)]
best_solution_value = objective_function(best_solution)

print(f"Best solution: x = {best_solution}, f(x) = {best_solution_value}")
