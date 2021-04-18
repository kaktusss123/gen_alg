import numpy as np
import ga


# Task 2
equation_inputs = [-4, 12, -3, 2, 8]
num_weights = 5


sol_per_pop = 10

pop_size = (sol_per_pop, num_weights)
new_population = np.random.uniform(low=-20.0, high=20.0, size=pop_size)


num_generations = 1000
num_parents_mating = 5


for generation in range(num_generations):
    print("Generation : ", generation)
    # Считаем fitness
    fitness = ga.cal_pop_fitness_task2(equation_inputs, new_population)

    # Выбираем лучших родителей
    parents = ga.select_mating_pool(new_population, fitness, num_parents_mating)

    # Скрещиваем лучших родителей
    offspring_crossover = ga.crossover(
        parents, offspring_size=(pop_size[0] - parents.shape[0], num_weights)
    )

    # Добавляем мутации
    offspring_mutation = ga.mutation1(offspring_crossover)

    # Создаем новую популяцию из лучших родителей и потомков
    new_population[: parents.shape[0], :] = parents
    new_population[parents.shape[0] :, :] = offspring_mutation

    # Лучший fitness в этом поколении
    print(
        "Best result : ",
        np.max(ga.cal_pop_fitness_task2(equation_inputs, new_population)),
    )

# Считаем fitness после прохода всех поколений
fitness = ga.cal_pop_fitness_task2(equation_inputs, new_population)
# Ищем индекс лучшей популяции
best_match_idx = np.where(fitness == np.max(fitness))

print("Best solution : ", new_population[best_match_idx, :])
print("Best solution fitness : ", fitness[best_match_idx])
