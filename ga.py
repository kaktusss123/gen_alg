import numpy as np
from string import ascii_lowercase

alphabet = ascii_lowercase + " "


def cal_pop_fitness_task1(equation_inputs: list, pop: np.ndarray) -> np.ndarray:
    """
    Считает fitness текущей популяции.

    :param equation_inputs: Столбец X
    :param pop: Столбец весов
    :return: Значение параметра fitness
    """
    fitness = np.sum(pop * equation_inputs, axis=1)
    return fitness


def cal_pop_fitness_task2(equation_inputs: list, pop: np.ndarray) -> np.ndarray:
    """
    Считает fitness текущей популяции.
    Y = w1x1 - w2x2 + w3x3 * w4x4 + w5x5

    :param equation_inputs: Столбец X
    :param pop: Столбец весов
    :return: Значение параметра fitness
    """
    mult = pop * equation_inputs
    fitness = mult[0] - mult[1] + mult[2] * mult[3] + mult[4]
    return fitness


def cal_pop_fitness_task3(equation_inputs: list, pop: np.ndarray) -> np.ndarray:
    """
    Считает fitness текущей популяции.
    intelligent information systems

    :param equation_inputs: Столбец X
    :param pop: Столбец весов
    :return: Значение параметра fitness
    """
    array = [alphabet.index(x) for x in "intelligent information systems"]
    diff = np.sum(np.abs(pop - array), axis=1)
    return -diff


def select_mating_pool(
    pop: np.ndarray, fitness: np.ndarray, num_parents: int
) -> np.ndarray:
    """
    Немного изменил функцию выбора пула лучших родителей. Выглядит красивее, а делает то же самое.
    Сортирует популяцию по fitness в обратном порядке. И выбирает num_parents лучших.

    :param pop: Популяция
    :param fitness: Столбец с оценкой
    :param num_parents: Количество родителей
    :return: Пул лучших родителей
    """
    idx = np.argsort(fitness)[::-1]
    return pop[idx][:num_parents, :]


def crossover(parents: np.ndarray, offspring_size: tuple):
    """
    Выполняет скрещивание родителей.

    :param parents: Массив родителей
    :param offspring_size: Размер потомков (кол-во потомков, кол-во генов)
    :return: Массив потомков
    """
    offspring = np.empty(offspring_size)
    # Делим наших родителей пополам
    crossover_point = np.uint8(offspring_size[1] / 2) + np.random.randint(1)

    for k in range(offspring_size[0]):
        # Индекс первого родителя
        parent1_idx = k % parents.shape[0]
        # Индекс второго родителя. Выбираются последовательно.
        parent2_idx = (k + 1) % parents.shape[0]
        # Первая половина потомка от первого родителя
        offspring[k, :crossover_point] = parents[parent1_idx, 0:crossover_point]
        # Вторая - от второго
        offspring[k, crossover_point:] = parents[parent2_idx, crossover_point:]
    return offspring


def mutation3(offspring_crossover: np.ndarray) -> np.ndarray:
    """
    Реализует функицю мутации.
    Случайным образом меняет один ген в каждом потомке.

    :param offspring_crossover: Массив потомков
    :return: Массив потомков с мутацией
    """
    # Мутация меняет один ген у каждого потомка
    for idx in range(offspring_crossover.shape[0]):
        gene = np.random.randint(31)
        offspring_crossover[idx, gene] = np.random.randint(27)
    return offspring_crossover


def mutation1(offspring_crossover: np.ndarray) -> np.ndarray:
    """
    Реализует функицю мутации.
    Случайным образом меняет один ген в каждом потомке.

    :param offspring_crossover: Массив потомков
    :return: Массив потомков с мутацией
    """
    # Мутация меняет один ген у каждого потомка
    for idx in range(offspring_crossover.shape[0]):
        # Прибавляем к нужному гену рандомную мутацию
        gene = np.random.randint(0, offspring_crossover.shape[1], 3)
        offspring_crossover[idx, gene] += np.random.uniform(-10.0, 10.0, 1)
    return offspring_crossover
