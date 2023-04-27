"""
Алгоритм ACO (Ant Colony Optimization)
"""

import numpy as np
from random import random


def inverse_distance(dist: np.ndarray) -> np.ndarray:
    """
    Вычисляет обратное расстояние между городами в матрице расстояний.
    Эта функция нужна в алгоритме ACO для того, чтобы определить начальное
    количество феромона на ребрах графа. Обратное расстояние между городами
    показывает, насколько привлекательно для муравья выбрать это ребро при
    построении маршрута. Чем меньше расстояние, тем больше обратное расстояние
    и тем больше феромона. Это способствует поиску более коротких и оптимальных
    маршрутов.
    :param dist: Матрица расстояний между городами.
    :return: Матрица обратных расстояний между городами
    """
    num = dist.shape[0]  # Количество городов
    reverse_dist = np.zeros(
        (num, num))  # Создаем пустую матрицу обратных расстояний
    for i in range(num):
        for j in range(num):
            if i != j:
                reverse_dist[i, j] = 1 / dist[
                    i, j]  # Вычисляем обратное расстояние по формуле 1 / d_ij
    return reverse_dist


def choose_next_city(visited: tuple, unvisited: tuple, ph: np.ndarray,
                     rev_dist: np.ndarray, a: float, b: float) -> int:
    """
    Выбирает следующий город с помощью жадного правила на основе феромона
    и обратного расстояния.
    Эта функция нужна в алгоритме ACO для того, чтобы имитировать поведение
    муравья при выборе следующего города для посещения. Муравей не выбирает
    город случайно, а учитывает два фактора: количество феромона на ребре,
    соединяющем текущий и следующий город, и расстояние между ними.
    Чем больше феромона и чем меньше расстояние, тем выше вероятность выбора
    этого города. Параметры a и b определяют, насколько сильно влияют феромон и
    расстояние на принятие решения.
    :param visited: Кортеж посещенных городов.
    :param unvisited: Кортеж непосещенных городов.
    :param ph: Матрица феромона между городами.
    :param rev_dist: Матрица обратных расстояний между городами.
    :param a: Параметр влияния феромона.
    :param b: Параметр влияния обратного расстояния.
    :return: Индекс следующего города
    """
    # Вычисляем привлекательность каждого непосещенного города по
    # формуле ph_ij^a * d_ij^b
    attractiveness = (np.power(ph[visited[-1], unvisited], a) * rev_dist[
        visited[-1], unvisited])

    # Нормализуем привлекательность так, чтобы сумма была равна 1
    attractiveness /= np.sum(attractiveness)

    # Генерируем случайное число от 0 до 1
    rand = random()

    # Проходим по всем непосещенным городам и выбираем тот,
    # который соответствует случайному числу
    for i in range(attractiveness.size):
        if rand < attractiveness[i]:
            next_city = unvisited[i]
            break
        rand -= attractiveness[i]
    else:
        next_city = unvisited[-1]
    return next_city


def length(route: tuple, dist: np.ndarray) -> float:
    """
    Вычисляет длину маршрута по матрице расстояний.
    :param route: Кортеж индексов городов в маршруте.
    :param dist: Матрица расстояний между городами
    :return: Длина маршрута
    """
    # Суммируем расстояния между соседними городами в маршруте
    length_between_cities = sum([dist[route[i], route[i + 1]] for i in
                                 range(len(route) - 1)])
    return length_between_cities


def update_pheromone(ph: np.ndarray, route: tuple, length: float,
                     q: float) -> np.ndarray:
    """
    Обновляет феромон на основе маршрута и его длины по
    формуле ph_ij = ph_ij + q /
    :param ph: Матрица феромона между городами
    :param route: Кортеж индексов городов в маршруте
    :param length: Длина маршрута
    :param q: Параметр влияния длины маршрута
    :return: Обновленная матрица феромона между городами
    """
    # Вычисляем относительное количество феромона по формуле q / L
    relative_q = q / length

    # Добавляем феромон к каждому ребру в маршруте
    for i in range(len(route) - 1):
        ph[route[i], route[i + 1]] += relative_q
    return ph


def aco(dist: np.ndarray, start: int, end: int, *, ants: int = 1, ages: int = 1,
        rho: float = 0.1, a: float = 1, b: float = 1, q: float = 1,
        ph_min: float = 0.01, ph_max: float = 1, elite: int = 0) -> tuple:
    """
    Решает задачу коммивояжера с помощью алгоритма муравьиной колонии.
    :param dist: Матрица расстояний между городами
    :param start: Индекс начального города.
    :param end: Индекс конечного города.
    :param ants: Количество муравьев в колонии.
    :param ages: Количество поколений муравьев.
    :param rho: Коэффициент испарения феромона.
    :param a: Параметр влияния феромона
    :param b: Параметр влияния обратного расстояния
    :param q: Параметр влияния длины маршрута
    :param ph_min: Минимальное значение феромона на ребре
    :param ph_max: Максимальное значение феромона на ребре
    :param elite: Количество элитных муравьев, которые оставляют больше феромона
    :return: кортеж индексов городов в лучшем маршруте и длину лучшего маршрута.
    """
    # Вычисляем обратное расстояние между городами
    rev_dist = inverse_distance(dist)
    # Получаем количество городов
    num = dist.shape[0]
    # Инициализируем матрицу феромона максимальным значением
    ph = np.ones((num, num)) * ph_max
    # Инициализируем лучший маршрут и его длину пустыми значениями
    best_route, best_length = tuple(), np.inf
    # Создаем множество вершин, исключая начальную и конечную
    vertexes = set(range(num)) - {end} | {start}
    # Получаем количество доступных вершин
    available = len(vertexes)
    # Повторяем для каждого поколения муравьев
    for _ in range(ages):
        # Повторяем для каждого муравья в колонии
        for _ in range(ants):
            # Инициализируем посещенные вершины начальной вершиной
            visited = (start,)
            # Повторяем пока не посетим все доступные вершины
            while len(visited) < available:
                # Получаем непосещенные вершины
                unvisited = tuple(vertexes - set(visited))
                # Выбираем следующую вершину с помощью жадного правила
                next_city = choose_next_city(visited, unvisited, ph, rev_dist,
                                             a, b)
                # Добавляем ее к посещенным вершинам
                visited += (next_city,)
            # Добавляем конечную вершину к посещенным вершинам
            visited += (end,)
            # Вычисляем длину маршрута
            route_length = length(visited, dist)
            # Обновляем феромон на основе маршрута и его длины
            ph = update_pheromone(ph, visited, route_length, q)
            # Если длина маршрута меньше лучшей длины, то обновляем
            # лучший маршрут и его длину
            if route_length < best_length:
                best_route, best_length = visited, route_length
        # Обновляем феромон на основе лучшего маршрута и его длины с
        # учетом элитных муравьев
        ph = update_pheromone(ph, best_route, best_length, q * elite)
        # Умножаем феромон на коэффициент испарения и обрезаем его по
        # минимальному и максимальному значению
        ph = np.clip(ph * (1 - rho), ph_min, ph_max)
    # Возвращаем лучший маршрут и его длину или None, если такого нет
    return (best_route, best_length) if best_route else (None, None)
