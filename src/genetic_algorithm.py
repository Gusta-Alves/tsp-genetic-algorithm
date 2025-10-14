# -*- coding: utf-8 -*-
"""
Genetic Algorithm functions for TSP.

Core genetic algorithm operations: fitness calculation, selection, crossover, mutation.
"""

import copy
import math
import random
from typing import List, Protocol, Tuple
from city import City
PROHIBITED_PENALTY = 100000
def calculate_distance(
    point1: City,
    point2: City,
    cities_location=None,
    vias_proibidas=None,
) -> float:
    """
    Distância Euclidiana entre dois pontos.
    Agora usa objetos City.
    """
    p1_coords = point1.get_coords()
    p2_coords = point2.get_coords()
    return math.hypot(p1_coords[0] - p2_coords[0], p1_coords[1] - p2_coords[1])


def calculate_fitness(
    path: List[City],
    cities_location=None,
    vias_proibidas=None,
    postos: List[Tuple[float, float]] = None,
) -> float:
    """
    Calcula fitness como soma das distâncias da rota.
    Função mantida para compatibilidade com código legado.
    """
    distance = 0.0
    since_last_refuel = 0.0
    n = len(path)
    visited = set()

    for i in range(n):
        # O caminho é circular, então a última cidade se conecta à primeira
        city_a = path[i]
        city_b = path[(i + 1) % n]

        if vias_proibidas and ((city_a, city_b) in vias_proibidas or (city_b, city_a) in vias_proibidas):
            return PROHIBITED_PENALTY

        d = calculate_distance(city_a, city_b, cities_location, vias_proibidas)
        distance += d

        # Penaliza rotas que visitam a mesma cidade mais de uma vez (exceto o depósito no final)
        if city_b in visited and city_b != path[0]:
            return PROHIBITED_PENALTY
        visited.add(city_b)

        if d < 1000000.0:
            since_last_refuel += d

        # Se passou do limite, força parada no posto mais próximo
        if postos and since_last_refuel > 900:  # MAX_DISTANCE
            posto_proximo = min(
                postos,
                key=lambda p: calculate_distance(city_a, p, cities_location, vias_proibidas),
            )
            d_posto = calculate_distance(
                city_a, posto_proximo, cities_location, vias_proibidas
            )
            d_volta = calculate_distance(
                posto_proximo, city_b, cities_location, vias_proibidas
            )
            distance += (
                d_posto + d_volta
            )  # adiciona caminho até o posto e de volta à rota
            since_last_refuel = 0.0  # reseta contador

    return distance


class FitnessCalculator:
    """Calculates fitness for TSP routes."""

    def calculate_fitness(self, route: List[Tuple[float, float]]) -> float:
        """
        Calculate fitness of a route.

        Args:
            route: The route to evaluate

        Returns:
            Fitness value (lower is better)
        """
        return self.problem.calculate_route_distance(route)


# ------------------------- POPULAÇÃO -------------------------


def generate_random_population(
    cities: List[City], population_size: int
) -> List[List[Tuple[float, float]]]:
    """
    Gera população aleatória de rotas.
    Mantém depósito fixo na primeira e última posição.
    Suporta tanto TSPProblem quanto lista de cidades (para compatibilidade).
    """
  
    depot = cities[0]
    cities_to_shuffle = cities[1:-1]  # Exclui depósito para embaralhar
    
    population = []
    for _ in range(population_size):
        route = cities_to_shuffle[:]
        random.shuffle(route)
        route = [depot] + route + [depot]
        population.append(route)
    return population


# ------------------------- HEURÍSTICAS -------------------------


def nearest_neighbor_heuristic(
    cities: List[City],
    start_city_index: int = 1,
    cities_compare=None,
    vias_proibidas=None,
) -> List[Tuple[float, float]]:
    """
    Heurística vizinho mais próximo, mantendo depósito fixo no início e fim.
    Suporta tanto TSPProblem quanto lista de cidades (para compatibilidade).
    """
   
    depot = cities[0]

    def calculate_dist(p1, p2):
        p1_coords, p2_coords = p1.get_coords(), p2.get_coords()
        return math.hypot(p1_coords[0] - p2_coords[0], p1_coords[1] - p2_coords[1])

    unvisited = cities[1:-1]  # exclui depósito
    current_city = cities[start_city_index]
    route = [depot, current_city]
    unvisited.remove(current_city)
    visited = {current_city}

    while unvisited:
        candidates = [city for city in unvisited if city not in visited]
        if not candidates:
            break

        nearest_city = min(
            candidates, key=lambda city: calculate_dist(current_city, city)
        )
        route.append(nearest_city)
        unvisited.remove(nearest_city)
        current_city = nearest_city
        visited.add(nearest_city)

    route.append(depot)
    return route


# ------------------------- SELEÇÃO -------------------------


def tournament_selection(
    population: List[List[City]],
    population_fitness: List[float],
    tournament_size: int = 3,
) -> List[City]:
    """Seleciona melhor indivíduo entre aleatoriamente escolhidos."""
    tournament_indices = random.sample(range(len(population)), tournament_size)
    tournament_fitness = [population_fitness[i] for i in tournament_indices]
    winner_index = tournament_indices[tournament_fitness.index(min(tournament_fitness))]
    return population[winner_index]


# ------------------------- CROSSOVER -------------------------


def order_crossover(
    parent1: List[City], parent2: List[City]
) -> Tuple[List[City], List[City]]:
    """
    Order Crossover (OX) mantendo depósito fixo no início/fim.
    """
    depot = parent1[0]
    p1 = parent1[1:-1]  # remove depósito
    p2 = parent2[1:-1]

    length = len(p1)
    start_index = random.randint(0, length - 1)
    end_index = random.randint(start_index + 1, length)

    # Cópia do segmento
    child1 = p1[start_index:end_index]
    child2 = p2[start_index:end_index]

    # Preencher com genes restantes
    remaining_genes1 = [gene for gene in p2 if gene not in child1]
    remaining_genes2 = [gene for gene in p1 if gene not in child2]

    # Inserir genes restantes nas posições corretas
    for i, gene in enumerate(remaining_genes1):
        child1.insert(i if i < start_index else i + (end_index - start_index), gene)
    for i, gene in enumerate(remaining_genes2):
        child2.insert(i if i < start_index else i + (end_index - start_index), gene)

    # Adicionar depósito no início e fim
    child1 = [depot] + child1 + [depot]
    child2 = [depot] + child2 + [depot]

    return child1, child2


# ------------------------- MUTATION -------------------------


def mutate(solution: List[City], mutation_probability: float) -> List[City]:
    """Mutação simples: swap, reverse ou 2-opt, mantendo depósito fixo."""
    mutated_solution = copy.deepcopy(solution)
    if random.random() < mutation_probability:
        if len(solution) < 4:  # menos de 2 cidades + depósito
            return solution
        mutation_type = random.choice(["swap", "reverse", "2opt"])
        # Sempre exclui depósito do início/fim
        sub = mutated_solution[1:-1]
        if mutation_type == "swap":
            i, j = random.sample(range(len(sub)), 2)
            sub[i], sub[j] = sub[j], sub[i]
        elif mutation_type in ["reverse", "2opt"]:
            i = random.randint(0, len(sub) - 2)
            j = random.randint(i + 1, len(sub))
            sub[i:j] = reversed(sub[i:j])
        mutated_solution[1:-1] = sub
    return mutated_solution


# ------------------------- SORT POPULATION -------------------------


def sort_population(
    population: List[List[City]], fitness: List[float]
) -> Tuple[List[List[City]], List[float]]:
    """Ordena população por fitness (menor é melhor)."""
    combined = list(zip(population, fitness))
    combined.sort(key=lambda x: x[1])
    sorted_population, sorted_fitness = zip(*combined)
    return list(sorted_population), list(sorted_fitness)
