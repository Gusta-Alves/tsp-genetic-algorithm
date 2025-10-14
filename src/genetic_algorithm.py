# -*- coding: utf-8 -*-
"""
Genetic Algorithm functions for TSP.

Core genetic algorithm operations: fitness calculation, selection, crossover, mutation.
"""

import copy
import math
import numpy as np
import random
from typing import List, Protocol, Tuple, Dict, Optional

from city import City
from typing import Dict, List, Tuple

_fitness_cache: Dict[Tuple[City, ...], float] = {}
_cache_hits: int = 0
_cache_misses: int = 0
#Cache desativado por estar interferindo nas restrições interativas
_cache_enabled: bool = False 
_distance_matrix = None


def set_distance_matrix(matrix):
    """
    Set global distance matrix for optimized distance calculations.

    Args:
        matrix: DistanceMatrix instance
    """
    global _distance_matrix
    _distance_matrix = matrix


def enable_fitness_cache(enabled: bool = True) -> None:
    """
    Habilita ou desabilita o cache de fitness.

    Args:
        enabled: True para habilitar, False para desabilitar
    """
    global _cache_enabled
    _cache_enabled = enabled
    if not enabled:
        clear_fitness_cache()


def clear_fitness_cache() -> None:
    """Limpa o cache de fitness."""
    global _fitness_cache, _cache_hits, _cache_misses
    _fitness_cache.clear()
    _cache_hits = 0
    _cache_misses = 0


def get_cache_stats() -> Dict[str, any]:
    """
    Retorna estatísticas do cache de fitness.

    Returns:
        Dict com hits, misses, size e hit_rate
    """
    total_requests = _cache_hits + _cache_misses
    hit_rate = (_cache_hits / total_requests * 100) if total_requests > 0 else 0.0

    return {
        "hits": _cache_hits,
        "misses": _cache_misses,
        "total_requests": total_requests,
        "hit_rate": hit_rate,
        "cache_size": len(_fitness_cache),
        "enabled": _cache_enabled,
    }


def create_distance_matrix(cities: List[City]) -> Tuple[np.ndarray, Dict[City, int]]:
    """Cria uma matriz de distância e um mapa de cidade para índice."""
    n = len(cities)
    dist_matrix = np.zeros((n, n))
    city_to_index = {city: i for i, city in enumerate(cities)}

    for i in range(n):
        for j in range(i, n):
            dist = calculate_distance(cities[i], cities[j])
            dist_matrix[i][j] = dist
            dist_matrix[j][i] = dist

    return dist_matrix, city_to_index


def calculate_distance(
    p1: City, p2: City, cities_location=None, vias_proibidas=None
) -> float:
    """
    Distância entre dois pontos.
    Usa matriz de distâncias se disponível, senão calcula Euclidiana.
    """
    p1_coords, p2_coords = p1.get_coords(), p2.get_coords()
    return math.hypot(p1_coords[0] - p2_coords[0], p1_coords[1] - p2_coords[1])


def calculate_fitness(
    path: List[City],
    distance_matrix: np.ndarray = None,
    city_to_index: dict = None,
    distance_limit: float = None,
    vias_proibidas: List[Tuple[City, City]] =None,
    postos: List[City] = None,
) -> float:
    """
    Calcula fitness como soma das distâncias da rota.

    Usa cache automático para evitar recalcular rotas idênticas.
    Cache hit rate típico: 30-50% em populações com elite preservado.

    Args:
        path: Lista de coordenadas da rota
        cities_location: Parâmetro legado (ignorado)
        vias_proibidas: Parâmetro legado (ignorado)
        postos: Lista de postos de abastecimento

    Returns:
        Fitness (distância total) da rota
    """
    global _cache_hits, _cache_misses

    # Cria chave para cache (tupla é hashable)
    route_key = tuple(path)

    # Verifica cache se habilitado
    if _cache_enabled and route_key in _fitness_cache:
        _cache_hits += 1
        return _fitness_cache[route_key]

    # Cache miss - calcula fitness
    _cache_misses += 1

    distance = 0.0
    since_last_refuel = 0.0
    visited = set()

    for i in range(len(path) - 1):
        a = path[i]
        b = path[i + 1]

        if vias_proibidas and ((a, b) in vias_proibidas or (b, a) in vias_proibidas):
            distance = 1_000_000.0 + (len(path) * 1000) # Penalidade grande
            if _cache_enabled:
                _fitness_cache[route_key] = distance
            return distance

        # Penaliza rotas que visitam a mesma cidade duas vezes (exceto o depósito no final)
        if b in visited and b != path[-1]:
            distance = 1_000_000.0 + (len(path) * 1000) # Penalidade grande
            if _cache_enabled:
                _fitness_cache[route_key] = distance
            return distance
        visited.add(b)

        idx_a = city_to_index.get(a)
        idx_b = city_to_index.get(b)        
        
        # Usa a matriz de distância se possível, senão calcula
        if idx_a is not None and idx_b is not None:         
            d = distance_matrix[idx_a, idx_b]
        else:            
            d = calculate_distance(a, b)

        since_last_refuel += d

        # Se passou do limite, força parada no posto mais próximo
        if postos and since_last_refuel > 900:  # MAX_DISTANCE
            # Encontra o posto mais próximo do ponto atual (a)
            posto_proximo = min(postos, key=lambda p: calculate_distance(a, p))
            
            # Calcula o custo do desvio: (a -> posto) + (posto -> b)
            d_posto = calculate_distance(a, posto_proximo)
            d_volta = calculate_distance(posto_proximo, b)
            
            # Adiciona a distância do desvio e subtrai a distância da rota original que foi substituída
            distance += (d_posto + d_volta)
            since_last_refuel = d_volta # A distância percorrida após abastecer é do posto até o próximo destino
        else:
            distance += d

    if distance_limit is not None and distance > distance_limit:
        # Penalidade severa para garantir que rotas que excedem o limite sejam descartadas.
        # O valor é alto o suficiente para superar qualquer rota válida.
        distance = 1_000_000.0 + (distance - distance_limit)

    # --- Fim do cálculo do Fitness ---

    if _cache_enabled:
        _fitness_cache[route_key] = distance

    return distance


# ------------------------- POPULAÇÃO -------------------------


def generate_random_population(
    cities_in_cluster: List[City], population_size: int
) -> List[List[Tuple[float, float]]]:
    """
    Gera população aleatória de rotas.
    Mantém depósito fixo na primeira e última posição.
    """
    # Modo legado
    cities = cities_in_cluster
    depot = cities[0]
    cities = cities[1:-1]  # Exclui depósito para embaralhar

    population = []
    for _ in range(population_size):
        route = cities[:]
        random.shuffle(route)
        route = [depot] + route + [depot]
        population.append(route)
    return population


# ------------------------- HEURÍSTICAS -------------------------


def nearest_neighbor_heuristic(
    cities_in_cluster: List[City],
    start_city_index: int = 1,
) -> List[Tuple[float, float]]:
    """
    Heurística vizinho mais próximo, mantendo depósito fixo no início e fim.
    """
    # Se o cluster tiver 0 ou 1 cidade (além do depósito), retorna a rota simples.
    if len(cities_in_cluster) <= 3: # Ex: [depot, city_A, depot] ou [depot, depot]
        return cities_in_cluster

    cities = cities_in_cluster
    depot = cities[0]

    def calculate_dist(p1, p2):
        return math.hypot(p1.x - p2.x, p1.y - p2.y)

    unvisited = cities[1:-1]  # exclui depósito do início e fim
    current_city = unvisited[start_city_index - 1] # Ajusta o índice para a lista fatiada
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
    population: List[List[Tuple[float, float]]],
    population_fitness: List[float],
    tournament_size: int = 3,
) -> List[Tuple[float, float]]:
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
    # Se os pais não tiverem cidades para cruzar (apenas depósito), retorna eles mesmos.
    if len(parent1) <= 2 or len(parent2) <= 2:
        return parent1, parent2

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
