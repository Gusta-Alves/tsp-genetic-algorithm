import copy
import math
import random
from typing import List, Tuple

# ------------------------- FUNÇÕES BÁSICAS -------------------------

def calculate_distance(point1: Tuple[float, float], point2: Tuple[float, float]) -> float:
    """Calcula distância Euclidiana entre dois pontos."""
    return math.sqrt((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2)

def calculate_fitness(path: List[Tuple[float, float]]) -> float:
    """Calcula fitness como soma das distâncias da rota."""
    distance = 0
    n = len(path)
    for i in range(n):
        distance += calculate_distance(path[i], path[(i + 1) % n])
    return distance

# ------------------------- POPULAÇÃO -------------------------

def generate_random_population(
    cities_location: List[Tuple[float, float]], population_size: int
) -> List[List[Tuple[float, float]]]:
    """
    Gera população aleatória de rotas.
    Mantém depósito fixo na primeira e última posição.
    """
    depot = cities_location[0]  # Considera primeiro ponto como depósito
    cities = cities_location[1:-1]  # Exclui depósito para embaralhar
    population = []
    for _ in range(population_size):
        route = cities[:]
        random.shuffle(route)
        route = [depot] + route + [depot]
        population.append(route)
    return population

# ------------------------- HEURÍSTICAS -------------------------

def nearest_neighbor_heuristic(
    cities_locations: List[Tuple[float, float]], start_city_index: int = 1
) -> List[Tuple[float, float]]:
    """Heurística vizinho mais próximo, mantendo depósito fixo no início e fim."""
    depot = cities_locations[0]
    unvisited = cities_locations[1:]  # exclui depósito
    current_city = cities_locations[start_city_index]
    route = [depot, current_city]
    unvisited.remove(current_city)

    while unvisited:
        nearest_city = min(
            unvisited, key=lambda city: calculate_distance(current_city, city)
        )
        route.append(nearest_city)
        unvisited.remove(nearest_city)
        current_city = nearest_city

    route.append(depot)  # volta ao depósito
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
    parent1: List[Tuple[float, float]], parent2: List[Tuple[float, float]]
) -> Tuple[List[Tuple[float, float]], List[Tuple[float, float]]]:
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

def mutate(
    solution: List[Tuple[float, float]], mutation_probability: float
) -> List[Tuple[float, float]]:
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
    population: List[List[Tuple[float, float]]], fitness: List[float]
) -> Tuple[List[List[Tuple[float, float]]], List[float]]:
    """Ordena população por fitness (menor é melhor)."""
    combined = list(zip(population, fitness))
    combined.sort(key=lambda x: x[1])
    sorted_population, sorted_fitness = zip(*combined)
    return list(sorted_population), list(sorted_fitness)
