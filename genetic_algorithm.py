import copy
import math
import random
from typing import List, Tuple

# ------------------------- RESTRIÇÕES -------------------------
# Edite estes pares com os índices das cidades do seu `att_48_cities_locations`
# Ex: (0,3) impede ir da cidade índice 0 para 3 (e vice-versa).

PROHIBITED_PENALTY = 1e6
MAX_DISTANCE = 900

# ------------------------- FUNÇÕES BÁSICAS -------------------------

def calculate_distance(point1: Tuple[float, float], point2: Tuple[float, float], cities_location=None, vias_proibidas=None) -> float:
    """
    Distância Euclidiana entre dois pontos.
    Se `cities_location` for passado, checa vias_proibidas usando os índices.
    Retorna PROHIBITED_PENALTY quando a aresta for proibida.
    """
    if cities_location is not None:
        try:
            city_to_index = {coord: i for i, coord in enumerate(cities_location)}
            idx1 = city_to_index.get(point1)
            idx2 = city_to_index.get(point2)

            if idx1 is not None and idx2 is not None:
                if (point1, point2) in vias_proibidas or (point2, point1) in vias_proibidas:
                    return PROHIBITED_PENALTY           
        except ValueError:
            # ponto não está na lista passada (fallback para distância normal)
            pass
    return math.hypot(point1[0] - point2[0], point1[1] - point2[1])

def calculate_fitness(path: List[Tuple[float, float]],
                      cities_location=None, vias_proibidas=None,
                      postos: List[Tuple[float, float]] = None) -> float:
    """
    Calcula fitness como soma das distâncias da rota.
    - Respeita vias proibidas se cities_location for dado.
    - Se houver postos definidos, o veículo deve parar para reabastecer
      sempre que ultrapassar DIST_MAX_SEM_ABASTECER sem abastecer.
    """
    distance = 0.0
    since_last_refuel = 0.0
    n = len(path)
    visited = set()

    for i in range(n):
        a = path[i]
        b = path[(i + 1) % n]

        d = calculate_distance(a, b, cities_location, vias_proibidas)
        distance += d

        if b in visited and b != path[0]:  # exclui depósito
            return PROHIBITED_PENALTY  # penalidade grande
        visited.add(b)

        if d < PROHIBITED_PENALTY:
            since_last_refuel += d

        # Penalidade se a cidade já foi visitada
       

        # Se passou do limite, força parada no posto mais próximo
        if postos and since_last_refuel > MAX_DISTANCE:
            posto_proximo = min(postos, key=lambda p: calculate_distance(a, p, cities_location, vias_proibidas))
            d_posto = calculate_distance(a, posto_proximo, cities_location, vias_proibidas)
            d_volta = calculate_distance(posto_proximo, b, cities_location, vias_proibidas)
            distance += d_posto + d_volta  # adiciona caminho até o posto e de volta à rota
            since_last_refuel = 0.0  # reseta contador

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
    cities_locations: List[Tuple[float, float]], start_city_index: int = 1, cities_compare: List[Tuple[float, float]] = None, vias_proibidas: List[Tuple[Tuple[float, float], Tuple[float, float]]] = None
) -> List[Tuple[float, float]]:
    """Heurística vizinho mais próximo, mantendo depósito fixo no início e fim."""
    depot = cities_locations[0]
    unvisited = cities_locations[1:]  # exclui depósito
    current_city = cities_locations[start_city_index]
    route = [depot, current_city]
    unvisited.remove(current_city)
    visited = {current_city}

    while unvisited:
        candidates = [city for city in unvisited if city not in visited]
        if not candidates:
            break  # caso não tenha candidatos (redundante, mas seguro)

        nearest_city = min(
            candidates, key=lambda city: calculate_distance(current_city, city, cities_compare, vias_proibidas)
        )
        route.append(nearest_city)
        unvisited.remove(nearest_city)
        current_city = nearest_city
        visited.add(nearest_city)

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
