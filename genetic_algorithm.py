import copy
import math
import random
from typing import List, Tuple

from numpy import cross

default_problems = {
    5: [(733, 251), (706, 87), (546, 97), (562, 49), (576, 253)],
    10: [
        (470, 169),
        (602, 202),
        (754, 239),
        (476, 233),
        (468, 301),
        (522, 29),
        (597, 171),
        (487, 325),
        (746, 232),
        (558, 136),
    ],
    12: [
        (728, 67),
        (560, 160),
        (602, 312),
        (712, 148),
        (535, 340),
        (720, 354),
        (568, 300),
        (629, 260),
        (539, 46),
        (634, 343),
        (491, 135),
        (768, 161),
    ],
    15: [
        (512, 317),
        (741, 72),
        (552, 50),
        (772, 346),
        (637, 12),
        (589, 131),
        (732, 165),
        (605, 15),
        (730, 38),
        (576, 216),
        (589, 381),
        (711, 387),
        (563, 228),
        (494, 22),
        (787, 288),
    ],
}


def generate_random_population(
    cities_location: List[Tuple[float, float]], population_size: int
) -> List[List[Tuple[float, float]]]:
    """
    Generate a random population of routes for a given set of cities.

    Parameters:
    - cities_location (List[Tuple[float, float]]): A list of tuples representing the locations of cities,
      where each tuple contains the latitude and longitude.
    - population_size (int): The size of the population, i.e., the number of routes to generate.

    Returns:
    List[List[Tuple[float, float]]]: A list of routes, where each route is represented as a list of city locations.
    """
    return [
        random.sample(cities_location, len(cities_location))
        for _ in range(population_size)
    ]


def nearest_neighbor_heuristic(
    cities_locations: List[Tuple[float, float]], start_city_index: int = 0
) -> List[Tuple[float, float]]:

    unvisited = cities_locations[:]
    current_city = cities_locations[start_city_index]
    unvisited.remove(current_city)
    route = [current_city]

    while unvisited:
        nearest_city = min(
            unvisited, key=lambda city: calculate_distance(current_city, city)
        )
        route.append(nearest_city)
        unvisited.remove(nearest_city)
        current_city = nearest_city

    return route


def convex_hull_heuristic(
    cities_locations: List[Tuple[float, float]],
) -> List[Tuple[float, float]]:
    """
    Generate a TSP route using convex hull heuristic.

    This heuristic works by:
    1. Finding the convex hull of all points
    2. Creating an initial tour using the convex hull points
    3. Inserting the remaining interior points optimally into the tour

    Parameters:
    - cities_locations (List[Tuple[float, float]]): A list of city coordinates

    Returns:
    List[Tuple[float, float]]: A tour route starting with the convex hull
    """
    if len(cities_locations) <= 3:
        return cities_locations[:]

    def cross_product(O, A, B):
        return (A[0] - O[0]) * (B[1] - O[1]) - (A[1] - O[1]) * (B[0] - O[0])

    def convex_hull(points):
        points = sorted(set(points))
        if len(points) <= 1:
            return points

        lower = []
        for p in points:
            while len(lower) >= 2 and cross_product(lower[-2], lower[-1], p) <= 0:
                lower.pop()
            lower.append(p)

        upper = []
        for p in reversed(points):
            while len(upper) >= 2 and cross_product(upper[-2], upper[-1], p) <= 0:
                upper.pop()
            upper.append(p)

        return lower[:-1] + upper[:-1]

    hull_points = convex_hull(cities_locations)

    if len(hull_points) == len(cities_locations):
        return hull_points

    tour = hull_points[:]

    interior_points = [point for point in cities_locations if point not in hull_points]

    for point in interior_points:
        best_position = 0
        best_increase = float("inf")

        for i in range(len(tour)):
            current_distance = calculate_distance(tour[i], tour[(i + 1) % len(tour)])
            new_distance = calculate_distance(tour[i], point) + calculate_distance(
                point, tour[(i + 1) % len(tour)]
            )
            increase = new_distance - current_distance

            if increase < best_increase:
                best_increase = increase
                best_position = i + 1

        tour.insert(best_position, point)

    return tour


def calculate_distance(
    point1: Tuple[float, float], point2: Tuple[float, float]
) -> float:
    """
    Calculate the Euclidean distance between two points.

    Parameters:
    - point1 (Tuple[float, float]): The coordinates of the first point.
    - point2 (Tuple[float, float]): The coordinates of the second point.

    Returns:
    float: The Euclidean distance between the two points.
    """
    return math.sqrt((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2)


def calculate_fitness(path: List[Tuple[float, float]]) -> float:
    """
    Calculate the fitness of a given path based on the total Euclidean distance.

    Parameters:
    - path (List[Tuple[float, float]]): A list of tuples representing the path,
      where each tuple contains the coordinates of a point.

    Returns:
    float: The total Euclidean distance of the path.
    """
    distance = 0
    n = len(path)
    for i in range(n):
        distance += calculate_distance(path[i], path[(i + 1) % n])

    return distance


def tournament_selection(
    population: List[List[Tuple[float, float]]],
    population_fitness: List[float],
    tournament_size: int = 3,
) -> List[Tuple[float, float]]:

    tournament_indices = random.sample(range(len(population)), tournament_size)
    tournament_fitness = [population_fitness[i] for i in tournament_indices]
    winner_index = tournament_indices[tournament_fitness.index(min(tournament_fitness))]
    return population[winner_index]


def order_crossover(
    parent1: List[Tuple[float, float]], parent2: List[Tuple[float, float]]
) -> List[Tuple[float, float]]:
    """
    Perform order crossover (OX) between two parent sequences to create a child sequence.

    Parameters:
    - parent1 (List[Tuple[float, float]]): The first parent sequence.
    - parent2 (List[Tuple[float, float]]): The second parent sequence.

    Returns:
    List[Tuple[float, float]]: The child sequence resulting from the order crossover.
    """
    length = len(parent1)

    # Choose two random indices for the crossover
    start_index = random.randint(0, length - 1)
    end_index = random.randint(start_index + 1, length)

    # Initialize the child with a copy of the substring from parent1
    child1 = parent1[start_index:end_index]
    child2 = parent2[start_index:end_index]

    # Fill in the remaining positions with genes from parent2
    remaining_positions1 = [
        i for i in range(length) if i < start_index or i >= end_index
    ]
    remaining_genes1 = [gene for gene in parent2 if gene not in child1]

    for position, gene in zip(remaining_positions1, remaining_genes1):
        child1.insert(position, gene)

    remaining_positions2 = [
        i for i in range(length) if i < start_index or i >= end_index
    ]
    remaining_genes2 = [gene for gene in parent2 if gene not in child2]

    for position, gene in zip(remaining_positions2, remaining_genes2):
        child2.insert(position, gene)

    return child1, child2


### demonstration: crossover test code
# Example usage:
# parent1 = [(1, 1), (2, 2), (3, 3), (4,4), (5,5), (6, 6)]
# parent2 = [(6, 6), (5, 5), (4, 4), (3, 3),  (2, 2), (1, 1)]

# # parent1 = [1, 2, 3, 4, 5, 6]
# # parent2 = [6, 5, 4, 3, 2, 1]


# child = order_crossover(parent1, parent2)
# print("Parent 1:", [0, 1, 2, 3, 4, 5, 6, 7, 8])
# print("Parent 1:", parent1)
# print("Parent 2:", parent2)
# print("Child   :", child)


# # Example usage:
# population = generate_random_population(5, 10)

# print(calculate_fitness(population[0]))


# population = [(random.randint(0, 100), random.randint(0, 100))
#           for _ in range(3)]


# TODO: implement a mutation_intensity and invert pieces of code instead of just swamping two.
def mutate(
    solution: List[Tuple[float, float]], mutation_probability: float
) -> List[Tuple[float, float]]:
    """
    Mutate a solution by inverting a segment of the sequence with a given mutation probability.

    Parameters:
    - solution (List[int]): The solution sequence to be mutated.
    - mutation_probability (float): The probability of mutation for each individual in the solution.

    Returns:
    List[int]: The mutated solution sequence.
    """
    mutated_solution = copy.deepcopy(solution)

    # Check if mutation should occur
    if random.random() < mutation_probability:

        # Ensure there are at least two cities to perform a swap
        if len(solution) < 2:
            return solution

        mutation_type = random.choice(["swap", "reverse", "2opt"])

        if mutation_type == "swap":
            i, j = random.sample(range(len(solution)), 2)
            mutated_solution[i], mutated_solution[j] = solution[j], solution[i]

        elif mutation_type == "reverse":
            i = random.randint(0, len(solution) - 2)
            j = random.randint(i + 1, len(solution))
            mutated_solution[i:j] = reversed(mutated_solution[i:j])
        elif mutation_type == "2opt":
            i = random.randint(0, len(solution) - 2)
            j = random.randint(i + 2, len(solution))
            mutated_solution[i:j] = reversed(mutated_solution[i:j])

    return mutated_solution


### Demonstration: mutation test code
# # Example usage:
# original_solution = [(1, 1), (2, 2), (3, 3), (4, 4)]
# mutation_probability = 1

# mutated_solution = mutate(original_solution, mutation_probability)
# print("Original Solution:", original_solution)
# print("Mutated Solution:", mutated_solution)


def sort_population(
    population: List[List[Tuple[float, float]]], fitness: List[float]
) -> Tuple[List[List[Tuple[float, float]]], List[float]]:
    """
    Sort a population based on fitness values.

    Parameters:
    - population (List[List[Tuple[float, float]]]): The population of solutions, where each solution is represented as a list.
    - fitness (List[float]): The corresponding fitness values for each solution in the population.

    Returns:
    Tuple[List[List[Tuple[float, float]]], List[float]]: A tuple containing the sorted population and corresponding sorted fitness values.
    """
    # Combine lists into pairs
    combined_lists = list(zip(population, fitness))

    # Sort based on the values of the fitness list
    sorted_combined_lists = sorted(combined_lists, key=lambda x: x[1])

    # Separate the sorted pairs back into individual lists
    sorted_population, sorted_fitness = zip(*sorted_combined_lists)

    return sorted_population, sorted_fitness


if __name__ == "__main__":
    N_CITIES = 10

    POPULATION_SIZE = 100
    N_GENERATIONS = 100
    MUTATION_PROBABILITY = 0.3
    cities_locations = [
        (random.randint(0, 100), random.randint(0, 100)) for _ in range(N_CITIES)
    ]

    # CREATE INITIAL POPULATION
    population = generate_random_population(cities_locations, POPULATION_SIZE)

    # Lists to store best fitness and generation for plotting
    best_fitness_values = []
    best_solutions = []

    for generation in range(N_GENERATIONS):

        population_fitness = [
            calculate_fitness(individual) for individual in population
        ]

        population, population_fitness = sort_population(population, population_fitness)

        best_fitness = calculate_fitness(population[0])
        best_solution = population[0]

        best_fitness_values.append(best_fitness)
        best_solutions.append(best_solution)

        print(f"Generation {generation}: Best fitness = {best_fitness}")

        new_population = [population[0]]  # Keep the best individual: ELITISM

        while len(new_population) < POPULATION_SIZE:

            # SELECTION
            parent1, parent2 = random.choices(
                population[:10], k=2
            )  # Select parents from the top 10 individuals

            # CROSSOVER
            child1, child2 = order_crossover(parent1, parent2)

            ## MUTATION
            child1 = mutate(child1, MUTATION_PROBABILITY)
            new_population.append(child1)

            # Add second child if there's space in population
            if len(new_population) < POPULATION_SIZE:
                child2 = mutate(child2, MUTATION_PROBABILITY)
                new_population.append(child2)

        print("generation: ", generation)
        population = new_population
