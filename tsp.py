import pygame
from pygame.locals import *
import random
import itertools
from genetic_algorithm import convex_hull_heuristic, mutate, nearest_neighbor_heuristic, order_crossover, generate_random_population, calculate_fitness, sort_population, default_problems, tournament_selection
from draw_functions import draw_paths, draw_plot, draw_cities
import sys
import numpy as np
import pygame

# Define constant values
# pygame
WIDTH, HEIGHT = 800, 400
NODE_RADIUS = 5
FPS = 30
PLOT_X_OFFSET = 450

# GA
N_CITIES = 15
POPULATION_SIZE = 150
N_GENERATIONS = None
MUTATION_PROBABILITY = 0.7
ELITE_SIZE = 10

# Define colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)
BLUE = (0, 0, 255)


# Initialize problem
# Using Random cities generation
# cities_locations = [(random.randint(NODE_RADIUS + PLOT_X_OFFSET, WIDTH - NODE_RADIUS), random.randint(NODE_RADIUS, HEIGHT - NODE_RADIUS))
#                     for _ in range(N_CITIES)]


# # Using Deault Problems: 10, 12 or 15
WIDTH, HEIGHT = 800, 400
cities_locations = default_problems[N_CITIES]


# Initialize Pygame
pygame.init()
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("TSP Solver using Pygame")
clock = pygame.time.Clock()
generation_counter = itertools.count(start=1)  # Start the counter at 1


# Create Initial Population
# Using heuristics like Nearest Neighbour and Convex Hull to initialize
population = []

# Add one convex hull solution
# population.append(convex_hull_heuristic(cities_locations))

# Add several nearest neighbor solutions starting from different cities
# for _ in range(POPULATION_SIZE//4):
#     population.append(nearest_neighbor_heuristic(cities_locations, random.randint(0, N_CITIES - 1)))

for i in range(N_CITIES):
    start_city_index = i % N_CITIES
    print(f"Starting NN heuristic from city index: {start_city_index}")
    population.append(nearest_neighbor_heuristic(cities_locations, start_city_index))

# Fill the rest with random solutions
population.extend(generate_random_population(cities_locations, POPULATION_SIZE - len(population)))

best_fitness_values = []
best_solutions = []


# Main game loop
running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_q:
                running = False

    generation = next(generation_counter)

    screen.fill(WHITE)

    population_fitness = [calculate_fitness(
        individual) for individual in population]

    population, population_fitness = sort_population(
        population,  population_fitness)

    best_fitness = calculate_fitness(population[0])
    best_solution = population[0]

    best_fitness_values.append(best_fitness)
    best_solutions.append(best_solution)

    draw_plot(screen, list(range(len(best_fitness_values))),
              best_fitness_values, y_label="Fitness - Distance (pxls)")

    draw_cities(screen, cities_locations, RED, NODE_RADIUS)
    draw_paths(screen, best_solution, BLUE, width=3)
    draw_paths(screen, population[1], rgb_color=(128, 128, 128), width=1)

    print(f"Generation {generation}: Best fitness = {round(best_fitness, 2)}")

    new_population = [*population[:ELITE_SIZE]]  # Keep the best 5 individuals: ELITE_SIZE

    while len(new_population) < POPULATION_SIZE:

        # selection
        # simple selection based on first 10 best solutions
        # parent1, parent2 = random.choices(population[:10], k=2)

        # solution based on fitness probability
        # probability = 1 / np.array(population_fitness)
        # parent1, parent2 = random.choices(population, weights=probability, k=2)

        # solution based on tournament selection
        parent1 = tournament_selection(population, population_fitness, tournament_size=5)
        parent2 = tournament_selection(population, population_fitness, tournament_size=5)

        # child1 = order_crossover(parent1, parent2)
        child1, child2 = order_crossover(parent1, parent2)

        child1 = mutate(child1, MUTATION_PROBABILITY)
        child2 = mutate(child2, MUTATION_PROBABILITY)
        
        new_population.append(child1)
        if len(new_population) < POPULATION_SIZE:
            new_population.append(child2)

    population = new_population

    pygame.display.flip()
    clock.tick(FPS)


# TODO: save the best individual in a file if it is better than the one saved.

# exit software
pygame.quit()
sys.exit()
