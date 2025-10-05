"""
TSP Solver com múltiplos veículos usando GA
- Depósito central
- Clusteriza cidades em 4 grupos
- Cada veículo percorre seu cluster
- Rotas coloridas
- Lado esquerdo da tela: gráfico moderno + tabela moderna
- Lado direito da tela: mapa das cidades
"""

import itertools
import sys

import numpy as np
import pygame
from pygame.locals import *

from benchmark_att48 import *
from draw_functions import draw_cities, draw_paths
from genetic_algorithm import (
    calculate_fitness,    
    generate_random_population,
    mutate,
    nearest_neighbor_heuristic,
    order_crossover,
    sort_population,
    tournament_selection,
)

from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg

# ------------------------- CONSTANTES -------------------------
SCREEN_WIDTH, SCREEN_HEIGHT = 1200, 650
NODE_RADIUS = 10
FPS = 30

POPULATION_SIZE = 100
MUTATION_PROBABILITY = 0.7
ELITE_SIZE = 10

NUM_VEHICLES = 4
VEHICLE_COLORS = [(255,0,0), (0,255,0), (0,0,255), (255,165,0)]

MARGIN = 50

# ------------------------- DIVISÃO DA TELA -------------------------
LEFT_PANEL_WIDTH = 500
RIGHT_PANEL_WIDTH = SCREEN_WIDTH - LEFT_PANEL_WIDTH 
GRAPH_HEIGHT = 400
TABLE_HEIGHT = 200

# ------------------------- PREPARAR CIDADES -------------------------
att_cities_locations = np.array(att_48_cities_locations)
max_x = max(point[0] for point in att_cities_locations)
max_y = max(point[1] for point in att_cities_locations)

scale_x = (RIGHT_PANEL_WIDTH - MARGIN*2 - NODE_RADIUS) / max_x
scale_y = (SCREEN_HEIGHT - MARGIN*2 - NODE_RADIUS) / max_y

cities_locations = [
    (int(point[0] * scale_x + LEFT_PANEL_WIDTH + MARGIN), int(point[1] * scale_y + MARGIN))
    for point in att_cities_locations
]

# ------------------------- DEPÓSITO CENTRAL -------------------------
depot_x = int(np.mean([x for x, y in cities_locations]))
depot_y = int(np.mean([y for x, y in cities_locations]))
depot = (depot_x, depot_y)

# ------------------------- CLUSTERIZAR CIDADES -------------------------
cities_array = np.array(cities_locations)
kmeans = KMeans(n_clusters=NUM_VEHICLES, random_state=42)
kmeans.fit(cities_array)
labels = kmeans.labels_

vehicle_clusters = [[] for _ in range(NUM_VEHICLES)]
for idx, label in enumerate(labels):
    vehicle_clusters[label].append(cities_locations[idx])

# Adicionar depósito no início e fim de cada cluster
for i in range(NUM_VEHICLES):
    vehicle_clusters[i] = [depot] + vehicle_clusters[i] + [depot]

# ------------------------- PYGAME -------------------------
pygame.init()
screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
pygame.display.set_caption("TSP Solver com múltiplos veículos")
clock = pygame.time.Clock()
generation_counter = itertools.count(start=1)

# ------------------------- POPULAÇÃO INICIAL -------------------------
vehicle_populations = []
for cluster in vehicle_clusters:
    population = []
    for i in range(1, len(cluster)-1):
        population.append(nearest_neighbor_heuristic(cluster, i))
    population.extend(generate_random_population(cluster, POPULATION_SIZE - len(population)))
    vehicle_populations.append(population)

vehicle_best_fitness = [[] for _ in range(NUM_VEHICLES)]
vehicle_best_solutions = [[] for _ in range(NUM_VEHICLES)]
vehicle_last_change = [0]*NUM_VEHICLES

# ------------------------- LOOP PRINCIPAL -------------------------
running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.KEYDOWN and event.key == pygame.K_q:
            running = False

    generation = next(generation_counter)
    screen.fill((255,255,255))

    vehicle_info = []

    # ----------------- PARA CADA VEÍCULO -----------------
    for v in range(NUM_VEHICLES):
        population = vehicle_populations[v]
        population_fitness = [calculate_fitness(ind) for ind in population]
        population, population_fitness = sort_population(population, population_fitness)

        best_fitness = calculate_fitness(population[0])
        best_solution = population[0]

        if len(vehicle_best_solutions[v]) == 0 or vehicle_best_solutions[v][-1] != best_solution:
            vehicle_last_change[v] = generation

        vehicle_best_fitness[v].append(best_fitness)
        vehicle_best_solutions[v].append(best_solution)
        num_cities = len(best_solution) - 2  # não contar depósito
        vehicle_info.append((v, best_fitness, num_cities, vehicle_last_change[v]))

        # Desenhar rotas e cidades no painel da direita
        draw_paths(screen, best_solution, VEHICLE_COLORS[v], width=3)        
        draw_cities(screen, vehicle_clusters[v], VEHICLE_COLORS[v], NODE_RADIUS, depot)

        # Evolução da população
        new_population = [population[0]]
        while len(new_population) < POPULATION_SIZE:
            parent1 = tournament_selection(population, population_fitness, tournament_size=5)
            parent2 = tournament_selection(population, population_fitness, tournament_size=5)
            child1, child2 = order_crossover(parent1, parent2)
            child1 = mutate(child1, MUTATION_PROBABILITY)
            child2 = mutate(child2, MUTATION_PROBABILITY)
            new_population.append(child1)
            if len(new_population) < POPULATION_SIZE:
                new_population.append(child2)
        vehicle_populations[v] = new_population

    # ----------------- DESENHAR GRÁFICO -----------------
    plt_fig, plt_ax = plt.subplots(figsize=(5,4), dpi=100)
    plt_ax.set_facecolor('#f9f9f9')  # fundo suave
    plt_ax.grid(True, linestyle='--', alpha=0.5)

    for v in range(NUM_VEHICLES):
        plt_ax.plot(
            list(range(len(vehicle_best_fitness[v]))),
            vehicle_best_fitness[v],
            color=np.array(VEHICLE_COLORS[v])/255,
            label=f'Veículo {v+1}',
            linewidth=3
        )

    plt_ax.set_xlabel("Generation", fontsize=12)
    plt_ax.set_ylabel("Fitness (Distance)", fontsize=12)
    plt_ax.tick_params(axis='both', labelsize=10)
    plt_ax.legend(frameon=False, fontsize=10)
    plt.tight_layout(pad=2)

    canvas = FigureCanvasAgg(plt_fig)
    canvas.draw()
    renderer = canvas.get_renderer()
    raw_data = renderer.buffer_rgba()
    size = canvas.get_width_height()
    surf = pygame.image.frombuffer(raw_data, size, "RGBA")
    screen.blit(surf, (10, 10)) 
    plt.close(plt_fig)

    # ----------------- DESENHAR TABELA -----------------
    vehicle_info.sort(key=lambda x: x[1])
    pygame.font.init()
    font = pygame.font.SysFont("Arial", 20, bold=False)
    start_x, start_y = 10, GRAPH_HEIGHT + 20
    col_widths = [80, 120, 150, 150]
    row_height = 35

    headers = ["Veículo", "Distância", "Cidades visitadas", "Última mudança"]

    # Cabeçalho
    for col, header in enumerate(headers):
        x = start_x + sum(col_widths[:col])
        y = start_y
        pygame.draw.rect(screen, (200,200,200), (x, y, col_widths[col], row_height))
        pygame.draw.rect(screen, (0,0,0), (x, y, col_widths[col], row_height), 2)
        text_surface = font.render(header, True, (0,0,0))
        screen.blit(text_surface, (x + 5, y + 5))

    start_y += row_height

    # Linhas da tabela (zebra) e cores por veículo
    for row, info in enumerate(vehicle_info):
        v, dist, num_cities, last_change = info
        values = [str(v+1), str(round(dist)), str(num_cities-1), str(last_change)]
        for col, val in enumerate(values):
            x = start_x + sum(col_widths[:col])
            y = start_y + row*row_height
            bg_color = (240, 240, 240) if row % 2 == 0 else (220, 220, 220)
            pygame.draw.rect(screen, bg_color, (x, y, col_widths[col], row_height))
            pygame.draw.rect(screen, (0,0,0), (x, y, col_widths[col], row_height), 2)
            text_surface = font.render(val, True, (0,0,0))
            screen.blit(text_surface, (x + 5, y + 5))

    pygame.display.flip()
    clock.tick(FPS)

pygame.quit()
sys.exit()
