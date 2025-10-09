# -*- coding: utf-8 -*-
"""
TSP Solver com múltiplos veículos usando GA
- Depósito central
- Clusteriza cidades em 4 grupos
- Cada veículo percorre seu cluster
- Rotas coloridas
- Lado esquerdo da tela: gráfico moderno + tabela moderna
- Lado direito da tela: mapa das cidades
- 3 checkboxes ao lado da tabela para controlar restrições
"""

import itertools
import sys

import matplotlib.pyplot as plt
import numpy as np
import pygame
from matplotlib.backends.backend_agg import FigureCanvasAgg
from pygame.locals import *
from sklearn.cluster import KMeans

from benchmark_att48 import *
from draw_functions import draw_cities, draw_paths
from genetic_algorithm import (
    calculate_distance,
    calculate_fitness,
    generate_random_population,
    mutate,
    nearest_neighbor_heuristic,
    order_crossover,
    sort_population,
    tournament_selection,
)
from llm_integration import get_llmSolution

# ------------------------- CONSTANTES -------------------------
SCREEN_WIDTH, SCREEN_HEIGHT = 1200, 650
NODE_RADIUS = 10
FPS = 30

POPULATION_SIZE = 100
MUTATION_PROBABILITY = 0.5
NUM_VEHICLES = 4
VEHICLE_COLORS = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 165, 0)]
MARGIN = 50
PROHIBITED_PENALTY = 1e6
MAX_DISTANCE = 900

LEFT_PANEL_WIDTH = 500
RIGHT_PANEL_WIDTH = SCREEN_WIDTH - LEFT_PANEL_WIDTH
GRAPH_HEIGHT = 400
TABLE_HEIGHT = 200

# ------------------------- RESTRIÇÕES -------------------------
checkboxes = [
    {
        "rect": pygame.Rect(LEFT_PANEL_WIDTH + 20, GRAPH_HEIGHT + 20, 20, 20),
        "text": "Via proibida",
        "value": lambda: restricao_via_proibida,
        "set": lambda val: set_restricao("vias", val),
    },
    {
        "rect": pygame.Rect(LEFT_PANEL_WIDTH + 20, GRAPH_HEIGHT + 50, 20, 20),
        "text": "Cidade prioritária",
        "value": lambda: restricao_cidade_prioritaria,
        "set": lambda val: set_restricao("prioridade", val),
    },
    {
        "rect": pygame.Rect(LEFT_PANEL_WIDTH + 20, GRAPH_HEIGHT + 80, 20, 20),
        "text": "Parada para Abastecer",
        "value": lambda: restricao_abastecimento,
        "set": lambda val: set_restricao("max", val),
    },
]

restricao_via_proibida = False
restricao_cidade_prioritaria = False
restricao_abastecimento = False


def aplicar_cidade_prioritaria(route, prioridades):
    """
    Garante que a cidade prioritária seja a primeira a ser visitada após o depósito.
    """
    depot = route[0]
    rest = route[1:-1]  # exclui depósito do início/fim
    for cidade in prioridades:
        if cidade in rest:
            rest.remove(cidade)
            rest = [cidade] + rest  # coloca prioridade no início
    return [depot] + rest + [depot]


def set_cidadesPrioritarias():
    global cidades_prioritarias
    cidades_prioritarias = []
    if restricao_cidade_prioritaria:
        cidades_prioritarias = [(944, 389), (674, 342), (900, 174), (1120, 387)]
        cidades_prioritarias = [tuple(c) for c in cidades_prioritarias]


def set_viaProibida():
    global vias_proibidas
    vias_proibidas = []
    if restricao_via_proibida:
        vias_proibidas = [
            ((814, 344), (877, 291)),
            ((779, 252), (784, 221)),
            ((1013, 165), (1014, 119)),
            ((1100, 443), (1108, 519)),
        ]


def inserir_paradas(route, postos, alcance_maximo):
    """
    Adiciona postos obrigatórios na rota quando a distância acumulada ultrapassa o alcance.
    """
    nova_rota = [route[0]]  # começar no depósito
    distancia_acumulada = 0

    for i in range(len(route) - 1):
        p1, p2 = route[i], route[i + 1]
        d = calculate_distance(p1, p2, cities_locations, vias_proibidas)

        if d < PROHIBITED_PENALTY:
            distancia_acumulada += d

        if distancia_acumulada > alcance_maximo:
            # Encontrar posto mais próximo do ponto atual (p1)
            posto_proximo = min(postos, key=lambda p: calculate_distance(p1, p))
            nova_rota.append(posto_proximo)
            distancia_acumulada = calculate_distance(posto_proximo, p2)

        nova_rota.append(p2)

    return nova_rota


def set_PostosAbastecimento():
    global postos_abastecimento
    postos_abastecimento = []
    if restricao_abastecimento:
        postos_abastecimento = [(710, 150), (840, 370), (1010, 220), (1060, 430)]
        postos_abastecimento = [tuple(c) for c in postos_abastecimento]


def set_restricao(tipo, val):
    global restricao_via_proibida, restricao_cidade_prioritaria, restricao_abastecimento
    if tipo == "vias":
        restricao_via_proibida = val
    elif tipo == "prioridade":
        restricao_cidade_prioritaria = val
    elif tipo == "max":
        restricao_abastecimento = val
    reiniciar_GA()  # reinicia apenas populações e histórico


# ------------------------- PREPARAR CIDADES -------------------------
def prepare_cities():
    global cities_locations, vehicle_clusters, vehicle_populations, depot

    att_cities_locations = np.array(att_48_cities_locations)
    max_x = max(point[0] for point in att_cities_locations)
    max_y = max(point[1] for point in att_cities_locations)

    scale_x = (RIGHT_PANEL_WIDTH - MARGIN * 2 - NODE_RADIUS) / max_x
    scale_y = (SCREEN_HEIGHT - MARGIN * 2 - NODE_RADIUS) / max_y

    cities_locations = [
        (
            int(point[0] * scale_x + LEFT_PANEL_WIDTH + MARGIN),
            int(point[1] * scale_y + MARGIN),
        )
        for point in att_cities_locations
    ]

    depot_x = int(np.mean([x for x, y in cities_locations]))
    depot_y = int(np.mean([y for x, y in cities_locations]))
    depot = (depot_x, depot_y)

    cities_array = np.array(cities_locations)
    kmeans = KMeans(n_clusters=NUM_VEHICLES, random_state=42)
    kmeans.fit(cities_array)
    labels = kmeans.labels_

    vehicle_clusters = [[] for _ in range(NUM_VEHICLES)]
    for idx, label in enumerate(labels):
        vehicle_clusters[label].append(cities_locations[idx])
    for i in range(NUM_VEHICLES):
        vehicle_clusters[i] = [depot] + vehicle_clusters[i] + [depot]

    vehicle_populations = []
    for cluster in vehicle_clusters:
        population = []
        # Heurísticas iniciais
        for i in range(1, len(cluster) - 1):
            sol = nearest_neighbor_heuristic(cluster, i)
            if restricao_cidade_prioritaria:
                sol = aplicar_cidade_prioritaria(sol, cidades_prioritarias)
            population.append(sol)
        # População aleatória restante
        random_population = generate_random_population(
            cluster, POPULATION_SIZE - len(population)
        )
        if restricao_cidade_prioritaria:
            random_population = [
                aplicar_cidade_prioritaria(p, cidades_prioritarias)
                for p in random_population
            ]
        population.extend(random_population)
        vehicle_populations.append(population)


# ------------------------- PYGAME -------------------------
pygame.init()
screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
pygame.display.set_caption("TSP Solver com múltiplos veículos")
clock = pygame.time.Clock()


# ------------------------- FUNÇÃO PARA REINICIAR POPULAÇÃO -------------------------
def reiniciar_GA():
    global vehicle_best_fitness, vehicle_best_solutions, vehicle_last_change, generation_counter
    vehicle_best_fitness = [[] for _ in range(NUM_VEHICLES)]
    vehicle_best_solutions = [[] for _ in range(NUM_VEHICLES)]
    vehicle_last_change = [0] * NUM_VEHICLES
    generation_counter = itertools.count(start=1)
    set_viaProibida()
    set_cidadesPrioritarias()
    set_PostosAbastecimento()
    prepare_cities()


# ------------------------- LOOP PRINCIPAL -------------------------

reiniciar_GA()

running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.KEYDOWN and event.key == pygame.K_q:
            running = False
        elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            for cb in checkboxes:
                if cb["rect"].collidepoint(event.pos):
                    cb["set"](not cb["value"]())

    generation = next(generation_counter)
    screen.fill((255, 255, 255))

    # ----------------- DESENHAR CHECKBOXES -----------------
    font = pygame.font.SysFont("Arial", 18)
    for cb in checkboxes:
        pygame.draw.rect(screen, (0, 0, 0), cb["rect"], 2)
        if cb["value"]():
            pygame.draw.line(
                screen,
                (0, 0, 0),
                (cb["rect"].x, cb["rect"].y),
                (cb["rect"].x + 20, cb["rect"].y + 20),
                2,
            )
            pygame.draw.line(
                screen,
                (0, 0, 0),
                (cb["rect"].x + 20, cb["rect"].y),
                (cb["rect"].x, cb["rect"].y + 20),
                2,
            )
        screen.blit(
            font.render(cb["text"], True, (0, 0, 0)),
            (cb["rect"].x + 25, cb["rect"].y - 2),
        )

    # ----------------- PARA CADA VEÍCULO -----------------
    vehicle_info = []
    best_solution = None
    for v in range(NUM_VEHICLES):
        population = vehicle_populations[v]

        population_fitness = [
            calculate_fitness(
                ind, cities_locations, vias_proibidas, postos_abastecimento
            )
            for ind in population
        ]
        population, population_fitness = sort_population(population, population_fitness)

        best_fitness = calculate_fitness(
            population[0], cities_locations, vias_proibidas, postos_abastecimento
        )
        best_solution = population[0]

        if (
            len(vehicle_best_solutions[v]) == 0
            or vehicle_best_solutions[v][-1] != best_solution
        ):
            vehicle_last_change[v] = generation

        vehicle_best_fitness[v].append(best_fitness)
        vehicle_best_solutions[v].append(best_solution)
        num_cities = len(best_solution) - 2
        vehicle_info.append((v, best_fitness, num_cities, vehicle_last_change[v]))

        draw_paths(
            screen,
            best_solution,
            VEHICLE_COLORS[v],
            width=3,
            vias_proibidas=vias_proibidas if restricao_via_proibida else [],
            cities_locations=cities_locations,
            postos_abastecimento=postos_abastecimento,
        )

        draw_cities(
            screen,
            vehicle_clusters[v],
            VEHICLE_COLORS[v],
            NODE_RADIUS,
            depot,
            cidades_prioritarias,
            postos_abastecimento,
        )

        # Evolução da população
        new_population = [population[0]]
        while len(new_population) < POPULATION_SIZE:
            parent1 = tournament_selection(
                population, population_fitness, tournament_size=5
            )
            parent2 = tournament_selection(
                population, population_fitness, tournament_size=5
            )
            child1, child2 = order_crossover(parent1, parent2)
            child1 = mutate(child1, MUTATION_PROBABILITY)
            child2 = mutate(child2, MUTATION_PROBABILITY)

            if restricao_cidade_prioritaria:
                child1 = aplicar_cidade_prioritaria(child1, cidades_prioritarias)
                child2 = aplicar_cidade_prioritaria(child2, cidades_prioritarias)

            new_population.append(child1)
            if len(new_population) < POPULATION_SIZE:
                new_population.append(child2)
        vehicle_populations[v] = new_population

    # ----------------- DESENHAR GRÁFICO -----------------
    plt_fig, plt_ax = plt.subplots(figsize=(5, 4), dpi=100)
    plt_ax.set_facecolor("#f9f9f9")
    plt_ax.grid(True, linestyle="--", alpha=0.5)
    for v in range(NUM_VEHICLES):
        plt_ax.plot(
            list(range(len(vehicle_best_fitness[v]))),
            vehicle_best_fitness[v],
            color=np.array(VEHICLE_COLORS[v]) / 255,
            label=f"Veículo {v+1}",
            linewidth=3,
        )
    plt_ax.set_xlabel("Generation", fontsize=12)
    plt_ax.set_ylabel("Fitness (Distance)", fontsize=12)
    plt_ax.tick_params(axis="both", labelsize=10)
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
    font = pygame.font.SysFont("Arial", 20)
    start_x, start_y = 10, GRAPH_HEIGHT + 20
    col_widths = [150, 120, 150, 80]
    row_height = 35
    headers = ["Veículo", "Distância", "Cidades", "Geração"]
    veiculos = ["Veículo 1", "Veículo 2", "Veículo 3", "Veículo 4"]

    # Cabeçalho
    for col, header in enumerate(headers):
        x = start_x + sum(col_widths[:col])
        y = start_y
        pygame.draw.rect(screen, (200, 200, 200), (x, y, col_widths[col], row_height))
        pygame.draw.rect(screen, (0, 0, 0), (x, y, col_widths[col], row_height), 2)
        text_surface = font.render(header, True, (0, 0, 0))
        screen.blit(text_surface, (x + 5, y + 5))

    start_y += row_height

    # Linhas da tabela (veículos)
    for row, info in enumerate(vehicle_info):
        v, dist, num_cities, last_change = info
        values = [v, str(round(dist)), str(num_cities - 1), str(last_change)]
        for col, val in enumerate(values):
            x = start_x + sum(col_widths[:col])
            y = start_y + row * row_height
            bg_color = (240, 240, 240) if row % 2 == 0 else (220, 220, 220)
            pygame.draw.rect(screen, bg_color, (x, y, col_widths[col], row_height))
            pygame.draw.rect(screen, (0, 0, 0), (x, y, col_widths[col], row_height), 2)

            # Desenhar círculo colorido na primeira coluna
            if col == 0:
                text_surface = font.render(veiculos[val], True, (0, 0, 0))
                screen.blit(text_surface, (x + 40, y + 5))
                # Risco menor e centralizado na célula
                line_length = 20  # tamanho fixo igual à legenda do gráfico
                line_x_start = x + 5  # margem da célula
                line_x_end = line_x_start + line_length
                line_y = y + row_height // 2
                pygame.draw.line(
                    screen,
                    VEHICLE_COLORS[v],
                    (line_x_start, line_y),
                    (line_x_end, line_y),
                    4,
                )
            else:
                text_surface = font.render(val, True, (0, 0, 0))
                screen.blit(text_surface, (x + 5, y + 5))

    # ----------------- LINHA DE TOTALIZADORES -----------------
    total_dist = sum(info[1] for info in vehicle_info)
    total_cities = sum(info[2] - 1 for info in vehicle_info)  # subtrair depósito

    y = start_y + len(vehicle_info) * row_height
    pygame.draw.rect(
        screen, (200, 200, 200), (start_x, y, sum(col_widths), row_height)
    )  # fundo
    pygame.draw.rect(
        screen, (0, 0, 0), (start_x, y, sum(col_widths), row_height), 2
    )  # borda

    total_values = ["Total", str(round(total_dist)), str(total_cities), ""]
    for col, val in enumerate(total_values):
        x = start_x + sum(col_widths[:col])
        pygame.draw.rect(screen, (200, 200, 200), (x, y, col_widths[col], row_height))
        pygame.draw.rect(screen, (0, 0, 0), (x, y, col_widths[col], row_height), 2)
        text_surface = font.render(val, True, (0, 0, 0))
        screen.blit(text_surface, (x + 5, y + 5))

    pygame.display.flip()
    clock.tick(FPS)

# get_llmSolution(best_solution)
pygame.quit()
sys.exit()
