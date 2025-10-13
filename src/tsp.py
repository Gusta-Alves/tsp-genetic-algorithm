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
import random
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

_isllmintegrationEnabled = False

if _isllmintegrationEnabled:
    from llm_integration import get_llmSolution
    from ui import render_markdown_line, ScrollableMarkdownArea

# ------------------------- CONSTANTES -------------------------
SCREEN_WIDTH, SCREEN_HEIGHT, RESULTS_AREA_HEIGHT = 1200, 700, 300
NODE_RADIUS = 10
FPS = 30

POPULATION_SIZE = 100
MUTATION_PROBABILITY = 0.5
NUM_VEHICLES = 4
NUM_CITIES = 48
VEHICLE_COLORS = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 165, 0),(160, 32, 240)]
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
        "enabled": True,
    },
    {
        "rect": pygame.Rect(LEFT_PANEL_WIDTH + 20, GRAPH_HEIGHT + 50, 20, 20),
        "text": "Cidade prioritária",
        "value": lambda: restricao_cidade_prioritaria,
        "set": lambda val: set_restricao("prioridade", val),
        "enabled": True,
    },
    {
        "rect": pygame.Rect(LEFT_PANEL_WIDTH + 20, GRAPH_HEIGHT + 80, 20, 20),
        "text": "Parada para Abastecer",
        "value": lambda: restricao_abastecimento,
        "set": lambda val: set_restricao("max", val),
        "enabled": True,
    },
]

input_values = {
    "veiculos": "4",
    "cidades": "48",
    #"limite_dist": "1000"  # exemplo de valor inicial
    }

# Retângulos dos inputs
input_rects = {
    "veiculos": pygame.Rect(LEFT_PANEL_WIDTH + 125, GRAPH_HEIGHT + 162, 28, 28),
    "cidades": pygame.Rect(LEFT_PANEL_WIDTH + 125, GRAPH_HEIGHT + 202, 28, 28),
    #"limite_dist": pygame.Rect(LEFT_PANEL_WIDTH + 20, GRAPH_HEIGHT + 230, 80, 28)
}


restricao_via_proibida = False
restricao_cidade_prioritaria = False
restricao_abastecimento = False

# Variáveis para armazenar as soluções finais exibidas
final_displayed_solutions = [None] * NUM_VEHICLES
final_displayed_fitness = [None] * NUM_VEHICLES


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
        cidades_prioritarias = [(944, 420), (674, 369), (900, 186), (1120, 418)]
        cidades_prioritarias = [tuple(c) for c in cidades_prioritarias]


def set_viaProibida():
    global vias_proibidas
    vias_proibidas = []
    if restricao_via_proibida:
        vias_proibidas = [
            ((814, 371), (877, 314)),
            ((779, 271), (784, 237)),
            ((1013, 176), (1014, 126)),
            ((1100, 480), (1108, 562)),
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
        postos_abastecimento = [(710, 150), (840, 400), (1140, 180), (1040, 520)]
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
                    
    #cidades_ativas = att_48_cities_locations[:NUM_CITIES]
    cidades_ativas = random.sample(att_48_cities_locations, NUM_CITIES)
    att_cities_locations = np.array(cidades_ativas)
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
    kmeans = KMeans(n_clusters=NUM_VEHICLES, random_state=72)
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
    global vehicle_best_fitness, vehicle_best_solutions, vehicle_last_change, generation_counter, vehicle_info
    vehicle_best_fitness = [[] for _ in range(NUM_VEHICLES)]
    vehicle_best_solutions = [[] for _ in range(NUM_VEHICLES)]
    vehicle_last_change = [0] * NUM_VEHICLES    
    generation_counter = itertools.count(start=1)    
    vehicle_info = []
    set_viaProibida()
    set_cidadesPrioritarias()
    set_PostosAbastecimento()    
    prepare_cities()

# ------------------------- LOOP PRINCIPAL -------------------------

reiniciar_GA()
isEditing = False
btn_color = (210, 210, 210)  # cinza padrão
color_veiculos = (210, 210, 210)
color_cidades = (210, 210, 210)
btn_text = "EDITAR"
running = True
active_input = None

while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.KEYDOWN and event.key == pygame.K_q:
            running = False
        elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            for cb in checkboxes:
                if not cb["enabled"]:
                    continue  # ignora se estiver desabilitado
                if cb["rect"].collidepoint(event.pos):
                    cb["set"](not cb["value"]())
            if btn_reset.collidepoint(event.pos):  # botão esquerdo do mouse 
                NUM_VEHICLES = 4
                NUM_CITIES = 48
                input_values["veiculos"] = str(NUM_VEHICLES)
                input_values["cidades"] = str(NUM_CITIES)
                for cb in checkboxes:                                      
                    cb["enabled"] = True                          
                    restricao_abastecimento = False
                    restricao_cidade_prioritaria = False
                    restricao_via_proibida = False                        
                reiniciar_GA()

            if btn_editar.collidepoint(event.pos):  # botão esquerdo do mouse                
                isEditing = not isEditing
                if isEditing:
                    btn_color = (120, 200, 120)  # verde para confirmar
                    btn_text = "CONFIRMAR"                                        
                else:
                    btn_color = (210, 210, 210)  # cinza padrão3
                    btn_text = "EDITAR"     
                    color_veiculos = (210,210,210)
                    color_cidades = (210,210,210)  
                    if input_values["veiculos"] == "":
                        input_values["veiculos"] = str(NUM_VEHICLES)
                    if input_values["cidades"] == "":
                        input_values["cidades"] = str(len(att_48_cities_locations))
                    NUM_CITIES = int(input_values["cidades"])
                    NUM_VEHICLES = int(input_values["veiculos"]) 

                    if NUM_VEHICLES*2 > NUM_CITIES:
                        NUM_CITIES = NUM_VEHICLES*2
                        input_values["cidades"] = str(NUM_CITIES)                
                for cb in checkboxes:
                    if NUM_VEHICLES == 4 and NUM_CITIES == 48:                        
                        cb["enabled"] = True      
                        print("Checkboxes habilitados")              
                    else:
                        restricao_abastecimento = False
                        restricao_cidade_prioritaria = False
                        restricao_via_proibida = False
                        cb["set"](False)
                        cb["enabled"] = False
                        print("Checkboxes desabilitados")
                            
                reiniciar_GA()
            # ----------------- CLIQUE NOS INPUTS -----------------
            mouse_pos = pygame.mouse.get_pos()
            mouse_pressed = pygame.mouse.get_pressed()

            if isEditing and mouse_pressed[0]:
                for key, rect in input_rects.items():
                    if rect.collidepoint(mouse_pos):
                        active_input = key
                        break
                else:
                    active_input = None

                if active_input == "veiculos":
                    input_values["veiculos"] = ""
                    color_veiculos = (255,255,255)
                    color_cidades = (210,210,210)
                elif active_input == "cidades":
                    input_values["cidades"] = ""
                    color_veiculos = (210,210,210)
                    color_cidades = (255,255,255)    

        elif event.type == pygame.KEYDOWN and isEditing and active_input:
            if event.key in (pygame.K_RETURN, pygame.K_KP_ENTER):
                active_input = None
            elif event.key == pygame.K_BACKSPACE:
                input_values[active_input] = input_values[active_input][:-1]
            elif event.unicode.isdigit() and len(input_values[active_input]) < 4:
                new_text = input_values[active_input] + event.unicode                
                try:
                    val = int(new_text)
                    if active_input == "veiculos" and 1 <= val <= 5:
                        input_values["veiculos"] = str(val)
                    elif active_input == "cidades" and 1 <= val <= 48:                       
                        input_values["cidades"] = str(val)
                except ValueError:
                    pass
            
    generation = next(generation_counter)
    screen.fill((255, 255, 255))

    # ----------------- DESENHAR CHECKBOXES -----------------
    font = pygame.font.SysFont("Arial", 18)
    for cb in checkboxes:
        # define cores conforme o estado enabled
        border_color = (0, 0, 0) if cb["enabled"] else (150, 150, 150)
        text_color = (0, 0, 0) if cb["enabled"] else (150, 150, 150)
        check_color = (0, 0, 0) if cb["enabled"] else (120, 120, 120)
        fill_color = (255, 255, 255) if cb["enabled"] else (220, 220, 220)

        # desenha o retângulo
        pygame.draw.rect(screen, fill_color, cb["rect"])
        pygame.draw.rect(screen, border_color, cb["rect"], 2)

        # desenha o X se marcado
        if cb["value"]():
            pygame.draw.line(
                screen,
                check_color,
                (cb["rect"].x, cb["rect"].y),
                (cb["rect"].x + 20, cb["rect"].y + 20),
                2,
            )
            pygame.draw.line(
                screen,
                check_color,
                (cb["rect"].x + 20, cb["rect"].y),
                (cb["rect"].x, cb["rect"].y + 20),
                2,
            )

        # desenha o texto
        screen.blit(
            font.render(cb["text"], True, text_color),
            (cb["rect"].x + 25, cb["rect"].y - 2),
        )

    # ----------------- BOTAO DE PAUSE -----------------        
    btn_editar = pygame.Rect(LEFT_PANEL_WIDTH + 20, GRAPH_HEIGHT + 130, 100, 30)  # Retângulo do botão

    # Desenhar botão
    pygame.draw.rect(screen, btn_color, btn_editar, border_radius=4)    # fundo
    pygame.draw.rect(screen, (0,0,0), btn_editar, 2, border_radius=4)         # borda

    # Texto centralizado
    text_surface = font.render(btn_text, True, (0,0,0))
    text_x = btn_editar.x + (btn_editar.width - text_surface.get_width()) // 2
    text_y = btn_editar.y + (btn_editar.height - text_surface.get_height()) // 2
    screen.blit(text_surface, (text_x, text_y))

    # ----------------- BOTAO DE RESET -----------------        
    btn_reset = pygame.Rect(LEFT_PANEL_WIDTH + 123, GRAPH_HEIGHT + 130, 30, 30)

    # Desenhar botão
    pygame.draw.rect(screen, btn_color, btn_reset, border_radius=4)    # fundo
    pygame.draw.rect(screen, (0,0,0), btn_reset, 2, border_radius=4)         # borda

    # Texto centralizado
    font2 = pygame.font.SysFont("Segoe UI Symbol", 20)  # fonte com ícones Unicode
    text_surface = font2.render("↻", True, (0,0,0))
    text_x = btn_reset.x + (btn_reset.width - text_surface.get_width()) // 2
    text_y = btn_reset.y + (btn_reset.height - text_surface.get_height()) // 2
    screen.blit(text_surface, (text_x, text_y))
      
    # ----------------- DESENHAR INPUTS -----------------            
    # Rótulos dos campos
    screen.blit(font.render("Nº de Veículos:", True, (0, 0, 0)), (LEFT_PANEL_WIDTH + 20, GRAPH_HEIGHT + 170))
    screen.blit(font.render("Nº de Cidades:", True, (0, 0, 0)), (LEFT_PANEL_WIDTH + 20, GRAPH_HEIGHT + 210))

    for key, rect in input_rects.items():
        # Desenha retângulo do input
        if key == "veiculos":
            pygame.draw.rect(screen, color_veiculos, rect)
        else:
            pygame.draw.rect(screen, color_cidades, rect)
        
        pygame.draw.rect(screen, (0,0,0), rect, 2)

        # Texto centralizado
        text_surface = font.render(input_values[key], True, (0,0,0))
        text_x = rect.x + 5
        text_y = rect.y + (rect.height - text_surface.get_height()) // 2
        screen.blit(text_surface, (text_x, text_y))

    if not isEditing:
        # ----------------- PARA CADA VEÍCULO -----------------
        vehicle_info = []
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

            final_displayed_solutions[v] = best_solution
            final_displayed_fitness[v] = best_fitness

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
    veiculos = ["Veículo 1", "Veículo 2", "Veículo 3", "Veículo 4","Veículo 5"]

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
        values = [v, str(round(dist)), str(num_cities), str(last_change)]
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
    total_cities = sum(info[2] for info in vehicle_info)  # subtrair depósito

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

# ------------------------- EXIBIR RESULTADOS FINAIS -------------------------
# end_time = time.time()
# print(f"Benchmark concluído após {MAX_GENERATIONS} gerações.")
# print(f"Tempo total: {end_time - start_time:.2f} segundos")
# print(f"Fitness final (distância total): {last_total_dist:.2f}")

# Prepara os dados para o LLM
solutions_data = []
for v in range(NUM_VEHICLES):
    best_solution = final_displayed_solutions[v]
    best_distance = final_displayed_fitness[v]
    # route = [city.name for city in best_solution]
    
    solutions_data.append({
        "veiculo": v + 1,
        "distancia": best_distance,
        # "rota": route
    })

if _isllmintegrationEnabled:
    # Obtém a resposta formatada do LLM
    llm_result = get_llmSolution(solutions_data)

    # Captura a tela atual antes de redimensionar
    old_screen = screen.copy()

    # Redimensiona a janela para incluir a área de resultados
    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT + RESULTS_AREA_HEIGHT))
    pygame.display.set_caption("TSP Solver - Resultados Finais")

    # Restaura o conteúdo anterior na parte superior
    screen.blit(old_screen, (0, 0))

    # Cria área de markdown com scroll na parte inferior
    results_y = SCREEN_HEIGHT
    markdown_area = ScrollableMarkdownArea(0, results_y, SCREEN_WIDTH, RESULTS_AREA_HEIGHT, screen)
    markdown_area.render_markdown(llm_result)

    pygame.display.flip()

    # Aguarda o usuário fechar a janela
    waiting = True
    while waiting:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                waiting = False
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_q:
                waiting = False
            elif event.type == pygame.MOUSEBUTTONDOWN:
                markdown_area.handle_scroll(event)
                markdown_area.render_markdown(llm_result)
                pygame.display.flip()
        clock.tick(30)

pygame.quit()
sys.exit()
