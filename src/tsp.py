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
import time

import matplotlib.pyplot as plt
import numpy as np
import pygame
from matplotlib.backends.backend_agg import FigureCanvasAgg
from pygame.locals import *
from sklearn.cluster import KMeans

from city import City
from benchmark_att48 import *
from constants import (
    CHECKBOX_OFFSET_X,
    CHECKBOX_OFFSET_Y_START,
    CHECKBOX_SPACING,
    FPS,
    GRAPH_HEIGHT,
    LEFT_PANEL_WIDTH,
    MARGIN,
    MUTATION_PROBABILITY,
    NODE_RADIUS,
    NUM_VEHICLES,
    POPULATION_SIZE,
    PROHIBITED_PENALTY,
    RIGHT_PANEL_WIDTH,
    SCREEN_HEIGHT,
    SCREEN_WIDTH,
    VEHICLE_COLORS,
    VEHICLE_DISTANCE_LIMITS,
    MAX_DISTANCE_LIMIT,
    MIN_DISTANCE_LIMIT,
    DISTANCE_LIMIT_STEP,
    RESULTS_AREA_HEIGHT,
)

# Armazena os limites de distância iniciais para permitir um reset completo
INITIAL_VEHICLE_DISTANCE_LIMITS = list(VEHICLE_DISTANCE_LIMITS)

# Benchmark configuration
MAX_GENERATIONS = None
last_total_dist = 0
_isllmintegrationEnabled = False

from distance_matrix import DistanceMatrix
from draw_functions import draw_cities, draw_paths
from genetic_algorithm import (
    calculate_distance,
    create_distance_matrix,
    calculate_fitness,
    generate_random_population,
    mutate,
    nearest_neighbor_heuristic,
    order_crossover,
    set_distance_matrix,
    sort_population,
    tournament_selection,
)
from ui import render_markdown_line, ScrollableMarkdownArea

if _isllmintegrationEnabled:
    from llm_integration import get_llmSolution

# ------------------------- NOMES DE CIDADES -------------------------
CITY_NAMES = [
    "Chicago", "New York", "Los Angeles", "Toronto", "Vancouver", "Mexico City",
    "London", "Paris", "Berlin", "Madrid", "Rome", "Moscow", "Istanbul",
    "Tokyo", "Beijing", "Shanghai", "Seoul", "Sydney", "Melbourne",
    "Cairo", "Johannesburg", "Lagos", "Nairobi",
    "Rio de Janeiro", "Sao Paulo", "Buenos Aires", "Lima", "Bogota",
    "Mumbai", "Delhi", "Bangkok", "Singapore", "Jakarta", "Kuala Lumpur",
    "San Francisco", "Boston", "Miami", "Houston", "Dallas", "Atlanta",
    "Seattle", "Denver", "Phoenix", "Philadelphia", "Washington",
    "Montreal", "Calgary", "Ottawa", "Edmonton"
]
# Garante que a lista de nomes seja embaralhada para cada execução
random.shuffle(CITY_NAMES)

# ------------------------- RESTRIÇÕES -------------------------
checkboxes = [
    {
        "rect": pygame.Rect(
            LEFT_PANEL_WIDTH + 50 + CHECKBOX_OFFSET_X, CHECKBOX_OFFSET_Y_START, 20, 20
        ),
        "text": "Via proibida",
        "value": lambda: restricao_via_proibida,
        "set": lambda val: set_restricao("vias", val),
    },
    {
        "rect": pygame.Rect(
            LEFT_PANEL_WIDTH + 50 + CHECKBOX_OFFSET_X,
            CHECKBOX_OFFSET_Y_START + CHECKBOX_SPACING,
            20,
            20,
        ),
        "text": "Cidade prioritária",
        "value": lambda: restricao_cidade_prioritaria,
        "set": lambda val: set_restricao("prioridade", val),
    },
    {
        "rect": pygame.Rect(
            LEFT_PANEL_WIDTH + 50 + CHECKBOX_OFFSET_X,
            CHECKBOX_OFFSET_Y_START + 2 * CHECKBOX_SPACING,
            20,
            20,
        ),
        "text": "Abastecimento (>=900)",
        "value": lambda: restricao_abastecimento,
        "set": lambda val: set_restricao("max", val),
    },
]

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
        raw_coords = [(944, 389), (674, 342), (900, 174), (1120, 387)]
        # Tenta encontrar cidades existentes nessas coordenadas, senão cria novas
        cidades_prioritarias = []
        for coords in raw_coords:
            found = next((city for city in cities_locations if city.get_coords() == coords), None)
            if found:
                cidades_prioritarias.append(found)


def set_viaProibida():
    global vias_proibidas
    vias_proibidas = []
    if restricao_via_proibida:
        raw_edges = [((814, 344),(877, 291)),
                    ((779, 252),(784, 221)),
                    ((1013, 165),(1014, 119)),
                    ((1100, 443),(1108, 519))]
        vias_proibidas = []
        for start_coords, end_coords in raw_edges:
            start_city = next((c for c in cities_locations if c.get_coords() == start_coords), None)
            end_city = next((c for c in cities_locations if c.get_coords() == end_coords), None)
            if start_city and end_city:
                vias_proibidas.append((start_city, end_city))


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
        raw_postos = [(710, 150), (840, 370), (1010, 220), (1060, 430)]
        postos_abastecimento = [City(name=f"Posto{i+1}", x=p[0], y=p[1]) for i, p in enumerate(raw_postos)]


def set_restricao(tipo, val):
    global restricao_via_proibida, restricao_cidade_prioritaria, restricao_abastecimento, VEHICLE_DISTANCE_LIMITS
    if tipo == "vias":
        restricao_via_proibida = val
    elif tipo == "prioridade":
        restricao_cidade_prioritaria = val
    elif tipo == "max":
        restricao_abastecimento = val

    # ORDEM: Se uma restrição for alterada, todos os limites de distância devem retornar aos valores iniciais.
    print("Restrição alterada. Resetando limites de distância para os valores iniciais.")
    VEHICLE_DISTANCE_LIMITS[:] = INITIAL_VEHICLE_DISTANCE_LIMITS

    reiniciar_GA()  # reinicia apenas populações e histórico


def create_initial_population_for_vehicle(cluster):
    """Cria uma população inicial para um único cluster de veículo."""
    population = []
    # Heurísticas iniciais
    # Limita o número de heurísticas para não demorar muito se o cluster for grande
    num_heuristic_seeds = min(len(cluster) - 2, 20)
    if num_heuristic_seeds > 0:
        # Usa uma amostra de cidades como semente para a heurística
        start_indices = random.sample(range(1, len(cluster) - 1), num_heuristic_seeds)
        for i in start_indices:
            sol = nearest_neighbor_heuristic(cluster, i)
            if restricao_cidade_prioritaria:
                sol = aplicar_cidade_prioritaria(sol, cidades_prioritarias)
            population.append(sol)

    # População aleatória restante
    remaining_size = POPULATION_SIZE - len(population)
    if remaining_size > 0:
        random_population = generate_random_population(cluster, remaining_size)
        if restricao_cidade_prioritaria:
            random_population = [aplicar_cidade_prioritaria(p, cidades_prioritarias) for p in random_population]
        population.extend(random_population)

    return population


def _estimate_cluster_distance(cluster, vehicle_idx, dist_matrix, city_to_idx):
    """Estima a distância de uma rota para um cluster usando heurística."""
    # Garante que o cluster tenha apenas cidades únicas, exceto o depósito
    unique_cities = []
    for city in cluster:
        if city not in unique_cities:
            unique_cities.append(city)

    if len(unique_cities) <= 1:  # Apenas depósito ou vazio
        return 0

    heuristic_route = nearest_neighbor_heuristic(unique_cities, 1)
    dist = calculate_fitness(
        heuristic_route,
        dist_matrix,
        city_to_idx,
        VEHICLE_DISTANCE_LIMITS[vehicle_idx],
        vias_proibidas,
        postos_abastecimento,
    )
    return dist


def rebalance_clusters(vehicle_clusters, distance_limits, depot, dist_matrix, city_to_idx, pull_threshold=0.9) -> bool:
    """
    Rebalanceia os clusters para tentar garantir que as rotas iniciais
    sejam factíveis dentro dos limites de distância dos veículos.
    Retorna True se bem-sucedido, False se alguma cidade não pôde ser alocada.
    """
    print("Iniciando rebalanceamento de clusters...")
    MAX_REBALANCE_ITERATIONS = 10
    unassigned_cities = []
    made_a_change = True
    iterations = 0

    while made_a_change and iterations < MAX_REBALANCE_ITERATIONS:
        made_a_change = False
        iterations += 1

        # Estima a carga atual de cada veículo
        estimated_distances = [_estimate_cluster_distance(c, i, dist_matrix, city_to_idx) for i, c in enumerate(vehicle_clusters)]

        # Encontra o veículo mais sobrecarregado
        overload = [(est - limit, i) for i, (est, limit) in enumerate(zip(estimated_distances, distance_limits))]
        overload.sort(key=lambda x: x[0], reverse=True)

        # 1. Lógica de "Push": Veículo sobrecarregado tenta se livrar de uma cidade
        if overload[0][0] > 0:
            overloaded_idx = overload[0][1]
            overloaded_cluster = vehicle_clusters[overloaded_idx]

            cities_only = [c for c in overloaded_cluster if c != depot]
            if cities_only:
                city_to_move = max(cities_only, key=lambda c: dist_matrix.get_distance(depot, c))
                slack = [
                    (limit - est, i) for i, (est, limit) in enumerate(zip(estimated_distances, distance_limits)) 
                    if i != overloaded_idx and est <= limit
                ]
                slack.sort(key=lambda x: x[0], reverse=True)
                
                moved = False
                for _, target_idx in slack:
                    temp_cluster = vehicle_clusters[target_idx] + [city_to_move]
                    new_estimated_dist = _estimate_cluster_distance(temp_cluster, target_idx, dist_matrix, city_to_idx)
                    if new_estimated_dist <= distance_limits[target_idx]:
                        # Garante que a cidade seja removida de qualquer outro cluster antes de ser adicionada
                        # Garante que a cidade seja removida de qualquer OUTRO cluster onde possa existir
                        for other_cluster in vehicle_clusters:
                            if city_to_move in other_cluster and other_cluster is not vehicle_clusters[overloaded_idx]:
                                other_cluster.remove(city_to_move)
    
                        vehicle_clusters[overloaded_idx].remove(city_to_move)
                        vehicle_clusters[target_idx].insert(-1, city_to_move)
                        print(f"Rebalance (Push): Movendo '{city_to_move.name}' do Veículo {overloaded_idx+1} para o Veículo {target_idx+1}")
                        made_a_change = True
                        moved = True
                        break
                
                if not moved:
                    print(f"Rebalance (Push) falhou. Esvaziando Veículo {overloaded_idx+1}.")
                    cities_to_unassign = [c for c in vehicle_clusters[overloaded_idx] if c != depot]
                    unassigned_cities.extend(cities_to_unassign)
                    vehicle_clusters[overloaded_idx] = [depot, depot]
                    # Remove as cidades do cluster original para que não sejam processadas novamente
                    for city_to_remove in cities_to_unassign:
                        overloaded_cluster.remove(city_to_remove)
                    made_a_change = True

        # 2. Lógica de "Pull": Veículo ocioso tenta pegar trabalho de um sobrecarregado
        underload = [(est / limit if limit > 0 else float('inf'), i) for i, (est, limit) in enumerate(zip(estimated_distances, distance_limits))]
        underload.sort(key=lambda x: x[0])

        # Se o veículo mais ocioso está abaixo do threshold e há outro veículo mais carregado que ele.
        # A lógica de "pull" agora tenta balancear mesmo que ninguém esteja sobrecarregado.
        if not made_a_change and underload and len(overload) > 1 and underload[0][0] < pull_threshold:
            pulling_vehicle_idx = underload[0][1]
            # Pega o veículo mais carregado do sistema como alvo para puxar uma cidade
            target_vehicle_idx = overload[0][1] 
            target_cities = [c for c in vehicle_clusters[target_vehicle_idx] if c != depot]

            if target_cities:
                pulling_vehicle_cities = [c for c in vehicle_clusters[pulling_vehicle_idx] if c != depot]
                pulling_vehicle_center = np.mean([c.get_coords() for c in pulling_vehicle_cities], axis=0) if pulling_vehicle_cities else depot.get_coords()
                city_to_pull = min(target_cities, key=lambda c: np.linalg.norm(np.array(c.get_coords()) - pulling_vehicle_center))

                temp_pull_cluster = vehicle_clusters[pulling_vehicle_idx] + [city_to_pull]
                new_pull_dist = _estimate_cluster_distance(temp_pull_cluster, pulling_vehicle_idx, dist_matrix, city_to_idx)

                if new_pull_dist <= distance_limits[pulling_vehicle_idx]:
                    # CORREÇÃO: Remover a cidade do cluster de origem antes de adicioná-la ao novo.
                    vehicle_clusters[target_vehicle_idx].remove(city_to_pull) 
                    vehicle_clusters[pulling_vehicle_idx].insert(-1, city_to_pull)
                    print(f"Rebalance (Pull): Veículo {pulling_vehicle_idx+1} puxou '{city_to_pull.name}' do Veículo {target_vehicle_idx+1}")
                    made_a_change = True

    # 3. Tenta realocar cidades que ficaram sem veículo
    if unassigned_cities:
        print(f"Tentando realocar {len(unassigned_cities)} cidades não alocadas.")
        remaining_unassigned = []
        for city in unassigned_cities:
            moved = False
            estimated_distances = [_estimate_cluster_distance(c, i, dist_matrix, city_to_idx) for i, c in enumerate(vehicle_clusters)]
            slack = [(limit - est, i) for i, (est, limit) in enumerate(zip(estimated_distances, distance_limits))]
            slack.sort(key=lambda x: x[0], reverse=True)

            for _, target_idx in slack:
                temp_cluster = vehicle_clusters[target_idx] + [city]
                new_estimated_dist = _estimate_cluster_distance(temp_cluster, target_idx, dist_matrix, city_to_idx)
                
                if new_estimated_dist <= distance_limits[target_idx]:
                    vehicle_clusters[target_idx].insert(-1, city)
                    print(f"Rebalance (Re-assign): Movendo cidade órfã '{city.name}' para o Veículo {target_idx + 1}")
                    moved = True
                    break
            if not moved:
                remaining_unassigned.append(city)
        if remaining_unassigned:
            print(f"AVISO: {len(remaining_unassigned)} cidades não puderam ser alocadas respeitando os limites.")
            return False

    return True


# ------------------------- PREPARAR CIDADES -------------------------
def prepare_cities():
    global cities_locations, vehicle_clusters, vehicle_populations, depot, distance_matrix, city_to_index

    att_cities_locations = np.array(att_48_cities_locations)
    max_x = max(point[0] for point in att_cities_locations)
    max_y = max(point[1] for point in att_cities_locations)

    scale_x = (RIGHT_PANEL_WIDTH - MARGIN * 2 - NODE_RADIUS) / max_x
    scale_y = (SCREEN_HEIGHT - MARGIN * 2 - NODE_RADIUS) / max_y

    cities_locations = [
        City(
            name=CITY_NAMES[i] if i < len(CITY_NAMES) else f"C{i+1}",
            x=int(point[0] * scale_x + LEFT_PANEL_WIDTH + MARGIN),
            y=int(point[1] * scale_y + MARGIN),
        )
        for i, point in enumerate(att_cities_locations)
    ]

    # Converte para array NumPy para cálculos eficientes
    cities_coords_array = np.array([city.get_coords() for city in cities_locations])
    # Calcula o centroide (ponto médio) de todas as cidades para definir o depósito.
    depot_coords = tuple(np.mean(cities_coords_array, axis=0).astype(int))
    depot = City(name="Depot", x=depot_coords[0], y=depot_coords[1])

    # Cria a matriz de distância para otimização (agora que o depot existe)
    all_cities_for_matrix = [depot] + cities_locations

    # Inicializar matriz de distância otimizada usando all_cities_for_matrix
    distance_matrix = DistanceMatrix(all_cities_for_matrix)
    set_distance_matrix(distance_matrix)
    city_to_index = distance_matrix.city_to_index

    # cities_array = np.array(cities_locations)
    kmeans = KMeans(n_clusters=NUM_VEHICLES, random_state=72)
    kmeans.fit(cities_coords_array)
    labels = kmeans.labels_

    vehicle_clusters = [[] for _ in range(NUM_VEHICLES)]
    for idx, label in enumerate(labels):
        vehicle_clusters[label].append(cities_locations[idx])
    for i in range(NUM_VEHICLES):
        vehicle_clusters[i] = [depot] + vehicle_clusters[i] + [depot]

    vehicle_populations = []
    for cluster in vehicle_clusters:
        vehicle_populations.append(create_initial_population_for_vehicle(cluster))


# ------------------------- PYGAME -------------------------
pygame.init()
screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
pygame.display.set_caption("TSP Solver com múltiplos veículos")
clock = pygame.time.Clock()

def show_popup_message(screen, message, reset_to_initial=False) -> bool:
    """
    Exibe uma mensagem de popup com um botão 'OK'.
    Retorna True se o botão foi clicado, indicando a necessidade de reset.
    """
    global VEHICLE_DISTANCE_LIMITS

    font_title = pygame.font.SysFont("Arial", 22, bold=True)
    font_button = pygame.font.SysFont("Arial", 20)

    # --- Mensagem ---
    text_surface = font_title.render(message, True, (255, 255, 255))

    # --- Botão OK ---
    button_text_surface = font_button.render("OK", True, (0, 0, 0))
    button_width, button_height = button_text_surface.get_width() + 40, button_text_surface.get_height() + 10

    # --- Cálculo do tamanho do Popup ---
    rect_width = max(text_surface.get_width() + 40, button_width + 40)
    rect_height = text_surface.get_height() + button_height + 40  # Espaço para texto, botão e padding

    # --- Posições ---
    popup_rect = pygame.Rect((SCREEN_WIDTH - rect_width) // 2, (SCREEN_HEIGHT - rect_height) // 2, rect_width, rect_height)
    text_rect = text_surface.get_rect(center=(popup_rect.centerx, popup_rect.top + 30))
    button_rect = pygame.Rect(popup_rect.centerx - button_width // 2, popup_rect.bottom - button_height - 15, button_width, button_height)
    button_text_rect = button_text_surface.get_rect(center=button_rect.center)

    # --- Loop de espera pelo clique ---
    waiting_for_click = True
    while waiting_for_click:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                if button_rect.collidepoint(event.pos):
                    if reset_to_initial:
                        return True  # Sinaliza para reiniciar

        # Desenha o popup
        pygame.draw.rect(screen, (200, 0, 0), popup_rect, border_radius=10)  # Fundo vermelho
        pygame.draw.rect(screen, (255, 255, 255), popup_rect, width=2, border_radius=10)  # Borda branca
        screen.blit(text_surface, text_rect)

        # Desenha o botão
        pygame.draw.rect(screen, (220, 220, 220), button_rect, border_radius=5)
        screen.blit(button_text_surface, button_text_rect)

        pygame.display.flip()
    return False

# ------------------------- FUNÇÃO PARA REINICIAR POPULAÇÃO -------------------------
def reiniciar_GA():
    global vehicle_best_fitness, vehicle_best_solutions, vehicle_last_change, generation_counter, distance_matrix, city_to_index
    vehicle_best_fitness = [[] for _ in range(NUM_VEHICLES)]
    vehicle_best_solutions = [[] for _ in range(NUM_VEHICLES)]
    vehicle_last_change = [0] * NUM_VEHICLES
    generation_counter = itertools.count(start=1)
    set_viaProibida()
    set_cidadesPrioritarias()
    set_PostosAbastecimento()
    distance_matrix = None
    city_to_index = None
    prepare_cities()
    # Rebalanceia os clusters com base nos limites de distância
    global vehicle_clusters
    if not rebalance_clusters(vehicle_clusters, VEHICLE_DISTANCE_LIMITS, depot, distance_matrix, city_to_index):        
        if show_popup_message(screen, "Não foi possível alocar todas as cidades!", reset_to_initial=True):
            print("Resetando todos os limites para os valores iniciais.")
            VEHICLE_DISTANCE_LIMITS[:] = INITIAL_VEHICLE_DISTANCE_LIMITS
            return True # Sinaliza para o loop principal que um reset completo é necessário


def rebalance_and_reset_all(is_initial_setup=False) -> bool:
    """
    Rebalanceia os clusters com base nos limites atuais e reinicia as populações
    e o histórico de todos os veículos.
    Retorna True se bem-sucedido, False se falhar.
    """
    global vehicle_clusters, vehicle_populations, vehicle_best_fitness, vehicle_best_solutions, vehicle_last_change, generation_counter

    # Faz uma cópia profunda para poder reverter se o rebalanceamento falhar
    original_clusters = [list(c) for c in vehicle_clusters]
    
    print("Rebalanceamento global acionado por mudança de limite.")
    
    # 1. Rebalanceia os clusters com os novos limites
    rebalance_successful = rebalance_clusters(vehicle_clusters, VEHICLE_DISTANCE_LIMITS, depot, distance_matrix, city_to_index)

    # 2. Cria novas populações para os clusters rebalanceados
    if not rebalance_successful:
        print("Rebalanceamento falhou. Revertendo clusters.")
        vehicle_clusters[:] = original_clusters # Restaura o estado anterior
        return False # Indica falha para o chamador

    vehicle_populations = [create_initial_population_for_vehicle(c) for c in vehicle_clusters]
    
    # 3. Reseta o histórico de otimização de todos os veículos
    vehicle_best_fitness = [[] for _ in range(NUM_VEHICLES)]
    vehicle_best_solutions = [[] for _ in range(NUM_VEHICLES)]
    vehicle_last_change = [0] * NUM_VEHICLES
    generation_counter = itertools.count(start=1)
    return True


# ------------------------- LOOP PRINCIPAL -------------------------

reiniciar_GA()

spinner_buttons = [] # Armazenará os rects dos botões +/-
start_time = time.time()
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
            # Lógica para os botões de ajuste de limite
            for button in spinner_buttons:
                if button['rect'].collidepoint(event.pos):
                    original_limits = list(VEHICLE_DISTANCE_LIMITS) # Backup
                    v_idx = button['vehicle_idx']
                    if button['type'] == '+':
                        new_limit = min(MAX_DISTANCE_LIMIT, original_limits[v_idx] + DISTANCE_LIMIT_STEP)
                    else: # type == '-'
                        new_limit = max(MIN_DISTANCE_LIMIT, original_limits[v_idx] - DISTANCE_LIMIT_STEP)
                    VEHICLE_DISTANCE_LIMITS[v_idx] = new_limit
                    if not rebalance_and_reset_all():
                        VEHICLE_DISTANCE_LIMITS[:] = original_limits # Reverte a mudança                        
                        if show_popup_message(screen, "Não foi possível alocar todas as cidades!", reset_to_initial=True):
                            print("Resetando todos os limites para os valores iniciais.")
                            VEHICLE_DISTANCE_LIMITS[:] = INITIAL_VEHICLE_DISTANCE_LIMITS
                            reiniciar_GA()
                    break # Evita processar outros botões no mesmo clique

    generation = next(generation_counter)
    # Stop after MAX_GENERATIONS
    if MAX_GENERATIONS is not None:
        if generation > MAX_GENERATIONS:
            running = False
            break

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
    for v in range(NUM_VEHICLES):
        population = vehicle_populations[v]

        population_fitness = [
            calculate_fitness(
                ind,
                distance_matrix,
                city_to_index,
                VEHICLE_DISTANCE_LIMITS[v],
                vias_proibidas,
                postos_abastecimento,
            )
            for ind in population
        ]
        population, population_fitness = sort_population(population, population_fitness)

        best_fitness = calculate_fitness(
            population[0],
            distance_matrix,
            city_to_index,
            VEHICLE_DISTANCE_LIMITS[v],
            vias_proibidas,
            postos_abastecimento,
        )
        best_solution = population[0]
        
        # ORDEM: Se a distância for maior que o limite, zera a rota.
        if best_fitness > VEHICLE_DISTANCE_LIMITS[v] and len(best_solution) > 2:
            # Imprime no console que o rebalanceamento falhou para este veículo.
            print(f"Veículo {v+1}: Não foi possível rebalancear. Rota ótima ({best_fitness:.2f}) excede o limite ({VEHICLE_DISTANCE_LIMITS[v]}). Zerando rota.")
            # Zera a rota para exibição
            best_solution = [depot, depot]
            best_fitness = 0.0
            # Esvazia o cluster para forçar o rebalanceamento das cidades em ciclos futuros
            vehicle_clusters[v] = [depot, depot]

        final_displayed_solutions[v] = best_solution
        final_displayed_fitness[v] = best_fitness

        if (
            len(vehicle_best_solutions[v]) == 0
            or vehicle_best_solutions[v][-1] != best_solution
        ):
            vehicle_last_change[v] = generation

        vehicle_best_fitness[v].append(best_fitness)
        vehicle_best_solutions[v].append(best_solution)
        num_cities = len(set(best_solution) - {depot})
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
    col_widths = [150, 120, 100, 100, 80]
    row_height = 35
    headers = ["Veículo", "Distância", "Limite", "Cidades", "Geração"]
    veiculos = ["Veículo 1", "Veículo 2", "Veículo 3", "Veículo 4"]
    spinner_buttons.clear() # Limpa os botões da iteração anterior

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
        v_idx, dist, num_cities, last_change = info
        limit = VEHICLE_DISTANCE_LIMITS[v_idx]
        values = [v_idx, str(round(dist)), str(limit), str(num_cities), str(last_change)]
        for col, val in enumerate(values):
            x = start_x + sum(col_widths[:col])
            y = start_y + row * row_height
            bg_color = (240, 240, 240) if row % 2 == 0 else (220, 220, 220)
            pygame.draw.rect(screen, bg_color, (x, y, col_widths[col], row_height))
            pygame.draw.rect(screen, (0, 0, 0), (x, y, col_widths[col], row_height), 2)

            # Desenhar círculo colorido na primeira coluna
            if col == 0:
                text_surface = font.render(veiculos[int(val)], True, (0, 0, 0))
                screen.blit(text_surface, (x + 40, y + 5))
                # Risco menor e centralizado na célula
                line_length = 20  # tamanho fixo igual à legenda do gráfico
                line_x_start = x + 5  # margem da célula
                line_x_end = line_x_start + line_length
                line_y = y + row_height // 2
                pygame.draw.line(
                    screen,
                    VEHICLE_COLORS[v_idx],
                    (line_x_start, line_y),
                    (line_x_end, line_y),
                    4,
                )
            elif headers[col] == "Limite":
                # Desenha o valor e os botões +/-
                text_surface = font.render(val, True, (0, 0, 0))
                screen.blit(text_surface, (x + 5, y + 5))

                button_w, button_h = 20, row_height - 10
                minus_rect = pygame.Rect(x + col_widths[col] - button_w * 2 - 5, y + 5, button_w, button_h)
                plus_rect = pygame.Rect(x + col_widths[col] - button_w - 5, y + 5, button_w, button_h)

                pygame.draw.rect(screen, (200, 200, 200), minus_rect)
                pygame.draw.rect(screen, (200, 200, 200), plus_rect)
                screen.blit(font.render("-", True, (0,0,0)), (minus_rect.x + 6, minus_rect.y - 2))
                screen.blit(font.render("+", True, (0,0,0)), (plus_rect.x + 5, plus_rect.y - 2))

                spinner_buttons.append({'rect': minus_rect, 'type': '-', 'vehicle_idx': v_idx})
                spinner_buttons.append({'rect': plus_rect, 'type': '+', 'vehicle_idx': v_idx})

            else:
                text_surface = font.render(val, True, (0, 0, 0))
                screen.blit(text_surface, (x + 5, y + 5))

    # ----------------- LINHA DE TOTALIZADORES -----------------
    total_dist = sum(info[1] for info in vehicle_info)
    last_total_dist = total_dist
    total_cities = sum(info[2] for info in vehicle_info)  # subtrair depósito

    y = start_y + len(vehicle_info) * row_height
    pygame.draw.rect(
        screen, (200, 200, 200), (start_x, y, sum(col_widths), row_height)
    )  # fundo
    pygame.draw.rect(
        screen, (0, 0, 0), (start_x, y, sum(col_widths), row_height), 2
    )  # borda

    total_values = ["Total", str(round(total_dist)), "", str(total_cities), ""]
    for col, val in enumerate(total_values):
        x = start_x + sum(col_widths[:col])
        pygame.draw.rect(screen, (200, 200, 200), (x, y, col_widths[col], row_height))
        pygame.draw.rect(screen, (0, 0, 0), (x, y, col_widths[col], row_height), 2)
        text_surface = font.render(val, True, (0, 0, 0))
        screen.blit(text_surface, (x + 5, y + 5))

    # ----------------- VERIFICAÇÃO DE CONSISTÊNCIA -----------------
    # Garante que o número total de cidades alocadas não diminuiu.
    # Se diminuir, é um sinal de erro no rebalanceamento.
    total_cidades_alocadas = sum(len(set(c) - {depot}) for c in vehicle_clusters)
    total_cidades_geradas = len(cities_locations)
    if total_cidades_alocadas < total_cidades_geradas:
        print(f"ERRO: Perda de cidades! Esperado: {total_cidades_geradas}, Encontrado: {total_cidades_alocadas}. Reiniciando.")
        if show_popup_message(screen, "Erro de consistência: cidades perdidas. Reiniciando.", reset_to_initial=True):
            VEHICLE_DISTANCE_LIMITS[:] = INITIAL_VEHICLE_DISTANCE_LIMITS
            reiniciar_GA()

    pygame.display.flip()
    clock.tick(FPS)

# ------------------------- EXIBIR RESULTADOS FINAIS -------------------------
end_time = time.time()
print(f"Benchmark concluído após {MAX_GENERATIONS} gerações.")
print(f"Tempo total: {end_time - start_time:.2f} segundos")
print(f"Fitness final (distância total): {last_total_dist:.2f}")

# Prepara os dados para o LLM
solutions_data = []
for v in range(NUM_VEHICLES):
    best_solution = final_displayed_solutions[v]
    best_distance = final_displayed_fitness[v]
    route = [city.name for city in best_solution]
    
    solutions_data.append({
        "veiculo": v + 1,
        "distancia": best_distance,
        "rota": route
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
