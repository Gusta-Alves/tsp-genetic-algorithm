# -*- coding: utf-8 -*-
"""
Created on Fri Dec 22 16:03:11 2023

@author: SérgioPolimante
"""
from typing import List, Tuple

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pygame
import pylab
from matplotlib.backends.backend_agg import FigureCanvasAgg
from genetic_algorithm import calculate_distance

matplotlib.use("Agg")

MAX_DISTANCE = 900

def draw_button(screen, rect, text, color_bg, color_text=(255,255,255)):
    pygame.draw.rect(screen, color_bg, rect)
    font = pygame.font.SysFont("Arial", 20)
    text_surface = font.render(text, True, color_text)
    text_rect = text_surface.get_rect(center=(rect[0]+rect[2]//2, rect[1]+rect[3]//2))
    screen.blit(text_surface, text_rect)

def draw_plot(
    screen: pygame.Surface,
    x: list,
    y: list,
    x_label: str = "Generation",
    y_label: str = "Fitness",
) -> None:
    """
    Draw a plot on a Pygame screen using Matplotlib.

    Parameters:
    - screen (pygame.Surface): The Pygame surface to draw the plot on.
    - x (list): The x-axis values.
    - y (list): The y-axis values.
    - x_label (str): Label for the x-axis (default is 'Generation').
    - y_label (str): Label for the y-axis (default is 'Fitness').
    """
    fig, ax = plt.subplots(figsize=(4, 4), dpi=100)
    ax.plot(x, y)
    ax.set_ylabel(y_label)
    ax.set_xlabel(x_label)
    plt.tight_layout()

    canvas = FigureCanvasAgg(fig)
    canvas.draw()
    renderer = canvas.get_renderer()
    raw_data = renderer.buffer_rgba()

    size = canvas.get_width_height()
    surf = pygame.image.frombuffer(raw_data, size, "RGBA")
    screen.blit(surf, (0, 0))


def draw_cities(
    screen: pygame.Surface,
    cities_locations: List[Tuple[int, int]],
    rgb_color: Tuple[int, int, int],
    node_radius: int,
    depot: Tuple[int, int] = None,
    cidades_prioritarias: List[Tuple[int, int]] = [],
    postos: List[Tuple[int, int]] = []
) -> None:
    """
    Draws circles representing cities on the given Pygame screen.

    Parameters:
    - screen (pygame.Surface): The Pygame surface on which to draw the cities.
    - cities_locations (List[Tuple[int, int]]): List of (x, y) coordinates representing the locations of cities.
    - rgb_color (Tuple[int, int, int]): Tuple of three integers (R, G, B) representing the color of the city circles.
    - node_radius (int): The radius of the city circles.

    Returns:
    None
    """
    #for city_location in cities_locations:
    #    color = (0, 0, 0) if depot is not None and city_location == depot else rgb_color
    #    pygame.draw.circle(screen, color, city_location, node_radius)
    
    tamanho = node_radius*2 
    for posto in postos:
        x, y = posto
        rect = pygame.Rect(x - tamanho//2, y - tamanho//2, tamanho, tamanho)
        pygame.draw.rect(screen, (0, 100, 0), rect)  # preenchimento
        pygame.draw.rect(screen, (0, 0, 0), rect, 2)   

    for city_location in cities_locations:
        if depot is not None and city_location == depot:
            color = (0,0,0)
        elif city_location in cidades_prioritarias:
            color = (128, 0, 128)  # roxo
        else:
            color = rgb_color
        pygame.draw.circle(screen, color, city_location, node_radius)


def draw_paths(
    screen: pygame.Surface,
    path: List[Tuple[int, int]],
    rgb_color: Tuple[int, int, int],
    width: int = 1,
    vias_proibidas: List[Tuple[int, int]] = None,
    cities_locations: List[Tuple[int, int]] = None,
    postos_abastecimento: List[Tuple[int, int]] = None,
):
    """
    Desenha um caminho. Se vias_proibidas e cities_locations forem passados,
    desenha essas arestas proibidas com cor diferente (roxo).
    Note: cities_locations deve conter os pontos correspondentes a path (pode ser scaled).
    """
    
    for start, end in vias_proibidas:
        pygame.draw.line(screen, (128, 0, 128), start, end, width=3) 

    since_last_refuel = 0
    for i in range(len(path) - 1):
        start = path[i]
        end = path[i + 1]
        color = rgb_color
        d = calculate_distance(start, end, cities_locations, vias_proibidas)
        since_last_refuel += d   

        if postos_abastecimento and since_last_refuel > MAX_DISTANCE:
            posto = min(postos_abastecimento, key=lambda p: calculate_distance(start, p, cities_locations))
            pygame.draw.line(screen, (0,128,128), start, posto, 2)   # linha até o posto
            pygame.draw.line(screen, (0,128,128), posto, end, 2)   # linha de volta à rota
            since_last_refuel = 0
        else:
            pygame.draw.line(screen, color, start, end, width)  


def draw_text(
    screen: pygame.Surface,
    text: str,
    color: pygame.Color,
    cities_locations: List[Tuple[int, int]] = None,
    height: int = 400,
) -> None:
    """
    Draw text on a Pygame screen.

    Parameters:
    - screen (pygame.Surface): The Pygame surface to draw the text on.
    - text (str): The text to be displayed.
    - color (pygame.Color): The color of the text.
    - cities_locations: List of city positions to calculate text position
    - height: Screen height for positioning
    """
    pygame.font.init()  # You have to call this at the start

    font_size = 15
    my_font = pygame.font.SysFont("Arial", font_size)
    text_surface = my_font.render(text, False, color)

    if cities_locations:
        text_position = (
            np.average(np.array(cities_locations)[:, 0]),
            height - 1.5 * font_size,
        )
    else:
        text_position = (10, height - 1.5 * font_size)

    screen.blit(text_surface, text_position)
