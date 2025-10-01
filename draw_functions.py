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

matplotlib.use("Agg")


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
    for city_location in cities_locations:
        pygame.draw.circle(screen, rgb_color, city_location, node_radius)


def draw_paths(
    screen: pygame.Surface,
    path: List[Tuple[int, int]],
    rgb_color: Tuple[int, int, int],
    width: int = 1,
):
    """
    Draw a path on a Pygame screen.

    Parameters:
    - screen (pygame.Surface): The Pygame surface to draw the path on.
    - path (List[Tuple[int, int]]): List of tuples representing the coordinates of the path.
    - rgb_color (Tuple[int, int, int]): RGB values for the color of the path.
    - width (int): Width of the path lines (default is 1).
    """
    pygame.draw.lines(screen, rgb_color, True, path, width=width)


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
