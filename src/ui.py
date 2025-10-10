# -*- coding: utf-8 -*-
"""
User Interface module for TSP Solver.

Handles Pygame-based visualization and user interaction.
"""

import sys
from typing import Callable, List, Tuple

import pygame

from constants import (
    CHECKBOX_OFFSET_X,
    CHECKBOX_OFFSET_Y_START,
    CHECKBOX_SPACING,
    FPS,
    GRAPH_HEIGHT,
    LEFT_PANEL_WIDTH,
    SCREEN_HEIGHT,
    SCREEN_WIDTH,
    VEHICLE_COLORS,
)
from draw_functions import draw_cities, draw_paths


class Checkbox:
    """Represents a checkbox UI element."""

    def __init__(
        self, rect: pygame.Rect, text: str, value_getter: Callable, setter: Callable
    ):
        self.rect = rect
        self.text = text
        self.value_getter = value_getter
        self.setter = setter

    def draw(self, screen: pygame.Surface, font: pygame.font.Font):
        """Draw the checkbox on the screen."""
        pygame.draw.rect(screen, (0, 0, 0), self.rect, 2)
        if self.value_getter():
            pygame.draw.line(
                screen,
                (0, 0, 0),
                (self.rect.x, self.rect.y),
                (self.rect.x + 20, self.rect.y + 20),
                2,
            )
            pygame.draw.line(
                screen,
                (0, 0, 0),
                (self.rect.x + 20, self.rect.y),
                (self.rect.x, self.rect.y + 20),
                2,
            )
        screen.blit(
            font.render(self.text, True, (0, 0, 0)), (self.rect.x + 25, self.rect.y - 2)
        )

    def handle_click(self, pos: Tuple[int, int]) -> bool:
        """Handle mouse click on checkbox."""
        if self.rect.collidepoint(pos):
            self.setter(not self.value_getter())
            return True
        return False


class TSPSolverUI:
    """Main UI class for the TSP solver application."""

    def __init__(self, vehicle_colors: List[Tuple[int, int, int]] = None):
        self.vehicle_colors = vehicle_colors or VEHICLE_COLORS
        self.screen = None
        self.clock = None
        self.font = None
        self.checkboxes: List[Checkbox] = []
        self.running = False

    def initialize(self):
        """Initialize Pygame and UI components."""
        pygame.init()
        self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
        pygame.display.set_caption("TSP Solver com múltiplos veículos")
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont("Arial", 18)
        self.running = True

    def add_checkbox(self, text: str, value_getter: Callable, setter: Callable):
        """Add a checkbox to the UI."""
        y_offset = CHECKBOX_OFFSET_Y_START + len(self.checkboxes) * CHECKBOX_SPACING
        rect = pygame.Rect(LEFT_PANEL_WIDTH + CHECKBOX_OFFSET_X, y_offset, 20, 20)
        checkbox = Checkbox(rect, text, value_getter, setter)
        self.checkboxes.append(checkbox)

    def handle_events(self) -> bool:
        """
        Handle Pygame events.

        Returns:
            True if should continue running, False to quit
        """
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_q:
                return False
            elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                for checkbox in self.checkboxes:
                    if checkbox.handle_click(event.pos):
                        return True  # Constraint changed, need restart
        return True

    def draw_checkboxes(self):
        """Draw all checkboxes."""
        for checkbox in self.checkboxes:
            checkbox.draw(self.screen, self.font)

    def draw_table(self, vehicle_info: List[Tuple], generation: int):
        """Draw the results table."""
        font = pygame.font.SysFont("Arial", 20)
        start_x, start_y = 10, GRAPH_HEIGHT + 20
        col_widths = [150, 120, 150, 80]
        row_height = 35
        headers = ["Veículo", "Distância", "Cidades", "Geração"]

        # Header
        for col, header in enumerate(headers):
            x = start_x + sum(col_widths[:col])
            y = start_y
            pygame.draw.rect(
                self.screen, (200, 200, 200), (x, y, col_widths[col], row_height)
            )
            pygame.draw.rect(
                self.screen, (0, 0, 0), (x, y, col_widths[col], row_height), 2
            )
            text_surface = font.render(header, True, (0, 0, 0))
            self.screen.blit(text_surface, (x + 5, y + 5))

        start_y += row_height

        # Data rows
        for row, info in enumerate(vehicle_info):
            v, dist, num_cities, last_change = info
            values = [v, str(round(dist)), str(num_cities - 1), str(last_change)]
            for col, val in enumerate(values):
                x = start_x + sum(col_widths[:col])
                y = start_y + row * row_height
                bg_color = (240, 240, 240) if row % 2 == 0 else (220, 220, 220)
                pygame.draw.rect(
                    self.screen, bg_color, (x, y, col_widths[col], row_height)
                )
                pygame.draw.rect(
                    self.screen, (0, 0, 0), (x, y, col_widths[col], row_height), 2
                )

                if col == 0:
                    text_surface = font.render(f"Veículo {val + 1}", True, (0, 0, 0))
                    self.screen.blit(text_surface, (x + 40, y + 5))
                    # Color indicator
                    line_length = 20
                    line_x_start = x + 5
                    line_x_end = line_x_start + line_length
                    line_y = y + row_height // 2
                    pygame.draw.line(
                        self.screen,
                        self.vehicle_colors[v],
                        (line_x_start, line_y),
                        (line_x_end, line_y),
                        4,
                    )
                else:
                    text_surface = font.render(val, True, (0, 0, 0))
                    self.screen.blit(text_surface, (x + 5, y + 5))

        # Totals row
        total_dist = sum(info[1] for info in vehicle_info)
        total_cities = sum(info[2] - 1 for info in vehicle_info)

        y = start_y + len(vehicle_info) * row_height
        pygame.draw.rect(
            self.screen, (200, 200, 200), (start_x, y, sum(col_widths), row_height)
        )
        pygame.draw.rect(
            self.screen, (0, 0, 0), (start_x, y, sum(col_widths), row_height), 2
        )

        total_values = ["Total", str(round(total_dist)), str(total_cities), ""]
        for col, val in enumerate(total_values):
            x = start_x + sum(col_widths[:col])
            pygame.draw.rect(
                self.screen, (200, 200, 200), (x, y, col_widths[col], row_height)
            )
            pygame.draw.rect(
                self.screen, (0, 0, 0), (x, y, col_widths[col], row_height), 2
            )
            text_surface = font.render(val, True, (0, 0, 0))
            self.screen.blit(text_surface, (x + 5, y + 5))

    def draw_map(
        self,
        vehicle_clusters: List[List[Tuple[float, float]]],
        vehicle_solutions: List[List[Tuple[float, float]]],
        depot: Tuple[float, float],
        prohibited_edges: List[Tuple],
        cities_locations: List[Tuple[float, float]],
        priority_cities: List[Tuple[float, float]],
        fuel_stations: List[Tuple[float, float]],
        problem=None,
    ):
        """Draw the city map with routes."""
        for v in range(len(vehicle_clusters)):
            color = self.vehicle_colors[v]
            solution = vehicle_solutions[v] if v < len(vehicle_solutions) else []
            cluster = vehicle_clusters[v]

            draw_paths(
                self.screen,
                solution,
                color,
                width=3,
                problem=problem,
                vias_proibidas=prohibited_edges,
                cities_locations=cities_locations,
                postos_abastecimento=fuel_stations,
            )
            draw_cities(
                self.screen, cluster, color, 10, depot, priority_cities, fuel_stations
            )

    def update_display(self):
        """Update the display."""
        pygame.display.flip()
        self.clock.tick(FPS)

    def quit(self):
        """Clean up and quit."""
        pygame.quit()
        sys.exit()
