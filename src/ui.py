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


class ScrollableMarkdownArea:
    """Area with scrollable markdown content."""
    
    def __init__(self, x, y, width, height, screen):
        self.x = x
        self.y = y
        self.width = width
        self.height = height
        self.screen = screen
        self.scroll_offset = 0
        self.content_height = 0
        self.max_scroll = 0
        self.scroll_speed = 20
        
    def handle_scroll(self, event):
        """Handle mouse wheel scroll events."""
        if event.type == pygame.MOUSEBUTTONDOWN:
            if event.button == 4:  # Scroll up
                self.scroll_offset = max(0, self.scroll_offset - self.scroll_speed)
            elif event.button == 5:  # Scroll down
                self.scroll_offset = min(self.max_scroll, self.scroll_offset + self.scroll_speed)
    
    def render_markdown(self, markdown_text):
        """Render markdown text with scroll support."""
        # Cria uma surface temporária para desenhar todo o conteúdo
        temp_surface = pygame.Surface((self.width, max(self.height, 2000)))
        temp_surface.fill((245, 245, 245))
        
        text_y = 10
        lines = markdown_text.split('\n')
        
        for line in lines:
            if text_y > 2000 - 50:  # Limite de segurança
                break
            increment = render_markdown_line(line, text_y, temp_surface, base_x=10, max_width=self.width - 40)
            text_y += increment
        
        self.content_height = text_y + 10
        self.max_scroll = max(0, self.content_height - self.height)
        
        # Desenha a área visível com scroll
        # Fundo
        pygame.draw.rect(self.screen, (245, 245, 245), (self.x, self.y, self.width, self.height))
        pygame.draw.line(self.screen, (0, 0, 0), (self.x, self.y), (self.x + self.width, self.y), 3)
        
        # Blit apenas a parte visível
        visible_rect = pygame.Rect(0, self.scroll_offset, self.width, self.height)
        self.screen.blit(temp_surface, (self.x, self.y), visible_rect)
        
        # Desenha scrollbar se necessário
        if self.content_height > self.height:
            self._draw_scrollbar()
    
    def _draw_scrollbar(self):
        """Draw a scrollbar on the right side."""
        scrollbar_width = 10
        scrollbar_x = self.x + self.width - scrollbar_width - 5
        scrollbar_y = self.y + 5
        scrollbar_height = self.height - 10
        
        # Fundo da scrollbar
        pygame.draw.rect(self.screen, (200, 200, 200), 
                        (scrollbar_x, scrollbar_y, scrollbar_width, scrollbar_height))
        
        # Thumb da scrollbar
        thumb_height = max(20, int(scrollbar_height * (self.height / self.content_height)))
        thumb_y = scrollbar_y + int((self.scroll_offset / self.max_scroll) * (scrollbar_height - thumb_height)) if self.max_scroll > 0 else scrollbar_y
        
        pygame.draw.rect(self.screen, (100, 100, 100), 
                        (scrollbar_x, thumb_y, scrollbar_width, thumb_height))

    
def render_markdown_line(line, y_pos, screen, base_x=20, max_width=1160):
    """Render a single line of markdown-like text with word wrapping."""
    import re
    
    # Fontes
    regular_font = pygame.font.SysFont("Arial", 12)
    bold_font = pygame.font.SysFont("Arial", 12, bold=True)
    header_font = pygame.font.SysFont("Arial", 14, bold=True)
    
    line = line.strip()
    if not line:
        return 15
    
    x_pos = base_x
    line_height = 18
    current_y = y_pos
    total_height = 0
    
    # Headers (## ou ###)
    if line.startswith('###'):
        text = line[3:].strip()
        surface = pygame.font.SysFont("Arial", 14, bold=True).render(text, True, (0, 0, 0))
        screen.blit(surface, (x_pos, current_y))
        return line_height
    elif line.startswith('##'):
        text = line[2:].strip()
        surface = header_font.render(text, True, (0, 0, 0))
        screen.blit(surface, (x_pos, current_y))
        return line_height + 2
    elif line.startswith('#'):
        text = line[1:].strip()
        surface = pygame.font.SysFont("Arial", 18, bold=True).render(text, True, (0, 0, 0))
        screen.blit(surface, (x_pos, current_y))
        return line_height + 5
    
    # Lista com marcadores
    indent = 0
    if line.startswith('- ') or line.startswith('* '):
        bullet_surface = regular_font.render("•", True, (0, 0, 0))
        screen.blit(bullet_surface, (x_pos, current_y))
        indent = 15
        x_pos += indent
        line = line[2:].strip()
    
    # Processa negrito (**texto**) e quebra de linha
    parts = re.split(r'(\*\*.*?\*\*)', line)
    
    for part in parts:
        if not part:
            continue
            
        is_bold = part.startswith('**') and part.endswith('**')
        text = part[2:-2] if is_bold else part
        font = bold_font if is_bold else regular_font
        
        # Quebra o texto em palavras para word wrapping
        words = text.split(' ')
        for i, word in enumerate(words):
            # Adiciona espaço antes da palavra (exceto a primeira)
            if i > 0:
                word = ' ' + word
            
            word_surface = font.render(word, True, (0, 0, 0))
            word_width = word_surface.get_width()
            
            # Verifica se precisa quebrar linha
            if x_pos + word_width > base_x + max_width:
                current_y += line_height
                total_height += line_height
                x_pos = base_x + indent  # Mantém indentação em linhas subsequentes
            
            screen.blit(word_surface, (x_pos, current_y))
            x_pos += word_width
    
    return line_height + total_height
