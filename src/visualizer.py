"""
Visualization module for TSP Solver.

This module handles all Pygame visualization aspects including
rendering cities, paths, and fitness evolution plots.
"""

from typing import List, Tuple

import pygame

from config import VisualizationConfig
from draw_functions import draw_cities, draw_paths, draw_plot


class TSPVisualizer:
    """Handles all visualization aspects using Pygame."""

    def __init__(self, config: VisualizationConfig):
        """
        Initialize the Pygame visualization.

        Args:
            config: Visualization configuration parameters
        """
        self.config = config
        pygame.init()
        self.screen = pygame.display.set_mode((config.width, config.height))
        pygame.display.set_caption("TSP Solver - Genetic Algorithm")
        self.clock = pygame.time.Clock()

    def handle_events(self) -> bool:
        """
        Handle Pygame events.

        Returns:
            True if the application should continue running, False to quit
        """
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_q:
                    return False
        return True

    def render(
        self,
        cities_locations: List[Tuple[int, int]],
        best_solution: List[Tuple[int, int]],
        second_best_solution: List[Tuple[int, int]],
        best_fitness_history: List[float],
        generation: int,
        best_fitness: float,
    ) -> None:
        """
        Render the current state of the algorithm.

        Displays:
        - Fitness evolution plot
        - City locations
        - Best solution path (blue)
        - Second best solution path (gray)

        Args:
            cities_locations: All city locations
            best_solution: Current best route
            second_best_solution: Second best route
            best_fitness_history: History of best fitness values
            generation: Current generation number
            best_fitness: Current best fitness value
        """
        self.screen.fill(self.config.white)

        # Draw fitness plot
        if best_fitness_history:
            draw_plot(
                self.screen,
                list(range(len(best_fitness_history))),
                best_fitness_history,
                y_label="Fitness - Distance (pixels)",
            )

        # Draw cities and paths
        draw_cities(
            self.screen, cities_locations, self.config.red, self.config.node_radius
        )

        if second_best_solution:
            draw_paths(
                self.screen, second_best_solution, rgb_color=self.config.gray, width=1
            )

        if best_solution:
            draw_paths(self.screen, best_solution, self.config.blue, width=3)

        # Update display
        pygame.display.flip()
        self.clock.tick(self.config.fps)

        # Print progress
        print(f"Generation {generation}: Best fitness = {best_fitness:.2f}")

    def cleanup(self) -> None:
        """Clean up Pygame resources."""
        pygame.quit()
