"""
Configuration classes for TSP Solver.

This module contains dataclasses that hold configuration parameters
for the genetic algorithm and visualization components.
"""

from dataclasses import dataclass
from typing import Tuple


@dataclass
class GAConfig:
    """Configuration parameters for the Genetic Algorithm."""

    population_size: int = 150
    mutation_probability: float = 0.7
    elite_size: int = 10
    tournament_size: int = 5


@dataclass
class VisualizationConfig:
    """Configuration parameters for Pygame visualization."""

    width: int = 1500
    height: int = 800
    node_radius: int = 10
    fps: int = 30
    plot_x_offset: int = 450

    # Colors
    white: Tuple[int, int, int] = (255, 255, 255)
    red: Tuple[int, int, int] = (255, 0, 0)
    blue: Tuple[int, int, int] = (0, 0, 255)
    gray: Tuple[int, int, int] = (128, 128, 128)
