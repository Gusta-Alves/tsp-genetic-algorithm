# -*- coding: utf-8 -*-
"""
Configuration module for TSP Genetic Algorithm.

Contains configuration classes and settings.
"""

from dataclasses import dataclass

from constants import (
    ELITE_SIZE,
    FPS,
    MUTATION_PROBABILITY,
    NODE_RADIUS,
    POPULATION_SIZE,
    SCREEN_HEIGHT,
    SCREEN_WIDTH,
    TOURNAMENT_SIZE,
)


@dataclass
class GAConfig:
    """Configuration parameters for the Genetic Algorithm."""

    population_size: int = POPULATION_SIZE
    mutation_probability: float = MUTATION_PROBABILITY
    elite_size: int = ELITE_SIZE
    tournament_size: int = TOURNAMENT_SIZE


@dataclass
class VisualizationConfig:
    """Configuration parameters for Pygame visualization."""

    screen_width: int = SCREEN_WIDTH
    screen_height: int = SCREEN_HEIGHT
    fps: int = FPS
    node_radius: int = NODE_RADIUS
