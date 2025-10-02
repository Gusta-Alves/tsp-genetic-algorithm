"""
Problem initialization module for TSP Solver.

This module handles the initialization of TSP problem instances,
including loading and scaling benchmark problems.
"""

from typing import List, Tuple

import numpy as np

from benchmark_att48 import att_48_cities_locations, att_48_cities_order
from config import VisualizationConfig
from genetic_algorithm import calculate_fitness


class ProblemInitializer:
    """Handles initialization of TSP problem instances."""

    @staticmethod
    def initialize_att48_benchmark(
        vis_config: VisualizationConfig,
    ) -> Tuple[List[Tuple[int, int]], List[Tuple[int, int]], float]:
        """
        Initialize the ATT48 benchmark problem.

        Scales the ATT48 benchmark cities to fit within the visualization window
        and calculates the fitness of the optimal solution.

        Args:
            vis_config: Visualization configuration with window dimensions

        Returns:
            Tuple containing:
                - List of scaled city locations
                - Target optimal solution path
                - Fitness value of the target solution
        """
        att_cities_array = np.array(att_48_cities_locations)
        max_x = max(point[0] for point in att_cities_array)
        max_y = max(point[1] for point in att_cities_array)

        scale_x = (
            vis_config.width - vis_config.plot_x_offset - vis_config.node_radius
        ) / max_x
        scale_y = vis_config.height / max_y

        cities_locations = [
            (
                int(point[0] * scale_x + vis_config.plot_x_offset),
                int(point[1] * scale_y),
            )
            for point in att_cities_array
        ]

        target_solution = [cities_locations[i - 1] for i in att_48_cities_order]
        fitness_target_solution = calculate_fitness(target_solution)

        print(
            f"ATT48 Benchmark - Optimal Solution Fitness: {fitness_target_solution:.2f}"
        )

        return cities_locations, target_solution, fitness_target_solution
