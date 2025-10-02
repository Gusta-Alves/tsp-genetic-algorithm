"""
Population initialization module for TSP Solver.

This module handles the creation of the initial population for the genetic algorithm,
using heuristics and random generation.
"""

from typing import List, Tuple

from genetic_algorithm import (
    generate_random_population,
    nearest_neighbor_heuristic,
)


class PopulationInitializer:
    """Handles creation of initial population for the genetic algorithm."""

    @staticmethod
    def create_initial_population(
        cities_locations: List[Tuple[int, int]], population_size: int
    ) -> List[List[Tuple[int, int]]]:
        """
        Create initial population using nearest neighbor heuristic and random solutions.

        Initializes the population by:
        1. Creating one nearest neighbor solution for each city as starting point
        2. Filling remaining slots with random solutions

        Args:
            cities_locations: List of (x, y) coordinates for all cities
            population_size: Desired size of the population

        Returns:
            List of individuals (routes) forming the initial population
        """
        population = []
        n_cities = len(cities_locations)

        # Create nearest neighbor solutions starting from each city
        for i in range(n_cities):
            print(f"Initializing NN heuristic from city index: {i}")
            population.append(
                nearest_neighbor_heuristic(cities_locations, start_city_index=i)
            )

        # Fill remaining slots with random solutions
        remaining_size = population_size - len(population)
        if remaining_size > 0:
            population.extend(
                generate_random_population(cities_locations, remaining_size)
            )

        return population
