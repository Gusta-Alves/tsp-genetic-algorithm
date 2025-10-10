# -*- coding: utf-8 -*-
"""
Distance Matrix module for TSP optimization.

This module implements a pre-calculated distance matrix for fast distance
lookups during genetic algorithm execution. The matrix stores all pairwise
distances between cities, eliminating the need for repeated Euclidean
distance calculations.
"""

import math
from typing import Dict, List, Tuple

import numpy as np


class DistanceMatrix:
    """
    Pre-calculated distance matrix for fast distance lookups.

    This class builds and maintains an NxN matrix of Euclidean distances
    between all pairs of cities, enabling O(1) distance lookups instead
    of O(n) calculations during genetic algorithm execution.
    """

    def __init__(self, cities: List[Tuple[float, float]]):
        """
        Initialize distance matrix with city coordinates.

        Args:
            cities: List of (x, y) coordinate tuples
        """
        self.cities = cities
        self.n_cities = len(cities)
        self.matrix = self._build_matrix()
        self.city_to_index = {city: i for i, city in enumerate(cities)}

    def _build_matrix(self) -> np.ndarray:
        """
        Build the distance matrix using Euclidean distance.

        Creates a symmetric NxN matrix where element [i,j] represents
        the Euclidean distance between city i and city j.

        Returns:
            NxN numpy array with pairwise distances
        """
        n = self.n_cities
        matrix = np.zeros((n, n), dtype=np.float64)

        for i in range(n):
            for j in range(i + 1, n):
                dist = math.hypot(
                    self.cities[i][0] - self.cities[j][0],
                    self.cities[i][1] - self.cities[j][1],
                )
                matrix[i][j] = dist
                matrix[j][i] = dist

        return matrix

    def get_distance(
        self, city1: Tuple[float, float], city2: Tuple[float, float]
    ) -> float:
        """
        Get pre-calculated distance between two cities.

        Args:
            city1: First city coordinates
            city2: Second city coordinates

        Returns:
            Distance between the cities
        """
        if city1 == city2:
            return 0.0

        idx1 = self.city_to_index.get(city1)
        idx2 = self.city_to_index.get(city2)

        if idx1 is None or idx2 is None:
            return math.hypot(city1[0] - city2[0], city1[1] - city2[1])

        return self.matrix[idx1][idx2]

    def get_distance_by_index(self, idx1: int, idx2: int) -> float:
        """
        Get distance using city indices directly.

        Args:
            idx1: Index of first city
            idx2: Index of second city

        Returns:
            Distance between the cities
        """
        return self.matrix[idx1][idx2]

    def get_total_distance(self, route: List[Tuple[float, float]]) -> float:
        """
        Calculate total distance for a route.

        Args:
            route: List of city coordinates in order

        Returns:
            Total distance of the route
        """
        total = 0.0
        n = len(route)

        for i in range(n):
            total += self.get_distance(route[i], route[(i + 1) % n])

        return total

    def get_nearest_city(
        self, city: Tuple[float, float], exclude: set = None
    ) -> Tuple[float, float]:
        """
        Find nearest city to given city.

        Args:
            city: City coordinates
            exclude: Set of cities to exclude from search

        Returns:
            Coordinates of nearest city
        """
        exclude = exclude or set()
        idx = self.city_to_index.get(city)

        if idx is None:
            return None

        min_dist = float("inf")
        nearest_idx = -1

        for i in range(self.n_cities):
            if self.cities[i] in exclude or i == idx:
                continue

            dist = self.matrix[idx][i]
            if dist < min_dist:
                min_dist = dist
                nearest_idx = i

        return self.cities[nearest_idx] if nearest_idx >= 0 else None

    def get_statistics(self) -> Dict:
        """
        Get statistics about the distance matrix.

        Returns:
            Dictionary with min, max, mean, median distances
        """
        upper_triangle = self.matrix[np.triu_indices(self.n_cities, k=1)]

        return {
            "n_cities": self.n_cities,
            "n_distances": len(upper_triangle),
            "min_distance": float(np.min(upper_triangle)),
            "max_distance": float(np.max(upper_triangle)),
            "mean_distance": float(np.mean(upper_triangle)),
            "median_distance": float(np.median(upper_triangle)),
            "std_distance": float(np.std(upper_triangle)),
        }
