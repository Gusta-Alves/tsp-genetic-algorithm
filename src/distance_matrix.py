# -*- coding: utf-8 -*-
"""
Distance Matrix module for TSP optimization.

This module implements a pre-calculated distance matrix for fast distance
lookups during genetic algorithm execution. The matrix stores all pairwise
distances between cities, eliminating the need for repeated Euclidean
distance calculations.
"""

import math
from typing import Dict, List, Tuple, Union

import numpy as np

# Import City class
try:
    from .city import City
except ImportError:
    from city import City


class DistanceMatrix:
    """
    Pre-calculated distance matrix for fast distance lookups.

    This class builds and maintains an NxN matrix of Euclidean distances
    between all pairs of cities, enabling O(1) distance lookups instead
    of O(n) calculations during genetic algorithm execution.
    
    Supports both City objects and coordinate tuples.
    """

    def __init__(self, cities: List[Union[City, Tuple[float, float]]]):
        """
        Initialize distance matrix with city coordinates.

        Args:
            cities: List of City objects or (x, y) coordinate tuples
        """
        self.cities = cities
        self.n_cities = len(cities)
        self.matrix = self._build_matrix()
        self.city_to_index = {city: i for i, city in enumerate(cities)}

    def _get_coords(self, city: Union[City, Tuple[float, float]]) -> Tuple[float, float]:
        """
        Extract coordinates from a City object or tuple.

        Args:
            city: City object or coordinate tuple

        Returns:
            (x, y) coordinate tuple
        """
        if isinstance(city, City):
            return (city.x, city.y)
        return city

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
                # Extract coordinates from City objects or tuples
                coords_i = self._get_coords(self.cities[i])
                coords_j = self._get_coords(self.cities[j])
                
                dist = math.hypot(
                    coords_i[0] - coords_j[0],
                    coords_i[1] - coords_j[1],
                )
                matrix[i][j] = dist
                matrix[j][i] = dist

        return matrix

    def __getitem__(self, key):
        """
        Support subscript notation for accessing the distance matrix.
        
        This allows using distance_matrix[i, j] syntax like numpy arrays.

        Args:
            key: Either a single index or a tuple of (idx1, idx2)

        Returns:
            Distance value or row from the matrix
        """
        return self.matrix[key]

    def get_distance(
        self, city1: Union[City, Tuple[float, float]], city2: Union[City, Tuple[float, float]]
    ) -> float:
        """
        Get pre-calculated distance between two cities.

        Args:
            city1: First city (City object or coordinates)
            city2: Second city (City object or coordinates)

        Returns:
            Distance between the cities
        """
        if city1 == city2:
            return 0.0

        idx1 = self.city_to_index.get(city1)
        idx2 = self.city_to_index.get(city2)

        if idx1 is None or idx2 is None:
            # Fallback: calculate distance directly
            coords1 = self._get_coords(city1)
            coords2 = self._get_coords(city2)
            return math.hypot(coords1[0] - coords2[0], coords1[1] - coords2[1])

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

    def get_total_distance(self, route: List[Union[City, Tuple[float, float]]]) -> float:
        """
        Calculate total distance for a route.

        Args:
            route: List of cities (City objects or coordinates) in order

        Returns:
            Total distance of the route
        """
        total = 0.0
        n = len(route)

        for i in range(n):
            total += self.get_distance(route[i], route[(i + 1) % n])

        return total

    def get_nearest_city(
        self, city: Union[City, Tuple[float, float]], exclude: set = None
    ) -> Union[City, Tuple[float, float]]:
        """
        Find nearest city to given city.

        Args:
            city: City (City object or coordinates)
            exclude: Set of cities to exclude from search

        Returns:
            Nearest city (same type as input)
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
