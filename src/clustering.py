# -*- coding: utf-8 -*-
"""
Clustering module for TSP with multiple vehicles.

Handles the clustering of cities into groups for multiple vehicles.
"""

from typing import List, Tuple

import numpy as np
from sklearn.cluster import KMeans

from tsp_problem import TSPProblem


class CityClusterer:
    """Handles clustering of cities for multiple vehicle routing."""

    def __init__(self, problem: TSPProblem, num_vehicles: int):
        """
        Initialize the clusterer.

        Args:
            problem: The TSP problem instance
            num_vehicles: Number of vehicles/routes to create
        """
        self.problem = problem
        self.num_vehicles = num_vehicles

    def cluster_cities(self) -> List[List[Tuple[float, float]]]:
        """
        Cluster cities into groups for each vehicle.

        Returns:
            List of city clusters, each containing depot + cities + depot
        """
        cities_array = np.array(self.problem.cities)
        kmeans = KMeans(n_clusters=self.num_vehicles, random_state=42)
        labels = kmeans.fit_predict(cities_array)

        clusters = [[] for _ in range(self.num_vehicles)]
        for idx, label in enumerate(labels):
            clusters[label].append(self.problem.cities[idx])

        # Add depot to start and end of each cluster
        depot = self.problem.depot
        for i in range(self.num_vehicles):
            clusters[i] = [depot] + clusters[i] + [depot]

        return clusters
