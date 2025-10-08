# -*- coding: utf-8 -*-
"""
TSP Problem module.

Defines the Traveling Salesman Problem structure and constraints.
"""

from typing import List, Optional, Tuple

from constants import MAX_DISTANCE, PROHIBITED_PENALTY


class TSPProblem:
    """Represents a Traveling Salesman Problem instance."""

    def __init__(
        self,
        cities: List[Tuple[float, float]],
        depot: Optional[Tuple[float, float]] = None,
        prohibited_edges: Optional[
            List[Tuple[Tuple[float, float], Tuple[float, float]]]
        ] = None,
        fuel_stations: Optional[List[Tuple[float, float]]] = None,
        priority_cities: Optional[List[Tuple[float, float]]] = None,
    ):
        """
        Initialize TSP problem.

        Args:
            cities: List of city coordinates
            depot: Depot location (defaults to first city)
            prohibited_edges: List of prohibited edges
            fuel_stations: List of fuel station locations
            priority_cities: List of priority cities
        """
        self.cities = cities
        self.depot = depot or cities[0]
        self.prohibited_edges = prohibited_edges or []
        self.fuel_stations = fuel_stations or []
        self.priority_cities = priority_cities or []

        # Create city to index mapping for efficient lookups
        self.city_to_index = {coord: i for i, coord in enumerate(cities)}

    def is_edge_prohibited(
        self, city1: Tuple[float, float], city2: Tuple[float, float]
    ) -> bool:
        """Check if an edge between two cities is prohibited."""
        return (city1, city2) in self.prohibited_edges or (
            city2,
            city1,
        ) in self.prohibited_edges

    def get_city_index(self, city: Tuple[float, float]) -> Optional[int]:
        """Get the index of a city."""
        return self.city_to_index.get(city)

    def is_priority_city(self, city: Tuple[float, float]) -> bool:
        """Check if a city is a priority city."""
        return city in self.priority_cities

    def get_nearest_fuel_station(
        self, location: Tuple[float, float]
    ) -> Optional[Tuple[float, float]]:
        """Get the nearest fuel station to a location."""
        if not self.fuel_stations:
            return None
        return min(
            self.fuel_stations,
            key=lambda station: self.calculate_distance(location, station),
        )

    def calculate_distance(
        self, point1: Tuple[float, float], point2: Tuple[float, float]
    ) -> float:
        """Calculate Euclidean distance between two points."""
        import math

        return math.hypot(point1[0] - point2[0], point1[1] - point2[1])

    def calculate_route_distance(self, route: List[Tuple[float, float]]) -> float:
        """
        Calculate the total distance of a route, considering constraints.

        Args:
            route: List of cities in order

        Returns:
            Total distance with penalties for violations
        """
        if len(route) < 2:
            return 0.0

        total_distance = 0.0
        distance_since_refuel = 0.0

        for i in range(len(route) - 1):
            city1, city2 = route[i], route[i + 1]

            # Check for prohibited edges
            if self.is_edge_prohibited(city1, city2):
                return PROHIBITED_PENALTY

            # Calculate segment distance
            segment_distance = self.calculate_distance(city1, city2)
            total_distance += segment_distance

            # Check for fuel constraints
            if (
                self.fuel_stations
                and distance_since_refuel + segment_distance > MAX_DISTANCE
            ):
                # Need to refuel - find nearest station
                station = self.get_nearest_fuel_station(city1)
                if station:
                    # Add distance to station and back to route
                    to_station = self.calculate_distance(city1, station)
                    from_station = self.calculate_distance(station, city2)
                    total_distance += to_station + from_station
                    distance_since_refuel = from_station
                else:
                    # No fuel station available
                    return PROHIBITED_PENALTY
            else:
                distance_since_refuel += segment_distance

        return total_distance
