# -*- coding: utf-8 -*-
"""
TSP Solver Application.

Main application class that coordinates all components for solving TSP with genetic algorithms.
"""

import itertools
from typing import List, Optional, Tuple

import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg

from benchmark_att48 import att_48_cities_locations
from clustering import CityClusterer
from config import (
    ELITE_SIZE,
    MUTATION_PROBABILITY,
    NUM_VEHICLES,
    POPULATION_SIZE,
    TOURNAMENT_SIZE,
    VEHICLE_COLORS,
)
from draw_functions import draw_plot
from ga_engine import GeneticAlgorithmEngine
from genetic_algorithm import (
    FitnessCalculator,
    generate_random_population,
    nearest_neighbor_heuristic,
)
from tsp_problem import TSPProblem
from ui import TSPSolverUI


class TSPSolverApp:
    """Main application for TSP solving with multiple vehicles."""

    def __init__(self):
        self.problem: Optional[TSPProblem] = None
        self.clusterer: Optional[CityClusterer] = None
        self.vehicle_clusters: List[List[Tuple[float, float]]] = []
        self.vehicle_populations: List[List[List[Tuple[float, float]]]] = []
        self.vehicle_engines: List[GeneticAlgorithmEngine] = []
        self.ui = TSPSolverUI(VEHICLE_COLORS)
        self.generation_counter = itertools.count(start=1)

        # State tracking
        self.vehicle_best_fitness = []
        self.vehicle_best_solutions = []
        self.vehicle_last_change = []

        # Constraints
        self.prohibited_edges = []
        self.priority_cities = []
        self.fuel_stations = []

    def initialize_problem(self):
        """Initialize the TSP problem with ATT48 data."""
        # Scale coordinates for display
        max_x = max(point[0] for point in att_48_cities_locations)
        max_y = max(point[1] for point in att_48_cities_locations)

        scale_x = (1200 - 500 - 100) / max_x  # RIGHT_PANEL_WIDTH - MARGIN
        scale_y = (650 - 100) / max_y

        scaled_cities = [
            (int(point[0] * scale_x + 500 + 50), int(point[1] * scale_y + 50))
            for point in att_48_cities_locations
        ]

        depot_x = int(sum(x for x, y in scaled_cities) / len(scaled_cities))
        depot_y = int(sum(y for x, y in scaled_cities) / len(scaled_cities))
        depot = (depot_x, depot_y)

        self.problem = TSPProblem(
            cities=scaled_cities,
            depot=depot,
            prohibited_edges=self.prohibited_edges,
            fuel_stations=self.fuel_stations,
            priority_cities=self.priority_cities,
        )

    def setup_clustering(self):
        """Setup city clustering for multiple vehicles."""
        self.clusterer = CityClusterer(self.problem, NUM_VEHICLES)
        self.vehicle_clusters = self.clusterer.cluster_cities()

    def initialize_populations(self):
        """Initialize populations for each vehicle."""
        self.vehicle_populations = []
        self.vehicle_engines = []

        for cluster in self.vehicle_clusters:
            # Create initial population with heuristics
            population = []
            for i in range(1, len(cluster) - 1):
                solution = nearest_neighbor_heuristic(self.problem, i)
                if self.priority_cities:
                    solution = self._apply_priority_constraint(solution)
                population.append(solution)

            # Fill with random solutions
            random_pop = generate_random_population(
                self.problem, POPULATION_SIZE - len(population)
            )
            if self.priority_cities:
                random_pop = [self._apply_priority_constraint(p) for p in random_pop]
            population.extend(random_pop)

            self.vehicle_populations.append(population)

            # Create GA engine for this vehicle
            fitness_calc = FitnessCalculator(self.problem)
            engine = GeneticAlgorithmEngine(
                population_size=POPULATION_SIZE,
                mutation_probability=MUTATION_PROBABILITY,
                elite_size=ELITE_SIZE,
                tournament_size=TOURNAMENT_SIZE,
                fitness_calculator=fitness_calc.calculate_fitness,
            )
            engine.set_population(population)
            self.vehicle_engines.append(engine)

        # Initialize tracking
        self.vehicle_best_fitness = [[] for _ in range(NUM_VEHICLES)]
        self.vehicle_best_solutions = [[] for _ in range(NUM_VEHICLES)]
        self.vehicle_last_change = [0] * NUM_VEHICLES

    def _apply_priority_constraint(
        self, route: List[Tuple[float, float]]
    ) -> List[Tuple[float, float]]:
        """Apply priority city constraint to a route."""
        depot = route[0]
        rest = route[1:-1]
        for city in self.priority_cities:
            if city in rest:
                rest.remove(city)
                rest = [city] + rest
        return [depot] + rest + [depot]

    def update_constraints(self, prohibited: bool, priority: bool, fuel: bool):
        """Update problem constraints."""
        if prohibited:
            self.prohibited_edges = [
                ((814, 344), (877, 291)),
                ((779, 252), (784, 221)),
                ((1013, 165), (1014, 119)),
                ((1100, 443), (1108, 519)),
            ]
        else:
            self.prohibited_edges = []

        if priority:
            self.priority_cities = [(944, 389), (674, 342), (900, 174), (1120, 387)]
        else:
            self.priority_cities = []

        if fuel:
            self.fuel_stations = [(710, 150), (840, 370), (1010, 220), (1060, 430)]
        else:
            self.fuel_stations = []

        # Reinitialize problem and populations
        self.initialize_problem()
        self.setup_clustering()
        self.initialize_populations()

    def run_generation(self) -> List[Tuple]:
        """Run one generation for all vehicles."""
        generation = next(self.generation_counter)
        vehicle_info = []

        for v in range(NUM_VEHICLES):
            engine = self.vehicle_engines[v]
            engine.evolve_generation()

            best_solution, best_fitness = engine.get_best_solution()
            num_cities = len(best_solution) - 2

            if (
                len(self.vehicle_best_solutions[v]) == 0
                or self.vehicle_best_solutions[v][-1] != best_solution
            ):
                self.vehicle_last_change[v] = generation

            self.vehicle_best_fitness[v].append(best_fitness)
            self.vehicle_best_solutions[v].append(best_solution)
            vehicle_info.append(
                (v, best_fitness, num_cities, self.vehicle_last_change[v])
            )

        return vehicle_info

    def draw_plot(self):
        """Draw the fitness evolution plot."""
        plt.figure(figsize=(5, 4), dpi=100)
        plt.gca().set_facecolor("#f9f9f9")
        plt.grid(True, linestyle="--", alpha=0.5)

        for v in range(NUM_VEHICLES):
            plt.plot(
                list(range(len(self.vehicle_best_fitness[v]))),
                self.vehicle_best_fitness[v],
                color=[c / 255 for c in VEHICLE_COLORS[v]],
                label=f"Veículo {v+1}",
                linewidth=3,
            )

        plt.xlabel("Generation", fontsize=12)
        plt.ylabel("Fitness (Distance)", fontsize=12)
        plt.tick_params(axis="both", labelsize=10)
        plt.legend(frameon=False, fontsize=10)
        plt.tight_layout(pad=2)

        canvas = FigureCanvasAgg(plt.gcf())
        canvas.draw()
        renderer = canvas.get_renderer()
        raw_data = renderer.buffer_rgba()
        size = canvas.get_width_height()
        surf = pygame.image.frombuffer(raw_data, size, "RGBA")
        self.ui.screen.blit(surf, (10, 10))
        plt.close()

    def run(self):
        """Main application loop."""
        self.ui.initialize()

        # Setup initial problem
        self.initialize_problem()
        self.setup_clustering()
        self.initialize_populations()

        # Setup UI checkboxes
        self.ui.add_checkbox(
            "Via proibida",
            lambda: bool(self.prohibited_edges),
            lambda val: self.update_constraints(
                val, bool(self.priority_cities), bool(self.fuel_stations)
            ),
        )
        self.ui.add_checkbox(
            "Cidade prioritária",
            lambda: bool(self.priority_cities),
            lambda val: self.update_constraints(
                bool(self.prohibited_edges), val, bool(self.fuel_stations)
            ),
        )
        self.ui.add_checkbox(
            "Parada para Abastecer",
            lambda: bool(self.fuel_stations),
            lambda val: self.update_constraints(
                bool(self.prohibited_edges), bool(self.priority_cities), val
            ),
        )

        while self.ui.running:
            if not self.ui.handle_events():
                break

            self.ui.screen.fill((255, 255, 255))

            # Draw UI elements
            self.ui.draw_checkboxes()
            vehicle_info = self.run_generation()
            self.draw_plot()
            self.ui.draw_table(vehicle_info, next(self.generation_counter) - 1)
            self.ui.draw_map(
                self.vehicle_clusters,
                [engine.get_best_solution()[0] for engine in self.vehicle_engines],
                self.problem.depot,
                self.prohibited_edges,
                self.problem.cities,
                self.priority_cities,
                self.fuel_stations,
                problem=self.problem,
            )

            self.ui.update_display()

        self.ui.quit()


if __name__ == "__main__":
    app = TSPSolverApp()
    app.run()
