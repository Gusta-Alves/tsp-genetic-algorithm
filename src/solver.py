"""
TSP Solver orchestrator module.

This module contains the main TSP Solver class that orchestrates
all components to solve the Traveling Salesman Problem using
genetic algorithms with visualization.
"""

from config import GAConfig, VisualizationConfig
from ga_engine import GeneticAlgorithmEngine
from population_initializer import PopulationInitializer
from problem_initializer import ProblemInitializer
from visualizer import TSPVisualizer


class TSPSolver:
    """Main TSP Solver orchestrating all components."""

    def __init__(self, ga_config: GAConfig, vis_config: VisualizationConfig):
        """
        Initialize the TSP Solver.

        Args:
            ga_config: Genetic algorithm configuration
            vis_config: Visualization configuration
        """
        self.ga_config = ga_config
        self.vis_config = vis_config
        self.ga_engine = GeneticAlgorithmEngine(ga_config)
        self.visualizer = TSPVisualizer(vis_config)
        self.cities_locations = []

    def setup_problem(self) -> None:
        """
        Set up the TSP problem instance.

        Initializes the ATT48 benchmark problem and creates the initial population.
        """
        # Initialize problem
        self.cities_locations, _, _ = ProblemInitializer.initialize_att48_benchmark(
            self.vis_config
        )

        # Create initial population
        initial_population = PopulationInitializer.create_initial_population(
            self.cities_locations, self.ga_config.population_size
        )
        self.ga_engine.set_population(initial_population)

    def run(self) -> None:
        """
        Run the main solver loop.

        Executes the genetic algorithm with real-time visualization until
        the user closes the window or presses 'Q'.
        """
        running = True

        while running:
            # Handle events
            running = self.visualizer.handle_events()

            if not running:
                break

            # Evolve one generation
            self.ga_engine.evolve_generation()

            # Get current state
            best_solution, best_fitness = self.ga_engine.get_best_solution()
            second_best = self.ga_engine.get_second_best_solution()

            # Render
            self.visualizer.render(
                self.cities_locations,
                best_solution,
                second_best,
                self.ga_engine.best_fitness_history,
                self.ga_engine.generation,
                best_fitness,
            )

        # Cleanup
        self.visualizer.cleanup()
        print(f"\nFinal Result:")
        print(f"Generations: {self.ga_engine.generation}")
        print(f"Best Fitness: {best_fitness:.2f}")
