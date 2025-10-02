"""
TSP Solver - Main Entry Point.

This is the main entry point for the TSP Solver application.
It initializes the configuration and starts the solver.
"""

import sys

from config import GAConfig, VisualizationConfig
from solver import TSPSolver


def main() -> None:
    """
    Main entry point for the TSP Solver application.

    Initializes configurations and starts the solver.
    """
    # Create configurations
    ga_config = GAConfig(
        population_size=150, mutation_probability=0.7, elite_size=10, tournament_size=5
    )

    vis_config = VisualizationConfig(
        width=1500, height=800, node_radius=10, fps=30, plot_x_offset=450
    )

    # Create and run solver
    solver = TSPSolver(ga_config, vis_config)
    solver.setup_problem()
    solver.run()

    sys.exit(0)


if __name__ == "__main__":
    main()
