# -*- coding: utf-8 -*-
"""
Genetic Algorithm Engine module for TSP Solver.

This module contains the core genetic algorithm engine that handles
population evolution through selection, crossover, and mutation operations.
"""

from typing import List, Tuple, Callable

from genetic_algorithm import (
    mutate,
    order_crossover,
    sort_population,
    tournament_selection,
)


class GeneticAlgorithmEngine:
    """Core genetic algorithm engine responsible for evolution operations."""

    def __init__(
        self,
        population_size: int,
        mutation_probability: float,
        elite_size: int,
        tournament_size: int,
        fitness_calculator: Callable[[List[Tuple[float, float]]], float],
    ):
        """
        Initialize the genetic algorithm engine.

        Args:
            population_size: Size of the population
            mutation_probability: Probability of mutation
            elite_size: Number of elite individuals to preserve
            tournament_size: Size of tournament for selection
            fitness_calculator: Function to calculate fitness
        """
        self.population_size = population_size
        self.mutation_probability = mutation_probability
        self.elite_size = elite_size
        self.tournament_size = tournament_size
        self.fitness_calculator = fitness_calculator

        self.population = []
        self.population_fitness = []
        self.generation = 0
        self.best_fitness_history = []
        self.best_solution_history = []

    def set_population(self, population: List[List[Tuple[float, float]]]) -> None:
        """
        Set the current population.

        Args:
            population: List of individuals (routes) to set as current population
        """
        self.population = population

    def evolve_generation(self) -> None:
        """
        Evolve the population by one generation.

        Performs the following steps:
        1. Calculate fitness for all individuals
        2. Sort population by fitness
        3. Record best solution
        4. Create new population through selection, crossover, and mutation
        """
        self.generation += 1

        # Evaluate fitness
        self.population_fitness = [
            self.fitness_calculator(individual) for individual in self.population
        ]

        # Sort by fitness (best first)
        self.population, self.population_fitness = sort_population(
            self.population, self.population_fitness
        )

        # Record best solution
        best_fitness = self.population_fitness[0]
        best_solution = self.population[0]
        self.best_fitness_history.append(best_fitness)
        self.best_solution_history.append(best_solution)

        # Create new generation
        self._create_new_generation()

    def _create_new_generation(self) -> None:
        """
        Create new generation using elitism, selection, crossover, and mutation.

        Implements elitism by preserving the best individuals, then fills the
        rest of the population through tournament selection, crossover, and mutation.
        """
        new_population = list(self.population[: self.elite_size])

        while len(new_population) < self.population_size:
            # Tournament selection
            parent1 = tournament_selection(
                self.population,
                self.population_fitness,
                tournament_size=self.tournament_size,
            )
            parent2 = tournament_selection(
                self.population,
                self.population_fitness,
                tournament_size=self.tournament_size,
            )

            # Crossover
            child1, child2 = order_crossover(parent1, parent2)

            # Mutation
            child1 = mutate(child1, self.mutation_probability)
            child2 = mutate(child2, self.mutation_probability)

            # Add to new population
            new_population.append(child1)
            if len(new_population) < self.population_size:
                new_population.append(child2)

        self.population = new_population

    def get_best_solution(self) -> Tuple[List[Tuple[float, float]], float]:
        """
        Get the best solution found so far.

        Returns:
            Tuple containing the best route and its fitness value
        """
        if not self.best_solution_history:
            return [], float("inf")
        return self.best_solution_history[-1], self.best_fitness_history[-1]

    def get_second_best_solution(self) -> List[Tuple[float, float]]:
        """
        Get the second best solution from current population.

        Returns:
            The second best route in the current population
        """
        if len(self.population) > 1:
            return self.population[1]
        return self.population[0] if self.population else []
