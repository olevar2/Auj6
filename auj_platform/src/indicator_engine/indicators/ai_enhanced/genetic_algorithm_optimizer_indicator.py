"""
Genetic Algorithm Optimizer for Trading Strategy Parameter Optimization

This module implements a sophisticated genetic algorithm optimizer designed to find optimal
parameters for trading strategies. It uses evolutionary computation principles to evolve
trading strategy parameters that maximize profitability while managing risk.

Key Features:
    - Multi-objective optimization (return, risk, drawdown)
    - Adaptive population management
    - Advanced selection mechanisms
    - Dynamic mutation rates
    - Constraint handling
    - Convergence detection
"""

import asyncio
import random
import logging
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
from abc import ABC, abstractmethod

from ..base.base_indicator import BaseIndicator

# Configure logging
logger = logging.getLogger(__name__)

@dataclass
class OptimizationTarget:
    """Defines an optimization target parameter."""
    parameter_name: str
    parameter_type: str  # 'continuous', 'integer', 'categorical'
    min_value: float
    max_value: float
    categorical_values: Optional[List] = None
    default_value: Optional[float] = None

@dataclass
class Individual:
    """Represents an individual in the genetic algorithm population."""
    genes: np.ndarray
    fitness: float = 0.0
    age: int = 0
    generation: int = 0
    parent_ids: Tuple[int, int] = (0, 0)

class FitnessFunction(ABC):
    """Abstract base class for fitness functions."""

    @abstractmethod
    def evaluate(self, individual: Individual, data: pd.DataFrame, **kwargs) -> float:
        """Evaluate the fitness of an individual."""
        pass

class GeneticAlgorithmOptimizer(BaseIndicator):
    """
    Advanced Genetic Algorithm Optimizer for trading strategy parameter optimization.

    This optimizer uses evolutionary computation to find optimal parameter combinations
    for trading strategies. It supports multiple objective optimization, adaptive
    parameters, and sophisticated selection mechanisms.
    """

    def __init__(self, symbol: str = "EURUSD", timeframe: str = "H1", **kwargs):
        super().__init__(symbol=symbol, timeframe=timeframe)

        # Core genetic algorithm parameters
        self.parameters = {
            'population_size': kwargs.get('population_size', 100),
            'max_generations': kwargs.get('max_generations', 50),
            'mutation_rate': kwargs.get('mutation_rate', 0.1),
            'crossover_rate': kwargs.get('crossover_rate', 0.8),
            'elitism_rate': kwargs.get('elitism_rate', 0.2),
            'tournament_size': kwargs.get('tournament_size', 5),
            'convergence_threshold': kwargs.get('convergence_threshold', 0.001),
            'diversity_threshold': kwargs.get('diversity_threshold', 0.1),
            'constraint_handling': kwargs.get('constraint_handling', True),
            'fitness_function': kwargs.get('fitness_function', None)
        }

        # Optimization targets and population
        self.optimization_targets: List[OptimizationTarget] = []
        self.population: List[Individual] = []
        self.best_individuals: List[Individual] = []

        # Algorithm state
        self.current_generation = 0
        self.is_optimizing = False
        self.optimization_results = {}

        # Adaptive parameters
        self.adaptive_mutation_rate = self.parameters['mutation_rate']
        self.adaptive_population_size = self.parameters['population_size']

        # Performance tracking
        self.fitness_history: List[List[float]] = []
        self.diversity_history: List[float] = []

        # Configure default optimization targets if none provided
        self._setup_default_targets()

    def _setup_default_targets(self):
        """Setup default optimization targets for common trading parameters."""
        default_targets = [
            OptimizationTarget(
                parameter_name='period',
                parameter_type='integer',
                min_value=5,
                max_value=100,
                default_value=20
            ),
            OptimizationTarget(
                parameter_name='threshold',
                parameter_type='continuous',
                min_value=0.1,
                max_value=5.0,
                default_value=1.0
            ),
            OptimizationTarget(
                parameter_name='stop_loss',
                parameter_type='continuous',
                min_value=0.01,
                max_value=0.1,
                default_value=0.02
            ),
            OptimizationTarget(
                parameter_name='take_profit',
                parameter_type='continuous',
                min_value=0.02,
                max_value=0.2,
                default_value=0.05
            )
        ]

        if not self.optimization_targets:
            self.optimization_targets = default_targets

    async def calculate(self, data: pd.DataFrame, **kwargs) -> Dict[str, Any]:
        """
        Main calculation method for genetic algorithm optimization.

        Args:
            data: Market data for optimization
            **kwargs: Additional parameters including market_regime

        Returns:
            Dict containing optimization results and best solutions
        """
        try:
            market_regime = kwargs.get('market_regime', 'trending_bull')

            if len(data) < 50:
                logger.warning("Insufficient data for genetic algorithm optimization")
                return self._get_default_output()

            # Initialize or continue optimization
            if not self.is_optimizing:
                self._initialize_optimization()

            # Run genetic algorithm
            results = await self._run_genetic_algorithm(data, market_regime)

            return results

        except Exception as e:
            logger.error(f"Error in genetic algorithm calculation: {e}")
            return self._handle_calculation_error(e)

    def _initialize_optimization(self):
        """Initialize the genetic algorithm optimization process."""
        self.is_optimizing = True
        self.current_generation = 0
        self.population = []
        self.best_individuals = []
        self.fitness_history = []
        self.diversity_history = []

        # Reset adaptive parameters
        self.adaptive_mutation_rate = self.parameters['mutation_rate']
        self.adaptive_population_size = self.parameters['population_size']

        # Initialize population
        self._initialize_population()

        logger.info(f"Initialized genetic algorithm with {len(self.population)} individuals")

    def _initialize_population(self):
        """Initialize the population with random individuals."""
        self.population = []

        for i in range(self.adaptive_population_size):
            genes = self._create_random_genes()
            individual = Individual(
                genes=genes,
                fitness=0.0,
                age=0,
                generation=0,
                parent_ids=(0, 0)
            )
            self.population.append(individual)

    def _create_random_genes(self) -> np.ndarray:
        """Create random genes for an individual."""
        genes = []

        for target in self.optimization_targets:
            if target.parameter_type == 'continuous':
                gene = random.uniform(target.min_value, target.max_value)
            elif target.parameter_type == 'integer':
                gene = random.randint(int(target.min_value), int(target.max_value))
            elif target.parameter_type == 'categorical':
                gene = random.choice(target.categorical_values)
            else:
                gene = target.default_value or 0.0

            genes.append(gene)

        return np.array(genes)

    async def _run_genetic_algorithm(self, data: pd.DataFrame, market_regime: str) -> Dict[str, Any]:
        """Run the main genetic algorithm optimization loop."""
        convergence_count = 0
        best_fitness = float('-inf')

        for generation in range(self.parameters['max_generations']):
            self.current_generation = generation

            # Evaluate fitness for all individuals
            await self._evaluate_population(data, market_regime)

            # Track fitness statistics
            generation_fitnesses = [ind.fitness for ind in self.population]
            self.fitness_history.append(generation_fitnesses)

            # Calculate diversity
            diversity = self._calculate_population_diversity()
            self.diversity_history.append(diversity)

            # Update best individual
            current_best = max(self.population, key=lambda x: x.fitness)
            if current_best.fitness > best_fitness:
                best_fitness = current_best.fitness
                convergence_count = 0
            else:
                convergence_count += 1

            # Check convergence
            if convergence_count >= 10:
                logger.info(f"Converged after {generation} generations")
                break

            # Adapt parameters based on performance
            self._adapt_parameters(generation, diversity)

            # Create next generation
            self._create_next_generation()

        # Compile results
        return self._compile_optimization_results(market_regime)

    async def _evaluate_population(self, data: pd.DataFrame, market_regime: str):
        """Evaluate fitness for all individuals in the population."""
        for individual in self.population:
            individual.fitness = await self._evaluate_individual(individual, data, market_regime)

    async def _evaluate_individual(self, individual: Individual, data: pd.DataFrame, market_regime: str) -> float:
        """Evaluate the fitness of a single individual."""
        try:
            # Convert genes to parameter dictionary
            parameters = self._genes_to_parameters(individual.genes)

            # Calculate fitness based on trading performance
            fitness = await self._calculate_trading_fitness(parameters, data, market_regime)

            return fitness

        except Exception as e:
            logger.error(f"Error evaluating individual: {e}")
            return 0.0

    def _genes_to_parameters(self, genes: np.ndarray) -> Dict[str, Any]:
        """Convert genes array to parameter dictionary."""
        parameters = {}

        for i, target in enumerate(self.optimization_targets):
            if i < len(genes):
                if target.parameter_type == 'integer':
                    parameters[target.parameter_name] = int(genes[i])
                else:
                    parameters[target.parameter_name] = genes[i]

        return parameters

    async def _calculate_trading_fitness(self, parameters: Dict[str, Any],
                                       data: pd.DataFrame, market_regime: str) -> float:
        """Calculate fitness based on trading strategy performance."""
        try:
            # Simple fitness calculation based on parameters
            # In a real implementation, this would run a backtest

            period = parameters.get('period', 20)
            threshold = parameters.get('threshold', 1.0)
            stop_loss = parameters.get('stop_loss', 0.02)
            take_profit = parameters.get('take_profit', 0.05)

            # Calculate simple moving average signals
            if 'close' in data.columns and len(data) >= period:
                sma = data['close'].rolling(window=period).mean()
                signals = (data['close'] > sma * (1 + threshold/100)).astype(int)

                # Simple profit calculation
                returns = data['close'].pct_change().fillna(0)
                strategy_returns = signals.shift(1) * returns

                # Risk-adjusted return
                total_return = strategy_returns.sum()
                volatility = strategy_returns.std()

                if volatility > 0:
                    sharpe_ratio = total_return / volatility
                else:
                    sharpe_ratio = 0

                # Penalize extreme parameters
                penalty = 0
                if stop_loss > 0.05:
                    penalty += 0.1
                if take_profit < 0.03:
                    penalty += 0.1

                fitness = sharpe_ratio - penalty

                return max(fitness, 0.0)

            return 0.0

        except Exception as e:
            logger.error(f"Error calculating trading fitness: {e}")
            return 0.0

    def _calculate_population_diversity(self) -> float:
        """Calculate genetic diversity of the population."""
        if len(self.population) < 2:
            return 0.0

        total_distance = 0.0
        count = 0

        for i in range(len(self.population)):
            for j in range(i + 1, len(self.population)):
                distance = np.linalg.norm(self.population[i].genes - self.population[j].genes)
                total_distance += distance
                count += 1

        return total_distance / count if count > 0 else 0.0

    def _adapt_parameters(self, generation: int, diversity: float):
        """Adapt algorithm parameters based on current performance."""
        # Adaptive mutation rate
        if diversity < self.parameters['diversity_threshold']:
            self.adaptive_mutation_rate = min(0.3, self.adaptive_mutation_rate * 1.1)
        else:
            self.adaptive_mutation_rate = max(0.01, self.adaptive_mutation_rate * 0.95)

        # Adaptive population size
        if generation > 20 and diversity < 0.05:
            self.adaptive_population_size = min(200, int(self.adaptive_population_size * 1.1))

    def _create_next_generation(self):
        """Create the next generation through selection, crossover, and mutation."""
        # Sort population by fitness
        self.population.sort(key=lambda x: x.fitness, reverse=True)

        # Elite selection
        elite_count = max(1, int(len(self.population) * self.parameters['elitism_rate']))
        new_population = self.population[:elite_count].copy()

        # Create offspring
        while len(new_population) < self.adaptive_population_size:
            # Tournament selection
            parent1 = self._tournament_selection()
            parent2 = self._tournament_selection()

            # Crossover
            if random.random() < self.parameters['crossover_rate']:
                offspring1, offspring2 = self._crossover(parent1, parent2)
            else:
                offspring1, offspring2 = parent1, parent2

            # Mutation
            offspring1 = self._mutate(offspring1)
            offspring2 = self._mutate(offspring2)

            new_population.extend([offspring1, offspring2])

        # Trim to desired population size
        self.population = new_population[:self.adaptive_population_size]

    def _tournament_selection(self) -> Individual:
        """Select an individual using tournament selection."""
        tournament_size = min(self.parameters['tournament_size'], len(self.population))
        tournament = random.sample(self.population, tournament_size)
        return max(tournament, key=lambda x: x.fitness)

    def _crossover(self, parent1: Individual, parent2: Individual) -> Tuple[Individual, Individual]:
        """Perform crossover between two parents."""
        # Single-point crossover
        crossover_point = random.randint(1, len(parent1.genes) - 1)

        offspring1_genes = np.concatenate([
            parent1.genes[:crossover_point],
            parent2.genes[crossover_point:]
        ])

        offspring2_genes = np.concatenate([
            parent2.genes[:crossover_point],
            parent1.genes[crossover_point:]
        ])

        offspring1 = Individual(
            genes=offspring1_genes,
            fitness=0.0,
            age=0,
            generation=self.current_generation + 1,
            parent_ids=(id(parent1), id(parent2))
        )

        offspring2 = Individual(
            genes=offspring2_genes,
            fitness=0.0,
            age=0,
            generation=self.current_generation + 1,
            parent_ids=(id(parent1), id(parent2))
        )

        return offspring1, offspring2

    def _mutate(self, individual: Individual) -> Individual:
        """Apply mutation to an individual."""
        mutated_genes = individual.genes.copy()

        for i in range(len(mutated_genes)):
            if random.random() < self.adaptive_mutation_rate:
                target = self.optimization_targets[i]

                if target.parameter_type == 'continuous':
                    # Gaussian mutation
                    mutation_strength = (target.max_value - target.min_value) * 0.1
                    mutated_genes[i] += random.gauss(0, mutation_strength)
                    mutated_genes[i] = np.clip(mutated_genes[i], target.min_value, target.max_value)

                elif target.parameter_type == 'integer':
                    # Random reset mutation
                    mutated_genes[i] = random.randint(int(target.min_value), int(target.max_value))

                elif target.parameter_type == 'categorical':
                    # Random categorical mutation
                    mutated_genes[i] = random.choice(target.categorical_values)

        return Individual(
            genes=mutated_genes,
            fitness=0.0,
            age=individual.age + 1,
            generation=self.current_generation + 1,
            parent_ids=individual.parent_ids
        )

    def _compile_optimization_results(self, market_regime: str) -> Dict[str, Any]:
        """Compile final optimization results."""
        # Get best individual
        best_individual = max(self.population, key=lambda x: x.fitness)
        best_parameters = self._genes_to_parameters(best_individual.genes)

        # Get top solutions
        sorted_population = sorted(self.population, key=lambda x: x.fitness, reverse=True)
        top_solutions = []
        for i, individual in enumerate(sorted_population[:5]):
            solution = {
                'rank': i + 1,
                'parameters': self._genes_to_parameters(individual.genes),
                'fitness': individual.fitness,
                'generation': individual.generation
            }
            top_solutions.append(solution)

        # Calculate statistics
        fitness_values = [ind.fitness for ind in self.population]
        diversity = self._calculate_population_diversity()

        results = {
            'best_solution': best_parameters,
            'best_solutions': top_solutions,
            'optimization_results': {
                'generations_completed': self.current_generation + 1,
                'final_population_size': len(self.population),
                'convergence_achieved': self.current_generation < self.parameters['max_generations'] - 1
            },
            'diversity_analysis': {
                'genetic_diversity': diversity,
                'diversity_history': self.diversity_history
            },
            'convergence_analysis': {
                'status': 'converged' if self.current_generation < self.parameters['max_generations'] - 1 else 'max_generations',
                'fitness_history': self.fitness_history
            },
            'fitness_statistics': {
                'best_fitness': best_individual.fitness,
                'mean_fitness': np.mean(fitness_values),
                'std_fitness': np.std(fitness_values),
                'median_fitness': np.median(fitness_values)
            },
            'parameter_recommendations': self._generate_parameter_recommendations(best_parameters),
            'quality_metrics': {
                'confidence': min(1.0, best_individual.fitness / 2.0),
                'overall_quality': self._calculate_solution_quality(best_individual)
            },
            'current_generation': self.current_generation,
            'population_size': len(self.population),
            'market_regime': market_regime,
            'optimization_confidence': self._calculate_optimization_confidence(),
            'convergence_status': 'converged' if self.current_generation < self.parameters['max_generations'] - 1 else 'max_generations_reached',
            'best_fitness': best_individual.fitness
        }

        return results

    def _generate_parameter_recommendations(self, best_parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Generate parameter recommendations based on optimization results."""
        recommendations = {}

        for param_name, value in best_parameters.items():
            target = next((t for t in self.optimization_targets if t.parameter_name == param_name), None)
            if target:
                # Calculate relative position within bounds
                relative_pos = (value - target.min_value) / (target.max_value - target.min_value)

                if relative_pos < 0.2:
                    recommendation = "Consider increasing this parameter"
                elif relative_pos > 0.8:
                    recommendation = "Consider decreasing this parameter"
                else:
                    recommendation = "Parameter appears well-optimized"

                recommendations[param_name] = {
                    'value': value,
                    'recommendation': recommendation,
                    'confidence': min(1.0, abs(0.5 - relative_pos) * 2)
                }

        return recommendations

    def _calculate_solution_quality(self, individual: Individual) -> float:
        """Calculate overall quality score for a solution."""
        fitness_score = min(1.0, individual.fitness / 2.0)
        age_score = min(1.0, individual.age / 10.0)

        return (fitness_score * 0.8) + (age_score * 0.2)

    def _calculate_optimization_confidence(self) -> float:
        """Calculate confidence in the optimization results."""
        if not self.fitness_history:
            return 0.0

        # Check for improvement over generations
        if len(self.fitness_history) > 5:
            recent_best = [max(generation) for generation in self.fitness_history[-5:]]
            improvement = (recent_best[-1] - recent_best[0]) / max(abs(recent_best[0]), 0.001)

            # Check for convergence
            convergence_score = 1.0 - min(1.0, abs(improvement) * 10)

            # Check diversity maintenance
            diversity_score = min(1.0, self.diversity_history[-1] / 0.5) if self.diversity_history else 0.0

            return (convergence_score * 0.7) + (diversity_score * 0.3)

        return 0.5

    def _get_default_output(self) -> Dict[str, Any]:
        """Return default output when insufficient data or error."""
        return {
            'best_solution': {},
            'best_solutions': [],
            'optimization_results': {'generations_completed': 0},
            'diversity_analysis': {'genetic_diversity': 0.0},
            'convergence_analysis': {'status': 'not_started'},
            'fitness_statistics': {'best_fitness': 0.0},
            'parameter_recommendations': {},
            'quality_metrics': {'confidence': 0.0, 'overall_quality': 0.0},
            'current_generation': 0,
            'population_size': 0,
            'market_regime': 'unknown',
            'optimization_confidence': 0.0,
            'convergence_status': 'not_started',
            'best_fitness': 0.0
        }

    def _handle_calculation_error(self, error: Exception) -> Dict[str, Any]:
        """Handle calculation errors gracefully."""
        logger.error(f"Genetic algorithm calculation error: {error}")
        return self._get_default_output()

    def reset_optimization(self):
        """Reset optimization state."""
        self.population = []
        self.best_individuals = []
        self.fitness_history = []
        self.diversity_history = []
        self.current_generation = 0
        self.is_optimizing = False
        self.optimization_results = {}

    def get_optimization_status(self) -> Dict[str, Any]:
        """Get current optimization status."""
        return {
            'is_optimizing': self.is_optimizing,
            'current_generation': self.current_generation,
            'population_size': len(self.population),
            'best_fitness': max([ind.fitness for ind in self.population]) if self.population else 0.0,
            'diversity': self._calculate_population_diversity()
        }

    def set_optimization_targets(self, targets: List[OptimizationTarget]):
        """Set custom optimization targets."""
        self.optimization_targets = targets
        if self.is_optimizing:
            logger.warning("Changing targets during optimization - consider resetting")

    def add_optimization_target(self, target: OptimizationTarget):
        """Add a new optimization target."""
        self.optimization_targets.append(target)
        if self.is_optimizing:
            logger.warning("Adding target during optimization - consider resetting")

    def get_parameter_bounds(self) -> Dict[str, Tuple[float, float]]:
        """Get parameter bounds for all optimization targets."""
        bounds = {}
        for target in self.optimization_targets:
            bounds[target.parameter_name] = (target.min_value, target.max_value)
        return bounds

    def validate_parameters(self, parameters: Dict[str, Any]) -> bool:
        """Validate if parameters are within bounds."""
        for target in self.optimization_targets:
            if target.parameter_name in parameters:
                value = parameters[target.parameter_name]
                if not (target.min_value <= value <= target.max_value):
                    return False
        return True
