# Multi-Vehicle TSP Solver Using Genetic Algorithm

This repository contains a Python implementation of a Traveling Salesman Problem (TSP) solver with multiple vehicles using a Genetic Algorithm (GA). The system uses clustering to divide cities among vehicles and optimizes routes simultaneously with real-time visualization.

## 🎯 Overview

The solver employs a Genetic Algorithm to iteratively evolve a population of candidate solutions towards optimal or near-optimal routes. The system supports:

- **Multiple Vehicles**: Up to 5 vehicles operating simultaneously
- **Intelligent Clustering**: Automatic city division using K-Means
- **Configurable Constraints**: Prohibited routes, priority cities, and distance limitations
- **Real-Time Visualization**: Interactive graphical interface with Pygame
- **Elitism**: Preservation of the best individual across generations

## 📁 Project Structure

```
src/
├── tsp.py                  # Main application with Pygame interface
├── genetic_algorithm.py    # Genetic algorithm operations (selection, crossover, mutation)
├── city.py                 # City class for city representation
├── draw_functions.py       # Drawing and visualization functions
├── benchmark_att48.py      # ATT48 benchmark dataset (48 cities)
├── llm_integration.py      # LLM integration for result analysis (optional)
└── ui.py                   # UI components (scroll area, markdown)
```

## 🚀 Key Features

### Genetic Algorithm
- **Population**: 100 individuals per vehicle
- **Elitism**: Best solution preserved per generation
- **Selection**: Tournament selection
- **Crossover**: Order Crossover (OX)
- **Mutation**: 50% rate with city swap
- **Initialization**: Nearest-neighbor heuristic + random population

### Operational Constraints
1. **Prohibited Routes**: Forbids specific routes between city pairs
2. **Priority City**: Forces priority visits to certain cities right after the depot
3. **Distance Limitation**: Adds mandatory stops at refueling stations when accumulated distance exceeds 900 units

### Interactive Interface
- **Left Panel**: Fitness evolution graph and vehicle information table
- **Right Panel**: Map visualization with color-coded routes per vehicle
- **Controls**: 
  - Checkboxes to enable/disable constraints
  - Inputs to configure number of vehicles (1-5) and cities (8-48)
  - EDIT and RESET buttons for real-time configuration

## 📦 Dependencies

```bash
numpy>=1.21.0
pygame>=2.0.0
matplotlib>=3.5.0
scikit-learn>=1.0.0
openai>=1.0.0           # Optional for LLM integration
python-dotenv>=1.0.0    # Optional for LLM integration
screeninfo>=0.8.1
```

## 🔧 Installation

1. Clone the repository:
```bash
git clone https://github.com/Gusta-Alves/tsp-genetic-algorithm.git
cd tsp-genetic-algorithm
```

2. Create and activate a virtual environment:
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## 🎮 How to Use

Run the main file:
```bash
python src/tsp.py
```

### Interface Controls

- **Checkboxes**: Click to enable/disable constraints (available only for 4 vehicles and 48 cities)
- **EDIT Button**: Activates edit mode to modify number of vehicles and cities
- **RESET Button**: Restores default settings (4 vehicles, 48 cities, no constraints)
- **Q Key**: Closes the application
- **X Click**: Closes the application

### Custom Configuration

During edit mode, you can:
- Set 1 to 5 vehicles
- Set 8 to 48 cities (minimum = vehicles × 2)
- Apply or remove operational constraints

## 📊 Dataset

The project uses the **ATT48** benchmark with 48 cities, a classic TSP problem widely used for testing optimization algorithms.

## 🎨 Visualization

- Each vehicle has a unique color
- The central depot is calculated as the centroid of all cities
- Routes are drawn in real-time showing the algorithm's evolution
- Fitness graph shows convergence over generations
- Table displays distance, number of cities, and last change per vehicle

## 🤖 LLM Integration (Optional)

The system can integrate with language models (OpenAI) for result analysis and formatting. To activate:

1. Set `_isllmintegrationEnabled = True` in `tsp.py`
2. Create a `.env` file with your API key:
```
OPENAI_API_KEY=your_key_here
```

## 📈 Configurable Parameters

Edit the constants in `tsp.py`:

```python
POPULATION_SIZE = 100        # Population size per vehicle
MUTATION_PROBABILITY = 0.5   # Mutation rate
NUM_VEHICLES = 4             # Number of vehicles (1-5)
NUM_CITIES = 48              # Number of cities (8-48)
MAX_DISTANCE = 900           # Maximum distance without refueling
MAX_GENERATIONS = 100        # Maximum number of generations
```

## 🏗️ Architecture

### City Class
```python
@dataclass(frozen=True)
class City:
    name: str
    x: int
    y: int
```
Immutable representation of cities with automatic hashing for use in sets and dictionaries.

### Main Flow
1. **Initialization**: Loads cities from ATT48 benchmark and applies K-Means clustering
2. **Preparation**: Creates initial populations for each vehicle using heuristics
3. **Evolution**: For each generation, applies selection, crossover, and mutation
4. **Elitism**: Preserves the 5 best individuals from each population
5. **Visualization**: Updates interface in real-time at 30 FPS
6. **Constraints**: Applies penalties and adjustments according to active configurations

## 🎓 Genetic Algorithm Concepts

- **Fitness**: Sum of route distances (lower is better)
- **Tournament Selection**: Chooses the best individuals from random subgroups
- **Order Crossover (OX)**: Preserves relative order of cities from parents
- **Swap Mutation**: Random exchange of positions between two cities
- **Elitism**: Ensures that the best solutions are not lost

## 🤝 Contributions

Contributions are welcome! Feel free to:
- Report bugs
- Suggest improvements
- Add new features
- Improve documentation

## 📄 License

This project is licensed under the [MIT License](LICENSE).

## 👥 Authors

Developed as a combinatorial optimization project using genetic algorithms and clustering techniques for solving multi-vehicle TSP.

---

**Note**: This is an educational project that demonstrates the application of genetic algorithms to route optimization problems with realistic operational constraints.