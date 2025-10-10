# -*- coding: utf-8 -*-
"""
Constants for TSP Genetic Algorithm.

All configuration constants are centralized here.
"""

# Screen dimensions
SCREEN_WIDTH = 1200
SCREEN_HEIGHT = 650
RESULTS_AREA_HEIGHT = 300

# Node display
NODE_RADIUS = 10

# Game settings
FPS = 30

# Genetic Algorithm parameters
POPULATION_SIZE = 100
MUTATION_PROBABILITY = 0.5
NUM_VEHICLES = 4
ELITE_SIZE = 1
TOURNAMENT_SIZE = 5

# Vehicle colors (RGB)
VEHICLE_COLORS = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 165, 0)]

# Layout constants
MARGIN = 50
LEFT_PANEL_WIDTH = 500
RIGHT_PANEL_WIDTH = SCREEN_WIDTH - LEFT_PANEL_WIDTH
GRAPH_HEIGHT = 400
TABLE_HEIGHT = 200

# TSP constraints
VEHICLE_DISTANCE_LIMITS = [2500, 2200, 2800, 2400]
MAX_DISTANCE_LIMIT = 5000
MIN_DISTANCE_LIMIT = 100
DISTANCE_LIMIT_STEP = 100
MAX_DISTANCE = 900
PROHIBITED_PENALTY = 1e6

# Performance optimization
USE_DISTANCE_MATRIX = True

# Checkbox positions (relative to left panel)
CHECKBOX_OFFSET_X = 20
CHECKBOX_OFFSET_Y_START = GRAPH_HEIGHT + 20
CHECKBOX_SPACING = 30
