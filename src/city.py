# -*- coding: utf-8 -*-
"""
Define a classe City para representar uma cidade no problema TSP.
"""

from typing import Tuple


class City:
    """Representa uma cidade com nome e coordenadas."""

    def __init__(self, name: str, x: int, y: int):
        self.name = name
        self.x = x
        self.y = y

    def get_coords(self) -> Tuple[int, int]:
        """Retorna as coordenadas da cidade como uma tupla."""
        return (self.x, self.y)

    def __repr__(self) -> str:
        """Representação em string do objeto City."""
        return f"City(name='{self.name}', x={self.x}, y={self.y})"

    def __eq__(self, other):
        """Verifica a igualdade baseada no nome e coordenadas."""
        return isinstance(other, City) and self.name == other.name and self.get_coords() == other.get_coords()

    def __hash__(self):
        """Gera um hash para o objeto, permitindo seu uso em sets e dicts."""
        return hash((self.name, self.x, self.y))