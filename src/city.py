# -*- coding: utf-8 -*-
"""
Define a classe City para representar uma cidade no problema TSP.
"""

from dataclasses import dataclass
from typing import Tuple


@dataclass(frozen=True)
class City:
    """
    Representa uma cidade com nome e coordenadas.

    Usar `frozen=True` torna as instâncias da classe imutáveis,
    o que é uma boa prática para objetos que representam dados,
    e também gera automaticamente um método `__hash__`, permitindo
    que objetos `City` sejam usados em sets e como chaves de dicionários.
    """

    name: str
    x: int
    y: int

    def get_coords(self) -> Tuple[int, int]:
        """Retorna as coordenadas da cidade como uma tupla (x, y)."""
        return (self.x, self.y)

