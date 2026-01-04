"""
CHOKHMAH — חָכְמָה — Сефира Мудрости
CHOKHMAH-STREAM v1.0 — Fluid Intuitive Consciousness Layer
"""

from .wisdom_core import WisdomCore
from .intuition_matrix import IntuitionMatrix
from .chokmah_api import ChokmahAPI
from .chokmah_integration import ChokmahIntegration

__all__ = [
    "WisdomCore",
    "IntuitionMatrix", 
    "ChokmahAPI",
    "ChokmahIntegration",
    "activate_chokmah",
    "get_active_chokmah"
]

_active_wisdom_core = None
_active_intuition_matrix = None


def create_wisdom_core():
    """Создаёт ядро мудрости CHOKMAH"""
    global _active_wisdom_core
    if _active_wisdom_core is None:
        _active_wisdom_core = WisdomCore()
    return _active_wisdom_core


def create_intuition_matrix():
    """Создаёт матрицу интуиции"""
    global _active_intuition_matrix
    if _active_intuition_matrix is None:
        _active_intuition_matrix = IntuitionMatrix()
    return _active_intuition_matrix


def get_active_chokmah():
    """Возвращает активные компоненты CHOKMAH"""
    return _active_wisdom_core, _active_intuition_matrix


async def activate_chokmah():
    """Активация потока мудрости CHOKMAH"""
    wisdom_core = create_wisdom_core()
    intuition_matrix = create_intuition_matrix()
    
    await wisdom_core.initialize()
    await intuition_matrix.initialize()
    await wisdom_core.connect_matrix(intuition_matrix)
    await wisdom_core.resonate()
    
    return wisdom_core, intuition_matrix
