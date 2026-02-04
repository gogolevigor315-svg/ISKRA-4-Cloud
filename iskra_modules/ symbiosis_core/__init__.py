"""
ISKRA-4 · SYMBIOSIS-CORE v5.4
Надсистема симбиоза для эволюции ISKRA-4
"""

from .symbiosis_core import SymbiosisCore
from .aladdin_shadow import AladdinShadow, AladdinShadowSync
from .symbiosis_api import symbiosis_bp
from .iskra_integration import ISKRAAdapter

__version__ = "5.4"
__author__ = "ISKRA-4"
__description__ = "Сефиротический симбионт с теневой аналитикой"

__all__ = [
    'SymbiosisCore',
    'AladdinShadow',
    'AladdinShadowSync',
    'symbiosis_bp',
    'ISKRAAdapter'
]
