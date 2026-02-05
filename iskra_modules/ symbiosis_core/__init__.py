"""
ISKRA-4 · SYMBIOSIS-CORE v5.4
Надсистема симбиоза для эволюции ISKRA-4
"""

from .symbiosis_core import SymbiosisCore
from .aladdin_shadow import AladdinShadow, AladdinShadowSync
from .symbiosis_api import (
    symbiosis_bp,
    get_symbiosis_engine,
    get_aladdin_shadow,
    get_session_manager,
    get_emergency_protocol,
    init_app
)
from .iskra_integration import ISKRAAdapter
from .session_manager import SessionManager
from .emergency_protocol import EmergencyProtocol

__version__ = "5.4"
__author__ = "ISKRA-4"
__description__ = "Сефиротический симбионт с теневой аналитикой"

__all__ = [
    'SymbiosisCore',
    'AladdinShadow',
    'AladdinShadowSync',
    'symbiosis_bp',
    'ISKRAAdapter',
    'SessionManager',
    'EmergencyProtocol',
    'get_symbiosis_engine',
    'get_aladdin_shadow',
    'get_session_manager',
    'get_emergency_protocol',
    'init_app'
]
