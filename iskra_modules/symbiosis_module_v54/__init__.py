# SYMBIOSIS-CORE v5.4 - ИНИЦИАЛИЗАЦИЯ ПАКЕТА
from .symbiosis_api import symbiosis_bp
from .symbiosis_core import SymbiosisCore
from .iskra_integration import ISKRAIntegration
from .session_manager import SessionManager
from .aladdin_shadow import AladdinShadow
from .emergency_protocol import EmergencyProtocol

__all__ = [
    'symbiosis_bp',
    'SymbiosisCore',
    'ISKRAIntegration',
    'SessionManager',
    'AladdinShadow',
    'EmergencyProtocol'
]

print("✅ SYMBIOSIS-CORE v5.4 загружен (symbiosis_module_v54)")
