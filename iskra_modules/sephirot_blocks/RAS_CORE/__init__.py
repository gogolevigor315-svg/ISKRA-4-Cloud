"""
ИНИЦИАЛИЗАЦИЯ RAS-CORE v4.1
Модуль сефиротического внимания с золотым углом устойчивости 14.4°
Версия 4.1.1 - Исправлены импорты и обработка отсутствующих классов
"""

from .constants import (
    GOLDEN_STABILITY_ANGLE,
    GOLDEN_STABILITY_TOLERANCE,
    MAX_REFLECTION_DEPTH,
    calculate_stability_factor,
    angle_to_priority,
    calculate_angle_boost,
    normalize_focus_vector,
    get_stability_level,
    calculate_composite_stability,
    SEPHIROTIC_TARGETS,
    DEFAULT_FOCUS_PATTERNS,
    FOCUS_VECTORS,
    PRIORITY_THRESHOLDS,
    STABILITY_THRESHOLDS,
    SLO_TARGETS,
    METRICS_WINDOW_SIZE,
    TRIAD_IDEAL_VALUES,
    TRIAD_BALANCE_THRESHOLD,
    REFLECTION_CONFIG
)

# ================================================================
# 1. RASConfig (ДОБАВЛЕНО - ОТСУТСТВОВАЛО В ЭКСПОРТЕ!)
# ================================================================

class RASConfig:
    """Конфигурация RAS-CORE системы с поддержкой угла 14.4°"""
    
    def __init__(
        self,
        stability_angle: float = 14.4,
        reflection_cycle_ms: int = 144,
        enable_self_reflection: bool = True,
        max_concurrent_signals: int = 10,
        triad_balancing_enabled: bool = True,
        personality_coherence_threshold: float = 0.7
    ):
        self.stability_angle = stability_angle
        self.reflection_cycle_ms = reflection_cycle_ms
        self.enable_self_reflection = enable_self_reflection
        self.max_concurrent_signals = max_concurrent_signals
        self.triad_balancing_enabled = triad_balancing_enabled
        self.personality_coherence_threshold = personality_coherence_threshold
        
    def to_dict(self) -> dict:
        """Преобразование конфигурации в словарь"""
        return {
            "stability_angle": self.stability_angle,
            "reflection_cycle_ms": self.reflection_cycle_ms,
            "enable_self_reflection": self.enable_self_reflection,
            "max_concurrent_signals": self.max_concurrent_signals,
            "triad_balancing_enabled": self.triad_balancing_enabled,
            "personality_coherence_threshold": self.personality_coherence_threshold
        }
    
    @classmethod
    def from_dict(cls, config_dict: dict) -> 'RASConfig':
        """Создание конфигурации из словаря"""
        return cls(**config_dict)

# ================================================================
# 2. УСЛОВНЫЕ ИМПОРТЫ ИЗ ras_core_v4_1.py
# ================================================================

# Базовые классы (должны существовать)
try:
    from .ras_core_v4_1 import EnhancedRASCore, RASSignal
    ENHANCED_RAS_CORE_AVAILABLE = True
    print("[RAS-CORE] ✅ EnhancedRASCore и RASSignal загружены")
except ImportError as e:
    print(f"[RAS-CORE] ⚠️  Ошибка загрузки EnhancedRASCore/RASSignal: {e}")
    
    # Fallback реализации
    class EnhancedRASCore:
        def __init__(self, config=None):
            self.config = config or RASConfig()
            self.active = False
            
        def activate(self):
            self.active = True
            return {"status": "activated", "angle": self.config.stability_angle}
    
    class RASSignal:
        def __init__(self, data, priority=0.5):
            self.data = data
            self.priority = priority
            
    ENHANCED_RAS_CORE_AVAILABLE = False

# Классы очередей (могут отсутствовать)
try:
    from .ras_core_v4_1 import PrioritySignalQueue
    PRIORITY_QUEUE_AVAILABLE = True
except ImportError:
    PrioritySignalQueue = None
    PRIORITY_QUEUE_AVAILABLE = False
    print("[RAS-CORE] ⚠️  PrioritySignalQueue не найден, используем None")

try:
    from .ras_core_v4_1 import StabilityAwarePriorityQueue
    STABILITY_QUEUE_AVAILABLE = True
except ImportError:
    StabilityAwarePriorityQueue = None
    STABILITY_QUEUE_AVAILABLE = False
    print("[RAS-CORE] ⚠️  StabilityAwarePriorityQueue не найден, используем None")

# Менеджер конфигурации
try:
    from .ras_core_v4_1 import RASConfigManager
    CONFIG_MANAGER_AVAILABLE = True
except ImportError:
    RASConfigManager = None
    CONFIG_MANAGER_AVAILABLE = False
    print("[RAS-CORE] ⚠️  RASConfigManager не найден, используем None")

# Паттерны обучения
try:
    from .ras_core_v4_1 import PatternLearner
    PATTERN_LEARNER_AVAILABLE = True
except ImportError:
    PatternLearner = None
    PATTERN_LEARNER_AVAILABLE = False
    print("[RAS-CORE] ⚠️  PatternLearner не найден, используем None")

# Роутер
try:
    from .ras_core_v4_1 import AngleAwareSephiroticRouter
    ROUTER_AVAILABLE = True
except ImportError:
    AngleAwareSephiroticRouter = None
    ROUTER_AVAILABLE = False
    print("[RAS-CORE] ⚠️  AngleAwareSephiroticRouter не найден, используем None")

# Метрики
try:
    from .ras_core_v4_1 import StabilityMetricsCollector
    METRICS_COLLECTOR_AVAILABLE = True
except ImportError:
    StabilityMetricsCollector = None
    METRICS_COLLECTOR_AVAILABLE = False
    print("[RAS-CORE] ⚠️  StabilityMetricsCollector не найден, используем None")

# Движок саморефлексии
try:
    from .ras_core_v4_1 import SelfReflectionEngine
    REFLECTION_ENGINE_AVAILABLE = True
except ImportError:
    SelfReflectionEngine = None
    REFLECTION_ENGINE_AVAILABLE = False
    print("[RAS-CORE] ⚠️  SelfReflectionEngine не найден, используем None")

# Монитор триады
try:
    from .ras_core_v4_1 import TriadStabilityMonitor
    TRIAD_MONITOR_AVAILABLE = True
except ImportError:
    TriadStabilityMonitor = None
    TRIAD_MONITOR_AVAILABLE = False
    print("[RAS-CORE] ⚠️  TriadStabilityMonitor не найден, используем None")

# Mock шина
try:
    from .ras_core_v4_1 import EnhancedMockBus
    MOCK_BUS_AVAILABLE = True
except ImportError:
    EnhancedMockBus = None
    MOCK_BUS_AVAILABLE = False
    print("[RAS-CORE] ⚠️  EnhancedMockBus не найден, используем None")

# Другие классы из ras_core_v4_1
try:
    from .ras_core_v4_1 import (
        RASPattern,
        RASActivation,
        RASReflection,
        create_ras_core,
        initialize_ras_with_angle
    )
    RAS_PATTERN_AVAILABLE = True
except ImportError:
    RASPattern = None
    RASActivation = None
    RASReflection = None
    create_ras_core = None
    initialize_ras_with_angle = None
    RAS_PATTERN_AVAILABLE = False
    print("[RAS-CORE] ⚠️  Дополнительные классы из ras_core_v4_1 не найдены")

# Интеграционный класс RAS-CORE
try:
    from .ras_integration import (
        RASIntegration,
        create_ras_integration,
        integrate_ras_with_sephirot
    )
    RAS_INTEGRATION_AVAILABLE = True
    print("[RAS-CORE] ✅ RASIntegration загружен")
except ImportError as e:
    print(f"[RAS-CORE] ⚠️  Ошибка загрузки RASIntegration: {e}")
    RASIntegration = None
    create_ras_integration = None
    integrate_ras_with_sephirot = None
    RAS_INTEGRATION_AVAILABLE = False

# API компоненты
try:
    from .ras_api import RASAPI, create_ras_api
    RAS_API_AVAILABLE = True
    print("[RAS-CORE] ✅ RASAPI загружен")
except ImportError:
    RASAPI = None
    create_ras_api = None
    RAS_API_AVAILABLE = False
    print("[RAS-CORE] ⚠️  RASAPI не найден")

# ================================================================
# 3. ФУНКЦИЯ ДЛЯ ПРОВЕРКИ ГОТОВНОСТИ RAS-CORE
# ================================================================

def is_ras_core_ready() -> dict:
    """
    Проверяет готовность всех компонентов RAS-CORE
    Возвращает словарь со статусами
    """
    return {
        "ras_config": True,  # Всегда доступен (локальный класс)
        "enhanced_ras_core": ENHANCED_RAS_CORE_AVAILABLE,
        "priority_queue": PRIORITY_QUEUE_AVAILABLE,
        "stability_queue": STABILITY_QUEUE_AVAILABLE,
        "config_manager": CONFIG_MANAGER_AVAILABLE,
        "pattern_learner": PATTERN_LEARNER_AVAILABLE,
        "router": ROUTER_AVAILABLE,
        "metrics_collector": METRICS_COLLECTOR_AVAILABLE,
        "reflection_engine": REFLECTION_ENGINE_AVAILABLE,
        "triad_monitor": TRIAD_MONITOR_AVAILABLE,
        "mock_bus": MOCK_BUS_AVAILABLE,
        "ras_pattern": RAS_PATTERN_AVAILABLE,
        "ras_integration": RAS_INTEGRATION_AVAILABLE,
        "ras_api": RAS_API_AVAILABLE,
        "fully_ready": (
            True and  # RASConfig всегда доступен
            ENHANCED_RAS_CORE_AVAILABLE and
            RAS_INTEGRATION_AVAILABLE
        )
    }

# ================================================================
# 4. ЭКСПОРТ ДОСТУПНЫХ КОМПОНЕНТОВ
# ================================================================

__all__ = [
    # 1. КОНФИГУРАЦИЯ (ГЛАВНОЕ ИСПРАВЛЕНИЕ!)
    "RASConfig",  # ← ТЕПЕРЬ В ЭКСПОРТЕ!
    
    # 2. КОНСТАНТЫ (всегда доступны)
    "GOLDEN_STABILITY_ANGLE",
    "calculate_stability_factor",
    "angle_to_priority",
    "normalize_focus_vector",
    "get_stability_level",
    "calculate_composite_stability",
    "SEPHIROTIC_TARGETS",
    "DEFAULT_FOCUS_PATTERNS",
    "FOCUS_VECTORS",
    "PRIORITY_THRESHOLDS",
    "STABILITY_THRESHOLDS",
    "TRIAD_IDEAL_VALUES",
    "TRIAD_BALANCE_THRESHOLD",
    
    # 3. КЛАССЫ ИЗ ras_core_v4_1.py
    "EnhancedRASCore",
    "RASSignal",
    "PrioritySignalQueue",
    "StabilityAwarePriorityQueue",
    "RASConfigManager",
    "PatternLearner",
    "AngleAwareSephiroticRouter",
    "StabilityMetricsCollector",
    "SelfReflectionEngine",
    "TriadStabilityMonitor",
    "EnhancedMockBus",
    "RASPattern",
    "RASActivation",
    "RASReflection",
    
    # 4. ИНТЕГРАЦИОННЫЕ КОМПОНЕНТЫ
    "RASIntegration",
    "create_ras_integration",
    "integrate_ras_with_sephirot",
    
    # 5. API КОМПОНЕНТЫ
    "RASAPI",
    "create_ras_api",
    
    # 6. ФУНКЦИИ СОЗДАНИЯ
    "create_ras_core",
    "initialize_ras_with_angle",
    
    # 7. УТИЛИТЫ
    "is_ras_core_ready",
    
    # 8. ФЛАГИ ДОСТУПНОСТИ
    "ENHANCED_RAS_CORE_AVAILABLE",
    "PRIORITY_QUEUE_AVAILABLE",
    "STABILITY_QUEUE_AVAILABLE",
    "CONFIG_MANAGER_AVAILABLE",
    "PATTERN_LEARNER_AVAILABLE",
    "ROUTER_AVAILABLE",
    "METRICS_COLLECTOR_AVAILABLE",
    "REFLECTION_ENGINE_AVAILABLE",
    "TRIAD_MONITOR_AVAILABLE",
    "MOCK_BUS_AVAILABLE",
    "RAS_PATTERN_AVAILABLE",
    "RAS_INTEGRATION_AVAILABLE",
    "RAS_API_AVAILABLE"
]

# ================================================================
# 5. ИНИЦИАЛИЗАЦИОННОЕ СООБЩЕНИЕ
# ================================================================

if __name__ != "__main__":
    readiness = is_ras_core_ready()
    ready_count = sum(1 for v in readiness.values() if isinstance(v, bool) and v)
    total_count = sum(1 for v in readiness.values() if isinstance(v, bool))
    
    print(f"[RAS-CORE] 📊 Готовность: {ready_count}/{total_count} компонентов")
    print(f"[RAS-CORE] ✅ RASConfig доступен: {readiness.get('ras_config', False)}")
    
    if readiness["fully_ready"]:
        print("[RAS-CORE] ✅ Полностью готов к активации личности")
    else:
        print("[RAS-CORE] ⚠️  Частично готов. Отсутствующие критические компоненты:")
        for name, status in readiness.items():
            if not status and name != "fully_ready" and name != "ras_config":
                print(f"  - ❌ {name}")
    
    print("[RAS-CORE] 🌟 Критические компоненты для личности:")
    print(f"  - RASConfig: {'✅' if readiness.get('ras_config') else '❌'}")
    print(f"  - EnhancedRASCore: {'✅' if ENHANCED_RAS_CORE_AVAILABLE else '❌'}")
    print(f"  - RASIntegration: {'✅' if RAS_INTEGRATION_AVAILABLE else '❌'}")
