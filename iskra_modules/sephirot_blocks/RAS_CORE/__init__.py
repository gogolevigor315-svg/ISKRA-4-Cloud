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
# УСЛОВНЫЕ ИМПОРТЫ ИЗ ras_core_v4_1.py
# ================================================================

# Базовые классы (должны существовать)
try:
    from .ras_core_v4_1 import EnhancedRASCore, RASSignal
    ENHANCED_RAS_CORE_AVAILABLE = True
    print("[RAS-CORE] ✅ EnhancedRASCore и RASSignal загружены")
except ImportError as e:
    print(f"[RAS-CORE] ⚠️  Ошибка загрузки EnhancedRASCore/RASSignal: {e}")
    EnhancedRASCore = None
    RASSignal = None
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

# ================================================================
# ФУНКЦИЯ ДЛЯ ПРОВЕРКИ ГОТОВНОСТИ RAS-CORE
# ================================================================

def is_ras_core_ready() -> dict:
    """
    Проверяет готовность всех компонентов RAS-CORE
    Возвращает словарь со статусами
    """
    return {
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
        "fully_ready": (
            ENHANCED_RAS_CORE_AVAILABLE and
            PRIORITY_QUEUE_AVAILABLE and
            STABILITY_QUEUE_AVAILABLE and
            CONFIG_MANAGER_AVAILABLE
        )
    }

# ================================================================
# ЭКСПОРТ ДОСТУПНЫХ КЛАССОВ И ФУНКЦИЙ
# ================================================================

__all__ = [
    # Константы (всегда доступны)
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
    
    # Классы (могут быть None)
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
    
    # Утилиты
    "is_ras_core_ready",
    
    # Флаги доступности
    "ENHANCED_RAS_CORE_AVAILABLE",
    "PRIORITY_QUEUE_AVAILABLE",
    "STABILITY_QUEUE_AVAILABLE",
    "CONFIG_MANAGER_AVAILABLE",
    "PATTERN_LEARNER_AVAILABLE",
    "ROUTER_AVAILABLE",
    "METRICS_COLLECTOR_AVAILABLE",
    "REFLECTION_ENGINE_AVAILABLE",
    "TRIAD_MONITOR_AVAILABLE",
    "MOCK_BUS_AVAILABLE"
]

# ================================================================
# ИНИЦИАЛИЗАЦИОННОЕ СООБЩЕНИЕ
# ================================================================

if __name__ != "__main__":
    readiness = is_ras_core_ready()
    ready_count = sum(1 for v in readiness.values() if isinstance(v, bool) and v)
    total_count = sum(1 for v in readiness.values() if isinstance(v, bool))
    
    print(f"[RAS-CORE] 📊 Готовность: {ready_count}/{total_count} компонентов")
    
    if readiness["fully_ready"]:
        print("[RAS-CORE] ✅ Полностью готов к активации личности")
    else:
        print("[RAS-CORE] ⚠️  Частично готов. Критические компоненты отсутствуют.")
        for name, status in readiness.items():
            if not status and name != "fully_ready":
                print(f"  - ❌ {name}: отсутствует")
