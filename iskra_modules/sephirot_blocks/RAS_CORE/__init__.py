"""
ИНИЦИАЛИЗАЦИЯ RAS-CORE v4.1
Модуль сефиротического внимания с золотым углом устойчивости 14.4°
Версия 4.1.2 - Полностью исправлены импорты и функции совместимости
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
# 1. RASConfig (ОСНОВНОЙ КЛАСС КОНФИГУРАЦИИ)
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
# 2. ФУНКЦИИ ДЛЯ СОВМЕСТИМОСТИ С СИСТЕМОЙ (ДОБАВЛЕНО СРАЗУ!)
# ================================================================

def get_config(config_name: str = "default") -> dict:
    """
    🔥 КРИТИЧЕСКИ ВАЖНАЯ ФУНКЦИЯ!
    Система ISKRA-4 ищет именно get_config()
    Возвращает конфигурацию RAS-CORE в формате словаря
    """
    config = RASConfig()
    result = {
        "status": "loaded",
        "config_name": config_name,
        "stability_angle": config.stability_angle,
        "reflection_cycle_ms": config.reflection_cycle_ms,
        "enable_self_reflection": config.enable_self_reflection,
        "max_concurrent_signals": config.max_concurrent_signals,
        "triad_balancing_enabled": config.triad_balancing_enabled,
        "personality_coherence_threshold": config.personality_coherence_threshold,
        "sephirotic_targets": SEPHIROTIC_TARGETS,
        "default_focus_patterns": DEFAULT_FOCUS_PATTERNS,
        "golden_angle": GOLDEN_STABILITY_ANGLE,
        "version": "4.1.2",
        "message": "RAS-CORE configuration loaded successfully"
    }
    return result

def get_ras_config() -> RASConfig:
    """
    Альтернативная функция для получения объекта конфигурации
    """
    return RASConfig()

def create_default_ras_config() -> dict:
    """Создает конфигурацию по умолчанию для инициализации"""
    return get_config("default")

# ================================================================
# 3. УСЛОВНЫЕ ИМПОРТЫ ИЗ ras_core_v4_1.py
# ================================================================

# Базовые классы (должны существовать)
try:
    from .ras_core_v4_1 import EnhancedRASCore, RASSignal
    ENHANCED_RAS_CORE_AVAILABLE = True
except ImportError as e:
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

try:
    from .ras_core_v4_1 import StabilityAwarePriorityQueue
    STABILITY_QUEUE_AVAILABLE = True
except ImportError:
    StabilityAwarePriorityQueue = None
    STABILITY_QUEUE_AVAILABLE = False

# Менеджер конфигурации
try:
    from .ras_core_v4_1 import RASConfigManager
    CONFIG_MANAGER_AVAILABLE = True
except ImportError:
    RASConfigManager = None
    CONFIG_MANAGER_AVAILABLE = False

# Паттерны обучения
try:
    from .ras_core_v4_1 import PatternLearner
    PATTERN_LEARNER_AVAILABLE = True
except ImportError:
    PatternLearner = None
    PATTERN_LEARNER_AVAILABLE = False

# Роутер
try:
    from .ras_core_v4_1 import AngleAwareSephiroticRouter
    ROUTER_AVAILABLE = True
except ImportError:
    AngleAwareSephiroticRouter = None
    ROUTER_AVAILABLE = False

# Метрики
try:
    from .ras_core_v4_1 import StabilityMetricsCollector
    METRICS_COLLECTOR_AVAILABLE = True
except ImportError:
    StabilityMetricsCollector = None
    METRICS_COLLECTOR_AVAILABLE = False

# Движок саморефлексии
try:
    from .ras_core_v4_1 import SelfReflectionEngine
    REFLECTION_ENGINE_AVAILABLE = True
except ImportError:
    SelfReflectionEngine = None
    REFLECTION_ENGINE_AVAILABLE = False

# Монитор триады
try:
    from .ras_core_v4_1 import TriadStabilityMonitor
    TRIAD_MONITOR_AVAILABLE = True
except ImportError:
    TriadStabilityMonitor = None
    TRIAD_MONITOR_AVAILABLE = False

# Mock шина
try:
    from .ras_core_v4_1 import EnhancedMockBus
    MOCK_BUS_AVAILABLE = True
except ImportError:
    EnhancedMockBus = None
    MOCK_BUS_AVAILABLE = False

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

# ================================================================
# 4. ИМПОРТЫ ИЗ ДРУГИХ МОДУЛЕЙ RAS_CORE
# ================================================================

# Интеграционный класс RAS-CORE
try:
    from .ras_integration import (
        RASIntegration,
        create_ras_integration,
        integrate_ras_with_sephirot
    )
    RAS_INTEGRATION_AVAILABLE = True
except ImportError as e:
    RASIntegration = None
    create_ras_integration = None
    integrate_ras_with_sephirot = None
    RAS_INTEGRATION_AVAILABLE = False

# API компоненты
try:
    from .ras_api import RASAPI, create_ras_api
    RAS_API_AVAILABLE = True
except ImportError:
    RASAPI = None
    create_ras_api = None
    RAS_API_AVAILABLE = False

# ================================================================
# 5. ФУНКЦИЯ ДЛЯ ПРОВЕРКИ ГОТОВНОСТИ RAS-CORE
# ================================================================

def is_ras_core_ready() -> dict:
    """
    Проверяет готовность всех компонентов RAS-CORE
    Возвращает словарь со статусами
    """
    return {
        "ras_config": True,
        "get_config": True,  # Теперь всегда доступна
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
            ENHANCED_RAS_CORE_AVAILABLE and
            RAS_INTEGRATION_AVAILABLE and
            RAS_API_AVAILABLE
        )
    }

# ================================================================
# 6. ЭКСПОРТ ДОСТУПНЫХ КОМПОНЕНТОВ
# ================================================================

__all__ = [
    # 1. КОНФИГУРАЦИЯ И ФУНКЦИИ СОВМЕСТИМОСТИ (ВАЖНО!)
    "RASConfig",
    "get_config",           # 🔥 СИСТЕМА ИЩЕТ ИМЕННО ЭТУ ФУНКЦИЮ
    "get_ras_config",
    "create_default_ras_config",
    
    # 2. КОНСТАНТЫ
    "GOLDEN_STABILITY_ANGLE",
    "calculate_stability_factor",
    "angle_to_priority",
    "calculate_angle_boost",
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
]

# ================================================================
# 7. ИНИЦИАЛИЗАЦИОННОЕ СООБЩЕНИЕ
# ================================================================

if __name__ != "__main__":
    readiness = is_ras_core_ready()
    ready_count = sum(1 for v in readiness.values() if isinstance(v, bool) and v)
    total_count = sum(1 for v in readiness.values() if isinstance(v, bool))
    
    print(f"[RAS-CORE] 📊 Готовность: {ready_count}/{total_count} компонентов")
    print(f"[RAS-CORE] ✅ get_config() доступна: {readiness.get('get_config', False)}")
    print(f"[RAS-CORE] ✅ RASConfig доступен: {readiness.get('ras_config', False)}")
    
    if readiness["fully_ready"]:
        print("[RAS-CORE] ✅ Полностью готов к активации личности")
    else:
        missing_critical = []
        for name, status in readiness.items():
            if not status and name in ["enhanced_ras_core", "ras_integration", "ras_api"]:
                missing_critical.append(name)
        
        if missing_critical:
            print("[RAS-CORE] ⚠️  Отсутствуют критические компоненты:")
            for name in missing_critical:
                print(f"  - ❌ {name}")
    
    # Выводим тестовую конфигурацию
    try:
        config = get_config()
        print(f"[RAS-CORE] 🧪 get_config() test: {config.get('status', 'unknown')}")
        print(f"[RAS-CORE] 🧪 Угол устойчивости: {config.get('stability_angle', 'unknown')}°")
    except Exception as e:
        print(f"[RAS-CORE] 🧪 get_config() test failed: {e}")
