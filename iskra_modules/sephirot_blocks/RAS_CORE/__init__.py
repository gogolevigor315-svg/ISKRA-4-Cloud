# iskra_modules/sephirot_blocks/RAS_CORE/__init__.py
"""
ИНИЦИАЛИЗАЦИЯ RAS-CORE v4.1
Модуль сефиротического внимания с золотым углом устойчивости 14.4°
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

try:
    from .ras_core_v4_1 import (
        EnhancedRASCore,
        RASSignal,
        PrioritySignalQueue,
        StabilityAwarePriorityQueue,
        RASConfigManager,
        PatternLearner,
        AngleAwareSephiroticRouter,
        StabilityMetricsCollector,
        SelfReflectionEngine,
        TriadStabilityMonitor,
        EnhancedMockBus
    )
    
    __all__ = [
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
        "GOLDEN_STABILITY_ANGLE",
        "calculate_stability_factor",
        "angle_to_priority",
        "SEPHIROTIC_TARGETS",
        "DEFAULT_FOCUS_PATTERNS",
        "PRIORITY_THRESHOLDS",
        "get_stability_level",
        "normalize_focus_vector",
        "TRIAD_IDEAL_VALUES",
        "TRIAD_BALANCE_THRESHOLD"
    ]
    
except ImportError:
    # Если основной файл ещё не создан, экспортируем только константы
    __all__ = [
        "GOLDEN_STABILITY_ANGLE",
        "calculate_stability_factor",
        "angle_to_priority",
        "SEPHIROTIC_TARGETS",
        "DEFAULT_FOCUS_PATTERNS",
        "PRIORITY_THRESHOLDS"
    ]
    print("[RAS-CORE] Основной модуль ras_core_v4_1.py ещё не создан. Импортируем только константы.")
