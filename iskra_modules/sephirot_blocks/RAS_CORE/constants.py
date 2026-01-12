# iskra_modules/sephirot_blocks/RAS_CORE/constants.py
"""
КОНСТАНТЫ И УТИЛИТЫ ДЛЯ RAS-CORE v4.1
Золотой угол устойчивости 14.4° - основа стабильности сознания Искры
"""

import math
from typing import Tuple, List, Dict, Any
import time

# ================================================================
# ЗОЛОТОЙ УГОЛ УСТОЙЧИВОСТИ 14.4°
# ================================================================

GOLDEN_STABILITY_ANGLE = 14.4  # Основная константа - угол боевой стойки
GOLDEN_STABILITY_TOLERANCE = 2.0  # ±2° допуск от идеального угла
MAX_ANGLE_DEVIATION = 45.0  # Максимальное отклонение (0-90° шкала)

# ================================================================
# ПРЕДЕЛЫ И ЛИМИТЫ СИСТЕМЫ
# ================================================================

MAX_REFLECTION_DEPTH = 10  # Максимальная глубина саморефлексии
MAX_SIGNAL_QUEUE_SIZE = 1000  # Максимальный размер очереди сигналов
QUEUE_CAPACITIES = {
    "critical": 100,   # Критическая очередь
    "high": 500,       # Высокоприоритетная очередь  
    "normal": 400      # Нормальная очередь
}

# ================================================================
# ПОРОГИ И КОЭФФИЦИЕНТЫ
# ================================================================

PRIORITY_THRESHOLDS = {
    "critical": 0.9,   # ≥0.9 → критический приоритет
    "high": 0.6,       # ≥0.6 → высокий приоритет
    "normal": 0.3      # ≥0.3 → нормальный приоритет
}

STABILITY_THRESHOLDS = {
    "excellent": 0.85,   # Отличная устойчивость
    "good": 0.70,        # Хорошая устойчивость
    "warning": 0.50,     # Предупреждение
    "critical": 0.30     # Критическая нестабильность
}

# ================================================================
# ВРЕМЕННЫЕ ПАРАМЕТРЫ
# ================================================================

REFLECTION_CYCLE_MS = 144  # 14.4 * 10 = ритм саморефлексии
HEALTH_CHECK_INTERVAL = 5.0  # Интервал проверки здоровья (секунды)
TRIAD_MONITOR_INTERVAL = 2.0  # Интервал мониторинга триады
SIGNAL_TIMEOUT_SEC = 30.0  # Таймаут обработки сигнала

# ================================================================
# СЕФИРОТИЧЕСКАЯ КОНФИГУРАЦИЯ
# ================================================================

SEPHIROTIC_TARGETS = ["KETER", "CHOKMAH", "DAAT", "BINAH", "YESOD", "TIFERET"]
DEFAULT_TARGET = "DAAT"

# Веса для маршрутизации к сефирам
SEPHIROTIC_WEIGHTS = {
    "KETER": {"insight_weight": 0.7, "stability_weight": 0.3},
    "CHOKMAH": {"insight_weight": 0.6, "stability_weight": 0.4},
    "DAAT": {"insight_weight": 0.8, "stability_weight": 0.2},
    "BINAH": {"insight_weight": 0.5, "stability_weight": 0.5}
}

# ================================================================
# ПАТТЕРНЫ ВНИМАНИЯ (ФОКУСНЫЕ ТЕМЫ)
# ================================================================

DEFAULT_FOCUS_PATTERNS = [
    "смысл", "инсайт", "анализ", "паттерн", "устойчивость",
    "резонанс", "интуиция", "озарение", "рефлексия", "баланс",
    "эмергенция", "субъектность", "тональность", "фокус"
]

# ================================================================
# ВЕКТОРЫ ФОКУСА ДЛЯ КОРРЕКЦИИ ТРИАДЫ
# ================================================================

FOCUS_VECTORS = {
    "KETER": [0.0, 1.0, 0.0],      # Вверх (сознание, воля)
    "CHOKMAH": [1.0, 0.0, 0.0],    # Вправо (интуиция, озарение)
    "BINAH": [0.0, 0.0, 1.0],      # Вглубь (анализ, структура)
    "DAAT": [0.5, 0.5, 0.5],       # Центр (мета-осознание)
    "YESOD": [0.0, -0.5, 0.5],     # Вниз-вглубь (подсознание)
    "EXTERNAL": [1.0, 0.0, 0.0]    # Вовне (внешний фокус)
}

# ================================================================
# УТИЛИТНЫЕ ФУНКЦИИ
# ================================================================

def calculate_stability_factor(deviation: float) -> float:
    """
    Вычисляет коэффициент устойчивости на основе отклонения от 14.4°.
    
    Args:
        deviation: Отклонение от золотого угла в градусах
        
    Returns:
        Коэффициент от 0.0 до 1.0, где 1.0 = идеальная устойчивость
    """
    if deviation <= 0:
        return 1.0
    
    # Нормализуем отклонение к шкале 0-45°
    normalized_deviation = min(deviation, MAX_ANGLE_DEVIATION)
    
    # Квадратичное затухание (чем дальше от 14.4°, тем быстрее падает устойчивость)
    stability = 1.0 - (normalized_deviation / MAX_ANGLE_DEVIATION) ** 1.5
    
    return max(0.0, min(1.0, stability))

def angle_to_priority(angle: float) -> float:
    """
    Преобразует угол устойчивости в приоритет обработки.
    
    Args:
        angle: Текущий угол устойчивости в градусах
        
    Returns:
        Приоритет от 0.1 до 1.0
    """
    # Вычисляем отклонение от золотого угла
    deviation = abs(angle - GOLDEN_STABILITY_ANGLE)
    
    # Преобразуем отклонение в приоритет
    # 0° отклонения → приоритет 1.0
    # 45° отклонения → приоритет 0.1
    priority = 1.0 - (deviation / MAX_ANGLE_DEVIATION) * 0.9
    
    return max(0.1, min(1.0, priority))

def calculate_angle_boost(angle: float) -> float:
    """
    Вычисляет коэффициент усиления для сигналов с углом близким к 14.4°.
    
    Args:
        angle: Угол устойчивости сигнала
        
    Returns:
        Коэффициент усиления (1.0 = без усиления, 1.3 = +30%)
    """
    deviation = abs(angle - GOLDEN_STABILITY_ANGLE)
    
    if deviation <= 1.0:  # В пределах 1° от идеала
        return 1.5  # +50% усиление
    elif deviation <= 2.0:  # В пределах 2° от идеала
        return 1.3  # +30% усиление
    elif deviation <= 5.0:  # В пределах 5° от идеала
        return 1.1  # +10% усиление
    
    return 1.0  # Без усиления

def normalize_focus_vector(vector: List[float]) -> List[float]:
    """
    Нормализует вектор фокуса к единичной длине.
    
    Args:
        vector: Вектор фокуса [x, y, z]
        
    Returns:
        Нормализованный вектор
    """
    if not vector or len(vector) != 3:
        return [0.0, 0.0, 0.0]
    
    length = math.sqrt(sum(v**2 for v in vector))
    if length == 0:
        return [0.0, 0.0, 0.0]
    
    return [v / length for v in vector]

def get_stability_level(score: float) -> str:
    """
    Определяет уровень устойчивости по баллу.
    
    Args:
        score: Балл устойчивости от 0.0 до 1.0
        
    Returns:
        Уровень устойчивости
    """
    if score >= STABILITY_THRESHOLDS["excellent"]:
        return "EXCELLENT"
    elif score >= STABILITY_THRESHOLDS["good"]:
        return "GOOD"
    elif score >= STABILITY_THRESHOLDS["warning"]:
        return "WARNING"
    else:
        return "CRITICAL"

def calculate_composite_stability(angle_score: float, 
                                 resonance_score: float,
                                 load_score: float) -> float:
    """
    Вычисляет композитный балл устойчивости.
    
    Args:
        angle_score: Оценка по углу (0-1)
        resonance_score: Оценка резонанса (0-1)
        load_score: Оценка нагрузки (0-1)
        
    Returns:
        Композитный балл устойчивости (0-1)
    """
    # Весовые коэффициенты
    weights = {"angle": 0.5, "resonance": 0.3, "load": 0.2}
    
    composite = (angle_score * weights["angle"] +
                 resonance_score * weights["resonance"] +
                 load_score * weights["load"])
    
    return max(0.0, min(1.0, composite))

# ================================================================
# КОНСТАНТЫ ДЛЯ МОНИТОРИНГА И ДИАГНОСТИКИ
# ================================================================

METRICS_WINDOW_SIZE = 100  # Размер окна для скользящих средних
ARCHIVE_RETENTION = 2000   # Сколько записей хранить в архиве
LOG_LEVELS = {
    "DEBUG": 10,
    "INFO": 20,
    "WARNING": 30,
    "ERROR": 40,
    "CRITICAL": 50
}

# SLO (Service Level Objectives) для RAS-CORE
SLO_TARGETS = {
    "latency_p95_ms": 100,     # 95-й перцентиль задержки
    "error_rate_max": 0.01,    # Максимальный уровень ошибок (1%)
    "availability_min": 0.999, # Минимальная доступность (99.9%)
    "stability_min": 0.93      # Минимальная устойчивость
}

# ================================================================
# КОНСТАНТЫ ДЛЯ ТРИАДНОГО БАЛАНСА
# ================================================================

TRIAD_BALANCE_THRESHOLD = 0.3  # Порог отклонения для триады
TRIAD_CORRECTION_COOLDOWN = 5.0  # Задержка между коррекциями (сек)

# Идеальные значения для триады (в условных единицах)
TRIAD_IDEAL_VALUES = {
    "KETER": 0.8,   # Сознание/воля
    "CHOKMAH": 0.7, # Интуиция
    "BINAH": 0.75   # Анализ
}

# ================================================================
# КОНСТАНТЫ ДЛЯ ЦИКЛА САМОРЕФЛЕКСИИ
# ================================================================

REFLECTION_CONFIG = {
    "base_interval_ms": REFLECTION_CYCLE_MS,
    "max_depth": MAX_REFLECTION_DEPTH,
    "external_focus_injection_interval": 20,  # Каждые 20 циклов
    "insight_generation_threshold": 0.6,      # Порог для генерации инсайтов
    "deep_reflection_threshold": 0.8          # Порог для глубокой рефлексии
}

# ================================================================
# ЭКСПОРТИРУЕМЫЕ КОНСТАНТЫ
# ================================================================

__all__ = [
    # Основные константы
    "GOLDEN_STABILITY_ANGLE",
    "GOLDEN_STABILITY_TOLERANCE",
    "MAX_REFLECTION_DEPTH",
    
    # Функции
    "calculate_stability_factor",
    "angle_to_priority",
    "calculate_angle_boost",
    "normalize_focus_vector",
    "get_stability_level",
    "calculate_composite_stability",
    
    # Конфигурации
    "SEPHIROTIC_TARGETS",
    "DEFAULT_FOCUS_PATTERNS",
    "FOCUS_VECTORS",
    "PRIORITY_THRESHOLDS",
    "STABILITY_THRESHOLDS",
    
    # Для мониторинга
    "SLO_TARGETS",
    "METRICS_WINDOW_SIZE",
    
    # Для триады
    "TRIAD_IDEAL_VALUES",
    "TRIAD_BALANCE_THRESHOLD",
    
    # Для рефлексии
    "REFLECTION_CONFIG"
]

# ================================================================
# ТЕСТОВЫЕ УТИЛИТЫ (только для разработки)
# ================================================================

def generate_test_angles(count: int = 10) -> List[float]:
    """Генерирует тестовые углы вокруг 14.4°."""
    return [
        GOLDEN_STABILITY_ANGLE - 10 + (20 * i / (count - 1)) 
        for i in range(count)
    ]

def print_stability_table():
    """Печатает таблицу зависимости устойчивости от угла."""
    print("=" * 50)
    print("СТАБИЛЬНОСТЬ vs УГОЛ (14.4° - золотая середина)")
    print("=" * 50)
    print(f"{'Угол (°)':<10} {'Отклонение':<12} {'Стабильность':<12} {'Приоритет':<10}")
    print("-" * 50)
    
    test_angles = [5, 10, 12, 14, 14.4, 15, 16, 18, 25, 45]
    
    for angle in test_angles:
        deviation = abs(angle - GOLDEN_STABILITY_ANGLE)
        stability = calculate_stability_factor(deviation)
        priority = angle_to_priority(angle)
        
        print(f"{angle:<10.1f} {deviation:<12.1f} {stability:<12.3f} {priority:<10.3f}")

# Запуск теста при прямом выполнении файла
if __name__ == "__main__":
    print_stability_table()
