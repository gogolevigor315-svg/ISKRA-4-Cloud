# ================================================================
# BINAH/__init__.py
# Активационный модуль сефиры BINAH для ISKRA-4
# ================================================================

"""
בינה (BINAH) — Сефира понимания, структурирования интуиции.
Версия: 1.3.0
Модули: ANALYTICS-MEGAFORGE, GÖDEL-SENTINEL, ISKRA-MIND, BINAH-RESONANCE-MONITOR
"""

import logging

logger = logging.getLogger(__name__)

def activate_binah(bus=None, **kwargs):
    """
    ОБЯЗАТЕЛЬНАЯ функция активации BINAH для импорта системой ISKRA-4.
    
    Аргументы:
        bus: Шина событий sephirot_bus
        **kwargs: Дополнительные параметры активации
    
    Возвращает:
        Словарь с результатом активации
    """
    try:
        # Импортируем фабричную функцию
        from .binah_core import build_binah_core
        
        # Создаем ядро BINAH
        core = build_binah_core(bus)
        
        # Применяем параметры активации если есть
        if kwargs:
            logger.info(f"BINAH activation parameters: {kwargs}")
            
            # Принудительный резонанс
            if 'force_resonance' in kwargs:
                new_res = float(kwargs['force_resonance'])
                core.force_resonance_update(new_res)
                logger.info(f"BINAH forced resonance: {new_res}")
            
            # Конфигурация монитора резонанса
            if 'resonance_monitor_config' in kwargs and core.resonance_monitor:
                config = kwargs['resonance_monitor_config']
                core.configure_resonance_monitor(**config)
                logger.info("BINAH resonance monitor configured")
        
        # Получаем состояние ядра
        core_state = core.get_state()
        
        result = {
            "status": "activated",
            "sephira": "BINAH",
            "version": "1.3.0",
            "resonance": core.resonance,
            "core_state": core_state,
            "modules": {
                "analytics": core_state["modules"]["analytics"],
                "godel": core_state["modules"]["godel"],
                "iskra_mind": core_state["modules"]["iskra_mind"],
                "resonance_monitor": core_state["modules"]["resonance_monitor"],
                "own_resonators": True
            },
            "capabilities": core_state["capabilities"],
            "target_resonance_for_daat": 0.85,
            "message": "בינה (BINAH) v1.3 активирована. Готова к структурированию интуиции от CHOKMAH.",
            "ritual_complete": True
        }
        
        logger.info(f"✅ BINAH activated successfully. Resonance: {core.resonance:.2f}")
        
        return result
        
    except ImportError as e:
        error_msg = f"❌ BINAH activation failed - import error: {e}"
        logger.error(error_msg)
        return {
            "status": "activation_failed",
            "sephira": "BINAH",
            "error": str(e),
            "message": error_msg
        }
        
    except Exception as e:
        error_msg = f"❌ BINAH activation failed: {e}"
        logger.error(error_msg)
        return {
            "status": "activation_failed",
            "sephira": "BINAH",
            "error": str(e),
            "message": error_msg
        }

# 🔥 ДОБАВЛЕНО: Функция get_binah для совместимости с системой
def get_binah(bus=None, **kwargs):
    """
    Алиас для activate_binah, требуется системой ISKRA-4.
    Многие модули ищут get_binah() вместо activate_binah().
    """
    logger.info("BINAH: get_binah() called (compatibility alias)")
    return activate_binah(bus, **kwargs)

# Экспорт основных классов для импорта извне
from .binah_core import (
    BinahCore,
    build_binah_core,
    IntuitionPacket,
    StructuredUnderstanding,
    BinahEthicalResonator,
    BinahSpiritualHarmonizer
)

from .binah_resonance_monitor import (
    BinahResonanceMonitor,
    ResonanceRecord,
    SeismicEvent,
    EmergentSignature
)

# Экспорт фабричных функций зависимостей
try:
    from .ANALYTICS_MEGAFORGE_3_4_Sephirotic_Analytical_Engine import build_analytics_megaforge, AnalyticsMegaForge
    ANALYTICS_EXPORTED = True
except ImportError:
    ANALYTICS_EXPORTED = False
    AnalyticsMegaForge = None
    build_analytics_megaforge = None

try:
    from .GÖDEL_SENTINEL_3_2_Sephirotic_Paradox_Guardian import build_godel_sentinel, GodelSentinel
    GODEL_EXPORTED = True
except ImportError:
    GODEL_EXPORTED = False
    GodelSentinel = None
    build_godel_sentinel = None

try:
    from .ISKRA_MIND_3_1_sephirotic_reflective import activate_iskra_mind, IskraMindCore
    ISKRA_MIND_EXPORTED = True
except ImportError:
    ISKRA_MIND_EXPORTED = False
    IskraMindCore = None
    activate_iskra_mind = None

# Определяем __all__ для чистого импорта
__all__ = [
    # Основные функции
    'activate_binah',
    'get_binah',  # 🔥 ДОБАВЛЕНО ДЛЯ СОВМЕСТИМОСТИ
    
    # Основные классы из binah_core
    'BinahCore',
    'build_binah_core',
    'IntuitionPacket',
    'StructuredUnderstanding',
    'BinahEthicalResonator',
    'BinahSpiritualHarmonizer',
    
    # Классы из монитора резонанса
    'BinahResonanceMonitor',
    'ResonanceRecord',
    'SeismicEvent',
    'EmergentSignature'
]

# Добавляем экспортированные зависимости если они доступны
if ANALYTICS_EXPORTED:
    __all__.extend(['build_analytics_megaforge', 'AnalyticsMegaForge'])

if GODEL_EXPORTED:
    __all__.extend(['build_godel_sentinel', 'GodelSentinel'])

if ISKRA_MIND_EXPORTED:
    __all__.extend(['activate_iskra_mind', 'IskraMindCore'])

# Экспортируем None-объекты для избежания ошибок импорта
__all__.extend(['AnalyticsMegaForge', 'GodelSentinel', 'IskraMindCore'])

# ================================================================
# ДОПОЛНИТЕЛЬНЫЕ ФУНКЦИИ ДЛЯ СОВМЕСТИМОСТИ С СЕФИРОТИЧЕСКОЙ СИСТЕМОЙ
# ================================================================

def get_binah_config():
    """Возвращает конфигурацию BINAH для системных нужд"""
    return {
        "sephira": "BINAH",
        "version": "1.3.0",
        "required_resonance": 0.55,
        "target_resonance": 0.85,
        "angle_alignment": 14.4,
        "modules_expected": 4,
        "modules_loaded": sum([ANALYTICS_EXPORTED, GODEL_EXPORTED, ISKRA_MIND_EXPORTED, True]),  # + resonance monitor
        "activation_function": "activate_binah",
        "compatibility_function": "get_binah"
    }

def check_binah_ready():
    """Проверка готовности BINAH к интеграции"""
    return {
        "ready": all([ANALYTICS_EXPORTED, GODEL_EXPORTED, ISKRA_MIND_EXPORTED]),
        "missing_modules": [
            "ANALYTICS_MEGAFORGE" if not ANALYTICS_EXPORTED else None,
            "GÖDEL_SENTINEL" if not GODEL_EXPORTED else None,
            "ISKRA_MIND" if not ISKRA_MIND_EXPORTED else None
        ],
        "resonance_monitor": True,
        "can_activate": True
    }

# Добавляем эти функции в экспорт
__all__.extend(['get_binah_config', 'check_binah_ready'])

# ================================================================
# ИНИЦИАЛИЗАЦИОННОЕ СООБЩЕНИЕ
# ================================================================

if __name__ != "__main__":
    # Выводим сообщение при импорте пакета
    print("[BINAH] Package __init__ loaded")
    print(f"[BINAH] Export stats: A={ANALYTICS_EXPORTED}, G={GODEL_EXPORTED}, I={ISKRA_MIND_EXPORTED}")
    print(f"[BINAH] Compatibility: get_binah() = {get_binah is not None}")
    print("[BINAH] Use: from BINAH import activate_binah, get_binah, BinahCore, build_binah_core")
else:
    print("[BINAH] __init__ running in standalone mode")
    print("[BINAH] This is a package initialization file, not a script")
    print("[BINAH] Testing activation...")
    result = activate_binah()
    print(f"[BINAH] Activation result: {result}")
