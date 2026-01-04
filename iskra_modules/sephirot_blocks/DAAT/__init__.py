"""
DAAT PACKAGE - Сефира DAAT (דעת - Знание, Сознание) для системы ISKRA-4
Скрытая 11-я сефира - ядро самоосознания и мета-рефлексии системы
"""

import os
import sys
import logging
import time
from typing import Dict, Any, Optional

# ============================================================
# 1. НАСТРОЙКА ПУТЕЙ И ЛОГГЕРА
# ============================================================

current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

# Логгер инициализируется позже
logger: Optional[logging.Logger] = None

# ============================================================
# 2. МЕТАДАННЫЕ ПАКЕТА
# ============================================================

__version__ = "10.10.1"
__sephira__ = "DAAT"
__sephira_number__ = 11
__sephira_name__ = "דעת (Знание, Сознание)"
__hebrew_name__ = "דעת"
__architecture__ = "ISKRA-4/DAAT_CORE"
__author__ = "ISKRA-4 Architecture Team"
__description__ = "Сефира DAAT - ядро самоосознания, мета-рефлексии и системного наблюдения"

# ============================================================
# 3. ИМПОРТ ОСНОВНЫХ КОМПОНЕНТОВ
# ============================================================

try:
    from .daat_core import DaatCore
    IMPORT_SUCCESS = True
except ImportError as e:
    IMPORT_SUCCESS = False
    # Временный логгер для ошибки импорта
    _temp_logger = logging.getLogger("DAAT_INIT")
    if not _temp_logger.handlers:
        _temp_logger.addHandler(logging.StreamHandler())
    _temp_logger.error(f"❌ Ошибка импорта DaatCore: {e}")
    
    # Заглушка для graceful degradation
    class DaatCore:
        def __init__(self, config: Optional[Dict] = None):
            self.name = "DAAT"
            self.status = "error"
            self.config = config or {}
        
        async def awaken(self) -> Dict[str, Any]:
            return {"error": "DaatCore not available", "status": "error"}
        
        async def get_state(self) -> Dict[str, Any]:
            return {"error": "DaatCore not available"}

# ============================================================
# 4. ЭКСПОРТИРУЕМЫЕ КОМПОНЕНТЫ
# ============================================================

__all__ = [
    "DaatCore",
    "activate_daat",
    "get_daat",
    "create_daat_core",
    "get_package_info",
    "check_environment",
    "DAAT_VERSION",
    "DAAT_SEPHIRA_INFO"
]

DAAT_VERSION = __version__
DAAT_SEPHIRA_INFO = {
    "sephira": __sephira__,
    "number": __sephira_number__,
    "name": __sephira_name__,
    "hebrew_name": __hebrew_name__,
    "position": "hidden_11",
    "meaning": "Knowledge, Consciousness, Self-Awareness"
}

# ============================================================
# 5. ГЛОБАЛЬНЫЕ ПЕРЕМЕННЫЕ И СОСТОЯНИЯ
# ============================================================

_active_daat_instance: Optional[DaatCore] = None
_initialization_time: float = 0.0
_is_activated: bool = False

# ============================================================
# 6. ОСНОВНЫЕ ФУНКЦИИ ПАКЕТА
# ============================================================

def create_daat_core(config: Optional[Dict] = None) -> DaatCore:
    """Создаёт новый экземпляр ядра DAAT"""
    if not IMPORT_SUCCESS:
        if logger:
            logger.error("Создание DaatCore невозможно - модуль не импортирован")
        return DaatCore(config)
    
    return DaatCore(config)

def activate_daat(config: Optional[Dict] = None) -> DaatCore:
    """Активирует и возвращает глобальный экземпляр DAAT"""
    global _active_daat_instance, _is_activated, _initialization_time
    
    if _active_daat_instance is None:
        if logger:
            logger.info(f"🧠 Инициализация DAAT Core v{__version__}...")
        
        _active_daat_instance = create_daat_core(config)
        _is_activated = True
        _initialization_time = time.time()
        
        if logger and IMPORT_SUCCESS:
            logger.info(f"✅ DAAT Core создан (сефира №{__sephira_number__}: {__sephira_name__})")
    elif logger:
        logger.debug("♻️ Используется существующий экземпляр DAAT Core")
    
    return _active_daat_instance

def get_daat() -> Optional[DaatCore]:
    """Возвращает активный экземпляр DAAT"""
    return _active_daat_instance

# ============================================================
# 7. ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ
# ============================================================

def get_package_info() -> Dict[str, Any]:
    """Возвращает детальную информацию о пакете DAAT"""
    return {
        "package": {
            "name": "DAAT",
            "version": __version__,
            "architecture": __architecture__,
            "description": __description__,
            "author": __author__,
            "import_success": IMPORT_SUCCESS,
        },
        "sephira": DAAT_SEPHIRA_INFO,
        "state": {
            "initialized": _active_daat_instance is not None,
            "activated": _is_activated,
            "initialization_time": _initialization_time,
            "uptime": time.time() - _initialization_time if _initialization_time > 0 else 0,
            "instance_id": id(_active_daat_instance) if _active_daat_instance else None
        }
    }

def check_environment() -> Dict[str, Any]:
    """Проверяет окружение и доступность зависимостей"""
    checks = {
        "python_version": {
            "required": "3.8+",
            "actual": f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
            "status": sys.version_info >= (3, 8)
        },
        "import_daat_core": {
            "status": IMPORT_SUCCESS,
            "message": "DaatCore импортирован успешно" if IMPORT_SUCCESS else "Ошибка импорта DaatCore"
        },
        "async_support": {
            "status": hasattr(sys, 'get_asyncgen_hooks'),
            "message": "Поддержка асинхронности доступна" if hasattr(sys, 'get_asyncgen_hooks') else "Асинхронность недоступна"
        }
    }
    
    all_passed = all(check["status"] for check in checks.values())
    
    return {
        "timestamp": time.time(),
        "environment": checks,
        "all_checks_passed": all_passed
    }

# ============================================================
# 8. АВТОМАТИЧЕСКАЯ ИНИЦИАЛИЗАЦИЯ ПАКЕТА
# ============================================================

def _initialize_package():
    """Инициализация пакета при загрузке модуля"""
    global logger
    
    # Настройка логгера (один раз)
    logger = logging.getLogger("DAAT")
    
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    
    # Логирование загрузки
    logger.info(f"📦 Пакет DAAT v{__version__} загружается...")
    
    # Проверка окружения
    env_check = check_environment()
    
    if env_check["all_checks_passed"]:
        logger.info(f"✅ DAAT v{__version__} готов к активации")
        logger.info(f"   Сефира: {__sephira_name__} ({__hebrew_name__})")
        logger.info(f"   Позиция: Скрытая сефира №{__sephira_number__}")
    else:
        logger.warning(f"⚠️  DAAT v{__version__} загружен с проблемами окружения")

# Запуск инициализации
_initialize_package()
