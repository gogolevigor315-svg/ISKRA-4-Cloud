"""
CHOKMAH PACKAGE - Сефира CHOKMAH (חָכְמָה - Мудрость) для системы ISKRA-4
Ядро интуитивного озарения и потокового понимания системы
"""

import os
import sys
import importlib
import logging
import time
from typing import Optional, Dict, Any, Tuple, List

# ============================================================
# 1. НАСТРОЙКА ПУТЕЙ
# ============================================================

current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

# ============================================================
# 2. МЕТАДАННЫЕ ПАКЕТА
# ============================================================

__version__ = "1.0.0"
__sephira__ = "CHOKMAH"
__sephira_number__ = 2
__sephira_name__ = "חָכְמָה (Мудрость)"
__architecture__ = "ISKRA-4/CHOKMAH_STREAM"
__author__ = "ISKRA-4 Architecture Team"
__description__ = "Сефира CHOKMAH - ядро интуитивного озарения и потокового понимания системы"

# ============================================================
# 3. ИМПОРТ ОСНОВНЫХ КОМПОНЕНТОВ
# ============================================================

try:
    from .wisdom_core import WisdomCore
    from .intuition_matrix import IntuitionMatrix
    from .chokmah_api import ChokmahAPI
    from .chokmah_integration import ChokmahIntegration
    
    IMPORT_SUCCESS = True
except ImportError as e:
    IMPORT_SUCCESS = False
    logging.error(f"Ошибка импорта компонентов CHOKMAH: {e}")
    
    # Заглушки при ошибке импорта
    class WisdomCore:
        def __init__(self, config=None):
            pass
        async def initialize(self):
            pass
        async def connect_matrix(self, matrix):
            pass
        async def resonate(self):
            pass
    
    IntuitionMatrix = type('IntuitionMatrix', (), {})
    ChokmahAPI = type('ChokmahAPI', (), {})
    ChokmahIntegration = type('ChokmahIntegration', (), {})

# ============================================================
# 4. ЭКСПОРТИРУЕМЫЕ КОМПОНЕНТЫ
# ============================================================

__all__ = [
    "WisdomCore",
    "IntuitionMatrix", 
    "ChokmahAPI",
    "ChokmahIntegration",
    "activate_chokmah",
    "get_active_chokmah",
    "get_package_info",
    "check_dependencies"
]

# ============================================================
# 5. ГЛОБАЛЬНЫЕ ПЕРЕМЕННЫЕ
# ============================================================

_active_wisdom_core: Optional[WisdomCore] = None
_active_intuition_matrix: Optional[IntuitionMatrix] = None

# ============================================================
# 6. ОСНОВНЫЕ ФУНКЦИИ ПАКЕТА
# ============================================================

def create_wisdom_core(config: Optional[Dict] = None) -> WisdomCore:
    """Создаёт и возвращает ядро мудрости CHOKMAH"""
    global _active_wisdom_core
    if _active_wisdom_core is None:
        _active_wisdom_core = WisdomCore(config)
        logging.getLogger("CHOKMAH").info("💡 Ядро мудрости CHOKMAH создано")
    return _active_wisdom_core

def create_intuition_matrix(config: Optional[Dict] = None) -> IntuitionMatrix:
    """Создаёт и возвращает матрицу интуиции"""
    global _active_intuition_matrix
    if _active_intuition_matrix is None:
        _active_intuition_matrix = IntuitionMatrix(config)
        logging.getLogger("CHOKMAH").info("🔮 Матрица интуиции инициализирована")
    return _active_intuition_matrix

def get_active_chokmah() -> Tuple[Optional[WisdomCore], Optional[IntuitionMatrix]]:
    """Возвращает активные компоненты CHOKMAH"""
    return _active_wisdom_core, _active_intuition_matrix

async def activate_chokmah(config: Optional[Dict] = None) -> Tuple[WisdomCore, IntuitionMatrix]:
    """
    Асинхронная активация потока мудрости CHOKMAH
    
    Args:
        config: Конфигурация для инициализации
        
    Returns:
        Кортеж (WisdomCore, IntuitionMatrix) — активированные компоненты
    """
    logger = logging.getLogger("CHOKMAH")
    logger.info("🌊 Активация CHOKMAH-STREAM...")
    
    wisdom_core = create_wisdom_core(config)
    intuition_matrix = create_intuition_matrix(config)
    
    try:
        await wisdom_core.initialize()
        await intuition_matrix.initialize()
        await wisdom_core.connect_matrix(intuition_matrix)
        await wisdom_core.resonate()
        
        logger.info(f"✅ CHOKMAH-STREAM v{__version__} активирован")
        return wisdom_core, intuition_matrix
        
    except Exception as e:
        logger.error(f"❌ Ошибка активации CHOKMAH: {e}")
        raise

# ============================================================
# 7. ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ
# ============================================================

def get_package_info() -> Dict[str, Any]:
    """Возвращает метаданные пакета CHOKMAH"""
    return {
        "name": "CHOKMAH",
        "version": __version__,
        "sephira": __sephira__,
        "sephira_number": __sephira_number__,
        "sephira_name": __sephira_name__,
        "architecture": __architecture__,
        "description": __description__,
        "author": __author__,
        "import_success": IMPORT_SUCCESS,
        "available_components": __all__,
        "active_components": {
            "wisdom_core": _active_wisdom_core is not None,
            "intuition_matrix": _active_intuition_matrix is not None
        }
    }

def check_dependencies() -> Dict[str, Any]:
    """Проверяет доступность зависимостей"""
    dependencies = {
        "asyncio": "встроен в Python 3.7+",
        "typing": "встроен в Python 3.5+",
        "logging": "встроен",
        "sys": "встроен",
        "os": "встроен",
        "time": "встроен",
    }
    
    results = {}
    all_available = True
    
    for dep, description in dependencies.items():
        try:
            importlib.import_module(dep)
            results[dep] = {"status": "available", "description": description}
        except ImportError:
            results[dep] = {"status": "missing", "description": description}
            all_available = False
    
    return {
        "dependencies": results,
        "all_available": all_available,
        "timestamp": time.time()
    }

# ============================================================
# 8. ИНИЦИАЛИЗАЦИЯ ПРИ ЗАГРУЗКЕ
# ============================================================

def _initialize_package():
    """Инициализация пакета при загрузке"""
    logger = logging.getLogger("CHOKMAH")
    
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    
    logger.info(f"Пакет CHOKMAH v{__version__} загружается...")
    
    deps = check_dependencies()
    
    if not deps["all_available"]:
        logger.warning("Не все обязательные зависимости доступны")
        for dep, info in deps["dependencies"].items():
            if info["status"] == "missing":
                logger.warning(f"  Отсутствует: {dep} - {info['description']}")
    
    if IMPORT_SUCCESS:
        logger.info(f"✅ Пакет CHOKMAH v{__version__} успешно загружен")
        logger.info(f"   Сефира: {__sephira_name__} ({__sephira__})")
        logger.info(f"   Архитектура: {__architecture__}")
    else:
        logger.error(f"❌ Пакет CHOKMAH v{__version__} загружен с ошибками импорта")

_initialize_package()
