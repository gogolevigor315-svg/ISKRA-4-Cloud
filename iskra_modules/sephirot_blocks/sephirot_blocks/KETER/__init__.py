"""
KETHER PACKAGE - Сефира KETHER (Венец) для системы ISKRA-4
Инициализация пакета и экспорт основных компонентов
"""

import os
import sys
import importlib
import logging
import time
from typing import Optional, Dict, Any

# ============================================================
# 1. НАСТРОЙКА ПУТЕЙ
# ============================================================

current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

# ============================================================
# 2. МЕТАДАННЫЕ ПАКЕТА
# ============================================================

__version__ = "2.0.0"
__sephira__ = "KETHER"
__sephira_number__ = 1
__sephira_name__ = "Венец (Кетер)"
__architecture__ = "ISKRA-4/KETHERIC_BLOCK"
__author__ = "ISKRA-4 Architecture Team"
__description__ = "Сефира KETHER - интеграционное ядро Ketheric Block для системы ISKRA-4"

# ============================================================
# 3. ИМПОРТ ОСНОВНЫХ КОМПОНЕНТОВ
# ============================================================

try:
    from .keter_core import KetherCore, create_keter_core, ModuleInfo, EnergyFlow, topological_sort
    from .keter_api import KetherAPI, KetherCoreWithAPI, create_keter_core_with_api, create_keter_api_gateway
    from .keter_integration import KeterIntegration, create_keter_integration, initialize_keter_with_iskra
    from .spirit_synthesis_core_v2_1 import create_spirit_synthesis_module
    from .spirit_core_v3_4 import SpiritCoreV3_4
    from .willpower_core_v3_2 import WillpowerCoreV3_2
    from .core_govx_3_1 import create_core_govx_module
    from .moral_memory_3_1 import create_moral_memory_module
    
    IMPORT_SUCCESS = True
    
except ImportError as e:
    IMPORT_SUCCESS = False
    logging.error(f"Ошибка импорта компонентов KETHER: {e}")
    
    class KetherCore:
        def __init__(self, config=None):
            pass
    
    def create_keter_core(config=None):
        return KetherCore(config)
    
    KetherAPI = type('KetherAPI', (), {})
    KeterIntegration = type('KeterIntegration', (), {})
    create_spirit_synthesis_module = lambda config=None: None
    SpiritCoreV3_4 = KetherCore
    WillpowerCoreV3_2 = KetherCore
    create_core_govx_module = lambda config=None: None
    create_moral_memory_module = lambda config=None: None

# ============================================================
# 4. ЭКСПОРТИРУЕМЫЕ КОМПОНЕНТЫ
# ============================================================

__all__ = [
    "KetherCore",
    "create_keter_core",
    "ModuleInfo",
    "EnergyFlow",
    "topological_sort",
    "KetherAPI",
    "KetherCoreWithAPI",
    "create_keter_core_with_api",
    "create_keter_api_gateway",
    "KeterIntegration",
    "create_keter_integration",
    "initialize_keter_with_iskra",
    "create_spirit_synthesis_module",
    "SpiritCoreV3_4",
    "WillpowerCoreV3_2",
    "create_core_govx_module",
    "create_moral_memory_module",
]

# ============================================================
# 5. ФУНКЦИИ ИНИЦИАЛИЗАЦИИ
# ============================================================

def get_package_info() -> Dict[str, Any]:
    return {
        "name": "KETHER",
        "version": __version__,
        "sephira": __sephira__,
        "sephira_number": __sephira_number__,
        "sephira_name": __sephira_name__,
        "architecture": __architecture__,
        "description": __description__,
        "author": __author__,
        "import_success": IMPORT_SUCCESS,
        "available_components": __all__
    }

def check_dependencies() -> Dict[str, Any]:
    dependencies = {
        "asyncio": "встроен в Python 3.7+",
        "typing": "встроен в Python 3.5+",
        "dataclasses": "встроен в Python 3.7+",
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
# 6. ИНИЦИАЛИЗАЦИЯ ПРИ ЗАГРУЗКЕ
# ============================================================

def _initialize_package():
    logger = logging.getLogger("KETHER")
    
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    
    logger.info(f"Пакет KETHER v{__version__} загружается...")
    
    deps = check_dependencies()
    
    if not deps["all_available"]:
        logger.warning("Не все обязательные зависимости доступны")
        for dep, info in deps["dependencies"].items():
            if info["status"] == "missing":
                logger.warning(f"  Отсутствует: {dep} - {info['description']}")
    
    if IMPORT_SUCCESS:
        logger.info(f"✅ Пакет KETHER v{__version__} успешно загружен")
        logger.info(f"   Сефира: {__sephira_name__} ({__sephira__})")
        logger.info(f"   Архитектура: {__architecture__}")
    else:
        logger.error(f"❌ Пакет KETHER v{__version__} загружен с ошибками импорта")

_initialize_package()
