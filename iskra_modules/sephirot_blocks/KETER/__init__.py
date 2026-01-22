"""
KETHER PACKAGE - Сефира KETHER (Венец) для системы ISKRA-4
Инициализация пакета и экспорт основных компонентов
ВЕРСИЯ С ИСПРАВЛЕННЫМИ ИМПОРТАМИ ДЛЯ API СОВМЕСТИМОСТИ
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
    from .keter_api import KetherAPI, KetherCoreWithAPI, create_keter_core_with_api, create_keter_api_gateway, create_keter_api_module
    from .keter_integration import KeterIntegration, create_keter_integration, initialize_keter_with_iskra
    from .spirit_synthesis_core_v2_1 import create_spirit_synthesis_module
    from .spirit_core_v3_4 import SpiritCoreV3_4, create_spirit_module
    from .willpower_core_v3_2 import WillpowerCoreV3_2, create_willpower_module
    from .core_govx_3_1 import create_core_govx_module
    from .moral_memory_3_1 import create_moral_memory_module
    
    # Импортируем глобальные функции get_module_instance из каждого модуля
    from .willpower_core_v3_2 import get_module_instance as get_willpower_instance
    from .spirit_core_v3_4 import get_module_instance as get_spirit_instance
    from .keter_api import get_module_instance as get_keter_api_instance
    from .core_govx_3_1 import get_module_instance as get_core_govx_instance
    
    # Проверяем наличие activate_keter в разных модулях
    try:
        from .keter_api import activate_keter
    except ImportError:
        try:
            from .keter_integration import activate_keter
        except ImportError:
            try:
                from .keter_core import activate_keter
            except ImportError:
                # Создаём заглушку если функция нигде не найдена
                def activate_keter():
                    """Активация сефиры KETHER (заглушка для совместимости)"""
                    return {
                        "status": "activated",
                        "sephira": "KETHER",
                        "message": "Kether activated (stub function)",
                        "version": __version__,
                        "timestamp": time.time()
                    }
    
    # Создаем алиасы для совместимости
    WillpowerCore = WillpowerCoreV3_2  # ← ВАЖНО: АЛИАС ДЛЯ СОВМЕСТИМОСТИ!
    
    IMPORT_SUCCESS = True
    
except ImportError as e:
    IMPORT_SUCCESS = False
    logging.error(f"Ошибка импорта компонентов KETHER: {e}")
    
    # Заглушки для совместимости
    class KetherCore:
        def __init__(self, config=None):
            pass
    
    def create_keter_core(config=None):
        return KetherCore(config)
    
    KetherAPI = type('KetherAPI', (), {})
    KeterIntegration = type('KeterIntegration', (), {})
    create_spirit_synthesis_module = lambda config=None: None
    
    SpiritCoreV3_4 = KetherCore
    create_spirit_module = lambda: None
    
    WillpowerCoreV3_2 = KetherCore
    WillpowerCore = WillpowerCoreV3_2  # ← АЛИАС ДЛЯ СОВМЕСТИМОСТИ И В fallback!
    create_willpower_module = lambda: None
    
    create_core_govx_module = lambda config=None: None
    create_moral_memory_module = lambda config=None: None
    
    # Заглушки для функций get_module_instance
    def get_willpower_instance():
        return {"status": "fallback", "module": "willpower_core_fallback"}
    
    def get_spirit_instance():
        return {"status": "fallback", "module": "spirit_core_fallback"}
    
    def get_keter_api_instance():
        return {"status": "fallback", "module": "keter_api_fallback"}
    
    def get_core_govx_instance():
        return {"status": "fallback", "module": "core_govx_fallback"}
    
    def activate_keter():
        """Заглушка для функции активации KETER"""
        return {
            "status": "error",
            "sephira": "KETHER",
            "message": "Kether package import failed",
            "timestamp": time.time()
        }

# ============================================================
# 4. ЭКСПОРТИРУЕМЫЕ КОМПОНЕНТЫ
# ============================================================

__all__ = [
    # Core components
    "KetherCore",
    "create_keter_core",
    "ModuleInfo",
    "EnergyFlow",
    "topological_sort",
    
    # API components
    "KetherAPI",
    "KetherCoreWithAPI",
    "create_keter_core_with_api",
    "create_keter_api_gateway",
    "create_keter_api_module",
    "activate_keter",
    
    # Integration
    "KeterIntegration",
    "create_keter_integration",
    "initialize_keter_with_iskra",
    
    # Specialized modules
    "create_spirit_synthesis_module",
    "SpiritCoreV3_4",
    "create_spirit_module",
    "WillpowerCoreV3_2",
    "WillpowerCore",  # ← ВАЖНО: ДОБАВЛЕН АЛИАС В ЭКСПОРТ!
    "create_willpower_module",
    "create_core_govx_module",
    "create_moral_memory_module",
    
    # Module instance getters (для API системы)
    "get_willpower_instance",
    "get_spirit_instance", 
    "get_keter_api_instance",
    "get_core_govx_instance",
]

# ============================================================
# 5. ФУНКЦИИ ИНИЦИАЛИЗАЦИИ
# ============================================================

def get_package_info() -> Dict[str, Any]:
    """Получение информации о пакете"""
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
    """Проверка зависимостей пакета"""
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
# 6. ДОПОЛНИТЕЛЬНЫЕ ФУНКЦИИ ДЛЯ API СОВМЕСТИМОСТИ
# ============================================================

def get_module_by_name(module_name: str):
    """
    Универсальная функция для получения экземпляра модуля по имени
    Используется API системой ISKRA-4
    
    Args:
        module_name: имя модуля (например, "willpower_core_v3_2")
    
    Returns:
        Экземпляр модуля или словарь с ошибкой
    """
    module_map = {
        "willpower_core_v3_2": get_willpower_instance,
        "spirit_core_v3_4": get_spirit_instance,
        "keter_api": get_keter_api_instance,
        "core_govx_3_1": get_core_govx_instance,
    }
    
    if module_name in module_map:
        return module_map[module_name]()
    else:
        return {
            "error": f"Модуль {module_name} не найден в KETHER",
            "available_modules": list(module_map.keys())
        }

# Добавляем в __all__
__all__.append('get_module_by_name')

# ============================================================
# 7. ИНИЦИАЛИЗАЦИЯ ПРИ ЗАГРУЗКЕ
# ============================================================

def _initialize_package():
    """Инициализация пакета при загрузке"""
    logger = logging.getLogger("KETHER")
    
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    
    logger.info(f"Пакет KETHER v{__version__} загружается...")
    
    # Проверяем зависимости
    deps = check_dependencies()
    
    if not deps["all_available"]:
        logger.warning("Не все обязательные зависимости доступны")
        for dep, info in deps["dependencies"].items():
            if info["status"] == "missing":
                logger.warning(f"  Отсутствует: {dep} - {info['description']}")
    
    # Логируем результат импорта
    if IMPORT_SUCCESS:
        logger.info(f"✅ Пакет KETHER v{__version__} успешно загружен")
        logger.info(f"   Сефира: {__sephira_name__} ({__sephira__})")
        logger.info(f"   Архитектура: {__architecture__}")
        logger.info(f"   Экспортировано компонентов: {len(__all__)}")
        
        # Проверяем наличие ключевых функций
        key_functions = {
            "activate_keter": 'activate_keter' in globals() and callable(activate_keter),
            "WillpowerCore": 'WillpowerCore' in globals(),
            "WillpowerCoreV3_2": 'WillpowerCoreV3_2' in globals(),
            "create_willpower_module": 'create_willpower_module' in globals() and callable(create_willpower_module),
            "create_spirit_module": 'create_spirit_module' in globals() and callable(create_spirit_module),
            "create_keter_api_module": 'create_keter_api_module' in globals() and callable(create_keter_api_module),
            "get_module_by_name": 'get_module_by_name' in globals() and callable(get_module_by_name),
        }
        
        for func, available in key_functions.items():
            status = "✅ доступна" if available else "❌ не найдена"
            logger.info(f"   {func}: {status}")
            
    else:
        logger.error(f"❌ Пакет KETHER v{__version__} загружен с ошибками импорта")

# ============================================================
# 8. ФУНКЦИИ ДЛЯ ВНЕШНЕГО ИСПОЛЬЗОВАНИЯ
# ============================================================

def get_keter():
    """
    Получение экземпляра KETER для внешних модулей
    
    Returns:
        KetherCore или словарь с информацией
    """
    try:
        from .keter_core import KetherCore
        core = KetherCore()
        return core
    except ImportError:
        return {
            "status": "keter_not_available",
            "message": "KetherCore не может быть загружен",
            "fallback": True,
            "version": __version__
        }

# Добавляем в __all__
__all__.append('get_keter')

# ============================================================
# 9. ДОПОЛНИТЕЛЬНЫЕ АЛИАСЫ ДЛЯ ПОЛНОЙ СОВМЕСТИМОСТИ
# ============================================================

# Для совместимости со старым кодом, который может использовать другие имена
KETHER = KetherCore  # Алиас для краткости
KETER = KetherCore  # Ещё один алиас

# Добавляем новые алиасы в __all__
__all__.extend(['KETHER', 'KETER'])

# ============================================================
# 10. ИНИЦИАЛИЗАЦИЯ ПРИ ЗАГРУЗКЕ МОДУЛЯ
# ============================================================

_initialize_package()

# Финальное сообщение
logger = logging.getLogger("KETHER")
logger.info("=" * 60)
logger.info(f"KETHER PACKAGE v{__version__} ГОТОВ К ИСПОЛЬЗОВАНИЮ")
logger.info(f"Алиас WillpowerCore создан: {WillpowerCore is WillpowerCoreV3_2}")
logger.info(f"API совместимость: {IMPORT_SUCCESS}")
logger.info(f"Функция get_module_by_name: {'✅ доступна' if 'get_module_by_name' in globals() else '❌ недоступна'}")
logger.info("=" * 60)
