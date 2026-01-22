"""
KETHER PACKAGE - Сефира KETHER (Венец) для системы ISKRA-4
ПОЛНАЯ ИСПРАВЛЕННАЯ ВЕРСИЯ - все 9 модулей, без кастраций
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
# 3. ИМПОРТ ВСЕХ 9 МОДУЛЕЙ KETER
# ============================================================

try:
    # 1. Ядро Keter
    from .keter_core import KetherCore, create_keter_core, ModuleInfo, EnergyFlow, topological_sort
    
    # 2. API Keter (исправлен)
    from .keter_api import KetherAPI, KetherCoreWithAPI, create_keter_core_with_api, create_keter_api_gateway
    
    # 3. Интеграция Keter
    from .keter_integration import KeterIntegration, create_keter_integration, initialize_keter_with_iskra
    
    # 4. Willpower Core (исправлен) - ОРИГИНАЛЬНЫЙ КЛАСС
    from .willpower_core_v3_2 import WILLPOWER_CORE_v32_KETER, WillpowerCoreV3_2
    
    # 5. Spirit Core (исправлен) - ОРИГИНАЛЬНЫЙ КЛАСС
    from .spirit_core_v3_4 import SPIRIT_CORE_v34_KETER, SpiritCoreV3_4
    
    # 6. Core GovX (исправлен) - ОРИГИНАЛЬНЫЙ КЛАСС
    from .core_govx_3_1 import CoreGovX31
    
    # 7. Spirit Synthesis
    from .spirit_synthesis_core_v2_1 import create_spirit_synthesis_module
    
    # 8. Moral Memory
    from .moral_memory_3_1 import create_moral_memory_module
    
    # 9. Импорт ТОЛЬКО функций get_module_instance() из 4 исправленных модулей
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
    WillpowerCore = WillpowerCoreV3_2
    
    IMPORT_SUCCESS = True
    
except ImportError as e:
    IMPORT_SUCCESS = False
    logging.error(f"Ошибка импорта компонентов KETHER: {e}")
    
    # Заглушки для совместимости (только при ошибке импорта)
    class KetherCore:
        def __init__(self, config=None):
            pass
    
    def create_keter_core(config=None):
        return KetherCore(config)
    
    KetherAPI = KetherCore
    KetherCoreWithAPI = KetherCore
    KeterIntegration = KetherCore
    
    # Оригинальные классы 4 исправленных модулей
    WILLPOWER_CORE_v32_KETER = KetherCore
    WillpowerCoreV3_2 = KetherCore
    WillpowerCore = WillpowerCoreV3_2
    
    SPIRIT_CORE_v34_KETER = KetherCore
    SpiritCoreV3_4 = KetherCore
    
    CoreGovX31 = KetherCore
    
    # Функции создания модулей
    create_keter_core_with_api = lambda config=None: KetherCore(config)
    create_keter_api_gateway = lambda config=None: KetherCore(config)
    create_keter_integration = lambda config=None: KetherCore(config)
    initialize_keter_with_iskra = lambda config=None: KetherCore(config)
    create_spirit_synthesis_module = lambda config=None: KetherCore(config)
    create_moral_memory_module = lambda config=None: KetherCore(config)
    
    # Функции get_module_instance для API
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
# 4. ЭКСПОРТИРУЕМЫЕ КОМПОНЕНТЫ (ВСЕ 9 МОДУЛЕЙ)
# ============================================================

__all__ = [
    # 1. Kether Core
    "KetherCore", "create_keter_core",
    "ModuleInfo", "EnergyFlow", "topological_sort",
    
    # 2. Kether API
    "KetherAPI", "KetherCoreWithAPI",
    "create_keter_core_with_api", "create_keter_api_gateway",
    "activate_keter",
    
    # 3. Keter Integration
    "KeterIntegration", "create_keter_integration",
    "initialize_keter_with_iskra",
    
    # 4. Willpower Core (исправлен)
    "WILLPOWER_CORE_v32_KETER", "WillpowerCoreV3_2",
    "WillpowerCore",  # Алиас
    
    # 5. Spirit Core (исправлен)
    "SPIRIT_CORE_v34_KETER", "SpiritCoreV3_4",
    
    # 6. Core GovX (исправлен)
    "CoreGovX31",
    
    # 7. Spirit Synthesis
    "create_spirit_synthesis_module",
    
    # 8. Moral Memory
    "create_moral_memory_module",
    
    # 9. Функции get_module_instance для API системы
    "get_willpower_instance",
    "get_spirit_instance", 
    "get_keter_api_instance",
    "get_core_govx_instance",
]

# ============================================================
# 5. ДОПОЛНИТЕЛЬНЫЕ ФУНКЦИИ
# ============================================================

def get_package_info() -> Dict[str, Any]:
    """Получение информации о пакете KETHER"""
    return {
        "name": "KETHER",
        "version": __version__,
        "sephira": __sephira__,
        "sephira_number": __sephira_number__,
        "sephira_name": __sephira_name__,
        "architecture": __architecture__,
        "description": __description__,
        "author": __author__,
        "modules_count": 9,
        "import_success": IMPORT_SUCCESS,
        "available_components": len(__all__)
    }

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

# Добавляем в экспорт
__all__.append('get_package_info')
__all__.append('get_module_by_name')

# ============================================================
# 6. ФУНКЦИИ ДЛЯ ВНЕШНЕГО ИСПОЛЬЗОВАНИЯ
# ============================================================

def get_keter():
    """
    Получение экземпляра KETER для внешних модулей
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

__all__.append('get_keter')

# ============================================================
# 7. ДОПОЛНИТЕЛЬНЫЕ АЛИАСЫ
# ============================================================

KETHER = KetherCore  # Алиас для краткости
KETER = KetherCore  # Ещё один алиас

__all__.extend(['KETHER', 'KETER'])

# ============================================================
# 8. ИНИЦИАЛИЗАЦИЯ ПРИ ЗАГРУЗКЕ
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
    
    if IMPORT_SUCCESS:
        logger.info(f"✅ Пакет KETHER v{__version__} успешно загружен")
        logger.info(f"   Сефира: {__sephira_name__} ({__sephira__})")
        logger.info(f"   Архитектура: {__architecture__}")
        logger.info(f"   Модулей: 9")
        logger.info(f"   Компонентов экспортировано: {len(__all__)}")
        
        # Ключевые функции
        key_funcs = {
            "WillpowerCore": WillpowerCore is WillpowerCoreV3_2,
            "activate_keter": 'activate_keter' in globals(),
            "get_module_by_name": 'get_module_by_name' in globals(),
        }
        
        for func, available in key_funcs.items():
            status = "✅ доступна" if available else "❌ не найдена"
            logger.info(f"   {func}: {status}")
    else:
        logger.error(f"❌ Пакет KETHER v{__version__} загружен с ошибками импорта")

# ============================================================
# 9. ЗАПУСК ИНИЦИАЛИЗАЦИИ
# ============================================================

_initialize_package()

# Финальное сообщение
logger = logging.getLogger("KETHER")
logger.info("=" * 60)
logger.info(f"KETHER PACKAGE v{__version__} ГОТОВ К ИСПОЛЬЗОВАНИЮ")
logger.info(f"Алиас WillpowerCore создан: {WillpowerCore is WillpowerCoreV3_2}")
logger.info(f"API совместимость: {IMPORT_SUCCESS}")
logger.info("=" * 60)
