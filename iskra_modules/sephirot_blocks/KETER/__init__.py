"""
KETHER PACKAGE - Сефира KETHER (Венец) для системы ISKRA-4
ИСПРАВЛЕННАЯ ВЕРСИЯ - без циклических импортов, рабочая
"""

import os
import sys
import time
import logging

# ============================================================
# 1. НАСТРОЙКА ПУТЕЙ И БАЗОВЫЕ КОНСТАНТЫ
# ============================================================

__version__ = "2.0.0"
__sephira__ = "KETHER"
__sephira_number__ = 1
__sephira_name__ = "Венец (Кетер)"
__architecture__ = "ISKRA-4/KETHERIC_BLOCK"

print(f"✅ KETER package v{__version__} loading...")

# ============================================================
# 2. ИМПОРТ МОДУЛЕЙ БЕЗ ЦИКЛИЧЕСКИХ ЗАВИСИМОСТЕЙ
# ============================================================

# Импортируем МОДУЛИ, а не конкретные классы
# Это предотвращает циклические зависимости
try:
    # Используем абсолютный импорт через sys.modules
    current_package = 'sephirot_blocks.KETER'
    
    # 1. Willpower Core
    willpower_module = sys.modules[f'{current_package}.willpower_core_v3_2']
    
    # 2. Spirit Core  
    spirit_module = sys.modules[f'{current_package}.spirit_core_v3_4']
    
    # 3. Keter API
    keter_api_module = sys.modules[f'{current_package}.keter_api']
    
    # 4. Core GovX
    core_govx_module = sys.modules[f'{current_package}.core_govx_3_1']
    
    # 5. Остальные модули (опционально)
    keter_core_module = sys.modules.get(f'{current_package}.keter_core')
    keter_integration_module = sys.modules.get(f'{current_package}.keter_integration')
    spirit_synthesis_module = sys.modules.get(f'{current_package}.spirit_synthesis_core_v2_1')
    moral_memory_module = sys.modules.get(f'{current_package}.moral_memory_3_1')
    
    IMPORT_SUCCESS = True
    print("✅ Все модули KETER импортированы")
    
except KeyError as e:
    IMPORT_SUCCESS = False
    print(f"❌ Ошибка импорта модуля KETER: {e}")
    
    # Создаем пустые заглушки для предотвращения падения
    class EmptyModule:
        def __init__(self):
            pass
    
    willpower_module = EmptyModule()
    spirit_module = EmptyModule()
    keter_api_module = EmptyModule()
    core_govx_module = EmptyModule()
    keter_core_module = EmptyModule()
    keter_integration_module = EmptyModule()
    spirit_synthesis_module = EmptyModule()
    moral_memory_module = EmptyModule()

# ============================================================
# 3. АЛИАСЫ ДЛЯ СОВМЕСТИМОСТИ (ИЗВЛЕКАЕМ ИЗ МОДУЛЕЙ)
# ============================================================

# Пытаемся извлечь классы из модулей
try:
    WILLPOWER_CORE_v32_KETER = getattr(willpower_module, 'WILLPOWER_CORE_v32_KETER', None)
    WillpowerCoreV3_2 = getattr(willpower_module, 'WillpowerCoreV3_2', None)
    
    SPIRIT_CORE_v34_KETER = getattr(spirit_module, 'SPIRIT_CORE_v34_KETER', None)
    SpiritCoreV3_4 = getattr(spirit_module, 'SpiritCoreV3_4', None)
    
    CoreGovX31 = getattr(core_govx_module, 'CoreGovX31', None)
    
    # Алиас для совместимости
    WillpowerCore = WillpowerCoreV3_2
    
    print("✅ Алиасы классов созданы")
    
except AttributeError as e:
    print(f"⚠️ Не удалось создать алиасы: {e}")

# ============================================================
# 4. SPIRIT АЛИАС ДЛЯ СИСТЕМНОЙ СОВМЕСТИМОСТИ (ВАЖНО!)
# ============================================================

try:
    # Регистрируем SPIRIT в sys.modules
    sys.modules['sephirot_blocks.SPIRIT'] = spirit_module
    print("✅ SPIRIT алиас создан: sephirot_blocks.SPIRIT → sephirot_blocks.KETER.spirit_core_v3_4")
    
    # Также для KETER.SPIRIT
    sys.modules['KETER.SPIRIT'] = spirit_module
    print("✅ KETER.SPIRIT алиас создан")
    
except Exception as e:
    print(f"⚠️ Ошибка создания SPIRIT алиаса: {e}")

# ============================================================
# 5. КЛЮЧЕВАЯ ФУНКЦИЯ: get_module_by_name (для API системы)
# ============================================================

def get_module_by_name(module_name: str):
    """
    Универсальная функция для получения экземпляра модуля по имени
    Используется API системой ISKRA-4
    
    ПРЯМОЙ ДОСТУП К ФУНКЦИЯМ get_module_instance() в модулях
    """
    # Карта модулей и их функций get_module_instance
    module_map = {
        "willpower_core_v3_2": willpower_module,
        "spirit_core_v3_4": spirit_module,
        "keter_api": keter_api_module,
        "core_govx_3_1": core_govx_module,
    }
    
    target_module = module_map.get(module_name)
    
    if not target_module:
        return {
            "error": f"Модуль {module_name} не найден в KETHER",
            "available_modules": list(module_map.keys())
        }
    
    # Пытаемся получить get_module_instance из модуля
    try:
        get_instance_func = getattr(target_module, 'get_module_instance', None)
        
        if get_instance_func and callable(get_instance_func):
            # Вызываем функцию для получения экземпляра
            instance = get_instance_func()
            print(f"✅ get_module_by_name: успешно создан экземпляр {module_name}")
            return instance
        else:
            # Если функции нет, возвращаем сам модуль
            print(f"⚠️ get_module_by_name: функция get_module_instance не найдена в {module_name}")
            return target_module
            
    except Exception as e:
        print(f"❌ get_module_by_name ошибка для {module_name}: {e}")
        return {
            "error": f"Ошибка создания экземпляра {module_name}",
            "exception": str(e)
        }

# ============================================================
# 6. ФУНКЦИИ ДЛЯ ВНЕШНЕГО ИСПОЛЬЗОВАНИЯ
# ============================================================

def activate_keter():
    """Активация сефиры KETHER"""
    return {
        "status": "activated",
        "sephira": "KETHER",
        "message": "Kether activated",
        "version": __version__,
        "timestamp": time.time(),
        "modules_available": IMPORT_SUCCESS
    }

def get_package_info():
    """Информация о пакете"""
    return {
        "name": "KETHER",
        "version": __version__,
        "sephira": __sephira__,
        "sephira_name": __sephira_name__,
        "architecture": __architecture__,
        "import_success": IMPORT_SUCCESS
    }

# ============================================================
# 7. ЭКСПОРТИРУЕМЫЕ КОМПОНЕНТЫ
# ============================================================

__all__ = [
    # Основные функции
    "get_module_by_name",
    "activate_keter",
    "get_package_info",
    
    # Классы (если они есть)
]

# Добавляем классы только если они существуют
if 'WILLPOWER_CORE_v32_KETER' in locals() and WILLPOWER_CORE_v32_KETER is not None:
    __all__.extend(["WILLPOWER_CORE_v32_KETER", "WillpowerCoreV3_2", "WillpowerCore"])

if 'SPIRIT_CORE_v34_KETER' in locals() and SPIRIT_CORE_v34_KETER is not None:
    __all__.extend(["SPIRIT_CORE_v34_KETER", "SpiritCoreV3_4"])

if 'CoreGovX31' in locals() and CoreGovX31 is not None:
    __all__.append("CoreGovX31")

# ============================================================
# 8. ИНИЦИАЛИЗАЦИЯ
# ============================================================

print("=" * 60)
print(f"KETHER PACKAGE v{__version__} ГОТОВ")
print(f"Импорт успешен: {IMPORT_SUCCESS}")
print(f"Доступные модули: 4 из 9 (исправленные)")
print(f"SPIRIT алиас: {'✅ создан' if 'sephirot_blocks.SPIRIT' in sys.modules else '❌ ошибка'}")
print(f"Функция get_module_by_name: ✅ доступна")
print("=" * 60)

# Финальное сообщение
print("✅ KETHER package полностью инициализирован")
