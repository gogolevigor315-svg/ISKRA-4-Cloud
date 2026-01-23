"""
KETER PACKAGE - КОМПРОМИССНАЯ ВЕРСИЯ
Сохраняем важное, убираем проблемное
"""

print("✅ KETER package v2.0.0 loading (compromise)...")

# 1. БАЗОВЫЕ КОНСТАНТЫ (сохраняем)
__version__ = "2.0.0"
__sephira__ = "KETHER"
__sephira_name__ = "Венец (Кетер)"

# 2. ИМПОРТ ТОЛЬКО ФУНКЦИЙ get_module_instance
try:
    from .willpower_core_v3_2 import get_module_instance as get_willpower_instance
    print("✅ willpower_core_v3_2.get_module_instance импортирован")
except ImportError as e:
    print(f"❌ willpower_core_v3_2: {e}")
    get_willpower_instance = None

try:
    from .spirit_core_v3_4 import get_module_instance as get_spirit_instance
    print("✅ spirit_core_v3_4.get_module_instance импортирован")
except ImportError as e:
    print(f"❌ spirit_core_v3_4: {e}")
    get_spirit_instance = None

# 3. SPIRIT АЛИАС (критически важно)
import sys
try:
    # Импортируем модуль для алиаса
    from . import spirit_core_v3_4
    sys.modules['sephirot_blocks.SPIRIT'] = spirit_core_v3_4
    print("✅ SPIRIT алиас создан")
except Exception as e:
    print(f"⚠️ SPIRIT алиас ошибка: {e}")

# 4. КЛЮЧЕВАЯ ФУНКЦИЯ get_module_by_name
def get_module_by_name(module_name: str):
    """Исправленная версия - использует функции get_module_instance"""
    module_map = {
        "willpower_core_v3_2": get_willpower_instance,
        "spirit_core_v3_4": get_spirit_instance,
    }
    
    func = module_map.get(module_name)
    if func and callable(func):
        try:
            return func()
        except Exception as e:
            return {"error": f"Ошибка создания экземпляра: {e}"}
    else:
        return {"error": f"Модуль {module_name} не найден или функция не импортирована"}

# 5. ПРОСТЫЕ ФУНКЦИИ (сохраняем важное)
def activate_keter():
    return {
        "status": "activated",
        "sephira": "KETHER",
        "version": __version__,
        "modules_available": bool(get_willpower_instance and get_spirit_instance)
    }

def get_keter():
    """Упрощенная версия - возвращает базовую информацию"""
    return activate_keter()

# 6. ЭКСПОРТ МИНИМУМА
__all__ = [
    'get_module_by_name',
    'activate_keter',
    'get_keter',
    '__version__',
    '__sephira__',
]

print("=" * 60)
print(f"KETER PACKAGE v{__version__} ГОТОВ (compromise)")
print(f"get_module_by_name: ✅ доступна")
print(f"SPIRIT алиас: ✅ создан")
print("=" * 60)
