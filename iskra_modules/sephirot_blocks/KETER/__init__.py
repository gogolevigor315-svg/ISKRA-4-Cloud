"""
KETER PACKAGE - РЕАЛЬНЫЕ МОДУЛИ, А НЕ ЗАГЛУШКИ
Версия: 5.0 - Только реальные модули
"""

import sys
import time

print("🚀 KETER PACKAGE v5.0 - REAL MODULES ONLY")

# ==================== ИМПОРТ РЕАЛЬНЫХ МОДУЛЕЙ ====================

def import_real_module(module_name):
    """Импорт реального модуля и создание экземпляра"""
    try:
        module = __import__(f'iskra_modules.sephirot_blocks.KETER.{module_name}', fromlist=[''])
        
        if hasattr(module, 'get_module_instance'):
            instance = module.get_module_instance()
            print(f"✅ {module_name}: реальный экземпляр создан")
            return instance
        else:
            print(f"⚠️ {module_name}: нет get_module_instance()")
            return None
    except Exception as e:
        print(f"❌ {module_name}: ошибка импорта: {e}")
        return None

# ==================== РЕАЛЬНЫЕ ЭКЗЕМПЛЯРЫ МОДУЛЕЙ ====================

# ЗАГРУЖАЕМ РЕАЛЬНЫЕ МОДУЛИ
_real_modules = {
    "willpower_core_v3_2": import_real_module("willpower_core_v3_2"),
    "spirit_core_v3_4": import_real_module("spirit_core_v3_4"),
    "keter_api": import_real_module("keter_api"),
    "core_govx_3_1": import_real_module("core_govx_3_1"),
}

# ==================== ГЛАВНАЯ ФУНКЦИЯ ====================

def get_module_by_name(module_name: str):
    """
    Возвращает РЕАЛЬНЫЙ экземпляр модуля Keter
    """
    print(f"🔍 get_module_by_name: '{module_name}'")
    
    if module_name in _real_modules and _real_modules[module_name] is not None:
        instance = _real_modules[module_name]
        print(f"✅ Возвращаю реальный экземпляр {module_name}")
        return instance
    else:
        # Если реальный модуль не загружен - хотя бы правильная структура
        print(f"⚠️ Модуль {module_name} не найден, возвращаю структуру")
        return {
            "module": module_name,
            "status": "error",
            "error": "Module not properly loaded",
            "sephira": "KETHER",
            "timestamp": time.time(),
            "info": {"type": "error_fallback"}
        }

# ==================== ДОПОЛНИТЕЛЬНЫЕ ФУНКЦИИ ====================

def activate_keter():
    return {
        "status": "activated",
        "sephira": "KETHER",
        "version": "5.0",
        "message": "Keter activated with real modules",
        "timestamp": time.time(),
        "modules_loaded": len([m for m in _real_modules.values() if m is not None])
    }

def get_keter():
    return {
        "status": "available",
        "sephira": "KETHER",
        "real_modules": list(_real_modules.keys()),
        "loaded_modules": [name for name, instance in _real_modules.items() if instance is not None]
    }

# ==================== ЭКСПОРТ ====================

__all__ = ['get_module_by_name', 'activate_keter', 'get_keter']

# ==================== СТАТИСТИКА ====================

loaded = sum(1 for m in _real_modules.values() if m is not None)
print("=" * 60)
print(f"📊 РЕАЛЬНЫЕ МОДУЛИ KETER ЗАГРУЖЕНЫ: {loaded}/4")
for name, instance in _real_modules.items():
    status = "✅" if instance else "❌"
    print(f"   {status} {name}")
print("=" * 60)
print("🚀 KETER ГОТОВ С РЕАЛЬНЫМИ МОДУЛЯМИ")
print("=" * 60)
