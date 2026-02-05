"""
SYMBIOSIS-CORE v5.4 - УПРОЩЁННАЯ ВЕРСИЯ ДЛЯ RENDER
"""

import sys
import time

print("🚨 SYMBIOSIS-CORE v5.4 - УПРОЩЁННЫЙ __init__.py загружен")

# ==================== ЗАГЛУШКИ ДЛЯ ВСЕХ МОДУЛЕЙ ====================

class SymbiosisCoreStub:
    def __init__(self):
        self.name = "SymbiosisCore_STUB"
        self.version = "5.4"
    
    def get_info(self):
        return {
            "name": "SymbiosisCore",
            "version": self.version,
            "status": "stub",
            "timestamp": time.time()
        }

class AladdinShadowStub:
    def __init__(self):
        self.name = "AladdinShadow_STUB"
    
    def get_info(self):
        return {
            "name": "AladdinShadow",
            "status": "stub",
            "timestamp": time.time()
        }

class ISKRAAdapterStub:
    def __init__(self):
        self.name = "ISKRAAdapter_STUB"
    
    def get_info(self):
        return {
            "name": "ISKRAAdapter",
            "status": "stub",
            "timestamp": time.time()
        }

# Создаём экземпляры заглушек
symbiosis_core_stub = SymbiosisCoreStub()
aladdin_shadow_stub = AladdinShadowStub()
iskra_adapter_stub = ISKRAAdapterStub()

# Регистрируем в sys.modules для импорта
sys.modules['iskra_modules.symbiosis_core.symbiosis_core'] = symbiosis_core_stub
sys.modules['iskra_modules.symbiosis_core.aladdin_shadow'] = aladdin_shadow_stub
sys.modules['iskra_modules.symbiosis_core.iskra_integration'] = iskra_adapter_stub

print("✅ Заглушки для SymbiosisCore, AladdinShadow, ISKRAAdapter зарегистрированы")

# ==================== ОСНОВНОЙ ИМПОРТ ДЛЯ BLUEPRINT ====================

# Только один импорт — тот, что реально нужен для iskra_full.py
try:
    from .symbiosis_api import symbiosis_bp
    print("✅ symbiosis_bp импортирован успешно")
except ImportError as e:
    print(f"⚠️ Ошибка импорта symbiosis_bp: {e}")
    # Создаём пустой blueprint-заглушку
    from flask import Blueprint
    symbiosis_bp = Blueprint('symbiosis_stub', __name__)
    
    @symbiosis_bp.route('/')
    def stub():
        return {"status": "stub", "module": "symbiosis_core"}

# ==================== ЭКСПОРТ ====================

__all__ = [
    'symbiosis_bp',
    'SymbiosisCoreStub',
    'AladdinShadowStub',
    'ISKRAAdapterStub'
]

print("=" * 60)
print("✅ SYMBIOSIS-CORE __init__.py УСПЕШНО ЗАГРУЖЕН")
print("✅ Заглушки для всех модулей")
print("✅ symbiosis_bp готов (реальный или stub)")
print("=" * 60)
