# iskra_modules/__init__.py
"""ISKRA Modules Package - Production v1.0"""
__version__ = '1.0.0'

# НЕ ИМПОРТИРУЕМ модули здесь - это вызывает циклические зависимости!
# Вместо этого просто указываем что доступно
__all__ = ['symbiosis_core', 'sephirot_blocks']

# Можем добавить ленивую загрузку если нужно
def __getattr__(name):
    if name == 'symbiosis_core':
        from . import symbiosis_core
        return symbiosis_core
    elif name == 'sephirot_blocks':
        from . import sephirot_blocks
        return sephirot_blocks
    raise AttributeError(f"module 'iskra_modules' has no attribute '{name}'")

print(f"✅ ISKRA Modules loaded (version: {__version__})")
