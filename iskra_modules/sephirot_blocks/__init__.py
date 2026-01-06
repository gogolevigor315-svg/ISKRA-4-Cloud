# -*- coding: utf-8 -*-
"""
Сефиротические блоки ISKRA-4
Модуль инициализации пакета sephirot_blocks.
"""

from importlib import import_module

__all__ = [
    "KETER",
    "CHOKMAH", 
    "DAAT",
    "sephirot_base",
    "sephirot_bus",
    "sephirotic_engine"
]

def load_block(name: str):
    """Импортирует указанный сефиротический блок."""
    try:
        return import_module(f"iskra_modules.sephirot_blocks.{name}")
    except Exception as e:
        print(f"⚠️ Не удалось импортировать блок {name}: {e}")
        return None
