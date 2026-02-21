#!/usr/bin/env python3
# =============================================================================
# ISKRA-4 CLOUD v10.10 — DIAGNOSTIC VERSION
# =============================================================================
import os
import sys
import asyncio
import logging
from datetime import datetime, timezone
from flask import Flask, jsonify, request
from flask_cors import CORS

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(name)s | %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("ISKRA-4")

app = Flask(__name__)
CORS(app)

# =============================================================================
# ДИАГНОСТИКА СТРУКТУРЫ ПАПОК
# =============================================================================
logger.info("=== ДИАГНОСТИКА ПУТЕЙ ===")
logger.info(f"Текущая директория: {os.getcwd()}")
logger.info(f"PYTHONPATH: {sys.path[:5]}")

logger.info("\n=== Содержимое папки iskra_modules ===")
if os.path.exists("iskra_modules"):
    print(os.listdir("iskra_modules"))
else:
    print("Папка iskra_modules НЕ НАЙДЕНА!")

logger.info("\n=== Проверка daat_core ===")
daat_path = "iskra_modules/daat_core"
if os.path.exists(daat_path):
    print("daat_core найдена:", os.listdir(daat_path))
    init_file = os.path.join(daat_path, "__init__.py")
    print(f"__init__.py существует: {os.path.exists(init_file)}")
else:
    print("Папка daat_core НЕ НАЙДЕНА!")

# =============================================================================
# ГЛОБАЛЬНОЕ СОСТОЯНИЕ
# =============================================================================
_system = {
    "version": "10.10 Diagnostic",
    "status": "initializing",
    "resonance": 0.82,
}

# =============================================================================
# ЗАПУСК
# =============================================================================
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    logger.info(f"🚀 ISKRA-4 Diagnostic версия запущена на порту {port}")
    
    app.run(host="0.0.0.0", port=port, debug=False)
