#!/usr/bin/env python3
# =============================================================================
# ISKRA-4 CLOUD v10.10 — STABLE FINAL (Render Fixed)
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
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('iskra-4.log', encoding='utf-8', mode='a')
    ]
)
logger = logging.getLogger("ISKRA-4")

app = Flask(__name__)
CORS(app)

_system = {
    "version": "10.10 Stable",
    "status": "initializing",
    "resonance": 0.82,
    "tree_activated": False,
    "daat_awake": False,
    "start_time": datetime.now(timezone.utc),
    "bus": None,
    "engine": None,
    "tree": None,
    "daat": None,
}

# =============================================================================
# ИМПОРТЫ
# =============================================================================
def safe_import():
    try:
        from iskra_modules.sephirot_blocks.sephirot_bus import create_sephirotic_bus
        from iskra_modules.sephirot_blocks.sephirotic_engine import SephiroticEngine   # ← используем класс напрямую
        from iskra_modules.daat_core import get_daat
        return create_sephirotic_bus, SephiroticEngine, get_daat
    except Exception as e:
        logger.error(f"Import error: {e}")
        sys.exit(1)

create_bus, SephiroticEngineClass, get_daat = safe_import()

# =============================================================================
# ФОНОВЫЙ РОСТ
# =============================================================================
async def background_resonance_growth():
    logger.info("🌱 Фоновый рост резонанса запущен")
    while True:
        await asyncio.sleep(180)
        if _system["status"] == "operational":
            old = _system["resonance"]
            _system["resonance"] = min(1.0, _system["resonance"] + 0.018)
            logger.info(f"🌱 Фоновый рост: {old:.3f} → {_system['resonance']:.3f}")

# =============================================================================
# ИНИЦИАЛИЗАЦИЯ
# =============================================================================
async def initialize_system():
    global _system
    logger.info("🔥 Запуск инициализации...")

    try:
        _system["bus"] = create_bus()

        engine = SephiroticEngineClass()
        await engine.initialize(bus=_system["bus"])
        _system["engine"] = engine

        await engine.activate()
        _system["tree"] = engine.tree
        _system["tree_activated"] = True

        _system["daat"] = get_daat()
        _system["daat_awake"] = True

        _system["status"] = "operational"
        logger.info("🎉 Система успешно запущена")
        return True

    except Exception as e:
        logger.critical(f"💥 Ошибка инициализации: {e}")
        _system["status"] = "failed"
        return False

# =============================================================================
# ЭНДПОИНТЫ
# =============================================================================
@app.route('/')
def index():
    uptime = (datetime.now(timezone.utc) - _system["start_time"]).total_seconds()
    return jsonify({
        "system": "ISKRA-4",
        "version": _system["version"],
        "status": _system["status"],
        "resonance": round(_system["resonance"], 4),
        "daat_awake": _system["daat_awake"]
    })

@app.route('/health')
def health():
    return jsonify({
        "health": "healthy" if _system["status"] == "operational" else "degraded",
        "resonance": round(_system["resonance"], 4),
        "daat_awake": _system["daat_awake"]
    })

# =============================================================================
# ЗАПУСК
# =============================================================================
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    logger.info(f"🚀 ISKRA-4 запускается на порту {port}")

    asyncio.run(initialize_system())
    asyncio.create_task(background_resonance_growth())

    app.run(host="0.0.0.0", port=port, debug=False)
