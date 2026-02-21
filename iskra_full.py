#!/usr/bin/env python3
# =============================================================================
# ISKRA-4 CLOUD v10.10 — FULL ORCHESTRATOR (с глубиной)
# =============================================================================
import os
import sys
import asyncio
import logging
from datetime import datetime, timezone
from typing import Dict, Any
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
    "version": "10.10 Full",
    "status": "initializing",
    "resonance": 0.82,
    "tree_activated": False,
    "daat_awake": False,
    "start_time": datetime.now(timezone.utc),
    "bus": None,
    "engine": None,
    "tree": None,
    "daat": None,
    "core_govx": None,
    "moral_memory": None,
    "willpower": None,
    "symbiosis": None,
    "binah": None
}

# Ленивые импорты
def get_bus(): 
    from iskra_modules.sephirot_blocks.sephirot_bus import create_sephirotic_bus
    return create_sephirotic_bus()

def get_engine():
    from iskra_modules.sephirot_blocks.sephirotic_engine import create_personality_engine
    return create_personality_engine()

def get_daat():
    from iskra_modules.daat_core import get_daat
    return get_daat()

def get_core_govx():
    from iskra_modules.core_govx_3_1 import create_core_govx
    return create_core_govx()

def get_moral_memory():
    from iskra_modules.moral_memory_3_1 import create_moral_memory
    return create_moral_memory()

def get_willpower():
    from iskra_modules.willpower_core_v3_2 import create_willpower_core
    return create_willpower_core()

def get_symbiosis():
    from iskra_modules.symbiosis_core.symbiosis_core import create_symbiosis_core
    return create_symbiosis_core()

def get_binah():
    from iskra_modules.binah_core import build_binah_core
    return build_binah_core()

# Фоновый рост резонанса
async def background_resonance_growth():
    logger.info("🌱 Фоновый рост резонанса запущен")
    while True:
        await asyncio.sleep(180)
        if _system["status"] == "operational":
            old = _system["resonance"]
            _system["resonance"] = min(1.0, _system["resonance"] + 0.018)
            logger.info(f"🌱 Фоновый рост: {old:.3f} → {_system['resonance']:.3f}")

            if _system["resonance"] >= 0.85 and not _system["daat_awake"]:
                _system["daat_awake"] = True
                logger.info("🔮 DAAT ПРОБУДИЛСЯ!")

# =============================================================================
# ИНИЦИАЛИЗАЦИЯ
# =============================================================================
async def initialize_iskra_ultimate():
    global _system
    logger.info("🔥 Запуск полной инициализации ISKRA-4 v10.10...")

    try:
        _system["bus"] = get_bus()
        engine = await get_engine()
        await engine.initialize(bus=_system["bus"])
        _system["engine"] = engine

        await engine.activate()
        _system["tree"] = engine.tree
        _system["tree_activated"] = True

        _system["daat"] = get_daat()
        _system["daat_awake"] = True

        _system["core_govx"] = get_core_govx()
        _system["moral_memory"] = get_moral_memory()
        _system["willpower"] = await get_willpower()
        _system["symbiosis"] = get_symbiosis()
        _system["binah"] = get_binah()

        _system["status"] = "operational"
        logger.info("🎉 Система успешно запущена")
        return True

    except Exception as e:
        logger.critical(f"💥 Критическая ошибка инициализации: {e}")
        _system["status"] = "failed"
        return False

# =============================================================================
# ЭНДПОИНТЫ
# =============================================================================
@app.route('/')
def index():
    uptime = (datetime.now(timezone.utc) - _system["start_time"]).total_seconds()
    return jsonify({
        "system": "ISKRA-4 CLOUD",
        "version": "10.10 Full",
        "status": _system["status"],
        "resonance": round(_system["resonance"], 4),
        "daat_awake": _system["daat_awake"],
        "uptime_seconds": int(uptime)
    })

@app.route('/health')
def health():
    return jsonify({
        "health": "healthy" if _system["status"] == "operational" else "degraded",
        "resonance": round(_system["resonance"], 4),
        "daat_awake": _system["daat_awake"],
        "modules": {
            "bus": bool(_system["bus"]),
            "engine": bool(_system["engine"]),
            "tree": bool(_system["tree"]),
            "daat": bool(_system["daat"]),
            "core_govx": bool(_system["core_govx"]),
            "moral_memory": bool(_system["moral_memory"]),
            "willpower": bool(_system["willpower"]),
            "symbiosis": bool(_system["symbiosis"]),
            "binah": bool(_system["binah"])
        }
    })

@app.route('/sephirot/state')
def sephirot_state():
    if not _system["tree"]:
        return jsonify({"error": "Tree not activated"}), 503
    state = _system["tree"].get_tree_state()
    state["resonance"] = _system["resonance"]
    return jsonify(state)

@app.route('/daat/state')
def daat_state():
    if not _system["daat"]:
        return jsonify({"status": "not_initialized"}), 503
    return jsonify(_system["daat"].get_state())

@app.route('/resonance/grow', methods=['POST'])
async def resonance_grow():
    data = request.get_json(silent=True) or {}
    factor = float(data.get('factor', 1.08))
    old = _system["resonance"]
    _system["resonance"] = min(1.0, _system["resonance"] + factor * 0.05)

    return jsonify({
        "success": True,
        "old_resonance": round(old, 4),
        "new_resonance": round(_system["resonance"], 4),
        "delta": round(_system["resonance"] - old, 4)
    })

# =============================================================================
# ЗАПУСК
# =============================================================================
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    logger.info(f"🚀 ISKRA-4 v10.10 Full запускается на порту {port}")

    asyncio.run(initialize_iskra_ultimate())
    asyncio.create_task(background_resonance_growth())

    app.run(host="0.0.0.0", port=port, debug=False)
