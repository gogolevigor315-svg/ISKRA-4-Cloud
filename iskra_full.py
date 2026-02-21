#!/usr/bin/env python3
# =============================================================================
# ISKRA-4 CLOUD v10.10 — FINAL ORCHESTRATOR + BACKGROUND GROWTH
# Полная версия с фоновым ростом резонанса и детальным мониторингом
# =============================================================================
import os
import sys
import asyncio
import logging
import time
from datetime import datetime, timezone
from typing import Dict, Any
import psutil
from flask import Flask, jsonify, request
from flask_cors import CORS

# =============================================================================
# ЛОГИРОВАНИЕ (в консоль + файл)
# =============================================================================
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(name)s | %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('iskra-4.log', encoding='utf-8', mode='a')
    ]
)
logger = logging.getLogger("ISKRA-4")

# =============================================================================
# FLASK
# =============================================================================
app = Flask(__name__)
CORS(app)

# =============================================================================
# ГЛОБАЛЬНОЕ СОСТОЯНИЕ
# =============================================================================
_system = {
    "version": "10.10 Final+",
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

# =============================================================================
# ЛЕНИВЫЕ ИМПОРТЫ
# =============================================================================
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

# =============================================================================
# ФОНОВЫЙ РОСТ РЕЗОНАНСА
# =============================================================================
async def background_resonance_growth():
    """Фоновый рост резонанса каждые 3 минуты"""
    logger.info("🌱 Запущен фоновый рост резонанса (каждые 3 минуты)")
    
    while True:
        try:
            await asyncio.sleep(180)  # 3 минуты
            
            if _system["status"] == "operational":
                old_res = _system["resonance"]
                _system["resonance"] = min(1.0, _system["resonance"] + 0.018)
                
                logger.info(f"🌱 Фоновый рост: {old_res:.3f} → {_system['resonance']:.3f}")
                
                # Проверка готовности DAAT
                if _system["resonance"] >= 0.85 and not _system["daat_awake"] and _system["daat"]:
                    _system["daat_awake"] = True
                    logger.info("🔮 DAAT ДОСТИГНУЛ ПОРОГА И ПРОБУДИЛСЯ!")
                    
        except asyncio.CancelledError:
            logger.info("🌱 Фоновая задача роста резонанса остановлена")
            break
        except Exception as e:
            logger.error(f"Ошибка в фоновом росте резонанса: {e}")

# =============================================================================
# АСИНХРОННАЯ ИНИЦИАЛИЗАЦИЯ
# =============================================================================
async def initialize_iskra_ultimate():
    global _system
    logger.info("🔥 ЗАПУСК ПОЛНОЙ ИНИЦИАЛИЗАЦИИ ISKRA-4 v10.10...")

    try:
        bus = get_bus()
        _system["bus"] = bus

        engine = await get_engine()
        await engine.initialize(bus=bus)
        _system["engine"] = engine

        await engine.activate()
        tree = engine.tree
        _system["tree"] = tree
        _system["tree_activated"] = True

        daat = get_daat()
        _system["daat"] = daat

        _system["core_govx"] = get_core_govx()
        _system["moral_memory"] = get_moral_memory()
        _system["willpower"] = await get_willpower()
        _system["symbiosis"] = get_symbiosis()
        _system["binah"] = get_binah()

        _system["status"] = "operational"

        logger.info(f"🎉 ISKRA-4 v10.10 УСПЕШНО ЗАПУЩЕНА | Резонанс: {_system['resonance']:.3f}")
        return True

    except Exception as e:
        logger.critical(f"💥 КРИТИЧЕСКАЯ ОШИБКА ИНИЦИАЛИЗАЦИИ: {e}")
        _system["status"] = "failed"
        return False

# =============================================================================
# ЗАПУСК ФОНОВЫХ ЗАДАЧ
# =============================================================================
@app.before_serving
async def startup():
    """Запуск при старте сервера"""
    await initialize_iskra_ultimate()
    asyncio.create_task(background_resonance_growth())

# =============================================================================
# ЭНДПОИНТЫ
# =============================================================================
@app.route('/')
def index():
    uptime = (datetime.now(timezone.utc) - _system["start_time"]).total_seconds()
    return jsonify({
        "system": "ISKRA-4 CLOUD",
        "version": "10.10 Final+",
        "status": _system["status"],
        "resonance": round(_system["resonance"], 4),
        "daat_awake": _system["daat_awake"],
        "tree_activated": _system["tree_activated"],
        "uptime_seconds": int(uptime)
    })

@app.route('/health')
def health():
    return jsonify({
        "health": "healthy" if _system["status"] == "operational" else "degraded",
        "resonance": round(_system["resonance"], 4),
        "daat_awake": _system["daat_awake"],
        "tree_activated": _system["tree_activated"],
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
    state["daat_integrated"] = "DAAT" in getattr(_system["tree"], "nodes", {})
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
    target = float(data.get('target', 0.85))

    if not _system["tree"]:
        return jsonify({"error": "Tree not ready"}), 503

    old_res = _system["resonance"]
    _system["resonance"] = min(1.0, _system["resonance"] + factor * 0.05)

    daat_ready = _system["resonance"] >= target

    return jsonify({
        "success": True,
        "old_resonance": round(old_res, 4),
        "new_resonance": round(_system["resonance"], 4),
        "delta": round(_system["resonance"] - old_res, 4),
        "daat_ready": daat_ready,
        "message": "Резонанс увеличен" if daat_ready else "Резонанс растёт..."
    })

@app.route('/activate', methods=['POST'])
async def activate():
    success = await initialize_iskra_ultimate()
    return jsonify({
        "success": success,
        "resonance": round(_system["resonance"], 4),
        "status": _system["status"]
    })

# =============================================================================
# ЗАПУСК
# =============================================================================
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    logger.info(f"🚀 ISKRA-4 v10.10 Final+ запускается на порту {port}")
    app.run(host="0.0.0.0", port=port, debug=False)
