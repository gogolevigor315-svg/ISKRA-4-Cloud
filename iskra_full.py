#!/usr/bin/env python3
# =============================================================================
# ISKRA-4 CLOUD v10.10 — FINAL STABLE VERSION (Render Fixed)
# Полная версия с защитой импортов и явной инициализацией
# =============================================================================
import os
import sys
import asyncio
import logging
from datetime import datetime, timezone
from flask import Flask, jsonify, request
from flask_cors import CORS

# =============================================================================
# ЛОГИРОВАНИЕ
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
# БЕЗОПАСНЫЕ ИМПОРТЫ
# =============================================================================
def safe_import():
    try:
        # Добавляем корень проекта в PYTHONPATH
        project_root = os.path.dirname(os.path.abspath(__file__))
        if project_root not in sys.path:
            sys.path.insert(0, project_root)

        from iskra_modules.sephirot_blocks.sephirot_bus import create_sephirotic_bus
        from iskra_modules.sephirot_blocks.sephirotic_engine import SephiroticEngine
        from iskra_modules.daat_core import get_daat

        logger.info("✅ Все ключевые модули успешно импортированы")
        return create_sephirotic_bus, SephiroticEngine, get_daat

    except Exception as e:
        logger.critical(f"❌ Критическая ошибка импорта: {e}")
        sys.exit(1)

create_bus, SephiroticEngineClass, get_daat = safe_import()

# =============================================================================
# ФОНОВЫЙ РОСТ РЕЗОНАНСА
# =============================================================================
async def background_resonance_growth():
    logger.info("🌱 Фоновый рост резонанса запущен (каждые 3 минуты)")
    while True:
        try:
            await asyncio.sleep(180)
            if _system["status"] == "operational":
                old = _system["resonance"]
                _system["resonance"] = min(1.0, _system["resonance"] + 0.018)
                logger.info(f"🌱 Фоновый рост: {old:.3f} → {_system['resonance']:.3f}")

                if _system["resonance"] >= 0.85 and not _system["daat_awake"] and _system["daat"]:
                    _system["daat_awake"] = True
                    logger.info("🔮 DAAT ПРОБУДИЛСЯ!")
        except asyncio.CancelledError:
            break
        except Exception as e:
            logger.error(f"Ошибка фонового роста: {e}")

# =============================================================================
# ИНИЦИАЛИЗАЦИЯ
# =============================================================================
async def initialize_system():
    global _system
    logger.info("🔥 Запуск инициализации ISKRA-4 v10.10...")

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
        logger.info(f"🎉 Система успешно запущена | Резонанс: {_system['resonance']:.3f}")
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
        "daat_awake": _system["daat_awake"],
        "tree_activated": _system["tree_activated"]
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
    logger.info(f"🚀 ISKRA-4 v10.10 запускается на порту {port}")

    asyncio.run(initialize_system())
    asyncio.create_task(background_resonance_growth())

    app.run(host="0.0.0.0", port=port, debug=False)
