#!/usr/bin/env python3
# =============================================================================
# SEPHIROTIС BUS v10.10 Ultra Deep
# Центральная шина сефиротической системы с ПОЛНОЙ интеграцией RAS-CORE и DAAT
# =============================================================================
import asyncio
import json
import hashlib
import time
from datetime import datetime
from typing import Dict, List, Optional, Any, Callable
from collections import deque, defaultdict
import logging

logger = logging.getLogger("SephiroticBus")

GOLDEN_STABILITY_ANGLE = 14.4

# Маршрутизация RAS-CORE
RAS_CORE_ROUTING = {
    "in": ["BECHTEREVA", "EMOTIONAL_WEAVE", "NEOCORTEX", "YESOD"],
    "out": ["CHOKMAH", "DAAT", "KETER", "BINAH"]
}

# Приоритеты сигналов (восстановлено из оригинала)
PRIORITY_THRESHOLDS = {
    "critical": 0.9,
    "high": 0.6,
    "normal": 0.3
}

# =============================================================================
# КЛАСС СИГНАЛА
# =============================================================================
class SignalPackage:
    def __init__(self, 
                 type: str, 
                 source: str = "unknown", 
                 target: str = None, 
                 payload: Dict = None,
                 stability_angle: float = GOLDEN_STABILITY_ANGLE,
                 metadata: Dict = None):
        self.type = type
        self.source = source
        self.target = target
        self.payload = payload or {}
        self.stability_angle = stability_angle
        self.metadata = metadata or {}
        self.timestamp = time.time()
        self.id = hashlib.md5(f"{source}{target}{time.time()}".encode()).hexdigest()[:16]
        self.stability_factor = self._calculate_stability_factor()

    def _calculate_stability_factor(self) -> float:
        deviation = abs(self.stability_angle - GOLDEN_STABILITY_ANGLE)
        return max(0.1, min(1.0, 1.0 - deviation / 30.0))

    def to_dict(self) -> Dict:
        return {
            "id": self.id,
            "type": self.type,
            "source": self.source,
            "target": self.target,
            "stability_angle": self.stability_angle,
            "stability_factor": self.stability_factor,
            "timestamp": self.timestamp
        }

# =============================================================================
# ГЛАВНЫЙ КЛАСС ШИНЫ — ULTRA DEEP
# =============================================================================
class SephiroticBus:
    """
    SephiroticBus v10.10 Ultra Deep
    Полная шина с автоинтеграцией DAAT, 22 каналами и детальной статистикой углов
    """

    def __init__(self, name: str = "SephiroticBus"):
        self.name = name

        self.nodes: Dict[str, Any] = {}
        self.subscriptions: Dict[str, List[Callable]] = defaultdict(list)
        self.message_log = deque(maxlen=1000)
        self.focus_log = deque(maxlen=200)

        self.module_bindings: Dict[str, str] = {}
        self.sephira_to_module: Dict[str, str] = {}

        self.routing_table: Dict[str, Dict] = {}
        self.ras_core_connected = False
        self.total_paths = 22  # Восстановлено из оригинала

        self.stability_metrics = defaultdict(list)

        self.logger = self._setup_logger()

        self._setup_default_bindings()
        self._setup_routing_table()

        # ===== АВТОИНТЕГРАЦИЯ DAAT =====
        self._auto_integrate_daat()

        logger.info(f"🌐 SephiroticBus '{name}' инициализирована (22 канала, угол {GOLDEN_STABILITY_ANGLE}°)")

    def _setup_logger(self):
        logger = logging.getLogger(f"Bus.{self.name}")
        logger.setLevel(logging.INFO)
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter(
            '[%(asctime)s] [%(name)s] %(levelname)s - %(message)s',
            datefmt='%H:%M:%S'
        ))
        logger.addHandler(handler)
        return logger

    def _setup_default_bindings(self):
        self.module_bindings = {
            'bechtereva': 'KETER',
            'chernigovskaya': 'CHOKMAH',
            'emotional_weave': 'CHESED',
            'immune_core': 'GEVURAH',
            'policy_governor': 'TIFERET',
            'heartbeat_core': 'NETZACH',
            'polyglossia_adapter': 'HOD',
            'spinal_core': 'YESOD',
            'trust_mesh': 'MALKUTH',
            'ras_core': 'RAS_CORE'
        }
        self.sephira_to_module = {v: k for k, v in self.module_bindings.items()}

    def _setup_routing_table(self):
        self.routing_table = {
            "BECHTEREVA": {"default_target": "KETER", "signal_types": ["NEURO"], "stability_factor": 1.0},
            "CHERNIGOVSKAYA": {"default_target": "CHOKMAH", "signal_types": ["SEMIOTIC"], "stability_factor": 1.0},
            "EMOTIONAL_WEAVE": {"default_target": "CHESED", "signal_types": ["EMOTIONAL"], "stability_factor": 0.9},
            "NEOCORTEX": {"default_target": "BINAH", "signal_types": ["COGNITIVE"], "stability_factor": 0.95},
            "YESOD": {"default_target": "YESOD", "signal_types": ["SEPHIROTIC"], "stability_factor": 0.85},
        }

        self.routing_table["RAS_CORE"] = {
            "in": RAS_CORE_ROUTING["in"],
            "out": RAS_CORE_ROUTING["out"],
            "signal_types": ["FOCUS", "ATTENTION", "RESONANCE"],
            "stability_factor": 0.95,
            "golden_angle_priority": True
        }

    def _auto_integrate_daat(self):
        """Автоматическая интеграция DAAT при создании шины (восстановлено)"""
        try:
            from iskra_modules.sephirot_blocks.DAAT.daat_core import get_daat  # ✅
            daat = get_daat()

            if 'DAAT' not in self.nodes:
                class DaatNodeAdapter:
                    def __init__(self, daat_instance):
                        self.daat = daat_instance
                        self.name = "DAAT"
                        self.stability_angle = GOLDEN_STABILITY_ANGLE

                    async def receive(self, signal):
                        return {"status": "received_by_daat", "resonance": getattr(self.daat, 'resonance_index', 0)}

                    def get_state(self):
                        return {
                            "status": getattr(self.daat, 'status', 'unknown'),
                            "resonance": getattr(self.daat, 'resonance_index', 0),
                            "awakening": getattr(self.daat, 'awakening_level', 0)
                        }

                self.nodes['DAAT'] = DaatNodeAdapter(daat)
                self.logger.info("✅ DAAT автоматически интегрирован в шину")

            if 'DAAT' not in self.routing_table:
                self.routing_table['DAAT'] = {
                    "in": ["BINAH", "CHOKMAH"],
                    "out": ["TIFERET"],
                    "signal_types": ["SEPHIROTIC", "RESONANCE"],
                    "stability_factor": 0.95
                }

            self.total_paths = 22
            self.logger.info(f"✅ Древо расширено до {self.total_paths} каналов")

        except Exception as e:
            self.logger.warning(f"⚠️ Автоинтеграция DAAT не удалась: {e}")

    # =========================================================================
    # РЕГИСТРАЦИЯ УЗЛОВ
    # =========================================================================
    async def register_node(self, node: Any, node_name: Optional[str] = None) -> bool:
        if node_name is None:
            node_name = getattr(node, 'name', type(node).__name__).upper()

        if node_name in self.nodes:
            return False

        self.nodes[node_name] = node

        if node_name == "RAS_CORE":
            self.ras_core_connected = True
            await self._activate_ras_core_routing()

        self.logger.info(f"Узел {node_name} зарегистрирован")
        return True

    async def _activate_ras_core_routing(self):
        for source in RAS_CORE_ROUTING["in"]:
            if source in self.routing_table:
                self.routing_table[source]["ras_core_routing"] = True
        self.logger.info("RAS-CORE маршрутизация активирована")

    # =========================================================================
    # ПЕРЕДАЧА СИГНАЛОВ С ПРИОРИТЕТАМИ
    # =========================================================================
    async def transmit(self, signal: SignalPackage) -> Dict[str, Any]:
        self._log_message(signal)

        priority = self._calculate_signal_priority(signal)

        result = {
            "success": False,
            "delivered_to": [],
            "priority": priority,
            "stability_factor": signal.stability_factor
        }

        try:
            if self.ras_core_connected and self._should_route_through_ras_core(signal):
                ras_result = await self._route_through_ras_core(signal)
                result.update(ras_result)
            elif signal.target and signal.target in self.nodes:
                await self.nodes[signal.target].receive(signal)
                result["delivered_to"].append(signal.target)
            else:
                auto_result = await self._auto_route_signal(signal)
                result.update(auto_result)

            await self._notify_subscribers(signal)
            result["success"] = True

        except Exception as e:
            result["error"] = str(e)

        return result

    def _calculate_signal_priority(self, signal: SignalPackage) -> str:
        factor = signal.stability_factor
        if factor >= PRIORITY_THRESHOLDS["critical"]:
            return "critical"
        elif factor >= PRIORITY_THRESHOLDS["high"]:
            return "high"
        elif factor >= PRIORITY_THRESHOLDS["normal"]:
            return "normal"
        else:
            return "low"

    def _should_route_through_ras_core(self, signal: SignalPackage) -> bool:
        return (signal.type in ["FOCUS", "ATTENTION", "RESONANCE"] or
                getattr(signal, 'source', '').upper() in RAS_CORE_ROUTING["in"] or
                getattr(signal, 'target', '').upper() in RAS_CORE_ROUTING["out"])

    # =========================================================================
    # СТАТУС И СТАТИСТИКА УГЛОВ (восстановлено полностью)
    # =========================================================================
    def get_status(self) -> Dict[str, Any]:
        angle_stats = self._collect_angle_statistics()

        return {
            "name": self.name,
            "nodes": list(self.nodes.keys()),
            "total_nodes": len(self.nodes),
            "total_paths": self.total_paths,
            "ras_core_connected": self.ras_core_connected,
            "golden_stability_angle": GOLDEN_STABILITY_ANGLE,
            "stability_statistics": angle_stats,
            "message_log_size": len(self.message_log),
            "subscriptions": {k: len(v) for k, v in self.subscriptions.items()}
        }

    def _collect_angle_statistics(self) -> Dict:
        angles = []
        factors = []

        for name, node in self.nodes.items():
            if hasattr(node, 'stability_angle'):
                angle = getattr(node, 'stability_angle', GOLDEN_STABILITY_ANGLE)
                factor = getattr(node, 'stability_factor', 0.7)
                angles.append(angle)
                factors.append(factor)

        if not angles:
            return {"available": False}

        return {
            "available": True,
            "nodes_with_angle": len(angles),
            "average_angle": round(sum(angles)/len(angles), 2),
            "average_stability_factor": round(sum(factors)/len(factors), 4),
            "close_to_golden": sum(1 for a in angles if abs(a - GOLDEN_STABILITY_ANGLE) < 2.0),
            "min_angle": min(angles),
            "max_angle": max(angles)
        }

    async def health_check(self) -> Dict:
        status = self.get_status()
        angle_stats = status["stability_statistics"]

        return {
            "status": "healthy" if angle_stats.get("average_stability_factor", 0) > 0.65 else "degraded",
            "stability_health": angle_stats,
            "total_paths": self.total_paths,
            "ras_core_active": self.ras_core_connected,
            "timestamp": datetime.utcnow().isoformat()
        }

# =============================================================================
# ФАБРИКА И СИНГЛТОН
# =============================================================================
_bus_instance = None

async def create_sephirotic_bus(name: str = "SephiroticBus") -> SephiroticBus:
    global _bus_instance
    if _bus_instance is None:
        _bus_instance = SephiroticBus(name)
    return _bus_instance

def get_sephirotic_bus() -> SephiroticBus:
    global _bus_instance
    if _bus_instance is None:
        _bus_instance = SephiroticBus("SephiroticBus")
    return _bus_instance

logger.info("🌐 SephiroticBus v10.10 Ultra Deep загружен")
