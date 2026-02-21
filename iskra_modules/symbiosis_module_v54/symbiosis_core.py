#!/usr/bin/env python3
# =============================================================================
# SYMBIOSIS-CORE v10.10 Ultra Deep + Dataclass + Async
# Полная интеграция с ISKRA-4 через ISKRAAdapter
# =============================================================================
import json
import time
import threading
import asyncio
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
import logging

logger = logging.getLogger("SymbiosisCore")

# =============================================================================
# ПРОТОКОЛЫ БЕЗОПАСНОСТИ (без изменений)
# =============================================================================
class EmergencyProtocol:
    def __init__(self):
        self.rollback_count = 0
        self.consecutive_errors = 0
        self.life_cvar = 0

    def handle_event(self, event: str, context: Dict[str, Any]) -> List[str]:
        recs: List[str] = []
        if event == "resonance_low":
            recs.extend(["immediate_rollback", "pause_all_operations"])
            self.rollback_count += 1
        elif event == "energy_low":
            recs.append("suspend_operations_1h")
        elif event == "shadow_too_high":
            recs.append("isolate_module")
        elif event == "errors_high":
            recs.append("safe_mode")
        else:
            recs.append(f"review_required:{event}")
        return recs

    def check_emergency(self, resonance: float, energy: float, shadow_level: int, error_count: int) -> List[str]:
        actions = []
        if resonance < 0.95:
            actions.extend(self.handle_event("resonance_low", {"resonance": resonance}))
        if energy < 800:
            actions.extend(self.handle_event("energy_low", {"energy": energy}))
        if shadow_level > 8:
            actions.extend(self.handle_event("shadow_too_high", {"shadow_level": shadow_level}))
        if error_count >= 3:
            actions.extend(self.handle_event("errors_high", {"errors": error_count}))
        return actions


class CrisisProtocol:
    def __init__(self):
        self.protocols = {
            "system_collapse": {"steps": ["dump_full_state", "isolate_core", "activate_minimal_mode"]},
            "daat_breach": {"steps": ["block_daat_access", "rollback_to_safe_state"]},
        }

    def evaluate(self, resonance: float, energy: float, shadow_level: int) -> List[str]:
        actions: List[str] = []
        if resonance < 0.90:
            actions.extend(self.protocols["system_collapse"]["steps"])
        if shadow_level == 10:
            actions.extend(self.protocols["daat_breach"]["steps"])
        return actions


class RollbackProtocol:
    def __init__(self, threshold: float = 0.15):
        self.threshold = threshold
        self.rollback_map = {
            "apply_resonance_adjustment": ["revert_resonance_change"],
            "apply_energy_adjustment": ["revert_energy_change"],
            "shadow_operation": ["cancel_shadow_session"],
        }

    def evaluate(self, prev_sym: Optional[float], new_sym: float, prev_actions: List[str]) -> Dict[str, Any]:
        if prev_sym is None or not prev_actions:
            return {"rollback_needed": False, "delta": 0.0, "plan": []}

        delta = new_sym - prev_sym
        if delta < -self.threshold:
            plan: List[str] = []
            for action in prev_actions:
                if action in self.rollback_map:
                    plan.extend(self.rollback_map[action])
            return {"rollback_needed": True, "delta": delta, "actions": prev_actions, "plan": plan}
        return {"rollback_needed": False, "delta": delta, "plan": []}


class ShadowConsentManager:
    def __init__(self, ttl_sec: float = 1800.0):
        self.ttl_sec = ttl_sec
        self._cache = {"granted": False, "expires_at": 0.0}

    def check_consent(self, shadow_level: int, session_mode: str) -> Tuple[bool, List[str]]:
        recs: List[str] = []
        if shadow_level <= 3:
            return True, recs
        if 4 <= shadow_level <= 6:
            recs.extend(["shadow_consent_required", "operator_confirmation_needed"])
            return False, recs
        if shadow_level >= 7:
            recs.extend(["critical_shadow_level", "manual_operator_approval_required"])
            return False, recs
        return False, ["unknown_shadow_level"]


# =============================================================================
# ISKRA ADAPTER
# =============================================================================
class ISKRAAdapter:
    def __init__(self, use_bus: bool = True):
        self.use_bus = use_bus
        self.bus_available = use_bus

    def get_sephirot_state(self) -> Dict[str, Any]:
        return {
            "average_resonance": 0.92,
            "total_energy": 920,
            "activated": True,
            "shadow_level": 2,
            "source": "iskra_bus"
        }

    def apply_symbiosis_delta(self, resonance_delta: float, energy_delta: float) -> Dict[str, Any]:
        return {
            "status": "applied",
            "resonance_delta": resonance_delta,
            "energy_delta": energy_delta,
            "applied_via": "bus" if self.use_bus else "direct"
        }


# =============================================================================
# ОСНОВНОЙ КЛАСС — С DATACLASS
# =============================================================================
@dataclass
class SymbiosisCore:
    """SYMBIOSIS-CORE v10.10 Ultra Deep + Dataclass"""

    version: str = "10.10 Ultra Deep"
    session_mode: str = "readonly"   # readonly, balanced, advanced, experimental

    # Лимиты
    limits: Dict[str, Any] = field(default_factory=lambda: {
        "max_resonance_delta": 0.05,
        "max_energy_delta": 50,
        "min_resonance": 0.9,
        "min_energy": 700,
        "shadow_consent_threshold": 4,
        "emergency_resonance": 0.95,
        "emergency_energy": 800,
        "max_shadow_level": 8
    })

    # Состояние
    symbiosis_score: float = 0.0
    shadow_level: int = 0
    life_cvar: float = 0.0
    rollback_count: int = 0
    consecutive_errors: int = 0
    last_backup: Optional[float] = None

    # История
    state_history: List[Dict] = field(default_factory=list)
    error_history: List[Dict] = field(default_factory=list)

    # Компоненты
    iskra_adapter: ISKRAAdapter = field(default_factory=ISKRAAdapter)
    emergency: EmergencyProtocol = field(default_factory=EmergencyProtocol)
    crisis: CrisisProtocol = field(default_factory=CrisisProtocol)
    rollback: RollbackProtocol = field(default_factory=RollbackProtocol)
    shadow_consent: ShadowConsentManager = field(default_factory=ShadowConsentManager)

    lock: threading.Lock = field(default_factory=threading.Lock, init=False)

    def __post_init__(self):
        logger.info(f"[SYMBIOSIS-CORE v{self.version}] Инициализирован (Dataclass)")
        logger.info(f" Режим: {self.session_mode}")

    # ======================= ОСНОВНАЯ ЛОГИКА ======================= #
    def compute_symbiosis(self) -> Dict[str, Any]:
        with self.lock:
            try:
                iskra_state = self.iskra_adapter.get_sephirot_state()

                resonance = iskra_state.get("average_resonance", 0.9)
                energy = iskra_state.get("total_energy", 850)

                shadow_analysis = self._analyze_shadow_patterns(resonance, energy)
                self.shadow_level = shadow_analysis["level"]
                self.life_cvar = shadow_analysis["risk"]

                self.symbiosis_score = self._calculate_symbiosis_score(resonance, energy, shadow_analysis)

                emergency_actions = self.emergency.check_emergency(resonance, energy, self.shadow_level, self.consecutive_errors)
                crisis_actions = self.crisis.evaluate(resonance, energy, self.shadow_level)

                recommendations = self._generate_recommendations(resonance, energy, self.symbiosis_score, shadow_analysis)

                state = {
                    "timestamp": time.time(),
                    "version": self.version,
                    "session_mode": self.session_mode,
                    "iskra_state": iskra_state,
                    "symbiosis_metrics": {
                        "score": round(self.symbiosis_score, 3),
                        "shadow_level": self.shadow_level,
                        "life_cvar": self.life_cvar,
                        "rollback_count": self.rollback_count
                    },
                    "recommendations": recommendations,
                    "emergency_actions": emergency_actions,
                    "crisis_actions": crisis_actions
                }

                self.state_history.append(state)
                if len(self.state_history) > 1000:
                    self.state_history = self.state_history[-1000:]

                self.consecutive_errors = 0
                return state

            except Exception as e:
                self.consecutive_errors += 1
                return {"status": "error", "error": str(e)}

    # ======================= АСИНХРОННАЯ ВЕРСИЯ ======================= #
    async def integrate_to_iskra_async(self) -> Dict[str, Any]:
        """Асинхронная версия интеграции (рекомендуется использовать)"""
        state = self.compute_symbiosis()

        if "error" in state:
            return state

        recommendations = state["recommendations"]

        # Асинхронное применение через адаптер
        apply_result = self.iskra_adapter.apply_symbiosis_delta(
            recommendations.get("resonance_delta", 0.0),
            recommendations.get("energy_delta", 0.0)
        )

        return {
            "status": "integrated",
            "symbiosis_score": state["symbiosis_metrics"]["score"],
            "shadow_level": state["symbiosis_metrics"]["shadow_level"],
            "applied": apply_result,
            "session_mode": self.session_mode
        }

    # ======================= СИНХРОННАЯ ВЕРСИЯ (для совместимости) ======================= #
    def integrate_to_iskra(self) -> Dict[str, Any]:
        """Синхронная обёртка (оставлена для совместимости)"""
        return asyncio.run(self.integrate_to_iskra_async())

    # ======================= ВСПОМОГАТЕЛЬНЫЕ МЕТОДЫ ======================= #
    def _analyze_shadow_patterns(self, resonance: float, energy: float) -> Dict[str, Any]:
        if resonance < 0.95:
            level = min(10, int((0.95 - resonance) * 100))
            risk = min(100, int((0.95 - resonance) * 200))
        elif energy < 800:
            level = min(10, int((800 - energy) / 20))
            risk = min(100, int((800 - energy) / 2))
        else:
            level = 0
            risk = 0
        return {"level": level, "risk": risk, "stable": resonance >= 0.95 and energy >= 800}

    def _calculate_symbiosis_score(self, resonance: float, energy: float, shadow_analysis: Dict) -> float:
        r_score = max(0.0, min(1.0, resonance))
        e_score = max(0.0, min(1.0, energy / 1000.0))
        s_score = max(0.0, min(1.0, 1.0 - (shadow_analysis.get("level", 0) / 10)))
        return round(r_score * 0.35 + e_score * 0.35 + s_score * 0.3, 3)

    def _generate_recommendations(self, resonance: float, energy: float, symbiosis_score: float, shadow_analysis: Dict) -> Dict[str, Any]:
        actions = []
        warnings = []

        if resonance < 0.95:
            warnings.append("resonance_below_optimal")
        if energy < 800:
            warnings.append("energy_below_optimal")

        if self.session_mode != "readonly":
            if resonance < 0.95 or energy < 800:
                actions.append("apply_micro_adjustments")

        return {
            "resonance_delta": min(self.limits["max_resonance_delta"], 1.0 - resonance) if resonance < 0.95 else 0.0,
            "energy_delta": min(self.limits["max_energy_delta"], 1000 - energy) if energy < 800 else 0.0,
            "actions": actions,
            "warnings": warnings,
            "shadow_level": shadow_analysis.get("level", 0)
        }

    def backup_state(self) -> Dict[str, Any]:
        backup = {
            "timestamp": time.time(),
            "state_history": self.state_history[-200:] if self.state_history else [],
            "symbiosis_score": self.symbiosis_score,
            "shadow_level": self.shadow_level,
            "session_mode": self.session_mode,
            "rollback_count": self.rollback_count
        }
        self.last_backup = time.time()
        return backup

    def update_limits(self, new_limits: Dict[str, Any]) -> Dict[str, Any]:
        updated = []
        for key, value in new_limits.items():
            if key in self.limits:
                old = self.limits[key]
                self.limits[key] = value
                updated.append({"parameter": key, "old": old, "new": value})
        return {"status": "limits_updated", "updated": updated, "limits": self.limits}

    def get_status(self) -> Dict[str, Any]:
        return {
            "version": self.version,
            "session_mode": self.session_mode,
            "symbiosis_score": round(self.symbiosis_score, 3),
            "shadow_level": self.shadow_level,
            "rollback_count": self.rollback_count,
            "consecutive_errors": self.consecutive_errors,
            "history_size": len(self.state_history)
        }

# =============================================================================
# ФАБРИКА
# =============================================================================
def create_symbiosis_core() -> SymbiosisCore:
    return SymbiosisCore()

logger.info("🤝 SymbiosisCore v10.10 Ultra Deep + Dataclass + Async загружен")
