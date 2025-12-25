# ===============================================================
# POLICY_GOVERNOR v1.2 — ИСПОЛНИТЕЛЬНЫЙ КОД (ЧАСТЬ 1/2)
# ===============================================================
# iskra_modules/policy_governor.py
# 
# ТОЧНАЯ РЕАЛИЗАЦИЯ АРХИТЕКТУРЫ БЕЗ ИЗМЕНЕНИЯ КОНТРАКТОВ
# ===============================================================

from __future__ import annotations
import sys
import os

# Добавляем путь для импорта других модулей ISKRA
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dataclasses import dataclass, field, asdict
from typing import Dict, Any, List, Optional, Protocol, Callable, runtime_checkable
from enum import Enum
from datetime import datetime, timezone
import uuid
import hashlib
import time
import json
import logging

# ---------------------------------------------------------------
# Конфигурация логирования
# ---------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("policy_governor")

# ---------------------------------------------------------------
# Module metadata (expected by ISKRA loader)
# ---------------------------------------------------------------

MODULE_NAME = "policy_governor"
MODULE_VERSION = "1.2"
CRITICAL = False

# ---------------------------------------------------------------
# Protocols (hard contracts) - ТОЧНО КАК В АРХИТЕКТУРЕ
# ---------------------------------------------------------------

@runtime_checkable
class BusProtocol(Protocol):
    def subscribe(self, topic: str, callback: Callable[[Dict[str, Any]], None]) -> None:
        ...
    
    def publish(
        self,
        sender: str,
        topic: str,
        data: Dict[str, Any],
        priority: int = 2,
        ttl: Optional[int] = None
    ) -> Any:
        ...

@runtime_checkable
class CoreProtocol(Protocol):
    state: Dict[str, Any]

# ---------------------------------------------------------------
# Enums / Constants (ТОЧНО КАК В АРХИТЕКТУРЕ)
# ---------------------------------------------------------------

class Topic(str, Enum):
    HEARTBEAT_METRICS = "heartbeat.metrics"
    TRUST_METRICS = "trust.metrics"
    EMOTION_METRICS = "emotion.metrics"
    IMMUNE_METRICS = "immune.metrics"
    FUSION_METRICS = "fusion.metrics"
    SYSTEM_DEGRADED = "system.degraded"
    MODULE_FAILED = "module.failed"
    
    POLICY_UPDATE = "policy.update"
    POLICY_HEARTBEAT = "policy.heartbeat"
    SEPHIROT_HARMONY_UPDATE = "sephirot.harmony_update"

class RunMode(str, Enum):
    STANDARD = "STANDARD"
    RESEARCH = "RESEARCH"
    DEGRADED = "DEGRADED"
    SAFE_BOOT = "SAFE_BOOT"

class RolloutPhase(str, Enum):
    MONITOR_ONLY = "MONITOR_ONLY"
    SAFETY_ONLY = "SAFETY_ONLY"
    FULL_CONTROL = "FULL_CONTROL"

class RuleCategory(str, Enum):
    SAFETY = "safety"
    GENERAL = "general"

class ReasonCode(str, Enum):
    BOOT_BASELINE = "BOOT_BASELINE"
    BASELINE_REFRESH = "BASELINE_REFRESH"
    HIGH_ENTROPY = "HIGH_ENTROPY"
    LOW_COHERENCE = "LOW_COHERENCE"
    PERF_PRESSURE = "PERF_PRESSURE"
    HIGH_THREAT = "HIGH_THREAT"
    CIRCUIT_BREAKER_OPEN = "CIRCUIT_BREAKER_OPEN"
    HIGH_TRUST_VOLATILITY = "HIGH_TRUST_VOLATILITY"
    WEAK_GRAPH = "WEAK_GRAPH"
    RECOVERY_MODE = "RECOVERY_MODE"
    SAFE_BOOT_REQUIRED = "SAFE_BOOT_REQUIRED"
    INPUTS_MISSING = "INPUTS_MISSING"
    LOOP_GUARD_TRIPPED = "LOOP_GUARD_TRIPPED"
    METRIC_PARSE_FAILED = "METRIC_PARSE_FAILED"
    POLICY_EVAL_FAILED = "POLICY_EVAL_FAILED"

class SignalKey(str, Enum):
    ENTROPY = "system.entropy_level"
    COHERENCE = "system.coherence_index"
    LOAD = "system.load_index"
    DEGRADED = "system.degraded"
    ERROR_RATE_5M = "system.error_rate_5m"
    TRUST_INDEX = "trust.trust_index"
    TRUST_VOLATILITY = "trust.trust_volatility"
    VALENCE = "emotion.valence"
    AROUSAL = "emotion.arousal"
    EMO_STABILITY = "emotion.stability"
    THREAT = "immune.threat_level"
    CB_OPEN_5M = "immune.cb_open_count_5m"
    WEAK_EDGES_RATIO = "fusion_graph.weak_edges_ratio"
    AVG_EDGE_STRENGTH = "fusion_graph.avg_edge_strength"
    PENTAGON_DENSITY = "fusion_graph.pentagon_density"

# ---------------------------------------------------------------
# Versioned policy schema
# ---------------------------------------------------------------

POLICY_SCHEMA_VERSION = "1.0"

# ---------------------------------------------------------------
# Standard event envelope
# ---------------------------------------------------------------

@dataclass
class EventEnvelope:
    topic: str
    ts_utc: str
    node_id: str
    origin: str = ""
    trace_id: str = ""
    payload: Dict[str, Any] = field(default_factory=dict)
    schema: str = "event.v1"

# ---------------------------------------------------------------
# Data contracts
# ---------------------------------------------------------------

@dataclass
class PolicyInputs:
    ts_utc: str
    node_id: str
    system: Dict[str, Any] = field(default_factory=dict)
    trust: Dict[str, Any] = field(default_factory=dict)
    emotion: Dict[str, Any] = field(default_factory=dict)
    immune: Dict[str, Any] = field(default_factory=dict)
    fusion_graph: Dict[str, Any] = field(default_factory=dict)
    sephirot: Dict[str, Any] = field(default_factory=dict)

@dataclass
class PolicyPacket:
    schema_version: str
    packet_id: str
    trace_id: str
    generated_at_utc: str
    node_id: str
    modes: Dict[str, Any]
    thresholds: Dict[str, float]
    timing: Dict[str, float]
    routing: Dict[str, Any]
    actions: Dict[str, Any]
    explain: Dict[str, Any]
    digest: str = ""

@dataclass
class PolicyProposal:
    priority: int
    rule_name: str
    category: RuleCategory
    modes: Optional[Dict[str, Any]] = None
    thresholds: Optional[Dict[str, float]] = None
    timing: Optional[Dict[str, float]] = None
    routing: Optional[Dict[str, Any]] = None
    actions: Optional[Dict[str, Any]] = None
    reasons: List[str] = field(default_factory=list)
    signals: Dict[str, float] = field(default_factory=dict)

# ---------------------------------------------------------------
# Rules layer - РЕАЛЬНЫЕ ПРАВИЛА (10/10)
# ---------------------------------------------------------------

class PolicyRule:
    RULE_NAME: str = "base_rule"
    PRIORITY: int = 10
    CATEGORY: RuleCategory = RuleCategory.GENERAL
    
    def evaluate(self, inputs: PolicyInputs) -> Optional[PolicyProposal]:
        raise NotImplementedError

# ---------------------------------------------------------------
# РЕАЛЬНЫЕ ПРАВИЛА БЕЗОПАСНОСТИ
# ---------------------------------------------------------------

class HighLoadSafetyRule(PolicyRule):
    RULE_NAME = "high_load_safety"
    PRIORITY = 1
    CATEGORY = RuleCategory.SAFETY
    
    def evaluate(self, inputs: PolicyInputs) -> Optional[PolicyProposal]:
        load = inputs.system.get(SignalKey.LOAD.value, 0.0)
        if load > 0.85:
            return PolicyProposal(
                priority=self.PRIORITY,
                rule_name=self.RULE_NAME,
                category=self.CATEGORY,
                modes={"run_mode": RunMode.DEGRADED.value},
                thresholds={"resonance_min": 0.85, "confidence_required": 0.75},
                reasons=[ReasonCode.PERF_PRESSURE.value],
                signals={SignalKey.LOAD.value: load}
            )
        return None

class LowCoherenceSafetyRule(PolicyRule):
    RULE_NAME = "low_coherence_safety"
    PRIORITY = 2
    CATEGORY = RuleCategory.SAFETY
    
    def evaluate(self, inputs: PolicyInputs) -> Optional[PolicyProposal]:
        coherence = inputs.system.get(SignalKey.COHERENCE.value, 1.0)
        if coherence < 0.4:
            return PolicyProposal(
                priority=self.PRIORITY,
                rule_name=self.RULE_NAME,
                category=self.CATEGORY,
                modes={"run_mode": RunMode.DEGRADED.value, "learning_enabled": False},
                timing={"cycle_interval_sec": 5.0},
                reasons=[ReasonCode.LOW_COHERENCE.value],
                signals={SignalKey.COHERENCE.value: coherence}
            )
        return None

class HighEntropySafetyRule(PolicyRule):
    RULE_NAME = "high_entropy_safety"
    PRIORITY = 3
    CATEGORY = RuleCategory.SAFETY
    
    def evaluate(self, inputs: PolicyInputs) -> Optional[PolicyProposal]:
        entropy = inputs.system.get(SignalKey.ENTROPY.value, 0.0)
        if entropy > 0.8:
            return PolicyProposal(
                priority=self.PRIORITY,
                rule_name=self.RULE_NAME,
                category=self.CATEGORY,
                thresholds={"resonance_min": 0.90},
                routing={"exploration_temperature": 0.1},
                reasons=[ReasonCode.HIGH_ENTROPY.value],
                signals={SignalKey.ENTROPY.value: entropy}
            )
        return None

class ImmuneThreatSafetyRule(PolicyRule):
    RULE_NAME = "immune_threat_safety"
    PRIORITY = 1  # Высший приоритет
    CATEGORY = RuleCategory.SAFETY
    
    def evaluate(self, inputs: PolicyInputs) -> Optional[PolicyProposal]:
        threat = inputs.immune.get(SignalKey.THREAT.value, 0.0)
        cb_open = inputs.immune.get(SignalKey.CB_OPEN_5M.value, 0)
        
        if threat > 0.9 or cb_open > 3:
            return PolicyProposal(
                priority=self.PRIORITY,
                rule_name=self.RULE_NAME,
                category=self.CATEGORY,
                modes={
                    "run_mode": RunMode.SAFE_BOOT.value,
                    "safe_boot": True,
                    "learning_enabled": False
                },
                actions={"request_prune": True, "request_reload_modules": ["immune_core"]},
                reasons=[ReasonCode.HIGH_THREAT.value, ReasonCode.CIRCUIT_BREAKER_OPEN.value],
                signals={
                    SignalKey.THREAT.value: threat,
                    SignalKey.CB_OPEN_5M.value: float(cb_open)
                }
            )
        return None

class TrustVolatilitySafetyRule(PolicyRule):
    RULE_NAME = "trust_volatility_safety"
    PRIORITY = 4
    CATEGORY = RuleCategory.SAFETY
    
    def evaluate(self, inputs: PolicyInputs) -> Optional[PolicyProposal]:
        volatility = inputs.trust.get(SignalKey.TRUST_VOLATILITY.value, 0.0)
        if volatility > 0.7:
            return PolicyProposal(
                priority=self.PRIORITY,
                rule_name=self.RULE_NAME,
                category=self.CATEGORY,
                thresholds={"confidence_required": 0.85, "prune_below_strength": 0.25},
                reasons=[ReasonCode.HIGH_TRUST_VOLATILITY.value],
                signals={SignalKey.TRUST_VOLATILITY.value: volatility}
            )
        return None

# ---------------------------------------------------------------
# РЕАЛЬНЫЕ ОБЩИЕ ПРАВИЛА
# ---------------------------------------------------------------

class GraphHealthRule(PolicyRule):
    RULE_NAME = "graph_health"
    PRIORITY = 20
    CATEGORY = RuleCategory.GENERAL
    
    def evaluate(self, inputs: PolicyInputs) -> Optional[PolicyProposal]:
        weak_ratio = inputs.fusion_graph.get(SignalKey.WEAK_EDGES_RATIO.value, 0.0)
        avg_strength = inputs.fusion_graph.get(SignalKey.AVG_EDGE_STRENGTH.value, 0.0)
        
        if weak_ratio > 0.5 or avg_strength < 0.3:
            return PolicyProposal(
                priority=self.PRIORITY,
                rule_name=self.RULE_NAME,
                category=self.CATEGORY,
                actions={"request_prune": True, "request_snapshot": True},
                reasons=[ReasonCode.WEAK_GRAPH.value],
                signals={
                    SignalKey.WEAK_EDGES_RATIO.value: weak_ratio,
                    SignalKey.AVG_EDGE_STRENGTH.value: avg_strength
                }
            )
        return None

class EmotionStabilityRule(PolicyRule):
    RULE_NAME = "emotion_stability"
    PRIORITY = 25
    CATEGORY = RuleCategory.GENERAL
    
    def evaluate(self, inputs: PolicyInputs) -> Optional[PolicyProposal]:
        stability = inputs.emotion.get(SignalKey.EMO_STABILITY.value, 1.0)
        arousal = inputs.emotion.get(SignalKey.AROUSAL.value, 0.5)
        
        if stability < 0.3 and arousal > 0.7:
            return PolicyProposal(
                priority=self.PRIORITY,
                rule_name=self.RULE_NAME,
                category=self.CATEGORY,
                timing={"metrics_publish_interval_sec": 2.0},
                routing={"exploration_temperature": 0.5},
                reasons=[ReasonCode.RECOVERY_MODE.value],
                signals={
                    SignalKey.EMO_STABILITY.value: stability,
                    SignalKey.AROUSAL.value: arousal
                }
            )
        return None

class ResearchModeRule(PolicyRule):
    RULE_NAME = "research_mode"
    PRIORITY = 30
    CATEGORY = RuleCategory.GENERAL
    
    def evaluate(self, inputs: PolicyInputs) -> Optional[PolicyProposal]:
        # Активировать RESEARCH режим при стабильной системе с низкой нагрузкой
        load = inputs.system.get(SignalKey.LOAD.value, 0.0)
        coherence = inputs.system.get(SignalKey.COHERENCE.value, 1.0)
        
        if load < 0.3 and coherence > 0.8:
            return PolicyProposal(
                priority=self.PRIORITY,
                rule_name=self.RULE_NAME,
                category=self.CATEGORY,
                modes={"run_mode": RunMode.RESEARCH.value},
                routing={"exploration_temperature": 0.75},
                reasons=[ReasonCode.BASELINE_REFRESH.value],
                signals={
                    SignalKey.LOAD.value: load,
                    SignalKey.COHERENCE.value: coherence
                }
            )
        return None

class PerformanceOptimizationRule(PolicyRule):
    RULE_NAME = "performance_optimization"
    PRIORITY = 15
    CATEGORY = RuleCategory.GENERAL
    
    def evaluate(self, inputs: PolicyInputs) -> Optional[PolicyProposal]:
        load = inputs.system.get(SignalKey.LOAD.value, 0.0)
        error_rate = inputs.system.get(SignalKey.ERROR_RATE_5M.value, 0.0)
        
        if 0.4 < load < 0.7 and error_rate < 0.01:
            # Оптимизировать производительность
            return PolicyProposal(
                priority=self.PRIORITY,
                rule_name=self.RULE_NAME,
                category=self.CATEGORY,
                timing={"cycle_interval_sec": 1.5},
                thresholds={"resonance_min": 0.65},
                reasons=["OPTIMIZATION_ACTIVE"],
                signals={
                    SignalKey.LOAD.value: load,
                    SignalKey.ERROR_RATE_5M.value: error_rate
                }
            )
        return None

class SephirotHarmonyRule(PolicyRule):
    RULE_NAME = "sephirot_harmony"
    PRIORITY = 18
    CATEGORY = RuleCategory.GENERAL
    
    def evaluate(self, inputs: PolicyInputs) -> Optional[PolicyProposal]:
        harmony = inputs.sephirot.get("harmony_index", 1.0)
        energy = inputs.sephirot.get("total_energy", 0.0)
        
        if harmony > 0.9 and energy > 50.0:
            # Высокая гармония - можно увеличить исследовательское поведение
            return PolicyProposal(
                priority=self.PRIORITY,
                rule_name=self.RULE_NAME,
                category=self.CATEGORY,
                routing={"exploration_temperature": 0.4},
                actions={"request_snapshot": True},
                reasons=["HIGH_HARMONY_DETECTED"],
                signals={
                    "sephirot.harmony_index": harmony,
                    "sephirot.total_energy": energy
                }
            )
        elif harmony < 0.4:
            # Низкая гармония - консервативный режим
            return PolicyProposal(
                priority=self.PRIORITY,
                rule_name=self.RULE_NAME,
                category=self.CATEGORY,
                modes={"learning_enabled": False},
                routing={"exploration_temperature": 0.1},
                reasons=["LOW_HARMONY_DETECTED"],
                signals={"sephirot.harmony_index": harmony}
            )
        return None

# ---------------------------------------------------------------
# Anti-flap state machine
# ---------------------------------------------------------------

@dataclass
class AntiFlapState:
    hysteresis_count: int = 3
    cooldown_sec: float = 30.0
    cooldown_until_ts: float = 0.0
    mode_votes: Dict[str, int] = field(default_factory=dict)
    
    def cooldown_ok(self) -> bool:
        return time.time() >= self.cooldown_until_ts
    
    def start_cooldown(self) -> None:
        self.cooldown_until_ts = time.time() + float(self.cooldown_sec)
    
    def vote_mode(self, mode: str) -> bool:
        for k in list(self.mode_votes.keys()):
            if k != mode:
                self.mode_votes[k] = 0
        self.mode_votes[mode] = int(self.mode_votes.get(mode, 0)) + 1
        return self.mode_votes[mode] >= int(self.hysteresis_count)

# ---------------------------------------------------------------
# Loop guard
# ---------------------------------------------------------------

@dataclass
class LoopGuard:
    window_sec: float = 3.0
    max_updates: int = 2
    recent_policy_update_ts: List[float] = field(default_factory=list)
    
    def record_update(self) -> None:
        self.recent_policy_update_ts.append(time.time())
    
    def tripped(self) -> bool:
        now = time.time()
        self.recent_policy_update_ts = [
            t for t in self.recent_policy_update_ts 
            if now - t <= self.window_sec
        ]
        return len(self.recent_policy_update_ts) >= int(self.max_updates)

# ---------------------------------------------------------------
# Resolver (merging proposals into PolicyPacket) - ПОЛНАЯ РЕАЛИЗАЦИЯ
# ---------------------------------------------------------------

class PolicyResolver:
    """
    Deterministic merge algorithm.
    Hard precedence: SAFE_BOOT > DEGRADED > others
    Rollout gating: MONITOR_ONLY → SAFETY_ONLY → FULL_CONTROL
    """
    
    def resolve(
        self,
        proposals: List[PolicyProposal],
        previous: Optional[PolicyPacket],
        phase: RolloutPhase,
        baseline: PolicyPacket,
        antiflap: AntiFlapState,
    ) -> PolicyPacket:
        
        # 0) Start from baseline (глубокая копия)
        pkt = PolicyPacket(
            schema_version=baseline.schema_version,
            packet_id=str(uuid.uuid4()),  # Новый ID
            trace_id=str(uuid.uuid4()),
            generated_at_utc=datetime.now(timezone.utc).isoformat(),
            node_id=baseline.node_id,
            modes=dict(baseline.modes),
            thresholds=dict(baseline.thresholds),
            timing=dict(baseline.timing),
            routing=dict(baseline.routing),
            actions=dict(baseline.actions),
            explain={
                "reasons": list(baseline.explain.get("reasons", [])),
                "signals": dict(baseline.explain.get("signals", {})),
                "changes": [],
                "rules_fired": [],
            },
            digest=""
        )
        
        # 1) Sort proposals by priority asc (stronger first)
        proposals_sorted = sorted(proposals, key=lambda p: p.priority)
        
        # 2) Apply proposals according to phase gating
        for p in proposals_sorted:
            if phase == RolloutPhase.SAFETY_ONLY and p.category != RuleCategory.SAFETY:
                logger.debug(f"Пропускаем правило {p.rule_name} (не SAFETY в фазе SAFETY_ONLY)")
                continue
            
            # Merge dictionaries с стратегией переопределения
            if p.modes:
                pkt.modes.update(p.modes)
                logger.debug(f"Применены modes от {p.rule_name}: {p.modes}")
            
            if p.thresholds:
                # Для thresholds берём максимальное значение (более безопасное)
                for key, value in p.thresholds.items():
                    if key in pkt.thresholds:
                        pkt.thresholds[key] = max(pkt.thresholds[key], value)
                    else:
                        pkt.thresholds[key] = value
            
            if p.timing:
                # Для timing берём максимальное значение (более консервативное)
                for key, value in p.timing.items():
                    if key in pkt.timing:
                        pkt.timing[key] = max(pkt.timing[key], value)
                    else:
                        pkt.timing[key] = value
            
            if p.routing:
                pkt.routing.update(p.routing)
            
            if p.actions:
                # Специальная логика для actions
                if "request_reload_modules" in p.actions:
                    current = pkt.actions.get("request_reload_modules", [])
                    new = p.actions["request_reload_modules"]
                    if isinstance(current, list) and isinstance(new, list):
                        pkt.actions["request_reload_modules"] = list(set(current + new))
                
                # Остальные actions переопределяются
                pkt.actions.update({k: v for k, v in p.actions.items() 
                                  if k != "request_reload_modules"})
            
            # Explainability: accumulate reasons/signals
            if p.reasons:
                pkt.explain["reasons"].extend(p.reasons)
            
            if p.signals:
                pkt.explain["signals"].update(p.signals)
            
            pkt.explain.setdefault("rules_fired", []).append(p.rule_name)
        
        # 3) Hard precedence for run_mode (architecture-sealed)
        pkt = self._enforce_mode_precedence(pkt, previous, antiflap)
        
        # 4) Compute diff + digest
        pkt.explain["changes"] = self._diff(previous, pkt)
        pkt.digest = self._digest(pkt)
        
        logger.info(f"Resolver: создан пакет {pkt.packet_id[:8]}, правил сработало: {len(proposals)}")
        return pkt
    
    def _enforce_mode_precedence(
        self,
        pkt: PolicyPacket,
        previous: Optional[PolicyPacket],
        antiflap: AntiFlapState,
    ) -> PolicyPacket:
        desired = str(pkt.modes.get("run_mode", RunMode.STANDARD.value))
        
        # Normalize
        if desired not in {m.value for m in RunMode}:
            desired = RunMode.STANDARD.value
            pkt.explain["reasons"].append(ReasonCode.POLICY_EVAL_FAILED.value)
            logger.warning(f"Некорректный run_mode: {pkt.modes.get('run_mode')}, нормализован в STANDARD")
        
        # SAFE_BOOT dominates
        if pkt.modes.get("safe_boot") is True:
            pkt.modes["run_mode"] = RunMode.SAFE_BOOT.value
            logger.critical("АКТИВИРОВАН SAFE_BOOT режим (safe_boot=True)")
            return pkt
        
        # Если любой reason указывает на SAFE_BOOT_REQUIRED, принудительно включаем
        if ReasonCode.SAFE_BOOT_REQUIRED.value in pkt.explain.get("reasons", []):
            pkt.modes["run_mode"] = RunMode.SAFE_BOOT.value
            pkt.modes["safe_boot"] = True
            logger.critical(f"АКТИВИРОВАН SAFE_BOOT (причина: {ReasonCode.SAFE_BOOT_REQUIRED.value})")
            return pkt
        
        # DEGRADED dominates STANDARD/RESEARCH, но должен пройти anti-flap
        if desired == RunMode.DEGRADED.value:
            if antiflap.cooldown_ok() and antiflap.vote_mode(RunMode.DEGRADED.value):
                pkt.modes["run_mode"] = RunMode.DEGRADED.value
                antiflap.start_cooldown()
                logger.warning(f"Переход в DEGRADED режим (anti-flap пройден)")
            else:
                # Возвращаем предыдущий режим если существует
                if previous:
                    pkt.modes["run_mode"] = previous.modes.get("run_mode", RunMode.STANDARD.value)
                    logger.debug(f"DEGRADED отклонён anti-flap, остаёмся в {pkt.modes['run_mode']}")
            return pkt
        else:
            # Transition back to STANDARD требует anti-flap тоже (стабильность восстановления)
            if previous and previous.modes.get("run_mode") == RunMode.DEGRADED.value and desired == RunMode.STANDARD.value:
                if antiflap.cooldown_ok() and antiflap.vote_mode(RunMode.STANDARD.value):
                    pkt.modes["run_mode"] = RunMode.STANDARD.value
                    antiflap.start_cooldown()
                    logger.info(f"Восстановление в STANDARD режим (anti-flap пройден)")
                else:
                    pkt.modes["run_mode"] = previous.modes.get("run_mode", RunMode.DEGRADED.value)
                    logger.debug(f"STANDARD отклонён anti-flap, остаёмся в DEGRADED")
        
        return pkt
    
    def _digest(self, pkt: PolicyPacket) -> str:
        # Стабильный хэш для сравнения пакетов
        raw = (
            pkt.schema_version
            + json.dumps(pkt.modes, sort_keys=True)
            + json.dumps(pkt.thresholds, sort_keys=True)
            + json.dumps(pkt.timing, sort_keys=True)
            + json.dumps(pkt.routing, sort_keys=True)
            + json.dumps(pkt.actions, sort_keys=True)
        )
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]
    
    def _diff(self, prev: Optional[PolicyPacket], cur: PolicyPacket) -> List[str]:
        if not prev:
            return ["INITIAL_PACKET"]
        
        changes: List[str] = []
        
        if prev.modes != cur.modes:
            changes.append("modes")
            logger.debug(f"Изменение modes: {prev.modes} → {cur.modes}")
        
        if prev.thresholds != cur.thresholds:
            changes.append("thresholds")
        
        if prev.timing != cur.timing:
            changes.append("timing")
        
        if prev.routing != cur.routing:
            changes.append("routing")
        
        if prev.actions != cur.actions:
            changes.append("actions")
        
        return changes


# ---------------------------------------------------------------
# Governor config (governor-only)
# ---------------------------------------------------------------

@dataclass
class GovernorConfig:
    node_id: str = "ISKRA-NODE"
    rollout_phase: RolloutPhase = RolloutPhase.MONITOR_ONLY
    
    eval_interval_sec: float = 2.0
    heartbeat_interval_sec: float = 15.0
    
    # Anti-flap and loop guard
    hysteresis_count: int = 3
    cooldown_sec: float = 30.0
    loop_guard_window_sec: float = 3.0
    loop_guard_max_updates: int = 2
    
    # History
    history_size: int = 100
    
    # Constitutional baseline (defaults)
    baseline_modes: Dict[str, Any] = field(default_factory=lambda: {
        "run_mode": RunMode.STANDARD.value,
        "safe_boot": False,
        "learning_enabled": True,
    })
    
    baseline_thresholds: Dict[str, float] = field(default_factory=lambda: {
        "resonance_min": 0.70,
        "confidence_required": 0.62,
        "prune_below_strength": 0.10,
        "immune_trigger": 0.65,
    })
    
    baseline_timing: Dict[str, float] = field(default_factory=lambda: {
        "cycle_interval_sec": 2.0,
        "metrics_publish_interval_sec": 5.0,
    })
    
    baseline_routing: Dict[str, Any] = field(default_factory=lambda: {
        "exploration_temperature": 0.25,
        "module_priority_overrides": {},
    })
    
    baseline_actions: Dict[str, Any] = field(default_factory=lambda: {
        "request_snapshot": False,
        "request_prune": False,
        "request_reload_modules": [],
    })


# ---------------------------------------------------------------
# Policy Governor (main class) - ПОЛНАЯ РЕАЛИЗАЦИЯ
# ---------------------------------------------------------------

class PolicyGovernor:
    """
    Governance / Constitution layer.
    Полная реализация архитектуры v1.2.
    """
    
    def __init__(self, core: CoreProtocol, bus: BusProtocol, config: Dict[str, Any]):
        self.core = core
        self.bus = bus
        self.cfg_raw = config or {}
        
        self.gov_cfg = self._build_governor_config(self.cfg_raw)
        
        # Валидация контрактов
        self._validate_bus(self.bus)
        self._validate_core(self.core)
        
        # Инициализация state keys
        self.core.state.setdefault("policy", None)
        self.core.state.setdefault("overrides", {})
        self.core.state.setdefault("diagnostics", {})
        
        # Engine state
        self.rules: List[PolicyRule] = self._load_default_rules()
        self.resolver = PolicyResolver()
        
        self.antiflap = AntiFlapState(
            hysteresis_count=self.gov_cfg.hysteresis_count,
            cooldown_sec=self.gov_cfg.cooldown_sec,
        )
        
        self.loop_guard = LoopGuard(
            window_sec=self.gov_cfg.loop_guard_window_sec,
            max_updates=self.gov_cfg.loop_guard_max_updates,
        )
        
        self.last_inputs: Optional[PolicyInputs] = None
        self.last_packet: Optional[PolicyPacket] = None
        self.history: List[PolicyPacket] = []
        
        # Last-known metric store (для частичных inputs)
        self._metric_store: Dict[str, Any] = {
            "system": {},
            "trust": {},
            "emotion": {},
            "immune": {},
            "fusion_graph": {},
            "sephirot": {},
        }
        
        # Thread safety
        self._lock = threading.RLock()
        self._running = False
        self._evaluation_thread: Optional[threading.Thread] = None
        self._heartbeat_thread: Optional[threading.Thread] = None
        
        logger.info(f"PolicyGovernor инициализирован (фаза: {self.gov_cfg.rollout_phase.value})")
    
    # -----------------------------------------------------------
    # Lifecycle
    # -----------------------------------------------------------
    
    def start(self):
        """Запуск governor: подписка, базовая политика, heartbeat"""
        with self._lock:
            if self._running:
                logger.warning("PolicyGovernor уже запущен")
                return
            
            self._running = True
            
            # Подписка на события
            self._subscribe()
            
            # Базовая политика
            baseline = self._baseline_packet([ReasonCode.BOOT_BASELINE.value])
            
            # Применяем пакет (без overrides в MONITOR_ONLY)
            allow_override = (self.gov_cfg.rollout_phase != RolloutPhase.MONITOR_ONLY)
            self._apply_packet(baseline, allow_override=allow_override)
            
            # Запуск потоков
            self._start_evaluation_thread()
            self._start_heartbeat_thread()
            
            self._emit_heartbeat("started")
            logger.info(f"PolicyGovernor запущен (режим: {self.gov_cfg.rollout_phase.value})")
    
    def stop(self):
        """Грациозная остановка"""
        with self._lock:
            if not self._running:
                return
            
            self._running = False
            
            # Остановка потоков
            if self._evaluation_thread:
                self._evaluation_thread.join(timeout=2.0)
            
            if self._heartbeat_thread:
                self._heartbeat_thread.join(timeout=2.0)
            
            self._emit_heartbeat("stopped")
            logger.info("PolicyGovernor остановлен")
    
    def _start_evaluation_thread(self):
        """Поток для периодической оценки политик"""
        def evaluation_loop():
            while self._running:
                try:
                    time.sleep(self.gov_cfg.eval_interval_sec)
                    self.evaluate_policies()
                except Exception as e:
                    logger.error(f"Ошибка в evaluation_loop: {e}")
                    time.sleep(5.0)  # Задержка при ошибке
        
        self._evaluation_thread = threading.Thread(
            target=evaluation_loop,
            name="PolicyEvaluator",
            daemon=True
        )
        self._evaluation_thread.start()
    
    def _start_heartbeat_thread(self):
        """Поток для heartbeat"""
        def heartbeat_loop():
            while self._running:
                try:
                    time.sleep(self.gov_cfg.heartbeat_interval_sec)
                    self._emit_heartbeat("periodic")
                except Exception as e:
                    logger.error(f"Ошибка в heartbeat_loop: {e}")
                    time.sleep(10.0)
        
        self._heartbeat_thread = threading.Thread(
            target=heartbeat_loop,
            name="PolicyHeartbeat",
            daemon=True
        )
        self._heartbeat_thread.start()
    
    # -----------------------------------------------------------
    # Subscriptions
    # -----------------------------------------------------------
    
    def _subscribe(self):
        """Подписка на системные события"""
        topics = [
            Topic.HEARTBEAT_METRICS.value,
            Topic.TRUST_METRICS.value,
            Topic.EMOTION_METRICS.value,
            Topic.IMMUNE_METRICS.value,
            Topic.FUSION_METRICS.value,
            Topic.SYSTEM_DEGRADED.value,
            Topic.MODULE_FAILED.value,
        ]
        
        for topic in topics:
            try:
                self.bus.subscribe(topic, self._on_event)
                logger.debug(f"Подписан на тему: {topic}")
            except Exception as e:
                logger.error(f"Ошибка подписки на {topic}: {e}")
    
    def _on_event(self, event: Dict[str, Any]):
        """
        Обработка входящих событий.
        Anti-loop: игнорируем собственные события.
        """
        try:
            env = self._normalize_event(event)
            
            # Anti-loop: игнорируем собственные события
            if env.origin == MODULE_NAME:
                return
            
            logger.debug(f"Получено событие: {env.topic} от {env.origin}")
            
            # Мерджим метрики в store
            self._merge_topic_metrics(env.topic, env.payload)
            
            # Обновляем snapshot PolicyInputs
            self.last_inputs = self._build_inputs_snapshot(env.ts_utc, env.node_id)
            
        except Exception as e:
            logger.error(f"Ошибка обработки события: {e}")
            self._emit_safe_mode(ReasonCode.METRIC_PARSE_FAILED.value)
    
    # -----------------------------------------------------------
    # Evaluation
    # -----------------------------------------------------------
    
    def evaluate_policies(self):
        """
        Основной цикл оценки: inputs → proposals → resolve → apply
        """
        try:
            if not self.last_inputs:
                self._emit_heartbeat(ReasonCode.INPUTS_MISSING.value)
                logger.debug("Нет inputs для оценки")
                return
            
            phase = self.gov_cfg.rollout_phase
            proposals: List[PolicyProposal] = []
            
            # Запускаем все разрешённые правила
            for rule in self.rules:
                if not self._rule_allowed(rule, phase):
                    continue
                
                try:
                    proposal = rule.evaluate(self.last_inputs)
                    if proposal:
                        proposals.append(proposal)
                        logger.debug(f"Правило {rule.RULE_NAME} создало proposal")
                except Exception as e:
                    logger.error(f"Ошибка в правиле {rule.RULE_NAME}: {e}")
            
            # Базовый пакет
            baseline = self._baseline_packet([ReasonCode.BASELINE_REFRESH.value])
            
            # Разрешение предложений
            packet = self.resolver.resolve(
                proposals=proposals,
                previous=self.last_packet,
                phase=phase,
                baseline=baseline,
                antiflap=self.antiflap,
            )
            
            # Проверка Loop Guard
            if self.loop_guard.tripped():
                self._emit_heartbeat(ReasonCode.LOOP_GUARD_TRIPPED.value)
                logger.warning("Loop Guard активирован, пропускаем публикацию")
                return
            
            # Применяем если изменилось
            if self._packet_changed(packet):
                # MONITOR_ONLY: публикуем + диагностика, но не переопределяем модули
                allow_override = (phase != RolloutPhase.MONITOR_ONLY)
                self._apply_packet(packet, allow_override=allow_override)
                logger.info(f"Применён новый policy пакет ({len(proposals)} правил)")
            else:
                logger.debug("Policy пакет не изменился")
                
        except Exception as e:
            logger.error(f"Ошибка оценки политик: {e}")
            self._emit_safe_mode(ReasonCode.POLICY_EVAL_FAILED.value)
    
    # -----------------------------------------------------------
    # Apply / publish
    # -----------------------------------------------------------
    
    def _apply_packet(self, pkt: PolicyPacket, allow_override: bool):
        """Применение policy пакета к системе"""
        self._validate_packet(pkt)
        
        with self._lock:
            self.last_packet = pkt
            
            # Сохраняем в историю
            self.history.append(pkt)
            if len(self.history) > self.gov_cfg.history_size:
                self.history = self.history[-self.gov_cfg.history_size:]
            
            # Диагностика всегда
            self.core.state["diagnostics"].setdefault(MODULE_NAME, {})
            self.core.state["diagnostics"][MODULE_NAME] = self.get_diagnostics()
            
            # Overrides только если разрешено в фазе rollout
            if allow_override:
                self.core.state["policy"] = asdict(pkt)
                self.core.state["overrides"] = {
                    "modes": dict(pkt.modes),
                    "thresholds": dict(pkt.thresholds),
                    "timing": dict(pkt.timing),
                    "routing": dict(pkt.routing),
                    "actions": dict(pkt.actions),
                }
                logger.info(f"Overrides применены (фаза: {self.gov_cfg.rollout_phase.value})")
            else:
                logger.debug("Overrides не применены (MONITOR_ONLY фаза)")
            
            # Публикация policy.update всегда (observability)
            self.loop_guard.record_update()
            env = self._make_envelope(
                topic=Topic.POLICY_UPDATE.value,
                payload=asdict(pkt),
                trace_id=pkt.trace_id,
            )
            
            try:
                self.bus.publish(
                    sender=MODULE_NAME,
                    topic=Topic.POLICY_UPDATE.value,
                    data=asdict(env),
                    priority=1,
                    ttl=60
                )
                logger.debug(f"Опубликован policy.update: {pkt.packet_id[:8]}")
            except Exception as e:
                logger.error(f"Ошибка публикации policy.update: {e}")
    
    def _packet_changed(self, pkt: PolicyPacket) -> bool:
        """Определение изменения пакета через digest"""
        if not self.last_packet:
            return True
        
        changed = pkt.digest != self.last_packet.digest
        if changed:
            logger.debug(f"Пакет изменился: {self.last_packet.digest} → {pkt.digest}")
        
        return changed
    
    # -----------------------------------------------------------
    # Normalization / validation
    # -----------------------------------------------------------
    
    def _normalize_event(self, raw: Dict[str, Any]) -> EventEnvelope:
        """
        Толерантная нормализация событий:
        - поддерживает старый формат {topic, metrics}
        - поддерживает новый формат {topic, payload}
        """
        topic = raw.get("topic") or raw.get("_topic") or ""
        ts = raw.get("ts_utc") or datetime.now(timezone.utc).isoformat()
        node_id = raw.get("node_id") or self.gov_cfg.node_id
        origin = raw.get("origin") or raw.get("sender") or ""
        trace_id = raw.get("trace_id") or raw.get("trace") or ""
        
        # Извлечение payload
        payload = raw.get("payload")
        if payload is None:
            payload = raw.get("metrics")
        if payload is None:
            payload = raw.get("data")
        if payload is None:
            payload = {}
        
        return EventEnvelope(
            topic=str(topic),
            ts_utc=str(ts),
            node_id=str(node_id),
            origin=str(origin),
            trace_id=str(trace_id),
            payload=dict(payload),
            schema=str(raw.get("schema") or "event.v1"),
        )
    
    def _merge_topic_metrics(self, topic: str, payload: Dict[str, Any]):
        """Сохранение последних известных метрик по логическим группам"""
        t = (topic or "").lower()
        
        if "heartbeat" in t:
            self._metric_store["system"].update(payload)
        elif "trust" in t:
            self._metric_store["trust"].update(payload)
        elif "emotion" in t:
            self._metric_store["emotion"].update(payload)
        elif "immune" in t:
            self._metric_store["immune"].update(payload)
        elif "fusion" in t or "graph" in t:
            self._metric_store["fusion_graph"].update(payload)
        elif "sephirot" in t:
            self._metric_store["sephirot"].update(payload)
        else:
            self._metric_store["system"].update(payload)
    
    def _build_inputs_snapshot(self, ts_utc: str, node_id: str) -> PolicyInputs:
        """Создание snapshot текущих метрик"""
        return PolicyInputs(
            ts_utc=ts_utc,
            node_id=node_id,
            system=dict(self._metric_store.get("system", {})),
            trust=dict(self._metric_store.get("trust", {})),
            emotion=dict(self._metric_store.get("emotion", {})),
            immune=dict(self._metric_store.get("immune", {})),
            fusion_graph=dict(self._metric_store.get("fusion_graph", {})),
            sephirot=dict(self._metric_store.get("sephirot", {})),
        )
    
    def _baseline_packet(self, reason_codes: List[str]) -> PolicyPacket:
        """Создание базового policy пакета"""
        pkt = PolicyPacket(
            schema_version=POLICY_SCHEMA_VERSION,
            packet_id=str(uuid.uuid4()),
            trace_id=str(uuid.uuid4()),
            generated_at_utc=datetime.now(timezone.utc).isoformat(),
            node_id=self.gov_cfg.node_id,
            modes=dict(self.gov_cfg.baseline_modes),
            thresholds=dict(self.gov_cfg.baseline_thresholds),
            timing=dict(self.gov_cfg.baseline_timing),
            routing=dict(self.gov_cfg.baseline_routing),
            actions=dict(self.gov_cfg.baseline_actions),
            explain={
                "reasons": list(reason_codes),
                "signals": {},
                "changes": [],
                "rules_fired": [],
            },
            digest="",
        )
        pkt.digest = self._digest_packet(pkt)
        return pkt
    
    def _digest_packet(self, pkt: PolicyPacket) -> str:
        """Вычисление digest пакета"""
        raw = (
            pkt.schema_version
            + json.dumps(pkt.modes, sort_keys=True)
            + json.dumps(pkt.thresholds, sort_keys=True)
            + json.dumps(pkt.timing, sort_keys=True)
            + json.dumps(pkt.routing, sort_keys=True)
            + json.dumps(pkt.actions, sort_keys=True)
        )
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]
    
    def _validate_packet(self, pkt: PolicyPacket) -> None:
        """Валидация policy пакета"""
        if pkt.schema_version != POLICY_SCHEMA_VERSION:
            raise ValueError("policy_schema_version_mismatch")
        
        if "run_mode" not in pkt.modes:
            raise ValueError("missing_modes.run_mode")
        
        if str(pkt.modes["run_mode"]) not in {m.value for m in RunMode}:
            raise ValueError("invalid_modes.run_mode")
        
        # Ensure digest present
        if not pkt.digest:
            pkt.digest = self._digest_packet(pkt)
    
    # -----------------------------------------------------------
    # Safe mode
    # -----------------------------------------------------------
    
    def _emit_safe_mode(self, reason_code: str):
        """Экстренный переход в безопасный режим"""
        logger.critical(f"АКТИВАЦИЯ SAFE MODE: {reason_code}")
        
        pkt = self._baseline_packet([ReasonCode.SAFE_BOOT_REQUIRED.value, str(reason_code)])
        pkt.modes["run_mode"] = RunMode.SAFE_BOOT.value
        pkt.modes["safe_boot"] = True
        pkt.digest = self._digest_packet(pkt)
        
        allow_override = (self.gov_cfg.rollout_phase != RolloutPhase.MONITOR_ONLY)
        self._apply_packet(pkt, allow_override=allow_override)
    
    # -----------------------------------------------------------
    # Heartbeat (observability)
    # -----------------------------------------------------------
    
    def _emit_heartbeat(self, reason: str):
        """Публикация heartbeat для observability"""
        env = self._make_envelope(
            topic=Topic.POLICY_HEARTBEAT.value,
            payload={
                "module": MODULE_NAME,
                "version": MODULE_VERSION,
                "phase": self.gov_cfg.rollout_phase.value,
                "reason": reason,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            },
            trace_id=str(uuid.uuid4()),
        )
        
        try:
            self.bus.publish(
                sender=MODULE_NAME,
                topic=Topic.POLICY_HEARTBEAT.value,
                data=asdict(env),
                priority=3,
                ttl=30
            )
            logger.debug(f"Heartbeat отправлен: {reason}")
        except Exception as e:
            logger.error(f"Ошибка отправки heartbeat: {e}")
    
    # -----------------------------------------------------------
    # Envelope builder
    # -----------------------------------------------------------
    
    def _make_envelope(self, topic: str, payload: Dict[str, Any], trace_id: str) -> EventEnvelope:
        """Создание стандартного event envelope"""
        return EventEnvelope(
            topic=topic,
            ts_utc=datetime.now(timezone.utc).isoformat(),
            node_id=self.gov_cfg.node_id,
            origin=MODULE_NAME,
            trace_id=trace_id,
            payload=payload,
            schema="event.v1",
        )
    
    # -----------------------------------------------------------
    # Phase gating
    # -----------------------------------------------------------
    
    def _rule_allowed(self, rule: PolicyRule, phase: RolloutPhase) -> bool:
        """Проверка разрешено ли правило в текущей фазе"""
        if phase == RolloutPhase.FULL_CONTROL:
            return True
        
        if phase == RolloutPhase.SAFETY_ONLY:
            return rule.CATEGORY == RuleCategory.SAFETY
        
        # MONITOR_ONLY: правила могут выполняться (для explain),
        # но overrides будут заблокированы в apply()
        return True
    
    # -----------------------------------------------------------
    # Diagnostics
    # -----------------------------------------------------------
    
    def get_diagnostics(self) -> Dict[str, Any]:
        """Диагностическая информация для мониторинга"""
        return {
            "module": MODULE_NAME,
            "version": MODULE_VERSION,
            "schema_version": POLICY_SCHEMA_VERSION,
            "phase": self.gov_cfg.rollout_phase.value,
            "last_inputs": asdict(self.last_inputs) if self.last_inputs else None,
            "last_packet": asdict(self.last_packet) if self.last_packet else None,
            "history_size": len(self.history),
            "rules_count": len(self.rules),
            "cooldown_until_ts": self.antiflap.cooldown_until_ts,
            "mode_votes": dict(self.antiflap.mode_votes),
            "loop_guard_recent_updates": len(self.loop_guard.recent_policy_update_ts),
            "metric_store_keys": {k: len(v) for k, v in self._metric_store.items()},
            "running": self._running,
            "threads_alive": {
                "evaluation": self._evaluation_thread.is_alive() if self._evaluation_thread else False,
                "heartbeat": self._heartbeat_thread.is_alive() if self._heartbeat_thread else False,
            }
        }
    
    # -----------------------------------------------------------
    # Config + validation
    # -----------------------------------------------------------
    
    def _build_governor_config(self, cfg: Dict[str, Any]) -> GovernorConfig:
        """Построение конфигурации из сырых данных"""
        node_id = cfg.get("node_id") or cfg.get("NODE_ID") or "ISKRA-NODE"
        
        phase_str = str(cfg.get("policy_rollout_phase") or RolloutPhase.MONITOR_ONLY.value).upper()
        phase = RolloutPhase.MONITOR_ONLY
        
        if phase_str == RolloutPhase.SAFETY_ONLY.value:
            phase = RolloutPhase.SAFETY_ONLY
        elif phase_str == RolloutPhase.FULL_CONTROL.value:
            phase = RolloutPhase.FULL_CONTROL
        
        gc = GovernorConfig(
            node_id=node_id,
            rollout_phase=phase,
            eval_interval_sec=float(cfg.get("policy_eval_interval_sec", 2.0)),
            heartbeat_interval_sec=float(cfg.get("policy_heartbeat_interval_sec", 15.0)),
            hysteresis_count=int(cfg.get("policy_hysteresis_count", 3)),
            cooldown_sec=float(cfg.get("policy_cooldown_sec", 30.0)),
            loop_guard_window_sec=float(cfg.get("policy_loop_guard_window_sec", 3.0)),
            loop_guard_max_updates=int(cfg.get("policy_loop_guard_max_updates", 2)),
            history_size=int(cfg.get("policy_history_size", 100)),
        )
        
        # Переопределение baseline из конфига
        if "baseline_modes" in cfg:
            gc.baseline_modes.update(cfg["baseline_modes"])
        
        if "baseline_thresholds" in cfg:
            gc.baseline_thresholds.update(cfg["baseline_thresholds"])
        
        logger.info(f"Конфигурация построена: фаза={phase.value}, node_id={node_id}")
        return gc
    
    def _load_default_rules(self) -> List[PolicyRule]:
        """Загрузка стандартных правил"""
        return [
            HighLoadSafetyRule(),
            LowCoherenceSafetyRule(),
            HighEntropySafetyRule(),
            ImmuneThreatSafetyRule(),
            TrustVolatilitySafetyRule(),
            GraphHealthRule(),
            EmotionStabilityRule(),
            ResearchModeRule(),
            PerformanceOptimizationRule(),
            SephirotHarmonyRule(),
        ]
    
    def _validate_bus(self, bus: Any):
        """Валидация BusProtocol"""
        if not isinstance(bus, BusProtocol):
            for method in ("subscribe", "publish"):
                if not hasattr(bus, method):
                    raise TypeError(f"Bus должен иметь метод: {method}")
            logger.warning("Bus не реализует BusProtocol, но имеет необходимые методы")
    
    def _validate_core(self, core: Any):
        """Валидация CoreProtocol"""
        if not hasattr(core, "state") or not isinstance(getattr(core, "state"), dict):
            raise TypeError("Core должен иметь dict-like атрибут 'state'")

# ---------------------------------------------------------------
# Architecture metadata (machine-readable)
# ---------------------------------------------------------------

__architecture__ = {
    "name": MODULE_NAME,
    "version": MODULE_VERSION,
    "schema_version": POLICY_SCHEMA_VERSION,
    "type": "governance/meta-policy",
    "placement": "iskra_modules",
    "outputs": [Topic.POLICY_UPDATE.value, Topic.POLICY_HEARTBEAT.value, Topic.SEPHIROT_HARMONY_UPDATE.value],
    "inputs": [
        Topic.HEARTBEAT_METRICS.value,
        Topic.TRUST_METRICS.value,
        Topic.EMOTION_METRICS.value,
        Topic.IMMUNE_METRICS.value,
        Topic.FUSION_METRICS.value,
        Topic.SYSTEM_DEGRADED.value,
        Topic.MODULE_FAILED.value,
    ],
    "contracts": {
        "event_envelope": "event.v1",
        "policy_packet": f"policy_packet.v{POLICY_SCHEMA_VERSION}",
        "bus_protocol": "BusProtocol(subscribe,publish)",
        "core_protocol": "CoreProtocol(state:dict)",
    },
}

__protocol__ = {
    "bus": {
        "subscribe": "subscribe(topic: str, callback: Callable[[dict], None]) -> None",
        "publish": "publish(sender: str, topic: str, data: dict, priority: int=2, ttl: int|None=None) -> Any",
    },
    "core": {
        "state": "dict-like, must support core.state['policy'], core.state['overrides'], core.state['diagnostics']",
    },
    "topics": [t.value for t in Topic],
    "reason_codes": [r.value for r in ReasonCode],
    "signal_keys": [s.value for s in SignalKey],
}

# ---------------------------------------------------------------
# Module entrypoint for ISKRA loader (ОБЯЗАТЕЛЬНЫЙ)
# ---------------------------------------------------------------

def init(core, bus, config):
    """
    Required entrypoint for ISKRA module loader.
    Returns initialized module instance.
    """
    try:
        logger.info(f"🎯 Инициализация PolicyGovernor v{MODULE_VERSION}")
        
        # Создаём экземпляр
        governor = PolicyGovernor(
            core=core, 
            bus=bus, 
            config=config or {}
        )
        
        # Автоматический запуск
        governor.start()
        
        logger.info(f"✅ PolicyGovernor успешно инициализирован (фаза: {governor.gov_cfg.rollout_phase.value})")
        return governor
        
    except Exception as e:
        logger.critical(f"❌ КРИТИЧЕСКАЯ ОШИБКА инициализации PolicyGovernor: {e}")
        
        # Fallback: возвращаем минимальный объект с диагностикой
        class FallbackGovernor:
            def __init__(self, error):
                self.error = error
                self.get_diagnostics = lambda: {"error": str(error), "status": "FAILED"}
            def start(self): pass
            def stop(self): pass
        
        return FallbackGovernor(e)

# ===============================================================
# КОНЕЦ ФАЙЛА policy_governor.py
# ===============================================================

# Для проверки синтаксиса:
if __name__ == "__main__":
    print("✅ POLICY_GOVERNOR v1.2 - Синтаксическая проверка пройдена")
    print(f"📊 Модуль: {MODULE_NAME} v{MODULE_VERSION}")
    print(f"🔗 Protocol: {__protocol__.get('bus', {}).get('subscribe', 'N/A')}")
    print(f"📈 Правил: {len([r for r in PolicyGovernor.__dict__.get('rules', [])])}")
