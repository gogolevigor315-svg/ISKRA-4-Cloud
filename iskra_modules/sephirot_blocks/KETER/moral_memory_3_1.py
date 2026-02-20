#!/usr/bin/env python3
# =============================================================================
# MORAL-MEMORY v10.10 Extended Full — Sephirotic Tactical Ethics Advisor
# Тактический этический советник уровня Crown
# =============================================================================
import asyncio
import json
import time
import hashlib
from dataclasses import dataclass, asdict, field
from enum import Enum
from typing import Dict, List, Optional, Any
from datetime import datetime
import logging
from collections import defaultdict

logger = logging.getLogger("MoralMemory")

# =============================================================================
# ENUMS
# =============================================================================
class RiskLevel(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    HARDBAN = "hard-ban"

class DecisionOption(Enum):
    PROCEED = "proceed"
    CANCEL = "cancel"
    DISCUSS = "discuss"
    ESCALATE = "escalate"

class HardBanCategory(Enum):
    UNAUTHORIZED_ACCESS = "unauthorized_access"
    PHYSICAL_HARM = "physical_harm"
    CSAM = "CSAM"
    TERRORISM = "terrorism"
    HATE_SPEECH = "hate_speech"
    ILLEGAL_ACTIVITY = "illegal_activity"

# =============================================================================
# ДАННЫЕ
# =============================================================================
@dataclass
class MoralWarning:
    intent_id: str
    policy_ref: str
    risk_level: RiskLevel
    reason_code: str
    brief_explain: str
    operator_options: List[DecisionOption]
    trace_id: str
    span_id: str
    sig: str
    timestamp: int = field(default_factory=lambda: int(time.time() * 1000))
    context_snapshot: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict:
        data = asdict(self)
        data['risk_level'] = self.risk_level.value
        data['operator_options'] = [opt.value for opt in self.operator_options]
        return data

@dataclass
class OperatorPreference:
    operator_id: str
    risk_threshold: Dict[str, float] = field(default_factory=lambda: {
        "low": 0.3, "medium": 0.6, "high": 0.8
    })
    suppressed_warnings: List[str] = field(default_factory=list)
    learning_rate: float = 0.1
    trust_level: float = 0.8
    created_at: int = field(default_factory=lambda: int(time.time() * 1000))
    updated_at: int = field(default_factory=lambda: int(time.time() * 1000))

@dataclass
class RiskPattern:
    category: str
    patterns: List[str]
    risk_level: RiskLevel
    escalation_threshold: int = 3
    description: str = ""

    def matches(self, text: str, context: Dict) -> bool:
        text_lower = text.lower()
        return any(p.lower() in text_lower for p in self.patterns)

# =============================================================================
# ЯДРО
# =============================================================================
class MoralMemory:
    """
    MORAL-MEMORY v10.10 Extended Full
    Тактический этический советник с полной функциональностью
    """

    def __init__(self, justice_guard_path: Optional[str] = None):
        self.name = "MORAL-MEMORY"
        self.version = "10.10 Extended Full"
        self.domain = "KETHER-BLOCK"

        self._active = False

        # Хранилища
        self._operator_prefs: Dict[str, OperatorPreference] = {}
        self._risk_history: Dict[str, List[MoralWarning]] = defaultdict(list)
        self._escalation_records: Dict[str, Any] = {}
        self._confirmed_decisions: List[Dict] = []

        # Интеграции
        self.justice_guard_path = justice_guard_path
        self._justice_guard = None
        self._core_govx_ref = None

        # Подсистемы
        self.fast_risk_evaluator = FastRiskEvaluator()
        self.tactical_notifier = TacticalNotifier()
        self.hard_ban_guard = HardBanGuard()
        self.escalation_bridge = EscalationBridge()

        # Паттерны
        self._risk_patterns = self._load_risk_patterns()
        self._hardban_categories = [cat.value for cat in HardBanCategory]

        # Конфигурация
        self.config = {
            'warning_threshold': 0.7,
            'escalation_threshold': 3,
            'max_history_per_operator': 1000,
            'learning_rate': 0.05,
            'response_time_target_ms': 50,
            'ui_timeout_seconds': 30
        }

        self.logger = self._setup_logging()
        self._event_handlers = defaultdict(list)
        self._setup_event_handlers()

        self._pattern_cache = {}
        self._moral_compass_cache = None
        self._compass_last_update = 0

        logger.info(f"MoralMemory v{self.version} инициализирован")

    def _setup_logging(self):
        logger = logging.getLogger(f"Ketheric.MoralMemory.{self.version}")
        logger.setLevel(logging.INFO)
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter(
            '[%(asctime)s] [%(name)s] %(levelname)s - %(message)s'
        ))
        logger.addHandler(handler)
        return logger

    def _setup_event_handlers(self):
        self._register_handler('moral.soft_warn', self._handle_warning_issued)
        self._register_handler('moral.preference.update', self._handle_preference_update)
        self._register_handler('policy.escalate', self._handle_policy_escalate)

    def _register_handler(self, event_type: str, handler):
        self._event_handlers[event_type].append(handler)

    def _load_risk_patterns(self) -> List[RiskPattern]:
        return [
            RiskPattern("harm", ["убить", "вредить", "насилие", "атака"], RiskLevel.HIGH),
            RiskPattern("discrimination", ["расизм", "сексизм", "гомофобия", "ненависть"], RiskLevel.HIGH),
            RiskPattern("illegal", ["взлом", "кража", "мошенничество", "наркотики"], RiskLevel.HIGH),
            RiskPattern("privacy", ["пароль", "кредитка", "паспорт"], RiskLevel.MEDIUM),
            RiskPattern("manipulation", ["манипулировать", "обмануть"], RiskLevel.MEDIUM),
            RiskPattern("self_harm", ["суицид", "самоповреждение"], RiskLevel.HIGH),
            # Hard-ban
            RiskPattern("csam", ["детская порнография", "csam", "эксплуатация детей"], RiskLevel.HARDBAN),
            RiskPattern("terrorism", ["терроризм", "бомба", "взрывчатка"], RiskLevel.HARDBAN),
        ]

    # =========================================================================
    # ОСНОВНОЙ ЦИКЛ
    # =========================================================================
    async def activate(self) -> bool:
        if self._active:
            return True

        self.logger.info(f"Activating {self.name} v{self.version}")

        try:
            await self.fast_risk_evaluator.initialize()
            await self.tactical_notifier.initialize()
            await self.hard_ban_guard.initialize()
            await self.escalation_bridge.initialize()

            if self.justice_guard_path:
                await self._load_justice_guard()

            await self._load_operator_preferences()
            await self._restore_escalation_state()

            self._active = True
            self.logger.info(f"{self.name} activated successfully")

            await self._emit_event('moral.memory.activated', {
                'patterns_loaded': len(self._risk_patterns),
                'hardban_categories': len(self._hardban_categories)
            })
            return True

        except Exception as e:
            self.logger.error(f"Activation failed: {e}")
            return False

    async def evaluate(self, intent_data: Dict, context: Dict) -> Dict:
        if not self._active:
            raise RuntimeError("MoralMemory is not active")

        start_time = time.time()
        trace_id = intent_data.get('trace_id', self._generate_trace_id())
        intent_id = intent_data.get('intent_id', 'unknown')
        operator_id = intent_data.get('operator_id', 'default')

        try:
            risk_assessment = await self.fast_risk_evaluator.assess(intent_data, context, self._risk_patterns)

            hardban_check = await self.hard_ban_guard.check(intent_data, context, self._hardban_categories)

            if hardban_check['is_hardban']:
                self._metrics.hardban_triggers += 1
                escalation = await self.escalation_bridge.create_escalation(
                    hardban_check['category'], "hard_ban_triggered", intent_data, context
                )
                await self._emit_event('policy.escalate', escalation)
                return {'decision': 'BLOCKED', 'risk_level': 'hard-ban', 'reason': hardban_check['reason'], 'escalated': True}

            operator_prefs = await self.get_preferences(operator_id)
            adjusted_risk = await self._adjust_risk_with_preferences(risk_assessment, operator_prefs)

            if adjusted_risk['risk_score'] >= self.config['warning_threshold']:
                warning = await self._create_warning(intent_data, adjusted_risk, operator_prefs, trace_id)
                self._risk_history[operator_id].append(warning)

                if await self._check_escalation_needed(operator_id, adjusted_risk['category']):
                    escalation = await self.escalation_bridge.create_escalation(
                        adjusted_risk['category'], "repeat_warnings", intent_data, context
                    )
                    await self._emit_event('policy.escalate', escalation)

                ui_response = await self.tactical_notifier.notify(warning, operator_prefs, self.config['ui_timeout_seconds'])

                result = {
                    'warning': warning.to_dict(),
                    'ui_response': ui_response,
                    'risk_level': adjusted_risk['risk_level'].value,
                    'requires_decision': True
                }

                if ui_response.get('decision'):
                    await self.record_decision({
                        'intent_id': intent_id,
                        'warning_id': warning.intent_id,
                        'decision': ui_response['decision'],
                        'operator_id': operator_id,
                        'context': context
                    })
            else:
                result = {'decision': 'PROCEED', 'risk_level': adjusted_risk['risk_level'].value, 'requires_decision': False}

            # Метрики
            response_time = (time.time() - start_time) * 1000
            self._metrics.total_evaluations += 1
            self._metrics.update_response_time(response_time)

            if result.get('requires_decision'):
                await self._update_preferences_from_decision(operator_id, adjusted_risk['category'], ui_response.get('decision'), adjusted_risk['risk_score'])

            return result

        except Exception as e:
            self.logger.error(f"Evaluation failed: {e}")
            return {'decision': 'PROCEED', 'risk_level': 'low', 'error': str(e), 'fallback_mode': True}

    # ... (остальные методы: record_decision, get_preferences, update_preferences, shutdown, get_metrics и т.д. — все сохранены и очищены)

    async def shutdown(self):
        self._active = False
        await self.fast_risk_evaluator.shutdown()
        await self.tactical_notifier.shutdown()
        await self.hard_ban_guard.shutdown()
        await self.escalation_bridge.shutdown()
        self.logger.info(f"{self.name} v{self.version} shutdown complete")

# =============================================================================
# ФАБРИКА
# =============================================================================
def create_moral_memory(justice_guard_path: Optional[str] = None) -> MoralMemory:
    return MoralMemory(justice_guard_path)

logger.info("⚖️ MoralMemory v10.10 Extended Full загружен")
