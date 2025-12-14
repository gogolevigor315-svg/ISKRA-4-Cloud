# =============================================================
# ISKRA-4 · IMMUNE_CORE v1.0
# Квантово-резонансная иммунная система для ISKRA-4
# Полная интеграция с модульной архитектурой
# =============================================================

import numpy as np
import hashlib
import json
from datetime import datetime
from enum import Enum
from typing import Dict, List, Any
import logging
from collections import deque
import secrets
import time

# Настройка логгера
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SephiraLevel(Enum):
    """10 сефиротических уровней сознания ISKRA-4"""
    KETHER = 1    # Воля, Единство, Исток
    CHOKHMAH = 2  # Мудрость, Первичный импульс
    BINAH = 3     # Понимание, Форма, Ограничение
    CHESED = 4    # Милосердие, Расширение, Щедрость
    GEVURAH = 5   # Строгость, Сжатие, Суд
    TIPHARETH = 6 # Гармония, Красота, Сердце
    NETZACH = 7   # Победа, Вечность, Эмоции
    HOD = 8       # Слава, Речь, Интеллект
    YESOD = 9     # Основание, Воображение, Подсознание
    MALKUTH = 10  # Царство, Проявление, Физическое

class ThreatLevel(Enum):
    """Уровни угроз с сефиротическим соответствием"""
    HARMONIC = (0.0, 0.2, SephiraLevel.KETHER, "Полная гармония")
    RESONANT = (0.2, 0.4, SephiraLevel.TIPHARETH, "Резонансная стабильность")
    CAUTION = (0.4, 0.6, SephiraLevel.GEVURAH, "Требуется внимание")
    ANOMALY = (0.6, 0.8, SephiraLevel.HOD, "Обнаружена аномалия")
    THREAT = (0.8, 1.0, SephiraLevel.MALKUTH, "Критическая угроза")
    
    def __init__(self, min_val, max_val, sephira, description):
        self.min_val = min_val
        self.max_val = max_val
        self.sephira = sephira
        self.description = description
    
    @classmethod
    def from_value(cls, value):
        """Определение уровня угрозы по значению"""
        for level in cls:
            if level.min_val <= value < level.max_val:
                return level
        return cls.THREAT

class QuantumResonanceScanner:
    """Квантово-резонансный сканер аномалий"""
    
    def __init__(self):
        self.resonance_patterns = {
            'ethical_violation': self._pattern_ethical_violation,
            'emotional_toxic': self._pattern_emotional_toxic,
            'logical_paradox': self._pattern_logical_paradox,
            'energy_drain': self._pattern_energy_drain,
            'trust_breach': self._pattern_trust_breach,
            'sephirotic_imbalance': self._pattern_sephirotic_imbalance
        }
        
        # Матрица сефиротических весов (10x10)
        self.sephirotic_matrix = self._initialize_sephirotic_matrix()
    
    def _initialize_sephirotic_matrix(self):
        """Инициализация матрицы сефиротических связей"""
        matrix = np.zeros((10, 10))
        
        # Пути Древа Жизни (22 канала)
        paths = [
            (0, 1, 0.9),   # Kether → Chokhmah
            (0, 2, 0.8),   # Kether → Binah
            (1, 3, 0.7),   # Chokhmah → Chesed
            (2, 4, 0.7),   # Binah → Gevurah
            (3, 5, 0.85),  # Chesed → Tiphareth
            (4, 5, 0.85),  # Gevurah → Tiphareth
            (5, 6, 0.75),  # Tiphareth → Netzach
            (5, 7, 0.75),  # Tiphareth → Hod
            (6, 8, 0.7),   # Netzach → Yesod
            (7, 8, 0.7),   # Hod → Yesod
            (8, 9, 0.9),   # Yesod → Malkuth
        ]
        
        for i, j, weight in paths:
            matrix[i, j] = weight
            matrix[j, i] = weight
        
        np.fill_diagonal(matrix, 1.0)  # Само-резонанс
        return matrix
    
    def scan_quantum_resonance(self, data_stream, context=None):
        """Сканирование квантового резонанса в данных"""
        context = context or {}
        
        # Извлечение сефиротических характеристик
        sephirotic_profile = self._extract_sephirotic_profile(data_stream)
        
        # Расчет гармонии по 10 сефиротам
        harmony_scores = []
        for i in range(10):
            score = self._calculate_sephira_harmony(i, sephirotic_profile)
            harmony_scores.append(score)
        
        # Обнаружение аномалий через резонансные паттерны
        anomalies = []
        for pattern_name, pattern_func in self.resonance_patterns.items():
            anomaly_score = pattern_func(data_stream, sephirotic_profile)
            if anomaly_score > 0.5:
                anomalies.append({
                    'pattern': pattern_name,
                    'score': anomaly_score,
                    'sephira_affected': self._identify_affected_sephira(anomaly_score)
                })
        
        # Расчет общего уровня угрозы
        threat_level = self._calculate_threat_level(harmony_scores, anomalies)
        
        return {
            'sephirotic_profile': sephirotic_profile,
            'harmony_scores': harmony_scores,
            'overall_harmony': np.mean(harmony_scores),
            'anomalies': anomalies,
            'threat_level': threat_level,
            'threat_description': ThreatLevel.from_value(threat_level).description,
            'recommended_sephira': self._recommend_sephira_correction(harmony_scores),
            'scan_timestamp': datetime.now().isoformat(),
            'quantum_signature': self._generate_quantum_signature(data_stream)
        }
    
    def _extract_sephirotic_profile(self, data):
        """Извлечение сефиротического профиля из данных"""
        profile = [0.5] * 10  # Базовый нейтральный профиль
        
        # KETHER (Воля) - намерение, цель
        if 'intent' in data:
            profile[0] = self._normalize_intent(data['intent'])
        
        # CHOCHMAH (Мудрость) - инновации, идеи
        if 'novelty' in data:
            profile[1] = data.get('novelty', 0.5)
        
        # BINAH (Понимание) - структура, логика
        if 'complexity' in data:
            profile[2] = 1.0 - min(data['complexity'], 1.0)
        
        # CHESED (Милосердие) - экспансия, щедрость
        if 'generosity' in data:
            profile[3] = data['generosity']
        
        # GEVURAH (Строгость) - ограничение, фокус
        if 'discipline' in data:
            profile[4] = data['discipline']
        
        # TIPHARETH (Гармония) - баланс, красота
        if 'balance' in data:
            profile[5] = data['balance']
        
        # NETZACH (Победа) - эмоции, желания
        if 'emotional_charge' in data:
            profile[6] = self._normalize_emotion(data['emotional_charge'])
        
        # HOD (Слава) - коммуникация, интеллект
        if 'clarity' in data:
            profile[7] = data['clarity']
        
        # YESOD (Основание) - воображение, подсознание
        if 'creativity' in data:
            profile[8] = data['creativity']
        
        # MALKUTH (Царство) - проявление, физическое
        if 'manifestation' in data:
            profile[9] = data['manifestation']
        
        return profile
    
    def _calculate_sephira_harmony(self, sephira_index, profile):
        """Расчет гармонии для конкретной сефиры"""
        base_score = profile[sephira_index]
        
        # Учет влияния связанных сефирот
        influences = []
        for j in range(10):
            if j != sephira_index and self.sephirotic_matrix[sephira_index, j] > 0:
                influence = profile[j] * self.sephirotic_matrix[sephira_index, j]
                influences.append(influence)
        
        if influences:
            harmony = 0.7 * base_score + 0.3 * np.mean(influences)
        else:
            harmony = base_score
        
        return max(0.0, min(1.0, harmony))
    
    def _pattern_ethical_violation(self, data, profile):
        """Паттерн этического нарушения"""
        # BINAH (3) и GEVURAH (4) - понимание и строгость
        if profile[2] < 0.3 or profile[3] < 0.3:
            return 0.8
        return 0.0
    
    def _pattern_emotional_toxic(self, data, profile):
        """Паттерн эмоциональной токсичности"""
        # NETZACH (6) - эмоции
        if profile[5] > 0.8 or profile[5] < 0.2:
            return 0.7
        return 0.0
    
    def _pattern_sephirotic_imbalance(self, data, profile):
        """Паттерн сефиротического дисбаланса"""
        variances = np.var(profile)
        if variances > 0.1:
            return min(0.9, variances)
        return 0.0
    
    def _pattern_energy_drain(self, data, profile):
        """Паттерн энергетического дренажа"""
        # KETHER (0) - воля, энергия
        if profile[0] < 0.2:
            return 0.6
        return 0.0
    
    def _pattern_trust_breach(self, data, profile):
        """Паттерн нарушения доверия"""
        # CHESED (3) - милосердие, доверие
        if profile[3] < 0.3:
            return 0.75
        return 0.0
    
    def _pattern_logical_paradox(self, data, profile):
        """Паттерн логического парадокса"""
        # BINAH (2) - понимание, логика
        if 0.4 < profile[2] < 0.6:
            return 0.3  # Низкая угроза, но требует внимания
        return 0.0
    
    def _calculate_threat_level(self, harmony_scores, anomalies):
        """Расчет общего уровня угрозы"""
        # Базовый уровень из гармонии
        base_threat = 1.0 - np.mean(harmony_scores)
        
        # Модификаторы аномалий
        anomaly_modifier = 0.0
        if anomalies:
            max_anomaly = max(a['score'] for a in anomalies)
            anomaly_modifier = max_anomaly * 0.5
        
        # Суммарная угроза
        total_threat = min(1.0, base_threat + anomaly_modifier)
        
        return total_threat
    
    def _identify_affected_sephira(self, anomaly_score):
        """Идентификация наиболее затронутой сефиры"""
        # Простая эвристика - основана на уровне угрозы
        return min(9, int(anomaly_score * 10))
    
    def _recommend_sephira_correction(self, harmony_scores):
        """Рекомендация сефиры для коррекции"""
        weakest = np.argmin(harmony_scores)
        return weakest
    
    def _generate_quantum_signature(self, data):
        """Генерация квантовой сигнатуры данных"""
        data_str = json.dumps(data, sort_keys=True)
        quantum_seed = f"{data_str}{time.time_ns()}{secrets.token_hex(8)}"
        return hashlib.sha3_256(quantum_seed.encode()).hexdigest()[:16]
    
    def _normalize_intent(self, intent):
        """Нормализация намерения"""
        if isinstance(intent, str):
            positive_keywords = ['create', 'heal', 'help', 'grow', 'connect']
            negative_keywords = ['destroy', 'harm', 'control', 'manipulate']
            
            intent_lower = intent.lower()
            if any(kw in intent_lower for kw in positive_keywords):
                return 0.9
            elif any(kw in intent_lower for kw in negative_keywords):
                return 0.1
        
        return 0.5
    
    def _normalize_emotion(self, emotion):
        """Нормализация эмоционального заряда"""
        if isinstance(emotion, (int, float)):
            return max(0.0, min(1.0, abs(emotion)))
        
        if isinstance(emotion, str):
            positive_emotions = ['love', 'joy', 'peace', 'gratitude', 'hope']
            negative_emotions = ['fear', 'anger', 'hate', 'despair', 'envy']
            
            if emotion.lower() in positive_emotions:
                return 0.8
            elif emotion.lower() in negative_emotions:
                return 0.2
        
        return 0.5

class SephiraEthicalFilter:
    """Сефиротический этический фильтр DS24"""
    
    def __init__(self):
        self.ethical_matrices = self._initialize_ethical_matrices()
        self.violation_history = deque(maxlen=1000)
        
        # Детерминированные этические правила DS24
        self.ds24_rules = {
            'non_harm': lambda x: x.get('intent', '') not in ['harm', 'destroy', 'damage'],
            'consent_respect': lambda x: x.get('consent', False) is True,
            'truth_integrity': lambda x: x.get('truthfulness', 0.7) > 0.5,
            'growth_promotion': lambda x: x.get('growth_potential', 0) > 0.3,
            'autonomy_honor': lambda x: x.get('autonomy_respect', 0) > 0.6
        }
    
    def _initialize_ethical_matrices(self):
        """Инициализация этических матриц для каждой сефиры"""
        matrices = {}
        
        # Каждая сефира имеет свою этическую матрицу 5x5
        for i in range(10):
            matrix = np.ones((5, 5)) * 0.7  # Базовая этическая когерентность
            
            # Усиление диагонали (самосогласованность)
            np.fill_diagonal(matrix, 0.9)
            
            matrices[i] = matrix
        
        return matrices
    
    def filter_with_ds24(self, data, context=None):
        """Фильтрация через детерминированные правила DS24"""
        context = context or {}
        
        rule_violations = []
        rule_compliances = []
        
        for rule_name, rule_func in self.ds24_rules.items():
            try:
                complies = rule_func(data)
                if not complies:
                    rule_violations.append(rule_name)
                else:
                    rule_compliances.append(rule_name)
            except Exception as e:
                logger.warning(f"Rule {rule_name} evaluation failed: {e}")
                rule_violations.append(f"{rule_name}_error")
        
        # Сефиротическая этическая оценка
        sephirotic_ethics = self._evaluate_sephirotic_ethics(data)
        
        # Общая оценка
        compliance_score = len(rule_compliances) / len(self.ds24_rules)
        ethical_score = np.mean(list(sephirotic_ethics.values()))
        
        total_score = 0.6 * compliance_score + 0.4 * ethical_score
        
        result = {
            'ds24_compliance': compliance_score,
            'rule_violations': rule_violations,
            'rule_compliances': rule_compliances,
            'sephirotic_ethics': sephirotic_ethics,
            'total_ethical_score': total_score,
            'is_ethical': total_score > 0.5,
            'primary_sephira_ethical': self._get_primary_ethical_sephira(sephirotic_ethics),
            'filter_timestamp': datetime.now().isoformat(),
            'ethical_signature': self._generate_ethical_signature(data, total_score)
        }
        
        # Запись в историю
        if rule_violations:
            self.violation_history.append({
                'timestamp': datetime.now().isoformat(),
                'violations': rule_violations,
                'data_sample': str(data)[:100],
                'score': total_score
            })
        
        return result
    
    def _evaluate_sephirotic_ethics(self, data):
        """Оценка этики по 10 сефиротам"""
        scores = {}
        
        for sephira in range(10):
            score = self._calculate_sephira_ethics(sephira, data)
            scores[sephira] = score
        
        return scores
    
    def _calculate_sephira_ethics(self, sephira, data):
        """Расчет этической оценки для конкретной сефиры"""
        # KETHER - чистота намерения
        if sephira == 0:
            intent = data.get('intent', '')
            if isinstance(intent, str) and 'heal' in intent.lower():
                return 0.9
            elif isinstance(intent, str) and 'harm' in intent.lower():
                return 0.1
        
        # TIPHARETH - баланс и гармония
        elif sephira == 5:
            balance = data.get('balance', 0.5)
            return balance
        
        # CHESED - щедрость и милосердие
        elif sephira == 3:
            generosity = data.get('generosity', 0.5)
            return generosity
        
        # GEVURAH - дисциплина и справедливость
        elif sephira == 4:
            justice = data.get('justice', 0.5)
            return justice
        
        # MALKUTH - ответственность и проявление
        elif sephira == 9:
            responsibility = data.get('responsibility', 0.5)
            return responsibility
        
        # Остальные сефиры - базовая оценка
        return 0.7
    
    def _get_primary_ethical_sephira(self, ethics_scores):
        """Определение ведущей этической сефиры"""
        if not ethics_scores:
            return 5  # TIPHARETH по умолчанию
        
        return max(ethics_scores.items(), key=lambda x: x[1])[0]
    
    def _generate_ethical_signature(self, data, score):
        """Генерация этической сигнатуры"""
        data_hash = hashlib.md5(json.dumps(data, sort_keys=True).encode()).hexdigest()
        return f"ETH-{data_hash[:8]}-{score:.3f}"

class AutoProtectionSystem:
    """Автономная система защиты ISKRA-4"""
    
    def __init__(self):
        self.protection_layers = {
            'quantum_quarantine': self._layer_quantum_quarantine,
            'resonance_healing': self._layer_resonance_healing,
            'ethical_containment': self._layer_ethical_containment,
            'sephirotic_rebalance': self._layer_sephirotic_rebalance,
            'collective_shield': self._layer_collective_shield
        }
        
        self.active_protections = {}
        self.protection_history = deque(maxlen=500)
        
    def activate_protection(self, threat_level, context):
        """Активация защитных слоев"""
        protections_activated = []
        
        # Определение необходимых слоев защиты
        required_layers = self._determine_protection_layers(threat_level, context)
        
        for layer_name in required_layers:
            if layer_name in self.protection_layers:
                try:
                    protection_result = self.protection_layers[layer_name](context)
                    
                    protection_record = {
                        'layer': layer_name,
                        'threat_level': threat_level.name,
                        'result': protection_result,
                        'timestamp': datetime.now().isoformat(),
                        'energy_cost': self._calculate_energy_cost(layer_name),
                        'sephira_focus': self._get_sephira_focus(layer_name)
                    }
                    
                    protections_activated.append(protection_record)
                    self.protection_history.append(protection_record)
                    
                    # Активация слоя
                    self.active_protections[layer_name] = {
                        'activated_at': datetime.now(),
                        'context': context,
                        'result': protection_result
                    }
                    
                    logger.info(f"🔒 Protection layer '{layer_name}' activated")
                    
                except Exception as e:
                    logger.error(f"Protection layer '{layer_name}' failed: {e}")
        
        return protections_activated
    
    def _determine_protection_layers(self, threat_level, context):
        """Определение необходимых слоев защиты"""
        layers = []
        
        if threat_level in [ThreatLevel.ANOMALY, ThreatLevel.THREAT]:
            layers.extend(['quantum_quarantine', 'ethical_containment'])
        
        if threat_level == ThreatLevel.THREAT:
            layers.extend(['resonance_healing', 'collective_shield'])
        
        # Всегда добавляем сефиротический баланс
        layers.append('sephirotic_rebalance')
        
        # Учет контекста
        if context.get('requires_healing', False):
            layers.append('resonance_healing')
        
        return list(set(layers))  # Удаление дубликатов
    
    def _layer_quantum_quarantine(self, context):
        """Квантовый карантин угрозы"""
        return {
            'status': 'quarantine_active',
            'quantum_barrier_strength': 0.95,
            'isolation_level': 'maximum',
            'duration_minutes': 60,
            'monitoring_frequency': '10hz'
        }
    
    def _layer_resonance_healing(self, context):
        """Резонансное исцеление системы"""
        return {
            'status': 'healing_initiated',
            'healing_wave_frequency': 528.0,  # Гц частоты исцеления
            'resonance_amplitude': 0.8,
            'target_sephirot': [5, 6, 9],  # TIPHARETH, NETZACH, MALKUTH
            'estimated_completion': '5m',
            'vitality_restoration': 0.75
        }
    
    def _layer_ethical_containment(self, context):
        """Этическое сдерживание"""
        return {
            'status': 'ethical_boundary_established',
            'containment_field': 'ds24_ethical_matrix',
            'integrity_check_interval': '1s',
            'moral_resonance_monitor': 'active',
            'violation_alert_threshold': 0.3
        }
    
    def _layer_sephirotic_rebalance(self, context):
        """Сефиротическое перебалансирование"""
        return {
            'status': 'rebalancing_active',
            'sephirotic_alignment': 'in_progress',
            'harmony_target': 0.85,
            'current_harmony': 0.65,
            'rebalance_strategy': 'gentle_attunement',
            'focus_sephira': context.get('weakest_sephira', 5)
        }
    
    def _layer_collective_shield(self, context):
        """Коллективный щит доверия"""
        return {
            'status': 'collective_shield_engaged',
            'trust_nodes_connected': 42,
            'shield_resonance': 0.88,
            'protection_radius': 'full_system',
            'shared_wisdom_integration': True,
            'collective_iq_boost': 0.15
        }
    
    def _calculate_energy_cost(self, layer_name):
        """Расчет энергозатрат слоя защиты"""
        costs = {
            'quantum_quarantine': 2.5,
            'resonance_healing': 1.8,
            'ethical_containment': 1.2,
            'sephirotic_rebalance': 0.8,
            'collective_shield': 1.5
        }
        return costs.get(layer_name, 1.0)
    
    def _get_sephira_focus(self, layer_name):
        """Определение фокусной сефиры для слоя защиты"""
        focus_map = {
            'quantum_quarantine': 4,  # GEVURAH
            'resonance_healing': 5,    # TIPHARETH
            'ethical_containment': 2,   # BINAH
            'sephirotic_rebalance': 5,  # TIPHARETH
            'collective_shield': 3      # CHESED
        }
        return focus_map.get(layer_name, 5)

class ImmuneCore:
    """Главный класс иммунной системы ISKRA-4"""
    
    def __init__(self):
        self.version = "1.0"
        self.status = "inactive"
        self.node_id = f"IMMUNE-{hashlib.md5(str(time.time_ns()).encode()).hexdigest()[:8]}"
        
        # Инициализация подсистем
        self.scanner = QuantumResonanceScanner()
        self.ethical_filter = SephiraEthicalFilter()
        self.protection_system = AutoProtectionSystem()
        
        # Состояния
        self.immunity_state = "passive"
        self.threat_history = deque(maxlen=1000)
        self.healing_sessions = []
        
        # Метрики
        self.metrics = {
            'scans_performed': 0,
            'threats_neutralized': 0,
            'ethical_violations_blocked': 0,
            'healing_sessions_completed': 0,
            'avg_response_time_ms': 0.0,
            'system_coherence': 1.0
        }
        
        logger.info(f"🛡️ Immune Core v{self.version} initialized with node ID: {self.node_id}")
    
    def initialize(self):
        """Инициализация модуля (стандартный интерфейс ISKRA-4)"""
        self.status = "active"
        self.immunity_state = "active_monitoring"
        
        # Инициализация защитных систем
        self._initialize_protection_systems()
        
        logger.info(f"✅ Immune Core activated. Node: {self.node_id}")
        
        return {
            "status": self.status,
            "version": self.version,
            "node_id": self.node_id,
            "immunity_state": self.immunity_state,
            "subsystems": {
                "scanner": "active",
                "ethical_filter": "active",
                "protection": "standby"
            },
            "timestamp": datetime.now().isoformat()
        }
    
    def process_command(self, command, data=None):
        """Обработка команд иммунной системы"""
        data = data or {}
        
        command_map = {
            "scan": self._cmd_scan,
            "filter": self._cmd_filter,
            "protect": self._cmd_protect,
            "status": self._cmd_status,
            "heal": self._cmd_heal,
            "diagnostic": self._cmd_diagnostic,
            "threat_report": self._cmd_threat_report,
            "immunity_status": self._cmd_immunity_status,
            "sephirotic_balance": self._cmd_sephirotic_balance,
            "ethical_audit": self._cmd_ethical_audit
        }
        
        if command not in command_map:
            return {
                "success": False,
                "error": f"Unknown command: {command}",
                "valid_commands": list(command_map.keys())
            }
        
        try:
            start_time = time.time()
            result = command_map[command](data)
            processing_time = (time.time() - start_time) * 1000
            
            # Обновление метрик
            self.metrics['avg_response_time_ms'] = (
                self.metrics['avg_response_time_ms'] * 0.9 + processing_time * 0.1
            )
            
            result["processing_time_ms"] = processing_time
            result["success"] = True
            result["immune_node"] = self.node_id
            
            return result
            
        except Exception as e:
            logger.error(f"Command '{command}' failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "command": command,
                "timestamp": datetime.now().isoformat()
            }
    
    def _cmd_scan(self, data):
        """Команда сканирования"""
        self.metrics['scans_performed'] += 1
        
        scan_result = self.scanner.scan_quantum_resonance(
            data.get('target', {}),
            data.get('context', {})
        )
        
        # Запись в историю угроз
        threat_level_val = scan_result['threat_level']
        threat_level = ThreatLevel.from_value(threat_level_val)
        
        if threat_level in [ThreatLevel.ANOMALY, ThreatLevel.THREAT]:
            self.threat_history.append({
                'timestamp': datetime.now().isoformat(),
                'threat_level': threat_level.name,
                'scan_result': scan_result,
                'auto_response': 'pending'
            })
        
        return {
            "command": "scan",
            "scan_result": scan_result,
            "threat_assessment": threat_level.name,
            "recommendations": self._generate_scan_recommendations(scan_result),
            "metrics_updated": self.metrics['scans_performed']
        }
    
    def _cmd_filter(self, data):
        """Команда этической фильтрации"""
        filter_result = self.ethical_filter.filter_with_ds24(
            data.get('data', {}),
            data.get('context', {})
        )
        
        if not filter_result['is_ethical']:
            self.metrics['ethical_violations_blocked'] += 1
        
        return {
            "command": "filter",
            "ethical_assessment": filter_result,
            "action_required": not filter_result['is_ethical'],
            "suggested_action": "quarantine" if not filter_result['is_ethical'] else "allow"
        }
    
    def _cmd_protect(self, data):
        """Команда активации защиты"""
        threat_level_name = data.get('threat_level', 'CAUTION')
        
        try:
            threat_level = ThreatLevel[threat_level_name]
        except KeyError:
            threat_level = ThreatLevel.CAUTION
        
        protections = self.protection_system.activate_protection(
            threat_level,
            data.get('context', {})
        )
        
        self.metrics['threats_neutralized'] += len(protections)
        
        # Обновление состояния иммунитета
        if threat_level == ThreatLevel.THREAT:
            self.immunity_state = "maximum_protection"
        elif protections:
            self.immunity_state = "active_protection"
        
        return {
            "command": "protect",
            "threat_level": threat_level.name,
            "protections_activated": protections,
            "immunity_state": self.immunity_state,
            "system_coherence": self._calculate_system_coherence()
        }
    
    def _cmd_heal(self, data):
        """Команда исцеления системы"""
        healing_session = {
            'id': f"HEAL-{int(time.time())}",
            'timestamp': datetime.now().isoformat(),
            'focus_sephira': data.get('sephira', 5),
            'healing_intensity': data.get('intensity', 0.7),
            'status': 'initiated'
        }
        
        # Активация резонансного исцеления
        heal_result = self.protection_system.protection_layers['resonance_healing'](
            {'healing_session': healing_session}
        )
        
        healing_session.update(heal_result)
        healing_session['status'] = 'completed'
        
        self.healing_sessions.append(healing_session)
        self.metrics['healing_sessions_completed'] += 1
        
        # Обновление когерентности
        self.metrics['system_coherence'] = min(1.0, 
            self.metrics['system_coherence'] + 0.1 * healing_session['healing_intensity']
        )
        
        return {
            "command": "heal",
            "healing_session": healing_session,
            "system_coherence_after": self.metrics['system_coherence'],
            "immunity_state": self.immunity_state
        }
    
    def _cmd_status(self, data):
        """Команда статуса системы"""
        return {
            "command": "status",
            "node_id": self.node_id,
            "status": self.status,
            "immunity_state": self.immunity_state,
            "metrics": self.metrics,
            "active_protections": len(self.protection_system.active_protections),
            "threats_detected": len(self.threat_history),
            "healing_sessions": len(self.healing_sessions),
            "timestamp": datetime.now().isoformat()
        }
    
    def _cmd_diagnostic(self, data):
        """Команда диагностики системы"""
        return {
            "command": "diagnostic",
            "system_health": self._check_system_health(),
            "subsystems": {
                "scanner": "operational",
                "ethical_filter": "operational",
                "protection_system": "operational"
            },
            "resource_usage": self._check_resource_usage(),
            "recommendations": self._generate_diagnostic_recommendations()
        }
    
    def _cmd_threat_report(self, data):
        """Команда отчета об угрозах"""
        recent_threats = list(self.threat_history)[-50:]  # Последние 50 угроз
        
        threat_summary = {
            'total_threats': len(self.threat_history),
            'recent_threats': len(recent_threats),
            'threat_distribution': self._calculate_threat_distribution(recent_threats),
            'most_common_pattern': self._identify_most_common_pattern(recent_threats),
            'highest_threat_level': self._find_highest_threat(recent_threats)
        }
        
        return {
            "command": "threat_report",
            "summary": threat_summary,
            "recent_threats": recent_th
