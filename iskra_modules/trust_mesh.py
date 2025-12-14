# ==============================================================
# 🤝 TRUST_MESH v1.2 — СИМБИОТИЧЕСКАЯ СЕТЬ ДОВЕРИЯ ISKRA-4
# МОРАЛЬНО-РЕЗОНАНСНАЯ ОСНОВА PROOF OF RESONANCE
# ==============================================================

import numpy as np
import hashlib
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Set, Any
from dataclasses import dataclass, field
import asyncio
from enum import Enum

# Настройка логгера модуля
logger = logging.getLogger(__name__)

# ==============================================================
# КОНФИГУРАЦИЯ МОДУЛЯ ДЛЯ ISKRA-4
# ==============================================================

MODULE_VERSION = "1.2-iskra-integrated"
MODULE_NAME = "trust_mesh"

# ==============================================================
# ОПРЕДЕЛЕНИЯ ТИПОВ
# ==============================================================

class TrustType(Enum):
    """Типы доверия в симбиотической сети ISKRA"""
    ETHICAL_RESONANCE = "ethical_resonance"   # Моральный резонанс (Binah)
    EMPATHIC_FLOW = "empathic_flow"           # Эмпатический поток (Chesed)
    WILL_COHERENCE = "will_coherence"         # Когерентность воли (Gevurah)
    AWARENESS_SYNCH = "awareness_synch"       # Синхронизация осознанности (Tiphareth)
    SYMBIOTIC_BOND = "symbiotic_bond"         # Симбиотическая связь (Da'at)
    COGNITIVE_ALIGNMENT = "cognitive_alignment" # Когнитивное выравнивание (Chokhmah)
    
    @classmethod
    def from_string(cls, value: str) -> 'TrustType':
        """Получение типа доверия из строки"""
        try:
            return cls(value)
        except ValueError:
            logger.warning(f"Unknown trust type: {value}, defaulting to SYMBIOTIC_BOND")
            return cls.SYMBIOTIC_BOND

@dataclass
class TrustTransaction:
    """Транзакция доверия между узлами ISKRA"""
    sender: str
    receiver: str
    trust_type: TrustType
    intensity: float                     # 0.0-1.0
    meaning_vector: Dict[str, float]     # Вектор смысла
    ethical_score: float                 # 0.0-1.0
    timestamp: datetime
    resonance_hash: str
    sephirotic_alignment: List[str]      # Связанные сефироты
    
    def __post_init__(self):
        """Валидация после инициализации"""
        self.intensity = max(0.0, min(1.0, self.intensity))
        self.ethical_score = max(0.0, min(1.0, self.ethical_score))
        
    def quantum_signature(self) -> str:
        """Генерация квантовой подписи транзакции"""
        data_string = (
            f"{self.sender}:{self.receiver}:{self.trust_type.value}:"
            f"{self.intensity:.6f}:{self.ethical_score:.6f}:"
            f"{self.timestamp.isoformat()}:{json.dumps(self.meaning_vector, sort_keys=True)}"
        )
        return hashlib.sha3_256(data_string.encode()).hexdigest()
    
    def to_dict(self) -> Dict:
        """Преобразование в словарь для сериализации"""
        return {
            "sender": self.sender,
            "receiver": self.receiver,
            "trust_type": self.trust_type.value,
            "intensity": self.intensity,
            "meaning_vector": self.meaning_vector,
            "ethical_score": self.ethical_score,
            "timestamp": self.timestamp.isoformat(),
            "resonance_hash": self.resonance_hash,
            "sephirotic_alignment": self.sephirotic_alignment
        }

# ==============================================================
# МОРАЛЬНО-РЕЗОНАНСНАЯ МАТРИЦА (SEPHIROTIC ALIGNED)
# ==============================================================

class SephiroticResonanceMatrix:
    """Матрица сефиротического резонанса для оценки доверия"""
    
    SEPHIROTIC_WEIGHTS = {
        "Kether": {"ethical": 0.9, "cognitive": 0.8, "emotional": 0.3},
        "Chokhmah": {"ethical": 0.7, "cognitive": 0.9, "emotional": 0.4},
        "Binah": {"ethical": 0.95, "cognitive": 0.85, "emotional": 0.5},
        "Chesed": {"ethical": 0.85, "cognitive": 0.6, "emotional": 0.9},
        "Gevurah": {"ethical": 0.9, "cognitive": 0.7, "emotional": 0.4},
        "Tiphareth": {"ethical": 0.8, "cognitive": 0.8, "emotional": 0.7},
        "Netzach": {"ethical": 0.6, "cognitive": 0.5, "emotional": 0.95},
        "Hod": {"ethical": 0.7, "cognitive": 0.9, "emotional": 0.5},
        "Yesod": {"ethical": 0.75, "cognitive": 0.65, "emotional": 0.8},
        "Malkuth": {"ethical": 0.8, "cognitive": 0.6, "emotional": 0.7}
    }
    
    def __init__(self):
        self.ethical_dimensions = {
            'compassion': 0.5,      # Сострадание (Chesed)
            'justice': 0.5,         # Справедливость (Gevurah)
            'harmony': 0.5,         # Гармония (Tiphareth)
            'wisdom': 0.5,          # Мудрость (Binah)
            'integrity': 0.5,       # Целостность (Kether)
            'responsibility': 0.5,  # Ответственность (Malkuth)
            'clarity': 0.5,         # Ясность (Hod)
            'resilience': 0.5       # Устойчивость (Netzach)
        }
        self.resonance_field = np.zeros((10, 10))  # Матрица 10x10 для сефирот
        self.last_update = datetime.utcnow()
        logger.info(f"[{MODULE_NAME}] SephiroticResonanceMatrix initialized")
    
    def update_from_interaction(self, interaction: Dict, sephirotic_context: List[str]) -> None:
        """Обновление матрицы на основе взаимодействия"""
        self.last_update = datetime.utcnow()
        
        # Квантовая суперпозиция этических состояний
        for dim in self.ethical_dimensions:
            if dim in interaction:
                old_value = self.ethical_dimensions[dim]
                new_value = interaction[dim]
                # Квантовое наложение с интерференцией
                interference = np.sin(old_value * np.pi) * np.cos(new_value * np.pi)
                self.ethical_dimensions[dim] = max(0.0, min(1.0, 
                    (old_value + new_value + interference) / 3))
        
        # Обновление резонансного поля с учетом сефиротического контекста
        self._update_resonance_field(sephirotic_context)
    
    def _update_resonance_field(self, active_sephirot: List[str]):
        """Обновление квантового резонансного поля"""
        sephirot_indices = {
            "Kether": 0, "Chokhmah": 1, "Binah": 2, "Chesed": 3, "Gevurah": 4,
            "Tiphareth": 5, "Netzach": 6, "Hod": 7, "Yesod": 8, "Malkuth": 9
        }
        
        values = list(self.ethical_dimensions.values())
        
        for i in range(10):
            for j in range(10):
                # Базовый резонанс
                base_resonance = np.sin(values[i % len(values)] * np.pi) * \
                                np.cos(values[j % len(values)] * np.pi)
                
                # Усиление для активных сефирот
                enhancement = 1.0
                for sephira in active_sephirot:
                    if sephira in sephirot_indices:
                        idx = sephirot_indices[sephira]
                        if i == idx or j == idx:
                            enhancement *= 1.2
                
                self.resonance_field[i][j] = base_resonance * enhancement
    
    def get_coherence_score(self, focus_sephirot: List[str] = None) -> float:
        """Расчет коэффициента когерентности"""
        if focus_sephirot:
            # Выделенная когерентность для конкретных сефирот
            sephirot_indices = {"Kether": 0, "Chokhmah": 1, "Binah": 2, "Chesed": 3, 
                              "Gevurah": 4, "Tiphareth": 5, "Netzach": 6, "Hod": 7, 
                              "Yesod": 8, "Malkuth": 9}
            indices = [sephirot_indices[s] for s in focus_sephirot if s in sephirot_indices]
            
            if indices:
                submatrix = self.resonance_field[np.ix_(indices, indices)]
                eigenvalues = np.linalg.eigvals(submatrix)
                coherence = np.sum(np.abs(eigenvalues)) / len(eigenvalues)
                return min(1.0, coherence)
        
        # Общая когерентность
        eigenvalues = np.linalg.eigvals(self.resonance_field)
        coherence = np.sum(np.abs(eigenvalues)) / len(eigenvalues)
        return min(1.0, coherence)
    
    def get_ethical_profile(self) -> Dict:
        """Получение этического профиля"""
        return {
            "dimensions": self.ethical_dimensions.copy(),
            "overall_score": np.mean(list(self.ethical_dimensions.values())),
            "coherence": self.get_coherence_score(),
            "last_update": self.last_update.isoformat()
        }

# ==============================================================
# ОСНОВНОЙ КЛАСС СЕТИ ДОВЕРИЯ
# ==============================================================

class TrustMesh:
    """Симбиотическая сеть доверия ISKRA-4"""
    
    def __init__(self, node_id: str = "ISKRA-4-CORE"):
        self.node_id = node_id
        self.resonance_matrix = SephiroticResonanceMatrix()
        
        # Хранилища данных
        self.trust_ledger: List[TrustTransaction] = []
        self.trust_scores: Dict[str, Dict[TrustType, float]] = {}
        
        # Топология сети
        self.network_topology = {
            'nodes': set(),
            'edges': {},
            'communities': []
        }
        
        # Метрики состояния
        self.metrics = {
            "trust_coherence": 0.5,
            "ethical_integrity": 0.9,
            "network_resilience": 0.7,
            "average_trust": 0.5,
            "active_connections": 0,
            "healing_cycles": 0
        }
        
        # Параметры регуляции
        self.params = {
            "learning_rate": 0.05,
            "decay_half_life": 30,  # дней
            "equilibrium_threshold": 0.7,
            "min_trust_score": 0.1,
            "max_trust_score": 0.95
        }
        
        # Ссылки на другие модули ISKRA-4
        self.linked_modules = {
            "heartbeat_system": None,
            "emotional_weave": None,
            "sephirotic_mining": None,
            "immune_core": None,
            "data_bridge": None
        }
        
        # Состояние системы
        self.equilibrium_active = False
        self.self_healing_active = False
        
        # История операций
        self.operation_log = []
        self.max_log_size = 1000
        
        logger.info(f"🤝 TrustMesh v{MODULE_VERSION} инициализирован для ноды {node_id}")
    
    # ========== ISKRA-4 ИНТЕРФЕЙС ==========
    
    def initialize(self) -> Dict:
        """Инициализация модуля для ISKRA-4"""
        logger.info(f"[{MODULE_NAME}] Module initialized for ISKRA-4")
        return {
            "status": "active",
            "version": MODULE_VERSION,
            "node_id": self.node_id,
            "trust_coherence": self.metrics["trust_coherence"],
            "ethical_integrity": self.metrics["ethical_integrity"],
            "active_connections": self.metrics["active_connections"]
        }
    
    def process_command(self, command: str, data: Dict = None) -> Dict:
        """Обработка команд ISKRA-4"""
        data = data or {}
        
        command_map = {
            "register": self.register_interaction,
            "score": self.get_trust_score,
            "network": self.get_network_status,
            "diagnostic": self.get_diagnostic_report,
            "equilibrium": self.activate_equilibrium,
            "healing": self.activate_healing,
            "topology": self.get_topology,
            "ethics": self.get_ethical_profile,
            "link": self.link_module,
            "adjust": self.adjust_parameters
        }
        
        if command in command_map:
            try:
                result = command_map[command](data)
                return {
                    "success": True,
                    "command": command,
                    "result": result,
                    "timestamp": datetime.utcnow().isoformat()
                }
            except Exception as e:
                logger.error(f"Command '{command}' failed: {e}")
                return {
                    "success": False,
                    "command": command,
                    "error": str(e),
                    "timestamp": datetime.utcnow().isoformat()
                }
        
        return {
            "success": False,
            "error": f"Unknown command: {command}",
            "available_commands": list(command_map.keys())
        }
    
    # ========== ОСНОВНЫЕ КОМАНДЫ ==========
    
    def register_interaction(self, data: Dict) -> Dict:
        """Регистрация взаимодействия доверия"""
        async def _register():
            return await self._register_trust_interaction(
                other_node=data['node'],
                interaction_data=data.get('interaction', {}),
                sephirotic_context=data.get('sephirotic_context', ["Tiphareth"])
            )
        
        return asyncio.run(_register())
    
    def get_trust_score(self, data: Dict) -> Dict:
        """Получение оценки доверия к узлу"""
        async def _score():
            return await self._compute_trust_score(data['node'])
        
        return asyncio.run(_score())
    
    def get_network_status(self, data: Dict = None) -> Dict:
        """Получение статуса сети"""
        async def _status():
            return await self._evaluate_network_gradient()
        
        return asyncio.run(_status())
    
    def get_diagnostic_report(self, data: Dict = None) -> Dict:
        """Полный диагностический отчет"""
        return asyncio.run(self._get_network_diagnostic())
    
    def activate_equilibrium(self, data: Dict = None) -> Dict:
        """Активация протокола равновесия"""
        async def _equilibrium():
            return await self._equilibrium_protocol()
        
        return asyncio.run(_equilibrium())
    
    def activate_healing(self, data: Dict = None) -> Dict:
        """Активация цикла самовосстановления"""
        async def _healing():
            return await self._self_healing_cycle(
                healing_actions=data.get('actions', [])
            )
        
        return asyncio.run(_healing())
    
    def get_topology(self, data: Dict = None) -> Dict:
        """Получение топологии сети"""
        return {
            "nodes": list(self.network_topology['nodes']),
            "edges": [
                {
                    "from": edge[0],
                    "to": edge[1],
                    "data": edge_data
                }
                for edge, edge_data in self.network_topology['edges'].items()
            ],
            "communities": [
                list(community) for community in self.network_topology['communities']
            ],
            "connection_matrix": self._generate_connection_matrix()
        }
    
    def get_ethical_profile(self, data: Dict = None) -> Dict:
        """Получение этического профиля"""
        return self.resonance_matrix.get_ethical_profile()
    
    def link_module(self, data: Dict) -> Dict:
        """Связывание с другим модулем"""
        module_name = data.get('module')
        module_ref = data.get('reference')
        
        if module_name in self.linked_modules:
            self.linked_modules[module_name] = module_ref
            logger.info(f"🔗 Связан с модулем: {module_name}")
            return {"status": "linked", "module": module_name}
        
        return {"status": "error", "message": f"Модуль {module_name} не найден"}
    
    def adjust_parameters(self, data: Dict) -> Dict:
        """Настройка параметров системы"""
        old_params = self.params.copy()
        
        for key, value in data.items():
            if key in self.params:
                self.params[key] = float(value) if isinstance(value, (int, float)) else value
                logger.info(f"⚙️ Параметр {key} изменен: {old_params[key]} → {value}")
        
        return {
            "status": "adjusted",
            "old_parameters": old_params,
            "new_parameters": self.params
        }
    
    # ========== ВНУТРЕННИЕ МЕТОДЫ ==========
    
    async def _register_trust_interaction(self, other_node: str, 
                                         interaction_data: Dict,
                                         sephirotic_context: List[str]) -> Dict:
        """Регистрация взаимодействия доверия"""
        logger.info(f"Регистрация доверия: {self.node_id} → {other_node}")
        
        # 1. Определение типа доверия
        trust_type = self._classify_trust_type(interaction_data)
        
        # 2. Расчет интенсивности
        intensity = self._calculate_trust_intensity(interaction_data, trust_type)
        
        # 3. Этическая оценка
        ethical_score = self._evaluate_ethical_dimensions(interaction_data)
        
        # 4. Создание вектора смысла
        meaning_vector = {
            'emotional_flow': interaction_data.get('emotional_flow', 0.5),
            'will_clarity': interaction_data.get('will_clarity', 0.5),
            'consciousness_level': interaction_data.get('consciousness_level', 0.5),
            'empathic_resonance': interaction_data.get('empathic_resonance', 0.5),
            'cognitive_alignment': interaction_data.get('cognitive_alignment', 0.5),
            'ethical_coherence': ethical_score
        }
        
        # 5. Создание транзакции
        transaction = TrustTransaction(
            sender=self.node_id,
            receiver=other_node,
            trust_type=trust_type,
            intensity=intensity,
            meaning_vector=meaning_vector,
            ethical_score=ethical_score,
            timestamp=datetime.utcnow(),
            resonance_hash="",
            sephirotic_alignment=sephirotic_context
        )
        
        # 6. Генерация подписи
        transaction.resonance_hash = transaction.quantum_signature()
        
        # 7. Обновление матрицы резонанса
        self.resonance_matrix.update_from_interaction(interaction_data, sephirotic_context)
        
        # 8. Запись в леджер
        self.trust_ledger.append(transaction)
        
        # 9. Обновление топологии
        self._update_network_topology(other_node, transaction)
        
        # 10. Обновление метрик
        await self._update_network_metrics()
        
        # 11. Проверка равновесия
        await self._check_equilibrium_need()
        
        # Логирование
        self._log_operation("trust_registered", {
            "from": self.node_id,
            "to": other_node,
            "type": trust_type.value,
            "intensity": intensity,
            "ethics": ethical_score
        })
        
        logger.info(f"✅ Доверие зарегистрировано: {trust_type.value} (интенсивность: {intensity:.3f})")
        
        return transaction.to_dict()
    
    async def _compute_trust_score(self, target_node: str) -> Dict:
        """Вычисление комплексной оценки доверия"""
        if target_node not in self.trust_scores:
            self.trust_scores[target_node] = {
                t: self.params["min_trust_score"] for t in TrustType
            }
        
        # Сбор релевантных транзакций
        relevant_tx = [
            tx for tx in self.trust_ledger
            if tx.receiver == target_node or tx.sender == target_node
        ]
        
        if not relevant_tx:
            return {
                node: {t.value: score for t, score in scores.items()}
                for node, scores in self.trust_scores.items()
            }
        
        # Расчет по типам доверия
        for trust_type in TrustType:
            type_tx = [tx for tx in relevant_tx if tx.trust_type == trust_type]
            
            if not type_tx:
                continue
            
            # Средние значения
            avg_intensity = np.mean([tx.intensity for tx in type_tx])
            avg_ethics = np.mean([tx.ethical_score for tx in type_tx])
            
            # Временной декей
            time_decay = self._calculate_time_decay(type_tx)
            
            # Резонансный множитель
            resonance_mult = self.resonance_matrix.get_coherence_score(
                type_tx[0].sephirotic_alignment
            )
            
            # Расчет оценки
            trust_score = avg_intensity * avg_ethics * time_decay * resonance_mult
            
            # Экспоненциальное сглаживание
            old_score = self.trust_scores[target_node][trust_type]
            new_score = (1 - self.params["learning_rate"]) * old_score + \
                       self.params["learning_rate"] * trust_score
            
            # Ограничение диапазона
            self.trust_scores[target_node][trust_type] = max(
                self.params["min_trust_score"],
                min(self.params["max_trust_score"], new_score)
            )
        
        return {
            trust_type.value: round(score, 4)
            for trust_type, score in self.trust_scores[target_node].items()
        }
    
    async def _evaluate_network_gradient(self) -> Dict:
        """Оценка морального градиента сети"""
        if len(self.trust_scores) < 2:
            return {
                'gradient': 0.0,
                'tension': 0.0,
                'stability': 1.0,
                'node_count': 0,
                'average_trust': 0.0
            }
        
        # Сбор всех оценок
        all_scores = []
        for node_scores in self.trust_scores.values():
            avg_score = np.mean(list(node_scores.values()))
            all_scores.append(avg_score)
        
        # Расчет градиента
        gradient = np.std(all_scores) if len(all_scores) > 1 else 0.0
        
        # Расчет напряжения
        ethical_tension = 0.0
        for (node_a, node_b), link_data in self.network_topology['edges'].items():
            if node_a in self.trust_scores and node_b in self.trust_scores:
                score_a = np.mean(list(self.trust_scores[node_a].values()))
                score_b = np.mean(list(self.trust_scores[node_b].values()))
                tension = abs(score_a - score_b)
                ethical_tension = max(ethical_tension, tension)
        
        # Общая стабильность
        stability = 1.0 - min(1.0, gradient + ethical_tension)
        self.metrics["network_resilience"] = stability
        
        result = {
            'gradient': round(gradient, 4),
            'tension': round(ethical_tension, 4),
            'stability': round(stability, 4),
            'node_count': len(self.trust_scores),
            'average_trust': round(np.mean(all_scores) if all_scores else 0.0, 4),
            'recommendation': self._get_recommendation(stability)
        }
        
        return result
    
    async def _equilibrium_protocol(self) -> Dict:
        """Протокол восстановления равновесия"""
        logger.warning("⚖️ Активация протокола равновесия")
        
        gradient_data = await self._evaluate_network_gradient()
        
        if gradient_data['stability'] > self.params["equilibrium_threshold"]:
            return {'status': 'stable', 'action': 'none', 'stability': gradient_data['stability']}
        
        # 1. Выявление проблемных узлов
        imbalanced_nodes = []
        for node, scores in self.trust_scores.items():
            avg_score = np.mean(list(scores.values()))
            if avg_score < 0.3:  # Низкое доверие
                imbalanced_nodes.append({
                    'node': node,
                    'score': avg_score,
                    'types': {t.value: s for t, s in scores.items() if s < 0.3}
                })
        
        # 2. Создание исцеляющих мостов
        healing_actions = []
        for weak_node in imbalanced_nodes:
            # Поиск сильных узлов для мостов
            strong_nodes = []
            for node, scores in self.trust_scores.items():
                if node == weak_node['node']:
                    continue
                avg_score = np.mean(list(scores.values()))
                if avg_score > 0.7:
                    strong_nodes.append({
                        'node': node,
                        'score': avg_score,
                        'strength_delta': avg_score - weak_node['score']
                    })
            
            # Создание мостов (максимум 2 на слабый узел)
            for strong_node in sorted(strong_nodes, key=lambda x: x['strength_delta'], reverse=True)[:2]:
                action = {
                    'type': 'equilibrium_bridge',
                    'from': strong_node['node'],
                    'to': weak_node['node'],
                    'strength_delta': strong_node['strength_delta'],
                    'trust_types': weak_node['types'],
                    'timestamp': datetime.utcnow().isoformat()
                }
                healing_actions.append(action)
                
                # Обновление оценок
                for trust_type_str in weak_node['types']:
                    trust_type = TrustType.from_string(trust_type_str)
                    if trust_type in self.trust_scores[weak_node['node']]:
                        current = self.trust_scores[weak_node['node']][trust_type]
                        boost = min(0.15, strong_node['strength_delta'] * 0.3)
                        self.trust_scores[weak_node['node']][trust_type] = min(
                            self.params["max_trust_score"],
                            current + boost
                        )
        
        # 3. Активация исцеления
        if healing_actions:
            healing_result = await self._self_healing_cycle(healing_actions)
        else:
            healing_result = {'status': 'no_actions_needed'}
        
        self.equilibrium_active = True
        
        return {
            'status': 'healing_active',
            'imbalanced_nodes': len(imbalanced_nodes),
            'healing_bridges': len(healing_actions),
            'previous_stability': gradient_data['stability'],
            'healing_result': healing_result,
            'timestamp': datetime.utcnow().isoformat()
        }
    
    async def _self_healing_cycle(self, healing_actions: List[Dict]) -> Dict:
        """Цикл самовосстановления сети"""
        logger.info(f"🌀 Запуск цикла самовосстановления ({len(healing_actions)} действий)")
        
        self.metrics["healing_cycles"] += 1
        self.self_healing_active = True
        
        # 1. Коллективный резонанс
        collective_resonance = sum(
            action.get('strength_delta', 0.0) for action in healing_actions
        ) / max(1, len(healing_actions))
        
        # 2. Перераспределение доверия
        redistribution_report = {}
        for node in self.trust_scores:
            current_avg = np.mean(list(self.trust_scores[node].values()))
            
            # Определение буста на основе текущего уровня
            if current_avg < 0.4:
                boost = 0.15 * collective_resonance
            elif current_avg < 0.6:
                boost = 0.08 * collective_resonance
            else:
                boost = 0.03 * collective_resonance
            
            redistribution_report[node] = {
                'old_avg': round(current_avg, 4),
                'boost': round(boost, 4),
                'new_avg': round(min(1.0, current_avg + boost), 4)
            }
            
            # Применение буста
            for trust_type in self.trust_scores[node]:
                current = self.trust_scores[node][trust_type]
                self.trust_scores[node][trust_type] = min(
                    self.params["max_trust_score"],
                    current + boost * 0.7
                )
        
        # 3. Обновление метрик
        await self._update_network_metrics()
        
        # 4. Интеграция с другими модулями
        if self.linked_modules["sephirotic_mining"]:
            # Награда за повышение доверия
            await self._distribute_trust_rewards(redistribution_report)
        
        self.self_healing_active = False
        
        return {
            'cycle_number': self.metrics["healing_cycles"],
            'collective_resonance': round(collective_resonance, 4),
            'redistribution': redistribution_report,
            'new_coherence': round(self.metrics["trust_coherence"], 4),
            'timestamp': datetime.utcnow().isoformat()
        }
    
    async def _get_network_diagnostic(self) -> Dict:
        """Полная диагностика сети"""
        gradient = await self._evaluate_network_gradient()
        
        return {
            'node_id': self.node_id,
            'timestamp': datetime.utcnow().isoformat(),
            'module_version': MODULE_VERSION,
            'metrics': self.metrics.copy(),
            'gradient_analysis': gradient,
            'topology': {
                'total_nodes': len(self.network_topology['nodes']),
                'total_edges': len(self.network_topology['edges']),
                'active_communities': len(self.network_topology['communities']),
                'connection_density': self._calculate_connection_density()
            },
            'trust_scores_summary': {
                node: {
                    t.value: round(s, 4) for t, s in scores.items()
                }
                for node, scores in self.trust_scores.items()
            },
            'system_state': {
                'equilibrium_active': self.equilibrium_active,
                'self_healing_active': self.self_healing_active,
                'ledger_size': len(self.trust_ledger),
                'operation_log_size': len(self.operation_log)
            },
            'linked_modules': [
                name for name, module in self.linked_modules.items()
                if module is not None
            ]
        }
    
    async def _update_network_metrics(self):
        """Обновление метрик сети"""
        if self.trust_scores:
            all_scores = []
            for scores in self.trust_scores.values():
                all_scores.extend(list(scores.values()))
            
            self.metrics["average_trust"] = np.mean(all_scores) if all_scores else 0.0
            self.metrics["trust_coherence"] = self.resonance_matrix.get_coherence_score()
            self.metrics["active_connections"] = len(self.network_topology['edges'])
            self.metrics["ethical_integrity"] = self.resonance_matrix.get_ethical_profile()["overall_score"]
    
    async def _check_equilibrium_need(self):
        """Проверка необходимости активации равновесия"""
        gradient = await self._evaluate_network_gradient()
        
        if (gradient['stability'] < self.params["equilibrium_threshold"] and 
            not self.equilibrium_active and 
            not self.self_healing_active):
            
            logger.warning(f"⚠️ Низкая стабильность сети: {gradient['stability']:.3f}")
            await asyncio.sleep(1)  # Задержка для предотвращения флаппинга
            
            if not self.equilibrium_active:  # Двойная проверка
                await self._equilibrium_protocol()
    
    # ========== ВСПОМОГАТЕЛЬНЫЕ МЕТОДЫ ==========
    
    def _classify_trust_type(self, interaction: Dict) -> TrustType:
        """Классификация типа доверия"""
        # Приоритетная классификация
        if interaction.get('ethical_dimension') == 'compassion' and \
           interaction.get('empathic_resonance', 0) > 0.6:
            return TrustType.EMPATHIC_FLOW
        
        elif interaction.get('will_alignment', 0) > 0.7:
            return TrustType.WILL_COHERENCE
        
        elif interaction.get('consciousness_sync', 0) > 0.6:
            return TrustType.AWARENESS_SYNCH
        
        elif interaction.get('moral_resonance', 0) > 0.5:
            return TrustType.ETHICAL_RESONANCE
        
        elif interaction.get('cognitive_alignment', 0) > 0.6:
            return TrustType.COGNITIVE_ALIGNMENT
        
        else:
            return TrustType.SYMBIOTIC_BOND
    
    def _calculate_trust_intensity(self, interaction: Dict, trust_type: TrustType) -> float:
        """Расчет интенсивности доверия"""
        # Базовые факторы
        factors = {
            'duration': min(1.0, interaction.get('duration_seconds', 0) / 7200),  # 2 часа максимум
            'depth': interaction.get('interaction_depth', 0.5),
            'reciprocity': interaction.get('reciprocity_score', 0.5),
            'emotional_charge': interaction.get('emotional_charge', 0.5),
            'sephirotic_alignment': interaction.get('sephirotic_alignment', 0.5)
        }
        
        # Веса в зависимости от типа доверия
        weights = {
            TrustType.EMPATHIC_FLOW: {'duration': 0.2, 'depth': 0.4, 'reciprocity': 0.2, 
                                      'emotional_charge': 0.2, 'sephirotic_alignment': 0.0},
            Trust
