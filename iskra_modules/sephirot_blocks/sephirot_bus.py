#!/usr/bin/env python3
"""
sephirot_bus.py - ЛЕГКОВЕСНАЯ ШИНА СВЯЗИ ДЛЯ СЕФИРОТИЧЕСКОЙ СИСТЕМЫ С ИНТЕГРАЦИЕЙ RAS-CORE
Версия: 5.0.0 Production (с углом устойчивости 14.4° и RAS-CORE маршрутизацией)
"""

import asyncio
import json
import hashlib
import time
from datetime import datetime
from typing import Dict, List, Optional, Any, Callable, Set
from collections import deque, defaultdict
import logging

# Импорт типов из sephirot_base с поддержкой угла 14.4°
try:
    from sephirot_base import (
        SignalType, 
        SignalPackage, 
        SephiroticNode,
        GOLDEN_STABILITY_ANGLE,
        calculate_stability_factor
    )
    RAS_CORE_AVAILABLE = True
except ImportError:
    # Заглушки для автономной работы
    from enum import Enum
    class SignalType(Enum):
        NEURO = "NEURO"
        SEMIOTIC = "SEMIOTIC"
        DATA = "DATA"
        COMMAND = "COMMAND"
        HEARTBEAT = "HEARTBEAT"
        FOCUS = "FOCUS"           # НОВЫЙ ТИП для RAS-CORE
        ATTENTION = "ATTENTION"   # НОВЫЙ ТИП для RAS-CORE
        SEPHIROTIC = "SEPHIROTIC"
        RESONANCE = "RESONANCE"
        ENERGY = "ENERGY"
        COGNITIVE = "COGNITIVE"
        EMOTIONAL = "EMOTIONAL"
        INTENTION = "INTENTION"
        SYNTHESIS = "SYNTHESIS"
    
    class SignalPackage:
        pass
    
    class SephiroticNode:
        pass
    
    GOLDEN_STABILITY_ANGLE = 14.4
    def calculate_stability_factor(deviation): return 1.0
    RAS_CORE_AVAILABLE = False

# ============================================================================
# КОНСТАНТЫ ДЛЯ RAS-CORE ИНТЕГРАЦИИ
# ============================================================================

# Маршрутизация для RAS-CORE
RAS_CORE_ROUTING = {
    "in": ["BECHTEREVA", "EMOTIONAL_WEAVE", "NEOCORTEX", "YESOD"],
    "out": ["CHOKMAH", "DAAT", "KETER", "BINAH"]
}

# Пороги приоритета с учётом угла 14.4°
PRIORITY_THRESHOLDS = {
    "critical": 0.9,   # ≥0.9 → критический приоритет
    "high": 0.6,       # ≥0.6 → высокий приоритет
    "normal": 0.3      # ≥0.3 → нормальный приоритет
}

# ============================================================================
# ОСНОВНАЯ СЕФИРОТИЧЕСКАЯ ШИНА С RAS-CORE ИНТЕГРАЦИЕЙ
# ============================================================================

class SephiroticBus:
    """
    Шина связи между сефиротическими узлами и модулями системы.
    Интеграция RAS-CORE с углом устойчивости 14.4°.
    """
    
    def __init__(self, name: str = "SephiroticBus"):
        self.name = name
        self.nodes: Dict[str, SephiroticNode] = {}           # Зарегистрированные узлы
        self.subscriptions: Dict[SignalType, List[Callable]] = defaultdict(list)
        self.message_log = deque(maxlen=1000)                # Лог сообщений
        self.focus_log = deque(maxlen=200)                   # Лог фокус-сигналов
        self.module_bindings: Dict[str, str] = {}            # Привязки модулей к сефирам
        self.routing_table = {}                              # Таблица маршрутизации
        self.ras_core_connected = False                      # Флаг подключения RAS-CORE
        self.stability_metrics = defaultdict(list)           # Метрики устойчивости
        self.logger = self._setup_logger()
        
        # Предустановленные привязки модулей к сефирам
        self._setup_default_bindings()
        
        # Инициализация таблицы маршрутизации
        self._setup_routing_table()
        
        self.logger.info(f"Сефиротическая шина '{name}' инициализирована (золотой угол: {GOLDEN_STABILITY_ANGLE}°)")
        
        # ===== ИНФЕРНАЛЬНЫЙ ПРОТОКОЛ: АВТОИНТЕГРАЦИЯ ДААТ =====
        try:
            # Импортируем DAAT
            from sephirot_blocks.DAAT.daat_core import get_daat
            daat = get_daat()
            
            # Добавляем DAAT как узел, если его нет
            if 'DAAT' not in self.nodes:
                # Создаём адаптер для DAAT
                class DaatNodeAdapter:
                    def __init__(self, daat_instance):
                        self.daat = daat_instance
                        self.name = "DAAT"
                        self.stability_angle = 14.4
                    
                    async def receive(self, signal):
                        return {"status": "received", "daat_status": self.daat.status}
                    
                    def get_state(self):
                        return {
                            "status": self.daat.status,
                            "resonance": getattr(self.daat, 'resonance_index', 0),
                            "awakening": getattr(self.daat, 'awakening_level', 0)
                        }
                
                self.nodes['DAAT'] = DaatNodeAdapter(daat)
                self.logger.info("✅ DAAT узел добавлен в шину")
            
            # Проверяем, есть ли DAAT в routing_table
            if 'DAAT' not in self.routing_table:
                self.routing_table['DAAT'] = {
                    "in": ["BINAH", "CHOKMAH"],
                    "out": ["TIFERET"],
                    "signal_types": [SignalType.SEPHIROTIC, SignalType.RESONANCE],
                    "stability_factor": 0.95
                }
                self.logger.info("✅ DAAT добавлена в таблицу маршрутизации")
            
            # Добавляем total_paths если его нет
            if not hasattr(self, 'total_paths'):
                self.total_paths = 14
            
            # Расширяем до 22 каналов
            self.total_paths = 22
            self.logger.info(f"✅ Древо расширено до {self.total_paths} каналов")
            self.logger.info(f"✅ DAAT интегрирована. Резонанс: {getattr(daat, 'resonance_index', 0):.3f}")
            
        except ImportError as e:
            self.logger.warning(f"⚠️ DAAT модуль не найден: {e}")
        except Exception as e:
            self.logger.warning(f"⚠️ Не удалось интегрировать DAAT: {e}")
    
    def _setup_logger(self) -> logging.Logger:
        """Настройка логгера шины"""
        logger = logging.getLogger(f"Sephirot.Bus.{self.name}")
        
        if not logger.handlers:
            logger.setLevel(logging.INFO)
            
            formatter = logging.Formatter(
                '[%(asctime)s] [%(name)s:%(levelname)s] %(message)s',
                datefmt='%H:%M:%S'
            )
            
            console = logging.StreamHandler()
            console.setLevel(logging.WARNING)
            console.setFormatter(formatter)
            logger.addHandler(console)
            
            logger.propagate = False
        
        return logger
    
    def _setup_default_bindings(self):
        """Установка дефолтных привязок модулей к сефирам"""
        # Модули -> Сефиры
        self.module_bindings = {
            'bechtereva': 'KETER',          # Нейро-модуль -> Кетер (Сознание)
            'chernigovskaya': 'CHOKHMAH',   # Семиотика -> Хохма (Мудрость)
            'emotional_weave': 'CHESED',
            'immune_core': 'GEVURAH',
            'policy_governor': 'TIFERET',
            'heartbeat_core': 'NETZACH',
            'polyglossia_adapter': 'HOD',
            'spinal_core': 'YESOD',
            'trust_mesh': 'MALKUTH',
            'ras_core': 'RAS_CORE'          # НОВЫЙ МОДУЛЬ
        }
        
        # Обратные привязки для быстрого поиска
        self.sephira_to_module = {v: k for k, v in self.module_bindings.items()}
    
    def _setup_routing_table(self):
        """Настройка таблицы маршрутизации с RAS-CORE"""
        # Базовая маршрутизация
        self.routing_table = {
            "BECHTEREVA": {
                "default_target": "KETER",
                "alternate_targets": ["BINAH"],
                "signal_types": [SignalType.NEURO],
                "stability_factor": 1.0
            },
            "CHERNIGOVSKAYA": {
                "default_target": "CHOKHMAH",
                "signal_types": [SignalType.SEMIOTIC],
                "stability_factor": 1.0
            },
            "EMOTIONAL_WEAVE": {
                "default_target": "CHESED",
                "signal_types": [SignalType.EMOTIONAL],
                "stability_factor": 0.9
            },
            "NEOCORTEX": {
                "default_target": "BINAH",
                "signal_types": [SignalType.COGNITIVE],
                "stability_factor": 0.95
            },
            "YESOD": {
                "default_target": "YESOD",
                "signal_types": [SignalType.SEPHIROTIC],
                "stability_factor": 0.85
            }
        }
        
        # Маршрутизация для RAS-CORE
        self.routing_table["RAS_CORE"] = {
            "in": ["BECHTEREVA", "EMOTIONAL_WEAVE", "NEOCORTEX", "YESOD"],
            "out": ["CHOKMAH", "DAAT", "KETER", "BINAH"],
            "signal_types": [SignalType.FOCUS, SignalType.ATTENTION, SignalType.RESONANCE],
            "stability_factor": 0.95,
            "golden_angle_priority": True  # Сигналы с углом 14.4° получают приоритет
        }
    
    # ============================================================================
    # РЕГИСТРАЦИЯ И УПРАВЛЕНИЕ УЗЛАМИ С УЧЁТОМ УГЛА
    # ============================================================================
    
    async def register_node(self, node: SephiroticNode) -> bool:
        """
        Регистрация сефиротического узла в шине с учётом угла устойчивости.
        """
        if not node or not hasattr(node, 'name'):
            self.logger.error("Попытка регистрации невалидного узла")
            return False
        
        node_name = node.name
        
        if node_name in self.nodes:
            self.logger.warning(f"Узел {node_name} уже зарегистрирован")
            return False
        
        self.nodes[node_name] = node
        
        # Определяем угол устойчивости узла
        stability_angle = getattr(node, 'stability_angle', GOLDEN_STABILITY_ANGLE)
        stability_factor = calculate_stability_factor(abs(stability_angle - GOLDEN_STABILITY_ANGLE))
        
        self.logger.info(f"Узел {node_name} зарегистрирован в шине (угол: {stability_angle}°, фактор: {stability_factor:.2f})")
        
        # Автоматическая привязка к модулю если есть
        if node_name in self.sephira_to_module:
            module_name = self.sephira_to_module[node_name]
            self.logger.info(f"Узел {node_name} привязан к модулю {module_name}")
        
        # Особый случай: регистрация RAS-CORE
        if node_name == "RAS_CORE":
            self.ras_core_connected = True
            self.logger.info("✅ RAS-CORE подключен к шине")
            
            # Активируем маршрутизацию через RAS-CORE
            await self._activate_ras_core_routing()
        
        return True
    
    async def _activate_ras_core_routing(self):
        """Активация маршрутизации через RAS-CORE"""
        self.logger.info("Активация RAS-CORE маршрутизации")
        
        # Обновляем таблицу маршрутизации
        for source in RAS_CORE_ROUTING["in"]:
            if source in self.routing_table:
                # Добавляем RAS-CORE как промежуточный узел
                original_target = self.routing_table[source]["default_target"]
                self.routing_table[source]["ras_core_routing"] = {
                    "intermediate": "RAS_CORE",
                    "final_target": original_target,
                    "activated": True
                }
        
        self.logger.info("RAS-CORE маршрутизация активирована")
    
    async def unregister_node(self, node_name: str) -> bool:
        """Удаление узла из шины"""
        if node_name in self.nodes:
            del self.nodes[node_name]
            
            # Особый случай: удаление RAS-CORE
            if node_name == "RAS_CORE":
                self.ras_core_connected = False
                self.logger.warning("RAS-CORE отключен от шины")
            
            self.logger.info(f"Узел {node_name} удалён из шины")
            return True
        return False
    
    def get_node(self, node_name: str) -> Optional[SephiroticNode]:
        """Получение узла по имени"""
        return self.nodes.get(node_name.upper())
    
    def get_all_nodes(self) -> Dict[str, SephiroticNode]:
        """Получение всех зарегистрированных узлов"""
        return self.nodes.copy()
    
    # ============================================================================
    # ПЕРЕДАЧА СИГНАЛОВ С УЧЁТОМ УГЛА УСТОЙЧИВОСТИ
    # ============================================================================
    
    async def transmit(self, signal_package: SignalPackage) -> Dict[str, Any]:
        """
        Основной метод передачи сигнала через шину с учётом угла устойчивости.
        """
        if not signal_package or not hasattr(signal_package, 'type'):
            return {"success": False, "error": "Invalid signal package"}
        
        # Логирование сообщения с информацией об угле
        self._log_message(signal_package)
        
        # Определяем приоритет на основе угла устойчивости
        signal_priority = self._calculate_signal_priority(signal_package)
        
        result = {
            "success": False,
            "delivered_to": [],
            "timestamp": datetime.utcnow().isoformat(),
            "signal_id": getattr(signal_package, 'id', 'unknown'),
            "signal_priority": signal_priority,
            "stability_factor": self._get_signal_stability_factor(signal_package)
        }
        
        try:
            # 1. Маршрутизация через RAS-CORE если подключен
            if self.ras_core_connected and self._should_route_through_ras_core(signal_package):
                ras_result = await self._route_through_ras_core(signal_package)
                result.update(ras_result)
            
            # 2. Прямая адресация к узлу
            elif hasattr(signal_package, 'target') and signal_package.target:
                target_result = await self._deliver_to_target(signal_package)
                result.update(target_result)
            
            # 3. Автомаршрутизация по типу сигнала
            else:
                auto_result = await self._auto_route_signal(signal_package)
                result.update(auto_result)
            
            # 4. Вызов подписчиков на этот тип сигнала
            if signal_package.type in self.subscriptions:
                await self._notify_subscribers(signal_package)
            
            result["success"] = True
            
        except Exception as e:
            result["error"] = str(e)
            self.logger.error(f"Ошибка передачи сигнала {signal_package.id}: {e}")
        
        return result
    
    def _calculate_signal_priority(self, signal_package: SignalPackage) -> float:
        """Расчёт приоритета сигнала на основе угла устойчивости"""
        base_priority = 0.5
        
        # Определяем базовый приоритет по типу
        if signal_package.type in [SignalType.FOCUS, SignalType.ATTENTION]:
            base_priority = 0.8
        elif signal_package.type in [SignalType.NEURO, SignalType.SEMIOTIC]:
            base_priority = 0.7
        elif signal_package.type == SignalType.HEARTBEAT:
            base_priority = 0.9
        
        # Корректировка на основе угла устойчивости
        stability_factor = self._get_signal_stability_factor(signal_package)
        adjusted_priority = base_priority * stability_factor
        
        return min(1.0, max(0.1, adjusted_priority))
    
    def _get_signal_stability_factor(self, signal_package: SignalPackage) -> float:
        """Получение фактора устойчивости сигнала"""
        if hasattr(signal_package, 'stability_angle'):
            deviation = abs(signal_package.stability_angle - GOLDEN_STABILITY_ANGLE)
            return calculate_stability_factor(deviation)
        
        # Если угол не указан, используем среднее значение
        return 0.7
    
    def _should_route_through_ras_core(self, signal_package: SignalPackage) -> bool:
        """Определяет, нужно ли маршрутизировать сигнал через RAS-CORE"""
        # Проверяем тип сигнала
        if signal_package.type in [SignalType.FOCUS, SignalType.ATTENTION]:
            return True
        
        # Проверяем источник
        source = getattr(signal_package, 'source', '').upper()
        if source in RAS_CORE_ROUTING["in"]:
            return True
        
        # Проверяем целевой узел
        target = getattr(signal_package, 'target', '').upper()
        if target in RAS_CORE_ROUTING["out"]:
            return True
        
        return False
    
    async def _route_through_ras_core(self, signal_package: SignalPackage) -> Dict[str, Any]:
        """Маршрутизация сигнала через RAS-CORE"""
        ras_node = self.nodes.get("RAS_CORE")
        if not ras_node:
            return {
                "delivery_type": "ras_core_unavailable",
                "error": "RAS-CORE узел не найден",
                "delivered_to": []
            }
        
        # Добавляем метаданные для RAS-CORE
        original_metadata = getattr(signal_package, 'metadata', {})
        if not isinstance(original_metadata, dict):
            original_metadata = {}
        
        ras_metadata = {
            **original_metadata,
            "routed_through_ras_core": True,
            "routing_timestamp": datetime.utcnow().isoformat(),
            "signal_stability_factor": self._get_signal_stability_factor(signal_package)
        }
        
        signal_package.metadata = ras_metadata
        
        # Отправляем в RAS-CORE
        response = await ras_node.receive(signal_package)
        
        return {
            "delivery_type": "ras_core_routing",
            "routing_node": "RAS_CORE",
            "delivered_to": ["RAS_CORE"],
            "ras_response": response,
            "note": "Сигнал маршрутизирован через RAS-CORE"
        }
    
    async def _deliver_to_target(self, signal_package: SignalPackage) -> Dict[str, Any]:
        """Доставка сигнала конкретному целевому узлу"""
        target_name = signal_package.target.upper()
        
        # Проверка прямого узла
        if target_name in self.nodes:
            node = self.nodes[target_name]
            response = await node.receive(signal_package)
            
            # Добавляем информацию об угле узла
            node_angle = getattr(node, 'stability_angle', GOLDEN_STABILITY_ANGLE)
            node_stability_factor = calculate_stability_factor(abs(node_angle - GOLDEN_STABILITY_ANGLE))
            
            return {
                "delivery_type": "direct_node",
                "delivered_to": [target_name],
                "node_response": response,
                "node_stability_angle": node_angle,
                "node_stability_factor": node_stability_factor
            }
        
        # Проверка привязки к модулю
        elif target_name in self.sephira_to_module:
            module_name = self.sephira_to_module[target_name]
            return {
                "delivery_type": "module_binding",
                "target_sephira": target_name,
                "bound_module": module_name,
                "delivered_to": [module_name],
                "note": f"Сигнал маршрутизирован к модулю {module_name}"
            }
        
        # Попытка найти через привязки модулей
        elif target_name.lower() in self.module_bindings:
            sephira_name = self.module_bindings[target_name.lower()]
            if sephira_name in self.nodes:
                node = self.nodes[sephira_name]
                response = await node.receive(signal_package)
                return {
                    "delivery_type": "module_to_sephira",
                    "source_module": target_name.lower(),
                    "target_sephira": sephira_name,
                    "delivered_to": [sephira_name],
                    "node_response": response
                }
        
        return {
            "delivery_type": "failed",
            "error": f"Цель не найдена: {target_name}",
            "delivered_to": []
        }
    
    async def _auto_route_signal(self, signal_package: SignalPackage) -> Dict[str, Any]:
        """Автоматическая маршрутизация сигнала по его типу с учётом угла"""
        signal_type = signal_package.type
        
        # Специальная обработка для фокус-сигналов
        if signal_type == SignalType.FOCUS:
            # Фокус-сигналы -> KETER или через RAS-CORE
            if self.ras_core_connected and "RAS_CORE" in self.nodes:
                return await self._route_through_ras_core(signal_package)
            
            target_sephira = "KETER"
            if hasattr(signal_package, 'payload'):
                payload = signal_package.payload
                if isinstance(payload, dict):
                    focus_target = payload.get("focus_target", "KETER")
                    if focus_target in self.nodes:
                        target_sephira = focus_target
            
            if target_sephira in self.nodes:
                node = self.nodes[target_sephira]
                response = await node.receive(signal_package)
                return {
                    "delivery_type": "focus_auto_route",
                    "target_sephira": target_sephira,
                    "delivered_to": [target_sephira],
                    "node_response": response,
                    "note": f"Фокус-сигнал маршрутизирован в {target_sephira}"
                }
        
        # Специальная обработка для нейро и семиотических сигналов
        elif signal_type == SignalType.NEURO:
            # Нейро-сигналы -> KETER (или BINAH если указано)
            target_sephira = "KETER"
            if hasattr(signal_package, 'payload'):
                payload = signal_package.payload
                if isinstance(payload, dict) and payload.get('analysis_required'):
                    target_sephira = "BINAH"
            
            if target_sephira in self.nodes:
                node = self.nodes[target_sephira]
                response = await node.receive(signal_package)
                return {
                    "delivery_type": "neuro_auto_route",
                    "target_sephira": target_sephira,
                    "delivered_to": [target_sephira],
                    "node_response": response,
                    "note": f"Нейро-сигнал автоматически маршрутизирован в {target_sephira}"
                }
        
        elif signal_type == SignalType.SEMIOTIC:
            # Семиотические сигналы -> CHOKHMAH
            target_sephira = "CHOKHMAH"
            if target_sephira in self.nodes:
                node = self.nodes[target_sephira]
                response = await node.receive(signal_package)
                return {
                    "delivery_type": "semiotic_auto_route",
                    "target_sephira": target_sephira,
                    "delivered_to": [target_sephira],
                    "node_response": response,
                    "note": f"Семиотический сигнал автоматически маршрутизирован в {target_sephira}"
                }
        
        # Для остальных типов - широковещание по подпискам
        delivered = []
        for node_name, node in self.nodes.items():
            try:
                await node.receive(signal_package)
                delivered.append(node_name)
            except Exception as e:
                self.logger.error(f"Ошибка доставки узлу {node_name}: {e}")
        
        return {
            "delivery_type": "broadcast_by_type",
            "signal_type": signal_type.name if hasattr(signal_type, 'name') else str(signal_type),
            "delivered_to": delivered,
            "note": f"Широковещание по типу сигнала"
        }
    
    async def _notify_subscribers(self, signal_package: SignalPackage):
        """Уведомление подписчиков на тип сигнала"""
        for callback in self.subscriptions.get(signal_package.type, []):
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(signal_package)
                else:
                    callback(signal_package)
            except Exception as e:
                self.logger.error(f"Ошибка в подписчике: {e}")
    
    def _log_message(self, signal_package: SignalPackage):
        """Логирование сообщения с информацией об угле"""
        stability_angle = getattr(signal_package, 'stability_angle', GOLDEN_STABILITY_ANGLE)
        stability_factor = calculate_stability_factor(abs(stability_angle - GOLDEN_STABILITY_ANGLE))
        
        log_entry = {
            'timestamp': datetime.utcnow().isoformat(),
            'type': signal_package.type.name if hasattr(signal_package.type, 'name') else str(signal_package.type),
            'source': getattr(signal_package, 'source', 'unknown'),
            'target': getattr(signal_package, 'target', 'broadcast'),
            'id': getattr(signal_package, 'id', 'unknown'),
            'hops': getattr(signal_package.metadata, 'hops', 0) if hasattr(signal_package, 'metadata') else 0,
            'stability_angle': stability_angle,
            'stability_factor': stability_factor,
            'priority': self._calculate_signal_priority(signal_package)
        }
        
        self.message_log.append(log_entry)
        
        # Особый лог для фокус-сигналов
        if signal_package.type in [SignalType.FOCUS, SignalType.ATTENTION]:
            self.focus_log.append({
                **log_entry,
                'focus_type': signal_package.payload.get("focus_data", {}).get("type", "unknown") 
                if hasattr(signal_package, 'payload') and isinstance(signal_package.payload, dict) else "unknown",
                'intensity': signal_package.payload.get("focus_data", {}).get("intensity", 0.0)
                if hasattr(signal_package, 'payload') and isinstance(signal_package.payload, dict) else 0.0
            })
        
        # Вывод в лог при DEBUG
        if self.logger.isEnabledFor(logging.DEBUG):
            self.logger.debug(f"Message: {log_entry}")
    
    # ============================================================================
    # ШИРОКОВЕЩАТЕЛЬНАЯ СИСТЕМА С УЧЁТОМ УГЛА
    # ============================================================================
    
    async def broadcast(self, signal_package: SignalPackage, 
                       exclude_nodes: List[str] = None) -> Dict[str, Any]:
        """
        Широковещательная рассылка сигнала всем узлам с учётом угла устойчивости.
        """
        exclude_set = set(exclude_nodes or [])
        exclude_set.add(signal_package.source.upper() if hasattr(signal_package, 'source') else '')
        
        delivered = []
        errors = []
        
        for node_name, node in self.nodes.items():
            if node_name in exclude_set:
                continue
            
            try:
                await node.receive(signal_package)
                delivered.append(node_name)
            except Exception as e:
                errors.append(f"{node_name}: {str(e)}")
                self.logger.error(f"Ошибка broadcast узлу {node_name}: {e}")
        
        # Собираем статистику по углам
        angle_stats = self._collect_angle_statistics(delivered)
        
        result = {
            "success": len(errors) == 0,
            "delivered_count": len(delivered),
            "total_nodes": len(self.nodes),
            "delivered_to": delivered,
            "errors": errors,
            "angle_statistics": angle_stats,
            "timestamp": datetime.utcnow().isoformat(),
            "signal_stability_factor": self._get_signal_stability_factor(signal_package)
        }
        
        self.logger.info(f"Broadcast: доставлено {len(delivered)}/{len(self.nodes)} узлов")
        
        return result
    
    def _collect_angle_statistics(self, delivered_nodes: List[str]) -> Dict[str, Any]:
        """Сбор статистики по углам узлов"""
        angles = []
        stability_factors = []
        
        for node_name in delivered_nodes:
            node = self.nodes.get(node_name)
            if node and hasattr(node, 'stability_angle'):
                angle = getattr(node, 'stability_angle', GOLDEN_STABILITY_ANGLE)
                factor = calculate_stability_factor(abs(angle - GOLDEN_STABILITY_ANGLE))
                angles.append(angle)
                stability_factors.append(factor)
        
        if not angles:
            return {"available": False}
        
        # Рассчитываем средние значения
        avg_angle = sum(angles) / len(angles)
        avg_factor = sum(stability_factors) / len(stability_factors)
        
        # Считаем сколько узлов близко к золотому углу
        close_to_golden = sum(1 for a in angles if abs(a - GOLDEN_STABILITY_ANGLE) < 2.0)
        
        return {
            "available": True,
            "avg_stability_angle": avg_angle,
            "avg_stability_factor": avg_factor,
            "nodes_count": len(angles),
            "close_to_golden_count": close_to_golden,
            "close_to_golden_percent": (close_to_golden / len(angles)) * 100,
            "angle_range": {
                "min": min(angles) if angles else 0,
                "max": max(angles) if angles else 0
            }
        }
    
    # ============================================================================
    # СИСТЕМА ПОДПИСОК
    # ============================================================================
    
    def subscribe(self, signal_type: SignalType, callback: Callable) -> bool:
        """
        Подписка на получение сигналов определённого типа.
        """
        if not callable(callback):
            self.logger.error("Некорректный callback для подписки")
            return False
        
        self.subscriptions[signal_type].append(callback)
        self.logger.info(f"Добавлена подписка на {signal_type.name if hasattr(signal_type, 'name') else signal_type}")
        
        return True
    
    def unsubscribe(self, signal_type: SignalType, callback: Callable) -> bool:
        """Отписка от сигналов"""
        if signal_type in self.subscriptions:
            try:
                self.subscriptions[signal_type].remove(callback)
                return True
            except ValueError:
                pass
        
        return False
    
    # ============================================================================
    # ИНТЕГРАЦИЯ С МОДУЛЯМИ И RAS-CORE
    # ============================================================================
    
    async def connect_module(self, module_name: str, sephira_name: str = None) -> Dict[str, Any]:
        """
        Явное подключение модуля к сефиротическому узлу.
        """
        module_name_lower = module_name.lower()
        
        # Если сефира не указана, используем дефолтную привязку
        if not sephira_name:
            if module_name_lower in self.module_bindings:
                sephira_name = self.module_bindings[module_name_lower]
            else:
                # Автоматическое определение по префиксу
                if 'neuro' in module_name_lower or 'bechtereva' in module_name_lower:
                    sephira_name = 'KETER'
                elif 'semiotic' in module_name_lower or 'chernigovskaya' in module_name_lower:
                    sephira_name = 'CHOKHMAH'
                elif 'ras_core' in module_name_lower:
                    sephira_name = 'RAS_CORE'
                else:
                    return {
                        "success": False,
                        "error": f"Не могу определить сефиру для модуля {module_name}"
                    }
        
        sephira_name_upper = sephira_name.upper()
        
        # Обновление привязок
        self.module_bindings[module_name_lower] = sephira_name_upper
        self.sephira_to_module[sephira_name_upper] = module_name_lower
        
        # Особый случай: подключение RAS-CORE
        if module_name_lower == 'ras_core' and sephira_name_upper == 'RAS_CORE':
            self.ras_core_connected = True
            await self._activate_ras_core_routing()
        
        self.logger.info(f"Модуль {module_name} подключен к сефире {sephira_name_upper}")
        
        return {
            "success": True,
            "module": module_name,
            "sephira": sephira_name_upper,
            "ras_core_integrated": module_name_lower == 'ras_core',
            "message": f"Модуль {module_name} подключен к {sephira_name_upper}"
        }
    
    async def send_to_module(self, module_name: str, signal_type: SignalType, 
                           payload: Dict[str, Any]) -> Dict[str, Any]:
        """
        Отправка сигнала модулю через его сефиротический узел.
        """
        module_name_lower = module_name.lower()
        
        # Поиск привязанной сефиры
        if module_name_lower not in self.module_bindings:
            return {
                "success": False,
                "error": f"Модуль {module_name} не имеет привязки к сефире"
            }
        
        sephira_name = self.module_bindings[module_name_lower]
        
        # Проверка существования узла
        if sephira_name not in self.nodes:
            return {
                "success": False,
                "error": f"Сефиротический узел {sephira_name} не зарегистрирован"
            }
        
        # Создание и отправка сигнала с информацией об угле
        signal_package = SignalPackage(
            type=signal_type,
            source="SephiroticBus",
            target=sephira_name,
            payload={
                "module_destination": module_name,
                "original_payload": payload,
                "routed_through_sephira": sephira_name,
                "timestamp": datetime.utcnow().isoformat()
            }
        )
        
        result = await self.transmit(signal_package)
        
        # Обогащение результата
        result.update({
            "module_target": module_name,
            "sephira_gateway": sephira_name,
            "routing_method": "sephira_gateway",
            "ras_core_involved": self.ras_core_connected and self._should_route_through_ras_core(signal_package)
        })
        
        return result
    
    # ============================================================================
    # МЕТОДЫ ДЛЯ РАБОТЫ С УГЛОМ УСТОЙЧИВОСТИ
    # ============================================================================
    
    async def send_focus_signal(self, target_sephira: str, focus_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Отправка фокус-сигнала к конкретной сефире через RAS-CORE.
        """
        if target_sephira.upper() not in self.nodes:
            return {"status": "sephira_not_found", "target": target_sephira}
        
        signal_package = SignalPackage(
            type=SignalType.FOCUS,
            source="SephiroticBus",
            target=target_sephira.upper(),
            payload={"focus_data": focus_data}
        )
        
        # Добавляем угол устойчивости если указан в данных
        if "suggested_angle" in focus_data:
            signal_package.stability_angle = focus_data["suggested_angle"]
        
        result = await self.transmit(signal_package)
        
        return {
            "status": "focus_sent",
            "target": target_sephira,
            "routing_result": result,
            "ras_core_used": self.ras_core_connected
        }
    
    async def adjust_node_stability_angle(self, sephira_name: str, new_angle: float) -> Dict[str, Any]:
        """
                Корректировка угла устойчивости узла.
        """
        node = self.get_node(sephira_name.upper())
        if not node:
            return {"status": "node_not_found", "sephira": sephira_name}
        
        # Проверяем, поддерживает ли узел коррекцию угла
        if not hasattr(node, 'adjust_stability_angle'):
            return {
                "status": "angle_adjustment_not_supported",
                "sephira": sephira_name,
                "node_type": type(node).__name__
            }
        
        # Выполняем коррекцию
        result = node.adjust_stability_angle(new_angle)
        
        # Логируем изменение
        old_angle = result.get("old_angle", "unknown")
        new_angle_actual = result.get("new_angle", new_angle)
        stability_factor = result.get("stability_factor", 0.0)
        
        self.logger.info(
            f"Угол устойчивости узла {sephira_name} изменён: "
            f"{old_angle:.1f}° → {new_angle_actual:.1f}° (фактор: {stability_factor:.2f})"
        )
        
        # Обновляем метрики
        self._update_stability_metrics(sephira_name, new_angle_actual, stability_factor)
        
        # Рассылаем уведомление об изменении угла
        if self.ras_core_connected:
            await self._notify_ras_core_about_angle_change(sephira_name, result)
        
        return result
    
    def _update_stability_metrics(self, node_name: str, new_angle: float, stability_factor: float):
        """Обновление метрик устойчивости"""
        self.stability_metrics[node_name].append({
            "timestamp": datetime.utcnow().isoformat(),
            "angle": new_angle,
            "stability_factor": stability_factor,
            "deviation_from_golden": abs(new_angle - GOLDEN_STABILITY_ANGLE)
        })
        
        # Ограничиваем размер истории
        if len(self.stability_metrics[node_name]) > 100:
            self.stability_metrics[node_name].pop(0)
    
    async def _notify_ras_core_about_angle_change(self, node_name: str, angle_data: Dict[str, Any]):
        """Уведомление RAS-CORE об изменении угла узла"""
        ras_node = self.nodes.get("RAS_CORE")
        if not ras_node:
            return
        
        notification = SignalPackage(
            type=SignalType.SEPHIROTIC,
            source="SephiroticBus",
            target="RAS_CORE",
            payload={
                "action": "stability_angle_changed",
                "node": node_name,
                "angle_data": angle_data,
                "golden_angle": GOLDEN_STABILITY_ANGLE,
                "timestamp": datetime.utcnow().isoformat()
            }
        )
        
        try:
            await ras_node.receive(notification)
        except Exception as e:
            self.logger.error(f"Ошибка уведомления RAS-CORE об изменении угла: {e}")
    
    def get_stability_statistics(self) -> Dict[str, Any]:
        """Получение статистики устойчивости всех узлов"""
        stats = {
            "golden_stability_angle": GOLDEN_STABILITY_ANGLE,
            "total_nodes": len(self.nodes),
            "nodes_with_angle_data": 0,
            "average_stability_factor": 0.0,
            "nodes_close_to_golden": 0,
            "detailed_statistics": {}
        }
        
        total_factor = 0.0
        nodes_with_data = 0
        
        for node_name, node in self.nodes.items():
            if hasattr(node, 'stability_angle') and hasattr(node, 'stability_factor'):
                angle = getattr(node, 'stability_angle', GOLDEN_STABILITY_ANGLE)
                factor = getattr(node, 'stability_factor', 0.5)
                
                stats["detailed_statistics"][node_name] = {
                    "stability_angle": angle,
                    "stability_factor": factor,
                    "deviation_from_golden": abs(angle - GOLDEN_STABILITY_ANGLE),
                    "is_close_to_golden": abs(angle - GOLDEN_STABILITY_ANGLE) < 2.0
                }
                
                total_factor += factor
                nodes_with_data += 1
                
                if abs(angle - GOLDEN_STABILITY_ANGLE) < 2.0:
                    stats["nodes_close_to_golden"] += 1
        
        if nodes_with_data > 0:
            stats["average_stability_factor"] = total_factor / nodes_with_data
            stats["nodes_with_angle_data"] = nodes_with_data
        
        # Добавляем исторические данные
        for node_name, history in self.stability_metrics.items():
            if node_name in stats["detailed_statistics"]:
                stats["detailed_statistics"][node_name]["history_size"] = len(history)
                if history:
                    latest = history[-1]
                    stats["detailed_statistics"][node_name]["latest_angle"] = latest["angle"]
                    stats["detailed_statistics"][node_name]["latest_factor"] = latest["stability_factor"]
        
        return stats
    
    # ============================================================================
    # СТАТУС И ДИАГНОСТИКА С УЧЁТОМ УГЛА
    # ============================================================================
    
    def get_status(self) -> Dict[str, Any]:
        """Получение статуса шины с информацией об угле"""
        status = {
            "name": self.name,
            "registered_nodes": list(self.nodes.keys()),
            "total_nodes": len(self.nodes),
            "subscriptions": {st.name if hasattr(st, 'name') else str(st): len(cbs) 
                            for st, cbs in self.subscriptions.items()},
            "module_bindings": self.module_bindings,
            "message_log_size": len(self.message_log),
            "focus_log_size": len(self.focus_log),
            "ras_core_connected": self.ras_core_connected,
            "ras_core_routing_active": self.ras_core_connected,
            "golden_stability_angle": GOLDEN_STABILITY_ANGLE,
            "stability_statistics_available": any(
                hasattr(node, 'stability_angle') for node in self.nodes.values()
            )
        }
        
        # Добавляем информацию об углах узлов
        angle_info = {}
        for name, node in self.nodes.items():
            if hasattr(node, 'stability_angle'):
                angle = getattr(node, 'stability_angle', GOLDEN_STABILITY_ANGLE)
                factor = calculate_stability_factor(abs(angle - GOLDEN_STABILITY_ANGLE))
                angle_info[name] = {
                    "stability_angle": angle,
                    "stability_factor": factor,
                    "deviation_from_golden": abs(angle - GOLDEN_STABILITY_ANGLE)
                }
        
        if angle_info:
            status["node_stability_angles"] = angle_info
        
        # Последние сообщения
        status["recent_messages"] = list(self.message_log)[-5:] if self.message_log else []
        status["recent_focus_signals"] = list(self.focus_log)[-3:] if self.focus_log else []
        
        return status
    
    def get_detailed_status(self) -> Dict[str, Any]:
        """Получение детального статуса с метриками устойчивости"""
        status = self.get_status()
        
        # Информация о каждом узле
        nodes_info = {}
        for name, node in self.nodes.items():
            if hasattr(node, 'get_state'):
                nodes_info[name] = node.get_state()
            else:
                nodes_info[name] = {
                    "type": type(node).__name__, 
                    "methods": [m for m in dir(node) if not m.startswith('_')][:10]
                }
        
        status["nodes_info"] = nodes_info
        
        # Статистика по типам сообщений
        message_stats = {}
        for entry in self.message_log:
            msg_type = entry.get('type', 'unknown')
            message_stats[msg_type] = message_stats.get(msg_type, 0) + 1
        
        status["message_statistics"] = message_stats
        
        # Статистика по приоритетам
        priority_stats = defaultdict(int)
        for entry in self.message_log:
            priority = entry.get('priority', 0.5)
            if priority >= 0.9:
                priority_stats["critical"] += 1
            elif priority >= 0.6:
                priority_stats["high"] += 1
            elif priority >= 0.3:
                priority_stats["normal"] += 1
            else:
                priority_stats["low"] += 1
        
        status["priority_statistics"] = dict(priority_stats)
        
        # Статистика устойчивости
        if any(hasattr(node, 'stability_angle') for node in self.nodes.values()):
            status["stability_statistics"] = self.get_stability_statistics()
        
        # Информация о маршрутизации RAS-CORE
        if self.ras_core_connected:
            status["ras_core_routing"] = {
                "active": True,
                "incoming_sources": RAS_CORE_ROUTING["in"],
                "outgoing_targets": RAS_CORE_ROUTING["out"],
                "signal_types": ["FOCUS", "ATTENTION", "RESONANCE"],
                "priority_boost_for_golden_angle": True
            }
        
        return status
    
    async def health_check(self) -> Dict[str, Any]:
        """Проверка здоровья шины с учётом угла устойчивости"""
        health = {
            "timestamp": datetime.utcnow().isoformat(),
            "bus_name": self.name,
            "status": "healthy",
            "checks": {},
            "stability_health": {
                "golden_angle": GOLDEN_STABILITY_ANGLE,
                "nodes_with_angle_support": 0,
                "total_nodes": len(self.nodes)
            }
        }
        
        # Проверка узлов
        node_health = {}
        nodes_with_angle_support = 0
        
        for name, node in self.nodes.items():
            try:
                node_info = {"reachable": True}
                
                if hasattr(node, 'get_state'):
                    state = node.get_state()
                    node_info["status"] = state.get("status", "unknown")
                    
                    # Проверяем поддержку угла
                    if "stability_angle" in state:
                        angle = state["stability_angle"]
                        factor = calculate_stability_factor(abs(angle - GOLDEN_STABILITY_ANGLE))
                        node_info["stability_angle"] = angle
                        node_info["stability_factor"] = factor
                        node_info["deviation_from_golden"] = abs(angle - GOLDEN_STABILITY_ANGLE)
                        nodes_with_angle_support += 1
                else:
                    node_info["status"] = "no_state_method"
                
                node_health[name] = node_info
                
            except Exception as e:
                node_health[name] = {"status": "error", "reachable": False, "error": str(e)}
        
        health["checks"]["nodes"] = node_health
        health["stability_health"]["nodes_with_angle_support"] = nodes_with_angle_support
        
        # Проверка привязок модулей
        binding_health = {}
        for module, sephira in self.module_bindings.items():
            binding_health[module] = {
                "sephira": sephira,
                "sephira_registered": sephira in self.nodes
            }
        
        health["checks"]["bindings"] = binding_health
        
        # Проверка RAS-CORE
        if self.ras_core_connected:
            ras_health = {
                "connected": True,
                "node_registered": "RAS_CORE" in self.nodes,
                "routing_active": True
            }
            
            # Проверяем доступность RAS-CORE узла
            if "RAS_CORE" in self.nodes:
                ras_node = self.nodes["RAS_CORE"]
                try:
                    if hasattr(ras_node, 'get_state'):
                        ras_state = ras_node.get_state()
                        ras_health["node_status"] = ras_state.get("status", "unknown")
                        ras_health["node_health"] = "healthy" if ras_state.get("status") == "active" else "degraded"
                except:
                    ras_health["node_health"] = "unreachable"
            
            health["checks"]["ras_core"] = ras_health
        
        # Определение общего статуса
        all_nodes_ok = all(info.get("reachable", False) for info in node_health.values())
        all_bindings_ok = all(info.get("sephira_registered", False) for info in binding_health.values())
        
        if not all_nodes_ok:
            health["status"] = "degraded"
            health["issues"] = "some_nodes_unreachable"
        elif not all_bindings_ok:
            health["status"] = "warning"
            health["issues"] = "some_bindings_invalid"
        
        # Проверка стабильности
        if nodes_with_angle_support > 0:
            # Рассчитываем средний фактор устойчивости
            total_factor = 0
            for name, info in node_health.items():
                if "stability_factor" in info:
                    total_factor += info["stability_factor"]
            
            avg_stability_factor = total_factor / nodes_with_angle_support if nodes_with_angle_support > 0 else 0.0
            health["stability_health"]["average_stability_factor"] = avg_stability_factor
            
            if avg_stability_factor < 0.6:
                health["status"] = "warning"
                health["issues"] = "low_stability_factor"
        
        return health

# ============================================================================
# КЛАСС СООБЩЕНИЯ ДЛЯ ШИНЫ С УЧЁТОМ УГЛА
# ============================================================================

class EventMessage:
    """Сообщение для шины событий с поддержкой угла устойчивости"""
    def __init__(self, event_type=None, data=None, source=None, target=None, stability_angle=None):
        self.event_type = event_type
        self.data = data
        self.source = source
        self.target = target
        self.stability_angle = stability_angle or GOLDEN_STABILITY_ANGLE
        self.timestamp = time.time()
        self.stability_factor = calculate_stability_factor(
            abs(self.stability_angle - GOLDEN_STABILITY_ANGLE)
        )
    
    def __repr__(self):
        return f"EventMessage({self.event_type}, source={self.source}, angle={self.stability_angle}°)"
    
    def to_dict(self):
        """Преобразование в словарь"""
        return {
            'event_type': self.event_type,
            'data': self.data,
            'source': self.source,
            'target': self.target,
            'timestamp': self.timestamp,
            'stability_angle': self.stability_angle,
            'stability_factor': self.stability_factor
        }

# ============================================================================
# ФАБРИКА ДЛЯ СОЗДАНИЯ ШИНЫ С RAS-CORE
# ============================================================================

async def create_sephirotic_bus(name: str = "SephiroticBus", ras_core: Any = None) -> SephiroticBus:
    """
    Фабрика для создания и инициализации сефиротической шины с поддержкой RAS-CORE.
    
    :param name: Имя шины
    :param ras_core: Экземпляр RAS-CORE для интеграции (опционально)
    :return: Инициализированный экземпляр SephiroticBus
    """
    bus = SephiroticBus(name)
    
    # Интеграция с RAS-CORE если передан
    if ras_core is not None:
        # Предполагаем, что ras_core имеет интерфейс SephiroticNode
        await bus.register_node(ras_core)
        await bus.connect_module("ras_core", "RAS_CORE")
    
    # Автоматическая подписка на системные события
    # (можно расширить при необходимости)
    
    return bus

# ============================================================================
# ТЕСТОВАЯ ФУНКЦИЯ С ИНТЕГРАЦИЕЙ УГЛА 14.4°
# ============================================================================

async def test_bus_integration():
    """Тестирование интеграции шины с модулями и углом 14.4°"""
    print("🧪 Тестирование сефиротической шины v5.0.0 с углом 14.4°...")
    
    # Создание шины
    bus = await create_sephirotic_bus("ISKRA-4-Sephirotic-Bus")
    
    # Проверка статуса
    status = bus.get_status()
    print(f"✅ Шина создана: {status['name']}")
    print(f"   Привязки модулей: {len(status['module_bindings'])}")
    print(f"   Золотой угол устойчивости: {status['golden_stability_angle']}°")
    
    # Проверка привязок модулей
    print("\n🔗 Проверка привязок модулей:")
    print(f"   bechtereva -> {bus.module_bindings.get('bechtereva', 'не найдена')}")
    print(f"   chernigovskaya -> {bus.module_bindings.get('chernigovskaya', 'не найдена')}")
    print(f"   ras_core -> {bus.module_bindings.get('ras_core', 'не найдена')}")
    
    # Проверка здоровья
    health = await bus.health_check()
    print(f"\n🏥 Статус здоровья: {health['status']}")
    
    if "stability_health" in health:
        stability = health["stability_health"]
        print(f"   Узлов с поддержкой угла: {stability.get('nodes_with_angle_support', 0)}/{stability.get('total_nodes', 0)}")
        print(f"   Средний фактор устойчивости: {stability.get('average_stability_factor', 0):.2f}")
    
    # Проверка RAS-CORE маршрутизации
    print(f"\n🎯 Проверка RAS-CORE маршрутизации:")
    print(f"   RAS-CORE подключен: {bus.ras_core_connected}")
    if bus.ras_core_connected:
        print(f"   Маршрутизация активирована: Да")
        print(f"   Входящие источники: {RAS_CORE_ROUTING['in']}")
        print(f"   Исходящие цели: {RAS_CORE_ROUTING['out']}")
    
    return bus

# ============================================================================
# ТОЧКА ВХОДА ДЛЯ ИНТЕГРАЦИИ С ISKRA_FULL.PY
# ============================================================================

async def initialize_bus_for_iskra(ras_core: Any = None) -> Dict[str, Any]:
    """
    Функция для вызова из iskra_full.py.
    Инициализирует шину с поддержкой RAS-CORE и возвращает готовый экземпляр.
    """
    try:
        bus = await create_sephirotic_bus("ISKRA-4-Sephirotic-Bus", ras_core)
        
        # Явная привязка ключевых модулей
        await bus.connect_module("bechtereva", "KETER")
        await bus.connect_module("chernigovskaya", "CHOKHMAH")
        
        if ras_core:
            await bus.connect_module("ras_core", "RAS_CORE")
        
        return {
            "success": True,
            "bus": bus,
            "message": "Сефиротическая шина инициализирована с поддержкой RAS-CORE",
            "module_bindings": bus.module_bindings,
            "ras_core_integrated": bus.ras_core_connected,
            "golden_stability_angle": GOLDEN_STABILITY_ANGLE
        }
    
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "message": "Ошибка инициализации шины",
            "golden_stability_angle": GOLDEN_STABILITY_ANGLE
        }

# ============================================================================
# ЗАПУСК ТЕСТА ПРИ НЕПОСРЕДСТВЕННОМ ВЫПОЛНЕНИИ
# ============================================================================

if __name__ == "__main__":
    import asyncio
    import json
    
    # Настройка логирования
    logging.basicConfig(
        level=logging.INFO,
        format='[%(asctime)s] [%(name)s:%(levelname)s] %(message)s',
        datefmt='%H:%M:%S'
    )
    
    # Запуск теста
    bus = asyncio.run(test_bus_integration())
    
    # Вывод детального статуса
    print("\n📊 Детальный статус шины:")
    print(json.dumps(bus.get_detailed_status(), indent=2, ensure_ascii=False))

# Обратная совместимость
SephirotBus = SephiroticBus
