#!/usr/bin/env python3
"""
sephirot_bus.py - ЛЕГКОВЕСНАЯ ШИНА СВЯЗИ ДЛЯ СЕФИРОТИЧЕСКОЙ СИСТЕМЫ
Интеграция: bechtereva -> KETER/BINAH, chernigovskaya -> CHOKHMAH
Версия: 4.0.0 Production
"""

import asyncio
import json
import hashlib
from datetime import datetime
from typing import Dict, List, Optional, Any, Callable
from collections import deque, defaultdict
import logging

# Импорт типов из sephirot_base
try:
    from sephirot_base import SignalType, SignalPackage, SephiroticNode
except ImportError:
    # Заглушки для автономной работы
    from enum import Enum
    class SignalType(Enum):
        NEURO = "NEURO"
        SEMIOTIC = "SEMIOTIC"
        DATA = "DATA"
        COMMAND = "COMMAND"
        HEARTBEAT = "HEARTBEAT"
    
    class SignalPackage:
        pass
    
    class SephiroticNode:
        pass

# ============================================================================
# ОСНОВНАЯ СЕФИРОТИЧЕСКАЯ ШИНА
# ============================================================================

class SephiroticBus:
    """
    Шина связи между сефиротическими узлами и модулями системы.
    Обеспечивает маршрутизацию сигналов bechtereva -> KETER, chernigovskaya -> CHOKHMAH.
    """
    
    def __init__(self, name: str = "SephiroticBus"):
        self.name = name
        self.nodes: Dict[str, SephiroticNode] = {}  # Зарегистрированные узлы
        self.subscriptions: Dict[SignalType, List[Callable]] = defaultdict(list)
        self.message_log = deque(maxlen=1000)  # Лог сообщений
        self.module_bindings: Dict[str, str] = {}  # Привязки модулей к сефирам
        self.logger = self._setup_logger()
        
        # Предустановленные привязки модулей к сефирам
        self._setup_default_bindings()
        
        self.logger.info(f"Сефиротическая шина '{name}' инициализирована")
    
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
            'bechtereva': 'KETER',     # Нейро-модуль -> Кетер (Сознание)
            'chernigovskaya': 'CHOKHMAH', # Семиотика -> Хохма (Мудрость)
            'emotional_weave': 'CHESED',
            'immune_core': 'GEVURAH',
            'policy_governor': 'TIFERET',
            'heartbeat_core': 'NETZACH',
            'polyglossia_adapter': 'HOD',
            'spinal_core': 'YESOD',
            'trust_mesh': 'MALKUTH'
        }
        
        # Обратные привязки для быстрого поиска
        self.sephira_to_module = {v: k for k, v in self.module_bindings.items()}
    
    # ============================================================================
    # РЕГИСТРАЦИЯ И УПРАВЛЕНИЕ УЗЛАМИ
    # ============================================================================
    
    async def register_node(self, node: SephiroticNode) -> bool:
        """
        Регистрация сефиротического узла в шине.
        
        :param node: Экземпляр SephiroticNode
        :return: Успешность регистрации
        """
        if not node or not hasattr(node, 'name'):
            self.logger.error("Попытка регистрации невалидного узла")
            return False
        
        node_name = node.name
        
        if node_name in self.nodes:
            self.logger.warning(f"Узел {node_name} уже зарегистрирован")
            return False
        
        self.nodes[node_name] = node
        self.logger.info(f"Узел {node_name} зарегистрирован в шине")
        
        # Автоматическая привязка к модулю если есть
        if node_name in self.sephira_to_module:
            module_name = self.sephira_to_module[node_name]
            self.logger.info(f"Узел {node_name} привязан к модулю {module_name}")
        
        return True
    
    async def unregister_node(self, node_name: str) -> bool:
        """Удаление узла из шины"""
        if node_name in self.nodes:
            del self.nodes[node_name]
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
    # ПЕРЕДАЧА СИГНАЛОВ
    # ============================================================================
    
    async def transmit(self, signal_package: SignalPackage) -> Dict[str, Any]:
        """
        Основной метод передачи сигнала через шину.
        Маршрутизирует сигналы к целевым узлам или модулям.
        
        :param signal_package: Пакет сигнала
        :return: Результат передачи
        """
        if not signal_package or not hasattr(signal_package, 'type'):
            return {"success": False, "error": "Invalid signal package"}
        
        # Логирование сообщения
        self._log_message(signal_package)
        
        result = {
            "success": False,
            "delivered_to": [],
            "timestamp": datetime.utcnow().isoformat(),
            "signal_id": getattr(signal_package, 'id', 'unknown')
        }
        
        try:
            # 1. Прямая адресация к узлу
            if hasattr(signal_package, 'target') and signal_package.target:
                target_result = await self._deliver_to_target(signal_package)
                result.update(target_result)
            
            # 2. Автомаршрутизация по типу сигнала
            else:
                auto_result = await self._auto_route_signal(signal_package)
                result.update(auto_result)
            
            # 3. Вызов подписчиков на этот тип сигнала
            if signal_package.type in self.subscriptions:
                await self._notify_subscribers(signal_package)
            
            result["success"] = True
            
        except Exception as e:
            result["error"] = str(e)
            self.logger.error(f"Ошибка передачи сигнала {signal_package.id}: {e}")
        
        return result
    
    async def _deliver_to_target(self, signal_package: SignalPackage) -> Dict[str, Any]:
        """Доставка сигнала конкретному целевому узлу"""
        target_name = signal_package.target.upper()
        
        # Проверка прямого узла
        if target_name in self.nodes:
            node = self.nodes[target_name]
            response = await node.receive(signal_package)
            return {
                "delivery_type": "direct_node",
                "delivered_to": [target_name],
                "node_response": response
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
        """Автоматическая маршрутизация сигнала по его типу"""
        signal_type = signal_package.type
        
        # Специальная обработка для нейро и семиотических сигналов
        if signal_type == SignalType.NEURO:
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
        """Логирование сообщения"""
        log_entry = {
            'timestamp': datetime.utcnow().isoformat(),
            'type': signal_package.type.name if hasattr(signal_package.type, 'name') else str(signal_package.type),
            'source': getattr(signal_package, 'source', 'unknown'),
            'target': getattr(signal_package, 'target', 'broadcast'),
            'id': getattr(signal_package, 'id', 'unknown'),
            'hops': getattr(signal_package.metadata, 'hops', 0) if hasattr(signal_package, 'metadata') else 0
        }
        
        self.message_log.append(log_entry)
        
        # Вывод в лог при DEBUG
        if self.logger.isEnabledFor(logging.DEBUG):
            self.logger.debug(f"Message: {log_entry}")
    
    # ============================================================================
    # ШИРОКОВЕЩАТЕЛЬНАЯ СИСТЕМА
    # ============================================================================
    
    async def broadcast(self, signal_package: SignalPackage, 
                       exclude_nodes: List[str] = None) -> Dict[str, Any]:
        """
        Широковещательная рассылка сигнала всем узлам.
        
        :param signal_package: Пакет сигнала
        :param exclude_nodes: Узлы для исключения из рассылки
        :return: Результат рассылки
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
        
        result = {
            "success": len(errors) == 0,
            "delivered_count": len(delivered),
            "total_nodes": len(self.nodes),
            "delivered_to": delivered,
            "errors": errors,
            "timestamp": datetime.utcnow().isoformat()
        }
        
        self.logger.info(f"Broadcast: доставлено {len(delivered)}/{len(self.nodes)} узлов")
        
        return result
    
    # ============================================================================
    # СИСТЕМА ПОДПИСОК
    # ============================================================================
    
    def subscribe(self, signal_type: SignalType, callback: Callable) -> bool:
        """
        Подписка на получение сигналов определённого типа.
        
        :param signal_type: Тип сигнала для подписки
        :param callback: Функция-обработчик
        :return: Успешность подписки
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
    # ИНТЕГРАЦИЯ С МОДУЛЯМИ
    # ============================================================================
    
    async def connect_module(self, module_name: str, sephira_name: str = None) -> Dict[str, Any]:
        """
        Явное подключение модуля к сефиротическому узлу.
        
        :param module_name: Имя модуля (например, 'bechtereva')
        :param sephira_name: Имя сефиры (например, 'KETER')
        :return: Результат подключения
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
                else:
                    return {
                        "success": False,
                        "error": f"Не могу определить сефиру для модуля {module_name}"
                    }
        
        sephira_name_upper = sephira_name.upper()
        
        # Обновление привязок
        self.module_bindings[module_name_lower] = sephira_name_upper
        self.sephira_to_module[sephira_name_upper] = module_name_lower
        
        self.logger.info(f"Модуль {module_name} подключен к сефире {sephira_name_upper}")
        
        return {
            "success": True,
            "module": module_name,
            "sephira": sephira_name_upper,
            "message": f"Модуль {module_name} подключен к {sephira_name_upper}"
        }
    
    async def send_to_module(self, module_name: str, signal_type: SignalType, 
                           payload: Dict[str, Any]) -> Dict[str, Any]:
        """
        Отправка сигнала модулю через его сефиротический узел.
        
        :param module_name: Имя модуля
        :param signal_type: Тип сигнала
        :param payload: Полезная нагрузка
        :return: Результат отправки
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
        
        # Создание и отправка сигнала
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
            "routing_method": "sephira_gateway"
        })
        
        return result
    
    # ============================================================================
    # СТАТУС И ДИАГНОСТИКА
    # ============================================================================
    
    def get_status(self) -> Dict[str, Any]:
        """Получение статуса шины"""
        return {
            "name": self.name,
            "registered_nodes": list(self.nodes.keys()),
            "total_nodes": len(self.nodes),
            "subscriptions": {st.name if hasattr(st, 'name') else str(st): len(cbs) 
                            for st, cbs in self.subscriptions.items()},
            "module_bindings": self.module_bindings,
            "message_log_size": len(self.message_log),
            "recent_messages": list(self.message_log)[-5:] if self.message_log else []
        }
    
    def get_detailed_status(self) -> Dict[str, Any]:
        """Получение детального статуса"""
        status = self.get_status()
        
        # Информация о каждом узле
        nodes_info = {}
        for name, node in self.nodes.items():
            if hasattr(node, 'get_state'):
                nodes_info[name] = node.get_state()
            else:
                nodes_info[name] = {"type": type(node).__name__, "methods": dir(node)[:10]}
        
        status["nodes_info"] = nodes_info
        
        # Статистика по типам сообщений
        message_stats = {}
        for entry in self.message_log:
            msg_type = entry.get('type', 'unknown')
            message_stats[msg_type] = message_stats.get(msg_type, 0) + 1
        
        status["message_statistics"] = message_stats
        
        return status
    
    async def health_check(self) -> Dict[str, Any]:
        """Проверка здоровья шины"""
        health = {
            "timestamp": datetime.utcnow().isoformat(),
            "bus_name": self.name,
            "status": "healthy",
            "checks": {}
        }
        
        # Проверка узлов
        node_health = {}
        for name, node in self.nodes.items():
            try:
                if hasattr(node, 'get_state'):
                    state = node.get_state()
                    node_health[name] = {
                        "status": state.get("status", "unknown"),
                        "reachable": True
                    }
                else:
                    node_health[name] = {"status": "no_state_method", "reachable": True}
            except Exception as e:
                node_health[name] = {"status": "error", "reachable": False, "error": str(e)}
        
        health["checks"]["nodes"] = node_health
        
        # Проверка привязок модулей
        binding_health = {}
        for module, sephira in self.module_bindings.items():
            binding_health[module] = {
                "sephira": sephira,
                "sephira_registered": sephira in self.nodes
            }
        
        health["checks"]["bindings"] = binding_health
        
        # Определение общего статуса
        all_nodes_ok = all(info.get("reachable", False) for info in node_health.values())
        all_bindings_ok = all(info.get("sephira_registered", False) for info in binding_health.values())
        
        if not all_nodes_ok or not all_bindings_ok:
            health["status"] = "degraded"
        
        return health

# ============================================================================
# ФАБРИКА ДЛЯ СОЗДАНИЯ ШИНЫ
# ============================================================================

async def create_sephirotic_bus(name: str = "SephiroticBus") -> SephiroticBus:
    """
    Фабрика для создания и инициализации сефиротической шины.
    
    :param name: Имя шины
    :return: Инициализированный экземпляр SephiroticBus
    """
    bus = SephiroticBus(name)
    
    # Автоматическая подписка на системные события
    # (можно расширить при необходимости)
    
    return bus

# ============================================================================
# ТЕСТОВАЯ ФУНКЦИЯ
# ============================================================================

async def test_bus_integration():
    """Тестирование интеграции шины с модулями"""
    print("🧪 Тестирование сефиротической шины...")
    
    # Создание шины
    bus = await create_sephirotic_bus()
    
    # Проверка статуса
    status = bus.get_status()
    print(f"✅ Шина создана: {status['name']}")
    print(f"   Привязки модулей: {len(status['module_bindings'])}")
    
    # Проверка привязок модулей
    print("\n🔗 Проверка привязок модулей:")
    print(f"   bechtereva -> {bus.module_bindings.get('bechtereva', 'не найдена')}")
    print(f"   chernigovskaya -> {bus.module_bindings.get('chernigovskaya', 'не найдена')}")
    
    # Проверка здоровья
    health = await bus.health_check()
    print(f"\n🏥 Статус здоровья: {health['status']}")
    
    return bus

# ============================================================================
# ТОЧКА ВХОДА ДЛЯ ИНТЕГРАЦИИ
# ============================================================================

async def initialize_bus_for_iskra() -> Dict[str, Any]:
    """
    Функция для вызова из iskra_full.py.
    Инициализирует шину и возвращает готовый экземпляр.
    """
    try:
        bus = await create_sephirotic_bus("ISKRA-4-Sephirotic-Bus")
        
        # Явная привязка ключевых модулей
        await bus.connect_module("bechtereva", "KETER")
        await bus.connect_module("chernigovskaya", "CHOKHMAH")
        
        return {
            "success": True,
            "bus": bus,
            "message": "Сефиротическая шина инициализирована",
            "module_bindings": bus.module_bindings
        }
    
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "message": "Ошибка инициализации шины"
        }

# ============================================================================
# ЗАПУСК ТЕСТА ПРИ НЕПОСРЕДСТВЕННОМ ВЫПОЛНЕНИИ
# ============================================================================

if __name__ == "__main__":
    import asyncio
    
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
    import json
    print(json.dumps(bus.get_detailed_status(), indent=2, ensure_ascii=False))
