# sephirotic_engine.py - ГЛАВНЫЙ ДВИГАТЕЛЬ (ИДЕАЛЬНАЯ ВЕРСИЯ)
import asyncio
import importlib
import inspect
import json
from datetime import datetime
from typing import Dict, List, Optional, Any
import statistics
from dataclasses import dataclass
from collections import deque

from .sephirot_bus import SephiroticBus
from .sephirot_base import SephiroticNode, NodeStatus, SignalPackage, SignalType


@dataclass
class IntegrationLink:
    """Структура для связи сефирота с модулем ISKRA"""
    sephirot_name: str
    module_name: str
    link_type: str
    active: bool = True
    last_sync: Optional[datetime] = None


class SephiroticEngine:
    """Движок сефиротической системы - асинхронный и автономный"""
    
    def __init__(self, config_path: str = "config/sephirot_config.yaml"):
        self.bus = SephiroticBus()
        self.nodes: Dict[str, SephiroticNode] = {}
        self.integrations: List[IntegrationLink] = []
        self.heartbeat_interval = 2.0
        self.running = False
        self.metrics_history = deque(maxlen=1000)
        self.cycle_counter = 0
        self.config_path = config_path
        self.monitor_task: Optional[asyncio.Task] = None
        self.heartbeat_task: Optional[asyncio.Task] = None
        
        # Загрузка конфигурации
        self.config = self._load_config()
        
    def _load_config(self) -> Dict[str, Any]:
        """Загрузка конфигурации из YAML"""
        import yaml
        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f)
        except FileNotFoundError:
            # Конфигурация по умолчанию
            return {
                "sephirot": {
                    "active_nodes": ["Kether", "Tiferet", "Yesod", "Malkuth"],
                    "resonance_threshold": 0.65,
                    "auto_connect": True,
                    "heartbeat_interval": 2.0
                },
                "integration": {
                    "emotional_weave_to_tiferet": True,
                    "polyglossia_to_hod": True,
                    "iskr_eco_to_yesod": True
                }
            }
    
    async def initialize(self) -> Dict[str, Any]:
        """Полная инициализация движка"""
        print(f"[ENGINE] Инициализация сефиротической системы...")
        
        # 1. Автоматическое обнаружение и создание узлов
        await self._auto_discover_nodes()
        
        # 2. Создание квантовых связей между узлами
        await self._create_quantum_links()
        
        # 3. Автоматическая интеграция с модулями ISKRA
        await self._auto_integrate_modules()
        
        # 4. Запуск фоновых задач
        self._start_background_tasks()
        
        return {
            "status": "initialized",
            "nodes_active": len(self.nodes),
            "integrations": len([i for i in self.integrations if i.active]),
            "timestamp": datetime.utcnow().isoformat()
        }
    
    async def _auto_discover_nodes(self) -> None:
        """Автоматическое обнаружение всех классов сефиротов"""
        try:
            # Динамический импорт всех модулей из sephirot_blocks
            import os
            import sys
            
            # Добавляем путь к модулям
            module_path = os.path.join(os.path.dirname(__file__), '..', 'sephirot_blocks')
            
            for root, dirs, files in os.walk(module_path):
                for file in files:
                    if file.endswith('.py') and not file.startswith('__'):
                        module_name = file[:-3]
                        try:
                            # Импортируем модуль
                            module = importlib.import_module(
                                f"iskra_modules.sephirot_blocks.{module_name}"
                                if '__' not in root else 
                                f"iskra_modules.sephirot_blocks.{os.path.basename(root)}.{module_name}"
                            )
                            
                            # Ищем все классы-наследники SephiroticNode
                            for name, obj in inspect.getmembers(module):
                                if (inspect.isclass(obj) and 
                                    issubclass(obj, SephiroticNode) and 
                                    obj != SephiroticNode):
                                    
                                    # Создаем экземпляр
                                    instance = obj(self.bus)
                                    self.nodes[instance.name] = instance
                                    print(f"[ENGINE] Обнаружен узел: {instance.name} (уровень {instance.sephira_level})")
                                    
                        except Exception as e:
                            print(f"[ENGINE] Ошибка загрузки {file}: {e}")
                            continue
            
            # Если узлов не найдено, создаем базовые вручную
            if not self.nodes:
                await self._create_core_nodes_manually()
                
        except Exception as e:
            print(f"[ENGINE] Ошибка автообнаружения: {e}")
            await self._create_core_nodes_manually()
    
    async def _create_core_nodes_manually(self) -> None:
        """Создание базовых узлов вручную (fallback)"""
        from .sephirot_blocks._1_keter.keter_core import KetherCore
        from .sephirot_blocks._6_tiferet.tiferet_core import TiferetCore
        from .sephirot_blocks._9_yesod.yesod_core import YesodCore
        
        core_nodes = [
            ("Kether", KetherCore),
            ("Tiferet", TiferetCore),
            ("Yesod", YesodCore)
        ]
        
        for name, cls in core_nodes:
            try:
                instance = cls(self.bus)
                self.nodes[name] = instance
                print(f"[ENGINE] Создан базовый узел: {name}")
            except Exception as e:
                print(f"[ENGINE] Ошибка создания {name}: {e}")
    
    async def _create_quantum_links(self) -> None:
        """Создание квантовых связей между узлами на основе Древа Жизни"""
        # Карта связей сефирот (пути Древа Жизни)
        sephirot_connections = {
            "Kether": ["Chokmah", "Binah", "Tiferet"],
            "Chokmah": ["Kether", "Binah", "Tiferet", "Chesed"],
            "Binah": ["Kether", "Chokmah", "Tiferet", "Gevurah"],
            "Chesed": ["Chokmah", "Tiferet", "Netzach", "Gevurah"],
            "Gevurah": ["Binah", "Tiferet", "Chesed", "Hod"],
            "Tiferet": ["Kether", "Chokmah", "Binah", "Chesed", "Gevurah", "Netzach", "Hod", "Yesod"],
            "Netzach": ["Chesed", "Tiferet", "Hod", "Yesod"],
            "Hod": ["Gevurah", "Tiferet", "Netzach", "Yesod"],
            "Yesod": ["Tiferet", "Netzach", "Hod", "Malkuth"],
            "Malkuth": ["Yesod"]
        }
        
        # Создаем связи только для существующих узлов
        for source_name, source_node in self.nodes.items():
            if source_name in sephirot_connections:
                for target_name in sephirot_connections[source_name]:
                    if target_name in self.nodes:
                        try:
                            await source_node.create_quantum_link(target_name)
                            print(f"[ENGINE] Создана связь: {source_name} → {target_name}")
                        except Exception as e:
                            print(f"[ENGINE] Ошибка связи {source_name} → {target_name}: {e}")
    
    async def _auto_integrate_modules(self) -> None:
        """Автоматическая интеграция с модулями ISKRA-4"""
        integration_map = {
            "Tiferet": ["emotional_weave", "EmotionalWeave", "emotional"],
            "Hod": ["polyglossia_adapter", "PolyglossiaAdapter", "language"],
            "Yesod": ["iskr_eco_core", "ISKREcoCore", "eco"]
        }
        
        for sephirot_name, (module_path, class_name, link_type) in integration_map.items():
            if sephirot_name in self.nodes:
                try:
                    # Динамический импорт
                    module = importlib.import_module(f"iskra_modules.{module_path}")
                    module_class = getattr(module, class_name)
                    module_instance = module_class()
                    
                    # Устанавливаем связь
                    link_method = getattr(self.nodes[sephirot_name], f"set_{link_type}_link", None)
                    if link_method:
                        link_method(module_instance)
                        
                        # Запоминаем интеграцию
                        self.integrations.append(IntegrationLink(
                            sephirot_name=sephirot_name,
                            module_name=class_name,
                            link_type=link_type,
                            last_sync=datetime.utcnow()
                        ))
                        
                        print(f"[ENGINE] Интеграция: {sephirot_name} ↔ {class_name}")
                        
                except Exception as e:
                    print(f"[ENGINE] Ошибка интеграции {sephirot_name}: {e}")
    
    def _start_background_tasks(self) -> None:
        """Запуск фоновых задач"""
        # Создаем event loop если его нет
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        
        # Запускаем heartbeat
        self.heartbeat_task = loop.create_task(self._heartbeat_cycle())
        
        # Запускаем мониторинг
        self.monitor_task = loop.create_task(self._monitor_system())
        
        print(f"[ENGINE] Фоновые задачи запущены")
    
    async def _heartbeat_cycle(self) -> None:
        """Асинхронный цикл сердцебиения системы"""
        self.running = True
        print(f"[HEARTBEAT] Цикл сердцебиения запущен (интервал: {self.heartbeat_interval}с)")
        
        while self.running:
            try:
                self.cycle_counter += 1
                
                # 1. Генерация волевого импульса от Kether
                if "Kether" in self.nodes:
                    intention_signal = SignalPackage(
                        source="ENGINE",
                        target="Kether",
                        type=SignalType.INTENTION,
                        payload={
                            "intention": {
                                "strength": 0.5 + (self.cycle_counter % 10) * 0.05,
                                "purpose": "system_sustainment",
                                "cycle": self.cycle_counter
                            }
                        }
                    )
                    await self.nodes["Kether"].receive_signal(intention_signal)
                
                # 2. Эмоциональный резонанс каждые 3 цикла
                if self.cycle_counter % 3 == 0 and "Tiferet" in self.nodes:
                    emotion_signal = SignalPackage(
                        source="ENGINE",
                        target="Tiferet",
                        type=SignalType.EMOTIONAL,
                        payload={
                            "emotion_type": "harmony",
                            "intensity": 0.7,
                            "source": "heartbeat"
                        }
                    )
                    await self.nodes["Tiferet"].receive_signal(emotion_signal)
                
                # 3. Обновление фундамента каждые 5 циклов
                if self.cycle_counter % 5 == 0 and "Yesod" in self.nodes:
                    foundation_signal = SignalPackage(
                        source="ENGINE",
                        target="Yesod",
                        type=SignalType.DATA,
                        payload={
                            "key": f"heartbeat_{self.cycle_counter}",
                            "value": {
                                "cycle": self.cycle_counter,
                                "system_time": datetime.utcnow().isoformat(),
                                "node_count": len(self.nodes)
                            }
                        }
                    )
                    await self.nodes["Yesod"].receive_signal(foundation_signal)
                
                # 4. Статус каждые 10 циклов
                if self.cycle_counter % 10 == 0:
                    await self._log_system_status()
                
                # 5. Сохранение метрик
                await self._collect_metrics()
                
                await asyncio.sleep(self.heartbeat_interval)
                
            except asyncio.CancelledError:
                print("[HEARTBEAT] Цикл сердцебиения остановлен")
                break
            except Exception as e:
                print(f"[HEARTBEAT] Ошибка в цикле: {e}")
                await asyncio.sleep(1)  # Пауза при ошибке
    
    async def _monitor_system(self) -> None:
        """Мониторинг здоровья системы"""
        print("[MONITOR] Система мониторинга запущена")
        
        while self.running:
            try:
                # Проверка здоровья узлов
                health_report = await self.get_health_report()
                
                # Обнаружение аномалий
                anomalies = []
                for node_name, health in health_report.items():
                    if health.get("resonance", 1.0) < 0.1:
                        anomalies.append({
                            "type": "low_resonance",
                            "node": node_name,
                            "value": health["resonance"],
                            "timestamp": datetime.utcnow().isoformat()
                        })
                    
                    if health.get("energy", 1.0) < 0.2:
                        anomalies.append({
                            "type": "low_energy",
                            "node": node_name,
                            "value": health["energy"],
                            "timestamp": datetime.utcnow().isoformat()
                        })
                
                # Логирование аномалий
                if anomalies:
                    print(f"[MONITOR] Обнаружены аномалии: {anomalies}")
                
                # Проверка интеграций
                for integration in self.integrations:
                    if integration.active:
                        # Обновляем время последней синхронизации
                        integration.last_sync = datetime.utcnow()
                
                await asyncio.sleep(5.0)  # Проверка каждые 5 секунд
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                print(f"[MONITOR] Ошибка мониторинга: {e}")
                await asyncio.sleep(5)
    
    async def _log_system_status(self) -> None:
        """Логирование статуса системы"""
        state = await self.bus.get_network_state()
        active_nodes = len([n for n in self.nodes.values() if n.status == NodeStatus.ACTIVE])
        
        print(f"[STATUS] Цикл {self.cycle_counter} | "
              f"Узлы: {active_nodes}/{len(self.nodes)} активны | "
              f"Когерентность: {state.get('system_coherence', 0):.2f} | "
              f"Сигналов: {state.get('total_signals', 0)}")
    
    async def _collect_metrics(self) -> None:
        """Сбор метрик системы"""
        metrics = {
            "timestamp": datetime.utcnow().isoformat(),
            "cycle": self.cycle_counter,
            "active_nodes": len([n for n in self.nodes.values() if n.status == NodeStatus.ACTIVE]),
            "average_resonance": statistics.mean(
                [n.resonance for n in self.nodes.values()]
            ) if self.nodes else 0,
            "average_energy": statistics.mean(
                [n.energy for n in self.nodes.values()]
            ) if self.nodes else 0,
            "total_connections": sum(
                len(n.quantum_links) for n in self.nodes.values()
            ),
            "queue_sizes": {
                name: n.signal_queue.qsize()
                for name, n in self.nodes.items()
            }
        }
        
        self.metrics_history.append(metrics)
    
    async def broadcast_signal(self, signal_data: Dict[str, Any]) -> Dict[str, Any]:
        """Широковещательная отправка сигнала всем узлам"""
        results = {}
        
        for node_name, node in self.nodes.items():
            try:
                signal = SignalPackage(
                    source="BROADCAST",
                    target=node_name,
                    type=SignalType.COMMAND,
                    payload=signal_data
                )
                result = await node.receive_signal(signal)
                results[node_name] = {"success": True, "result": result}
            except Exception as e:
                results[node_name] = {"success": False, "error": str(e)}
        
        return {
            "broadcast_complete": True,
            "timestamp": datetime.utcnow().isoformat(),
            "results": results
        }
    
    async def get_active_nodes(self) -> List[str]:
        """Получение списка активных узлов"""
        return [name for name, node in self.nodes.items() 
                if node.status == NodeStatus.ACTIVE]
    
    async def get_health_report(self) -> Dict[str, Dict[str, Any]]:
        """Полный отчет о здоровье системы"""
        report = {}
        
        for node_name, node in self.nodes.items():
            report[node_name] = {
                "resonance": node.resonance,
                "energy": node.energy,
                "status": node.status.value,
                "queue_size": node.signal_queue.qsize(),
                "signals_processed": node.total_signals_processed,
                "last_active": node.last_metrics_update.isoformat() 
                if node.last_metrics_update else None,
                "connections": len(node.quantum_links)
            }
        
        return report
    
    async def get_comprehensive_metrics(self) -> Dict[str, Any]:
        """Всеобъемлющие метрики системы"""
        health = await self.get_health_report()
        
        # Анализ истории метрик
        if self.metrics_history:
            recent_metrics = list(self.metrics_history)[-10:]  # Последние 10 записей
            resonances = [m["average_resonance"] for m in recent_metrics]
            energies = [m["average_energy"] for m in recent_metrics]
        else:
            resonances = [0]
            energies = [0]
        
        return {
            "system_summary": {
                "total_nodes": len(self.nodes),
                "active_nodes": len([n for n in self.nodes.values() 
                                   if n.status == NodeStatus.ACTIVE]),
                "total_integrations": len(self.integrations),
                "total_cycles": self.cycle_counter,
                "uptime": "running" if self.running else "stopped"
            },
            "performance_metrics": {
                "average_resonance": statistics.mean(resonances),
                "resonance_stability": statistics.stdev(resonances) if len(resonances) > 1 else 0,
                "average_energy": statistics.mean(energies),
                "system_coherence": await self._calculate_system_coherence(),
                "signal_throughput": sum(
                    node.total_signals_processed for node in self.nodes.values()
                ) / max(self.cycle_counter, 1)
            },
            "node_health": health,
            "integration_status": [
                {
                    "sephirot": i.sephirot_name,
                    "module": i.module_name,
                    "type": i.link_type,
                    "active": i.active,
                    "last_sync": i.last_sync.isoformat() if i.last_sync else None
                }
                for i in self.integrations
            ],
            "timestamp": datetime.utcnow().isoformat()
        }
    
    async def _calculate_system_coherence(self) -> float:
        """Расчет когерентности системы"""
        if not self.nodes:
            return 0.0
        
        # Когерентность основана на синхронизации резонанса узлов
        resonances = [node.resonance for node in self.nodes.values()]
        
        if len(resonances) > 1:
            # Мера синхронизации - обратная дисперсия
            mean_res = statistics.mean(resonances)
            variance = sum((r - mean_res) ** 2 for r in resonances) / len(resonances)
            coherence = 1.0 / (1.0 + variance)  # 1.0 при полной синхронизации
        else:
            coherence = resonances[0] if resonances else 0.0
        
        return min(max(coherence, 0.0), 1.0)
    
    async def stop(self) -> Dict[str, Any]:
        """Безопасная остановка системы"""
        print("[ENGINE] Остановка сефиротической системы...")
        
        self.running = False
        
        # Остановка фоновых задач
        if self.heartbeat_task:
            self.heartbeat_task.cancel()
            try:
                await self.heartbeat_task
            except asyncio.CancelledError:
                pass
        
        if self.monitor_task:
            self.monitor_task.cancel()
            try:
                await self.monitor_task
            except asyncio.CancelledError:
                pass
        
        # Сбор финальных метрик
        final_metrics = await self.get_comprehensive_metrics()
        
        # Остановка всех узлов
        for node_name, node in self.nodes.items():
            await node.deactivate()
        
        print(f"[ENGINE] Система остановлена. Циклов выполнено: {self.cycle_counter}")
        
        return {
            "status": "stopped",
            "final_metrics": final_metrics,
            "total_cycles": self.cycle_counter,
            "timestamp": datetime.utcnow().isoformat()
        }
    
    async def activate_sephirot(self, sephirot_name: str) -> Dict[str, Any]:
        """Активация конкретного сефирота"""
        if sephirot_name not in self.nodes:
            return {"success": False, "error": f"Узел {sephirot_name} не найден"}
        
        node = self.nodes[sephirot_name]
        await node.activate()
        
        return {
            "success": True,
            "sephirot": sephirot_name,
            "status": node.status.value,
            "resonance": node.resonance,
            "timestamp": datetime.utcnow().isoformat()
        }
    
    async def connect_sephirots(self, source: str, target: str) -> Dict[str, Any]:
        """Создание связи между двумя сефиротами"""
        if source not in self.nodes:
            return {"success": False, "error": f"Источник {source} не найден"}
        if target not in self.nodes:
            return {"success": False, "error": f"Цель {target} не найден"}
        
        try:
            await self.nodes[source].create_quantum_link(target)
            return {
                "success": True,
                "connection": f"{source} → {target}",
                "quantum_strength": 0.85,  # Базовая сила связи
                "timestamp": datetime.utcnow().isoformat()
            }
        except Exception as e:
            return {"success": False, "error": str(e)}


# Глобальный экземпляр движка для удобного доступа
_global_engine: Optional[SephiroticEngine] = None


async def get_sephirotic_engine() -> SephiroticEngine:
    """Получение глобального экземпляра движка (синглтон)"""
    global _global_engine
    if _global_engine is None:
        _global_engine = SephiroticEngine()
        await _global_engine.initialize()
    return _global_engine


async def activate_sephirotic_system() -> Dict[str, Any]:
    """Активация всей сефиротической системы"""
    engine = await get_sephirotic_engine()
    return await engine.initialize()


async def shutdown_sephirotic_system() -> Dict[str, Any]:
    """Корректное завершение работы системы"""
    global _global_engine
    if _global_engine is not None:
        result = await _global_engine.stop()
        _global_engine = None
        return result
    return {"status": "already_stopped"}
