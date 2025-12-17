# sephirot_bus.py - СОВЕРШЕННАЯ ЖИВАЯ СИСТЕМА СВЯЗЕЙ
import asyncio
import json
import hashlib
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Set, Tuple
from dataclasses import dataclass, field
from collections import deque, defaultdict
import statistics

from .sephirot_base import (
    SephiroticNode, QuantumLink, SignalPackage, 
    SignalType, NodeStatus, ResonancePhase, NodeMetrics
)


@dataclass
class ChannelPath:
    """Структура пути Древа Жизни (1 из 22 каналов)"""
    hebrew_letter: str           # Ивритская буква пути
    from_sephira: str           # Исходная сефира
    to_sephira: str             # Целевая сефира
    strength: float = 0.8       # Базовая сила связи (0.0-1.0)
    bidirectional: bool = True  # Двунаправленный канал
    description: str = ""       # Описание пути
    activation_count: int = 0   # Количество активаций
    last_used: Optional[datetime] = None
    resonance_factor: float = 1.0  # Текущий резонансный фактор
    energy_decay: float = 0.95     # Коэффициент затухания сигнала
    max_bandwidth: int = 100       # Максимальная пропускная способность
    current_load: int = 0          # Текущая нагрузка
    
    def __post_init__(self):
        if not self.description:
            self.description = f"Путь {self.hebrew_letter}: {self.from_sephira} → {self.to_sephira}"
    
    def can_transmit(self, signal_strength: float = 1.0) -> Tuple[bool, str]:
        """Проверка возможности передачи сигнала"""
        if self.current_load >= self.max_bandwidth:
            return False, "channel_overloaded"
        
        effective_strength = self.strength * self.resonance_factor * signal_strength
        if effective_strength < 0.1:
            return False, "signal_too_weak"
        
        return True, "can_transmit"
    
    def calculate_signal_loss(self, distance: int = 1) -> float:
        """Расчет потери сигнала при передаче"""
        base_loss = (1 - self.strength) * 0.3
        resonance_loss = (1 - self.resonance_factor) * 0.2
        distance_loss = (distance - 1) * 0.1
        load_loss = (self.current_load / self.max_bandwidth) * 0.4
        
        total_loss = min(base_loss + resonance_loss + distance_loss + load_loss, 0.9)
        return total_loss
    
    def update_resonance(self, success: bool) -> None:
        """Обновление резонансного фактора на основе успешности передачи"""
        if success:
            # Усиление резонанса при успешной передаче
            self.resonance_factor = min(1.0, self.resonance_factor + 0.01)
            self.strength = min(1.0, self.strength + 0.005)
        else:
            # Ослабление при неудаче
            self.resonance_factor = max(0.1, self.resonance_factor - 0.02)
            self.strength = max(0.3, self.strength - 0.01)
        
        # Автоматическое восстановление со временем
        if self.last_used:
            hours_since_use = (datetime.utcnow() - self.last_used).total_seconds() / 3600
            if hours_since_use > 1:
                recovery = hours_since_use * 0.01
                self.resonance_factor = min(1.0, self.resonance_factor + recovery)
                self.strength = min(1.0, self.strength + recovery * 0.5)


@dataclass
class SignalTrace:
    """Трассировка прохождения сигнала через сеть"""
    signal_id: str
    source: str
    original_package: SignalPackage
    path_taken: List[Dict[str, Any]] = field(default_factory=list)
    start_time: datetime = field(default_factory=datetime.utcnow)
    end_time: Optional[datetime] = None
    success: bool = False
    total_hops: int = 0
    average_strength: float = 0.0
    bottlenecks: List[str] = field(default_factory=list)
    
    def add_hop(self, channel: ChannelPath, strength_after: float, 
                processing_time: float, node_status: str):
        """Добавление шага в трассировку"""
        self.path_taken.append({
            "channel": channel.hebrew_letter,
            "from": channel.from_sephira,
            "to": channel.to_sephira,
            "strength_before": channel.strength,
            "strength_after": strength_after,
            "processing_time": processing_time,
            "node_status": node_status,
            "timestamp": datetime.utcnow().isoformat(),
            "resonance_factor": channel.resonance_factor
        })
        self.total_hops += 1
    
    def complete(self, success: bool = True):
        """Завершение трассировки"""
        self.end_time = datetime.utcnow()
        self.success = success
        
        if self.path_taken:
            strengths = [hop["strength_after"] for hop in self.path_taken]
            self.average_strength = statistics.mean(strengths)
            
            # Определение узких мест (сигнал < 0.3)
            self.bottlenecks = [
                hop["channel"] for hop in self.path_taken 
                if hop["strength_after"] < 0.3
            ]
    
    def to_dict(self) -> Dict[str, Any]:
        """Конвертация в словарь"""
        return {
            "signal_id": self.signal_id,
            "source": self.source,
            "signal_type": self.original_package.type.value,
            "start_time": self.start_time.isoformat(),
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "duration_seconds": (
                (self.end_time - self.start_time).total_seconds() 
                if self.end_time else None
            ),
            "success": self.success,
            "total_hops": self.total_hops,
            "average_strength": self.average_strength,
            "bottlenecks": self.bottlenecks,
            "path_details": self.path_taken
        }


class QuantumChannelOptimizer:
    """Оптимизатор квантовых каналов с динамической подстройкой"""
    
    def __init__(self):
        self.channel_metrics: Dict[str, Dict[str, Any]] = {}
        self.optimization_history: deque = deque(maxlen=1000)
        self.last_optimization: Optional[datetime] = None
        
    async def analyze_channel_performance(self, channels: List[ChannelPath], 
                                         traces: List[SignalTrace]) -> Dict[str, Any]:
        """Анализ производительности каналов"""
        
        channel_stats = {}
        for channel in channels:
            channel_key = f"{channel.from_sephira}->{channel.to_sephira}"
            
            # Сбор статистики по этому каналу
            channel_traces = [
                trace for trace in traces
                if any(
                    hop["channel"] == channel.hebrew_letter 
                    for hop in trace.path_taken
                )
            ]
            
            if channel_traces:
                success_rate = len([t for t in channel_traces if t.success]) / len(channel_traces)
                avg_strength = statistics.mean([
                    hop["strength_after"]
                    for trace in channel_traces
                    for hop in trace.path_taken
                    if hop["channel"] == channel.hebrew_letter
                ]) if any(trace.path_taken for trace in channel_traces) else 0
                
                avg_processing_time = statistics.mean([
                    hop["processing_time"]
                    for trace in channel_traces
                    for hop in trace.path_taken
                    if hop["channel"] == channel.hebrew_letter
                ]) if any(trace.path_taken for trace in channel_traces) else 0
            else:
                success_rate = 0
                avg_strength = 0
                avg_processing_time = 0
            
            channel_stats[channel_key] = {
                "hebrew_letter": channel.hebrew_letter,
                "success_rate": success_rate,
                "average_strength": avg_strength,
                "average_processing_time": avg_processing_time,
                "current_strength": channel.strength,
                "resonance_factor": channel.resonance_factor,
                "load_percentage": (channel.current_load / channel.max_bandwidth) * 100,
                "activation_count": channel.activation_count,
                "recommendation": self._generate_recommendation(
                    success_rate, avg_strength, channel.current_load, channel.max_bandwidth
                )
            }
        
        return channel_stats
    
    def _generate_recommendation(self, success_rate: float, avg_strength: float,
                                current_load: int, max_bandwidth: int) -> str:
        """Генерация рекомендаций по оптимизации канала"""
        load_percentage = (current_load / max_bandwidth) * 100
        
        if success_rate < 0.5:
            return "increase_strength_or_rest"
        elif avg_strength < 0.3:
            return "improve_resonance"
        elif load_percentage > 80:
            return "increase_bandwidth_or_balance_load"
        elif success_rate > 0.9 and avg_strength > 0.8:
            return "optimal"
        else:
            return "monitor"
    
    async def optimize_channels(self, channels: List[ChannelPath], 
                               force: bool = False) -> List[ChannelPath]:
        """Оптимизация каналов на основе метрик"""
        
        now = datetime.utcnow()
        
        # Проверка необходимости оптимизации (не чаще чем раз в 5 минут)
        if (not force and self.last_optimization and 
            (now - self.last_optimization).total_seconds() < 300):
            return channels
        
        optimized_channels = []
        changes_made = []
        
        for channel in channels:
            original_strength = channel.strength
            original_resonance = channel.resonance_factor
            
            # Анализ метрик канала
            channel_key = f"{channel.from_sephira}->{channel.to_sephira}"
            metrics = self.channel_metrics.get(channel_key, {})
            
            success_rate = metrics.get("success_rate", 0.5)
            avg_strength = metrics.get("average_strength", 0.5)
            load_percentage = metrics.get("load_percentage", 0)
            
            # Применение оптимизаций
            if success_rate < 0.3:
                # Низкая успешность - снижаем нагрузку
                channel.max_bandwidth = max(50, channel.max_bandwidth - 20)
                channel.strength = max(0.3, channel.strength - 0.1)
                changes_made.append(f"{channel.hebrew_letter}: reduced_load")
            
            elif success_rate > 0.8 and avg_strength > 0.7:
                # Высокая успешность - можно повысить нагрузку
                if load_percentage < 60:
                    channel.max_bandwidth = min(200, channel.max_bandwidth + 20)
                    channel.strength = min(1.0, channel.strength + 0.05)
                    changes_made.append(f"{channel.hebrew_letter}: increased_capacity")
            
            # Автоматическая балансировка резонанса
            if avg_strength < 0.4:
                channel.resonance_factor = min(1.0, channel.resonance_factor + 0.05)
            elif avg_strength > 0.9:
                channel.resonance_factor = max(0.5, channel.resonance_factor - 0.03)
            
            # Запись изменений
            if (abs(channel.strength - original_strength) > 0.01 or
                abs(channel.resonance_factor - original_resonance) > 0.01):
                changes_made.append(
                    f"{channel.hebrew_letter}: strength {original_strength:.2f}→{channel.strength:.2f}, "
                    f"resonance {original_resonance:.2f}→{channel.resonance_factor:.2f}"
                )
            
            optimized_channels.append(channel)
        
        # Запись истории оптимизации
        if changes_made:
            self.optimization_history.append({
                "timestamp": now.isoformat(),
                "changes": changes_made,
                "total_channels": len(channels),
                "channels_optimized": len([c for c in changes_made if "strength" in c or "resonance" in c])
            })
        
        self.last_optimization = now
        return optimized_channels


class SephiroticBus:
    """Совершенная центральная шина для 10 узлов и 22 путей Древа Жизни"""
    
    def __init__(self, config_file: str = "config/sephirot_channels.json"):
        self.nodes: Dict[str, SephiroticNode] = {}
        self.channels: List[ChannelPath] = []
        self.signal_traces: Dict[str, SignalTrace] = {}
        self.message_log: deque = deque(maxlen=500)
        self.feedback_queue: asyncio.Queue = asyncio.Queue(maxsize=1000)
        self.optimizer = QuantumChannelOptimizer()
        self.load_balancer = ChannelLoadBalancer()
        
        # Метрики шины
        self.metrics = {
            "total_signals": 0,
            "successful_signals": 0,
            "failed_signals": 0,
            "average_latency": 0.0,
            "peak_load": 0,
            "system_coherence_history": deque(maxlen=100),
            "channel_health_history": defaultdict(lambda: deque(maxlen=50))
        }
        
        # Инициализация
        self._load_all_channels()
        self._init_background_tasks()
        
        print(f"[BUS] Инициализирована сефиротическая шина с {len(self.channels)} каналами")
    
    def _load_all_channels(self):
        """Загрузка всех 22 каналов Древа Жизни"""
        # Полный набор 22 путей (ивритские буквы + сефирот)
        all_channels = [
            # 3 вертикальных столпа
            ChannelPath("Aleph", "Kether", "Chokhmah", 0.95, True, "Путь от Короны к Мудрости"),
            ChannelPath("Beth", "Kether", "Binah", 0.95, True, "Путь от Короны к Пониманию"),
            ChannelPath("Gimel", "Kether", "Tiferet", 0.9, True, "Путь от Короны к Красоте"),
            
            # Горизонтальные пути верхнего треугольника
            ChannelPath("Daleth", "Chokhmah", "Binah", 0.85, True, "Путь от Мудрости к Пониманию"),
            ChannelPath("He", "Chokhmah", "Tiferet", 0.8, True, "Путь от Мудрости к Красоте"),
            ChannelPath("Vav", "Chokhmah", "Chesed", 0.75, True, "Путь от Мудрости к Милосердию"),
            
            # Средние пути
            ChannelPath("Zayin", "Binah", "Tiferet", 0.8, True, "Путь от Понимания к Красоте"),
            ChannelPath("Cheth", "Binah", "Gevurah", 0.75, True, "Путь от Понимания к Строгости"),
            ChannelPath("Teth", "Chesed", "Gevurah", 0.7, True, "Путь от Милосердия к Строгости"),
            
            # Центральные пути
            ChannelPath("Yod", "Chesed", "Tiferet", 0.85, True, "Путь от Милосердия к Красоте"),
            ChannelPath("Kaph", "Gevurah", "Tiferet", 0.85, True, "Путь от Строгости к Красоте"),
            ChannelPath("Lamed", "Chesed", "Netzach", 0.8, True, "Путь от Милосердия к Победе"),
            
            # Нижние пути
            ChannelPath("Mem", "Gevurah", "Hod", 0.8, True, "Путь от Строгости к Славе"),
            ChannelPath("Nun", "Tiferet", "Netzach", 0.9, True, "Путь от Красоты к Победе"),
            ChannelPath("Samekh", "Tiferet", "Yesod", 0.9, True, "Путь от Красоты к Основанию"),
            
            ChannelPath("Ayin", "Tiferet", "Hod", 0.9, True, "Путь от Красоты к Славе"),
            ChannelPath("Pe", "Netzach", "Hod", 0.85, True, "Путь от Победы к Славе"),
            ChannelPath("Tzaddi", "Netzach", "Yesod", 0.8, True, "Путь от Победы к Основанию"),
            
            ChannelPath("Qoph", "Hod", "Yesod", 0.85, True, "Путь от Славы к Основанию"),
            ChannelPath("Resh", "Netzach", "Malkuth", 0.75, True, "Путь от Победы к Царству"),
            ChannelPath("Shin", "Hod", "Malkuth", 0.75, True, "Путь от Славы к Царству"),
            ChannelPath("Tav", "Yesod", "Malkuth", 0.95, True, "Путь от Основания к Царству"),
        ]
        
        self.channels = all_channels
        print(f"[BUS] Загружены {len(self.channels)}/22 каналов Древа Жизни")
    
    def _init_background_tasks(self):
        """Инициализация фоновых задач шины"""
        self.background_tasks = {
            "optimization": None,
            "feedback_processing": None,
            "metrics_aggregation": None,
            "health_check": None
        }
    
    async def start_background_tasks(self):
        """Запуск фоновых задач"""
        loop = asyncio.get_event_loop()
        
        self.background_tasks["optimization"] = loop.create_task(
            self._periodic_optimization()
        )
        
        self.background_tasks["feedback_processing"] = loop.create_task(
            self._process_feedback_queue()
        )
        
        self.background_tasks["metrics_aggregation"] = loop.create_task(
            self._aggregate_metrics()
        )
        
        self.background_tasks["health_check"] = loop.create_task(
            self._health_check_cycle()
        )
        
        print("[BUS] Фоновые задачи запущены")
    
    async def register_node(self, node: SephiroticNode) -> bool:
        """Регистрация сефиротического узла с проверкой дубликатов"""
        if node.name in self.nodes:
            print(f"[BUS] Предупреждение: узел {node.name} уже зарегистрирован")
            return False
        
        self.nodes[node.name] = node
        
        # Создание обратных связей для узла
        await self._create_node_feedback_channels(node.name)
        
        print(f"[BUS] Зарегистрирован узел: {node.name} (уровень {node.sephira_level})")
        return True
    
    async def _create_node_feedback_channels(self, node_name: str):
        """Создание каналов обратной связи для узла"""
        # Находим все входящие каналы для узла
        incoming_channels = [
            channel for channel in self.channels 
            if channel.to_sephira == node_name
        ]
        
        for channel in incoming_channels:
            # Проверяем, существует ли уже обратный канал
            reverse_exists = any(
                c.from_sephira == node_name and c.to_sephira == channel.from_sephira
                for c in self.channels
            )
            
            if not reverse_exists and channel.bidirectional:
                # Создаем обратный канал с немного другими параметрами
                reverse_channel = ChannelPath(
                    hebrew_letter=f"{channel.hebrew_letter}_R",
                    from_sephira=node_name,
                    to_sephira=channel.from_sephira,
                    strength=channel.strength * 0.9,  # Обратный путь слабее
                    bidirectional=True,
                    description=f"Обратный путь от {node_name} к {channel.from_sephira}",
                    energy_decay=channel.energy_decay * 1.1  # Большее затухание
                )
                self.channels.append(reverse_channel)
    
    async def transmit(self, from_node: str, signal_package: SignalPackage, 
                      max_hops: int = 5, require_confirmation: bool = False) -> Dict[str, Any]:
        """Асинхронная передача сигнала через соответствующие каналы с трассировкой"""
        
        # Генерация ID сигнала для трассировки
        signal_id = hashlib.md5(
            f"{from_node}_{signal_package.type}_{datetime.utcnow().isoformat()}".encode()
        ).hexdigest()[:16]
        
        # Создание трассировки
        trace = SignalTrace(
            signal_id=signal_id,
            source=from_node,
            original_package=signal_package
        )
        self.signal_traces[signal_id] = trace
        
        # Проверка безопасности сигнала
        safety_check = await self._validate_signal(signal_package)
        if not safety_check["valid"]:
            trace.complete(False)
            return {
                "success": False,
                "error": f"Signal validation failed: {safety_check['reason']}",
                "signal_id": signal_id,
                "trace": trace.to_dict()
            }
        
        # Основной цикл передачи
        current_node = from_node
        remaining_hops = max_hops
        current_signal = signal_package.copy()
        current_strength = 1.0
        
        start_time = datetime.utcnow()
        
        while remaining_hops > 0 and current_node in self.nodes:
            # Находим все исходящие каналы из текущего узла
            outgoing_channels = [
                channel for channel in self.channels
                if channel.from_sephira == current_node and channel.to_sephira in self.nodes
            ]
            
            if not outgoing_channels:
                break
            
            # Балансировка нагрузки: выбираем лучший канал
            selected_channel = await self.load_balancer.select_best_channel(
                outgoing_channels, current_signal.type, current_strength
            )
            
            if not selected_channel:
                break
            
            # Проверка возможности передачи
            can_transmit, reason = selected_channel.can_transmit(current_strength)
            if not can_transmit:
                trace.bottlenecks.append(f"{selected_channel.hebrew_letter}:{reason}")
                break
            
            # Подготовка к передаче
            target_node = self.nodes[selected_channel.to_sephira]
            processing_start = datetime.utcnow()
            
            try:
                # Обновление метаданных сигнала
                current_signal.metadata["hops"] = max_hops - remaining_hops + 1
                current_signal.metadata["current_strength"] = current_strength
                current_signal.metadata["channel"] = selected_channel.hebrew_letter
                
                # Передача сигнала целевому узлу (асинхронно)
                result = await target_node.receive_signal(current_signal)
                
                # Расчет времени обработки
                processing_time = (datetime.utcnow() - processing_start).total_seconds()
                
                # Обновление канала
                selected_channel.activation_count += 1
                selected_channel.current_load += 1
                selected_channel.last_used = datetime.utcnow()
                
                # Расчет затухания сигнала
                signal_loss = selected_channel.calculate_signal_loss(max_hops - remaining_hops)
                current_strength *= (1 - signal_loss)
                
                # Обновление трассировки
                trace.add_hop(
                    channel=selected_channel,
                    strength_after=current_strength,
                    processing_time=processing_time,
                    node_status=target_node.status.value
                )
                
                # Обновление резонанса канала на основе успешности
                selected_channel.update_resonance(True)
                
                # Переход к следующему узлу
                current_node = selected_channel.to_sephira
                
                # Проверка завершения (достигнута цель или сигнал слишком слаб)
                if (current_node == signal_package.target or 
                    current_strength < 0.1 or 
                    (require_confirmation and result.get("confirmed", False))):
                    break
                
                remaining_hops -= 1
                
                # Небольшая задержка между переходами
                await asyncio.sleep(0.001)
                
            except Exception as e:
                # Ошибка обработки
                selected_channel.update_resonance(False)
                trace.add_hop(
                    channel=selected_channel,
                    strength_after=current_strength * 0.5,  # Сильное затухание при ошибке
                    processing_time=(datetime.utcnow() - processing_start).total_seconds(),
                    node_status="error"
                )
                break
            
            finally:
                # Снижение нагрузки на канал
                selected_channel.current_load = max(0, selected_channel.current_load - 1)
        
        # Завершение трассировки
        trace.complete(trace.total_hops > 0)
        
        # Обновление метрик шины
        await self._update_transmission_metrics(
            success=trace.success,
            duration=(datetime.utcnow() - start_time).total_seconds(),
            hops=trace.total_hops,
            final_strength=current_strength
        )
        
        # Логирование
        await self._log_transmission(trace)
        
        return {
            "success": trace.success,
            "signal_id": signal_id,
            "final_node": current_node,
            "final_strength": current_strength,
            "total_hops": trace.total_hops,
            "bottlenecks": trace.bottlenecks,
            "trace": trace.to_dict(),
            "timestamp": datetime.utcnow().isoformat()
        }
    
    async def _validate_signal(self, signal_package: SignalPackage) -> Dict[str, Any]:
        """Проверка безопасности и валидности сигнала"""
        
        # Проверка размера payload
        if hasattr(signal_package, 'payload'):
            import sys
            payload_size = sys.getsizeof(str(signal_package.payload))
            if payload_size > 1024 * 1024:  # 1 MB лимит
                return {"valid": False, "reason": "payload_too_large"}
        
        # Проверка типа сигнала
        if not isinstance(signal_package.type, SignalType):
            return {"valid": False, "reason": "invalid_signal_type"}
        
        # Проверка TTL если есть
        if hasattr(signal_package, 'ttl'):
            if signal_package.ttl <= 0:
                return {"valid": False, "reason": "signal_expired"}
        
        return {"valid": True, "reason": "ok"}
    
    async def broadcast_quantum_sync(self, sync_signal: SignalPackage, 
                                    group_name: str = "all") -> Dict[str, Dict[str, Any]]:
        """Квантовая синхронизация состояний узлов в группе"""
        
        results = {}
        
        if group_name == "all":
            target_nodes = list(self.nodes.keys())
        else:
            # Определение узлов в группе
            target_nodes = []
            for channel in self.channels:
                if group_name in channel.description or channel.from_sephira in group_name:
                    target_nodes.append(channel.from_sephira)
                    target_nodes.append(channel.to_sephira)
            target_nodes = list(set(target_nodes))
        
        # Параллельная синхронизация
        tasks = []
        for node_name in target_nodes:
            if node_name in self.nodes:
                task = self._perform_single_sync(node_name, sync_signal)
                tasks.append(task)
        
        # Ожидание всех синхронизаций
        sync_results = await asyncio.gather(*tasks, return_exceptions=True)
        
        for i, node_name in enumerate(target_nodes):
            if i < len(sync_results):
                results[node_name] = (
                    sync_results[i] if not isinstance(sync_results[i], Exception)
                    else {"error": str(sync_results[i]), "sync_successful": False}
                )
        
        return results
    
    async def _perform_single_sync(self, node_name: str, sync_signal: SignalPackage) -> Dict[str, Any]:
        """Выполнение синхронизации с одним узлом"""
        node = self.nodes[node_name]
        
        # Специальный сигнал синхронизации
        sync_package = sync_signal.copy()
        sync_package.target = node_name
        sync_package.metadata["sync_mode"] = True
        
        try:
            result = await node.receive_signal(sync_package)
            
            # Проверка успешности синхронизации
            current_resonance = node.resonance if hasattr(node, 'resonance') else 0.5
            sync_successful = result.get("sync_accepted", False) or current_resonance > 0.6
            
            return {
                "sync_successful": sync_successful,
                "current_resonance": current_resonance,
                "node_status": node.status.value,
                "response": result,
                "resonance_gain": min(0.1, (1 - current_resonance) * 0.2) if sync_successful else 0
            }
        except Exception as e:
            return {"error": str(e), "sync_successful": False}
    
    async def propagate_feedback(self, to_node: str, feedback: Dict[str, Any]) -> Dict[str, Any]:
        """Распространение обратной связи к узлу"""
        
        if to_node not in self.nodes:
            return {"error": f"Node {to_node} not found", "success": False}
        
        # Создание сигнала обратной связи
        feedback_signal = SignalPackage(
            source="BUS_FEEDBACK",
            target=to_node,
            type=SignalType.COMMAND,
            payload=feedback,
            metadata={"feedback_loop": True, "timestamp": datetime.utcnow().isoformat()}
        )
        
        # Помещение в очередь обратной связи
        try:
            await self.feedback_queue.put({
                "signal": feedback_signal,
                "priority": feedback.get("priority", 5)
            })
            return {"success": True, "queued": True, "timestamp": datetime.utcnow().isoformat()}
        except asyncio.QueueFull:
            return {"error": "Feedback queue is full", "success": False}
    
    async def _process_feedback_queue(self):
        """Обработка очереди обратной связи"""
        while True:
            try:
                # Получение с приоритетом
                feedback_item = await self.feedback_queue.get()
                feedback_signal = feedback_item["signal"]
                
                # Передача обратной связи
                await self.transmit(
                    from_node=feedback_signal.source,
                    signal_package=feedback_signal,
                    max_hops=3,
                    require_confirmation=True
                )
                
                self.feedback_queue.task_done()
                
            except Exception as e:
                print(f"[BUS] Ошибка обработки обратной связи: {e}")
            
            await asyncio.sleep(0.1)  # Предотвращение перегрузки
    
    async def get_signal_trace(self, signal_id: str) -> Optional[Dict[str, Any]]:
        """Получение детальной трассировки сигнала"""
        if signal_id in self.signal_traces:
            trace = self.signal_traces[signal_id]
            return trace.to_dict()
        return None
    
    async def get_network_state(self) -> Dict[str, Any]:
        """Расширенное состояние сети"""
        active_nodes = [
            name for name, node in self.nodes.items()
            if node.status == NodeStatus.ACTIVE
        ]
        
        # Расчет когерентности с учетом резонанса
        coherence = await self._calculate_enhanced_coherence()
        
        # Статистика каналов
        channel_stats = {
            "total": len(self.channels),
            "active": len([c for c in self.channels 
                          if c.from_sephira in self.nodes and c.to_sephira in self.nodes]),
            "avg_strength": statistics.mean([c.strength for c in self.channels]) 
                          if self.channels else 0,
            "avg_resonance": statistics.mean([c.resonance_factor for c in self.channels]) 
                           if self.channels else 0,
            "total_load": sum(c.current_load for c in self.channels)
        }
        
        return {
            "timestamp": datetime.utcnow().isoformat(),
            "nodes_registered": list(self.nodes.keys()),
            "nodes_active": active_nodes,
            "active_node_count": len(active_nodes),
            "total_node_count": len(self.nodes),
            "channel_statistics": channel_stats,
            "system_coherence": coherence,
            "coherence_level": self._coherence_to_level(coherence),
            "recent_signals": len(self.message_log),
            "feedback_queue_size": self.feedback_queue.qsize(),
            "active_traces": len(self.signal_traces),
            "metrics": {
                "success_rate": (
                    self.metrics["successful_signals"] / self.metrics["total_signals"] 
                    if self.metrics["total_signals"] > 0 else 0
                ),
                "average_latency": self.metrics["average_latency"],
                "peak_load": self.metrics["peak_load"]
            }
        }
    
    async def _calculate_enhanced_coherence(self) -> float:
        """Расширенный расчет когерентности сети"""
        if not self.nodes or not self.channels:
            return 0.0
        
        # 1. Базовая когерентность (активные связи)
        active_channels = [
            c for c in self.channels
            if c.from_sephira in self.nodes and c.to_sephira in self.nodes
        ]
        base_coherence = len(active_channels) / len(self.channels)
        
        # 2. Резонансная когерентность
        resonance_values = [c.resonance_factor for c in active_channels]
        resonance_coherence = (
            statistics.mean(resonance_values) 
            if resonance_values else 0
        )
        
        # 3. Энергетическая когерентность
        active_nodes = [
            node for node in self.nodes.values()
            if node.status == NodeStatus.ACTIVE
        ]
        if active_nodes:
            energy_values = [node.energy for node in active_nodes if hasattr(node, 'energy')]
            energy_coherence = (
                statistics.mean(energy_values) 
               
