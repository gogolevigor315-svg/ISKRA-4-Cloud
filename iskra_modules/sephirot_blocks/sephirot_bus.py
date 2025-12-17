 # sephirot_bus.py - СОВЕРШЕННАЯ СЕФИРОТИЧЕСКАЯ ШИНА (ИДЕАЛЬНАЯ ВЕРСИЯ)
import asyncio
import json
import hashlib
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Set, Tuple
from dataclasses import dataclass, field
from collections import deque, defaultdict
import statistics
import yaml
from enum import Enum

from .sephirot_base import (
    SephiroticNode, QuantumLink, SignalPackage, 
    SignalType, NodeStatus, ResonancePhase, NodeMetrics
)


class ChannelDirection(Enum):
    """Направление канала"""
    FORWARD = "forward"      # Прямое направление
    REVERSE = "reverse"      # Обратное направление
    BIDIRECTIONAL = "bidirectional"  # Двусторонний


@dataclass
class QuantumChannel:
    """Квантовый канал Древа Жизни с полной динамикой"""
    
    # Идентификаторы
    id: str
    hebrew_letter: str
    from_sephira: str
    to_sephira: str
    
    # Динамические параметры
    base_strength: float = 0.8          # Базовая сила (0.0-1.0)
    current_strength: float = 0.8       # Текущая сила с учетом резонанса
    resonance_factor: float = 1.0       # Фактор резонанса (0.1-2.0)
    energy_decay: float = 0.95          # Коэффициент затухания энергии
    learning_rate: float = 0.01         # Скорость обучения канала
    
    # Настройки
    direction: ChannelDirection = ChannelDirection.BIDIRECTIONAL
    max_bandwidth: int = 100            # Макс сигналов/сек
    current_load: int = 0               # Текущая нагрузка
    is_active: bool = True              # Активен ли канал
    
    # Метаданные
    description: str = ""
    created: datetime = field(default_factory=datetime.utcnow)
    last_used: Optional[datetime] = None
    last_optimized: Optional[datetime] = None
    
    # Метрики
    total_transmissions: int = 0
    successful_transmissions: int = 0
    failed_transmissions: int = 0
    avg_latency: float = 0.0
    avg_signal_strength: float = 0.0
    
    # История
    strength_history: deque = field(default_factory=lambda: deque(maxlen=100))
    resonance_history: deque = field(default_factory=lambda: deque(maxlen=100))
    latency_history: deque = field(default_factory=lambda: deque(maxlen=100))
    
    def __post_init__(self):
        """Пост-инициализация"""
        if not self.description:
            self.description = f"Путь {self.hebrew_letter}: {self.from_sephira} → {self.to_sephira}"
        
        # Инициализация истории
        self.strength_history.append(self.current_strength)
        self.resonance_history.append(self.resonance_factor)
    
    async def can_transmit(self, signal_strength: float = 1.0) -> Tuple[bool, str, float]:
        """
        Проверка возможности передачи с возвратом эффективной силы
        
        Returns:
            Tuple[bool, str, float]: (может передавать, причина, эффективная сила)
        """
        # Проверка активности
        if not self.is_active:
            return False, "channel_inactive", 0.0
        
        # Проверка перегрузки
        load_percentage = self.current_load / self.max_bandwidth if self.max_bandwidth > 0 else 0
        if load_percentage > 0.9:
            return False, "channel_overloaded", 0.0
        
        # Расчет эффективной силы
        effective_strength = (
            self.current_strength * 
            self.resonance_factor * 
            signal_strength * 
            (1 - load_percentage * 0.5)
        )
        
        if effective_strength < 0.05:
            return False, "signal_too_weak", effective_strength
        
        return True, "can_transmit", effective_strength
    
    async def calculate_signal_transform(self, signal_package: SignalPackage, 
                                        distance: int = 1) -> Tuple[SignalPackage, float, Dict[str, Any]]:
        """
        Расчет трансформации сигнала при прохождении через канал
        
        Returns:
            Tuple[modified_signal, remaining_strength, diagnostics]
        """
        diagnostics = {
            "channel_id": self.id,
            "base_strength": self.current_strength,
            "resonance_factor": self.resonance_factor,
            "distance": distance
        }
        
        # Копируем сигнал для модификации
        modified_signal = signal_package.copy()
        
        # Расчет потерь
        distance_loss = 0.1 * (distance - 1)
        load_loss = (self.current_load / self.max_bandwidth) * 0.3
        resonance_gain = (self.resonance_factor - 1.0) * 0.2
        
        total_loss = max(0.0, distance_loss + load_loss - resonance_gain)
        remaining_strength = 1.0 - total_loss
        
        # Применение потерь к силе сигнала
        if hasattr(modified_signal, 'strength'):
            modified_signal.strength *= remaining_strength
        
        # Модификация payload на основе характеристик канала
        if hasattr(modified_signal, 'payload'):
            # Добавление метаданных о прохождении через канал
            channel_info = {
                "channel_id": self.id,
                "hebrew_letter": self.hebrew_letter,
                "strength_impact": remaining_strength,
                "resonance_impact": self.resonance_factor,
                "timestamp": datetime.utcnow().isoformat()
            }
            
            if "channel_history" not in modified_signal.payload:
                modified_signal.payload["channel_history"] = []
            modified_signal.payload["channel_history"].append(channel_info)
            
            # Усиление/ослабление определенных типов сигналов
            if modified_signal.type == SignalType.EMOTIONAL:
                # Эмоциональные сигналы усиливаются резонансом
                if "intensity" in modified_signal.payload:
                    modified_signal.payload["intensity"] *= (1.0 + (self.resonance_factor - 1.0) * 0.5)
            
            elif modified_signal.type == SignalType.INTENTION:
                # Намерения усиливаются силой канала
                if "strength" in modified_signal.payload:
                    modified_signal.payload["strength"] *= self.current_strength
        
        diagnostics.update({
            "total_loss": total_loss,
            "remaining_strength": remaining_strength,
            "modified_signal_type": modified_signal.type.value
        })
        
        return modified_signal, remaining_strength, diagnostics
    
    async def update_from_transmission(self, success: bool, latency: float, 
                                      final_strength: float, signal_type: SignalType):
        """Обновление параметров канала на основе результата передачи"""
        self.total_transmissions += 1
        
        if success:
            self.successful_transmissions += 1
            
            # Усиление канала при успешной передаче
            learning_adjustment = self.learning_rate * final_strength
            
            # Разное обучение для разных типов сигналов
            if signal_type == SignalType.QUANTUM_SYNC:
                learning_adjustment *= 1.5  # Квантовая синхронизация учит быстрее
            elif signal_type == SignalType.EMOTIONAL:
                learning_adjustment *= 1.2  # Эмоции также хорошо учат
            
            self.current_strength = min(1.0, self.current_strength + learning_adjustment)
            self.resonance_factor = min(2.0, self.resonance_factor + learning_adjustment * 0.5)
            
        else:
            self.failed_transmissions += 1
            
            # Ослабление при неудаче, но не слишком резкое
            penalty = self.learning_rate * 0.5
            self.current_strength = max(0.1, self.current_strength - penalty)
            self.resonance_factor = max(0.1, self.resonance_factor - penalty * 0.3)
        
        # Обновление средней латенции
        if self.avg_latency == 0:
            self.avg_latency = latency
        else:
            self.avg_latency = (self.avg_latency * 0.9) + (latency * 0.1)
        
        # Обновление средней силы
        if self.avg_signal_strength == 0:
            self.avg_signal_strength = final_strength
        else:
            self.avg_signal_strength = (self.avg_signal_strength * 0.9) + (final_strength * 0.1)
        
        # Сохранение в историю
        self.strength_history.append(self.current_strength)
        self.resonance_history.append(self.resonance_factor)
        self.latency_history.append(latency)
        
        self.last_used = datetime.utcnow()
        
        # Автоматическая оптимизация каждые 100 передач
        if self.total_transmissions % 100 == 0:
            await self.auto_optimize()
    
    async def auto_optimize(self):
        """Автоматическая оптимизация параметров канала"""
        if len(self.strength_history) < 10:
            return
        
        # Анализ трендов
        recent_strengths = list(self.strength_history)[-10:]
        avg_recent = statistics.mean(recent_strengths)
        
        # Если сила снижается, пробуем увеличить learning rate
        if avg_recent < self.current_strength * 0.9:
            self.learning_rate = min(0.1, self.learning_rate * 1.1)
        
        # Если сила стабильна, уменьшаем learning rate для стабилизации
        elif abs(avg_recent - self.current_strength) < 0.05:
            self.learning_rate = max(0.001, self.learning_rate * 0.9)
        
        # Автоматическое восстановление если канал почти мертв
        if self.current_strength < 0.2 and self.resonance_factor < 0.3:
            await self.emergency_recovery()
        
        self.last_optimized = datetime.utcnow()
    
    async def emergency_recovery(self):
        """Экстренное восстановление канала"""
        print(f"[CHANNEL] Экстренное восстановление канала {self.id}")
        
        # Сброс до базовых значений с небольшим усилением
        self.current_strength = self.base_strength * 1.1
        self.resonance_factor = 1.0
        self.learning_rate = 0.02  # Увеличиваем скорость обучения
        
        # Очистка части истории
        if len(self.strength_history) > 50:
            self.strength_history = deque(list(self.strength_history)[-50:], maxlen=100)
        
        # Временное увеличение пропускной способности
        old_bandwidth = self.max_bandwidth
        self.max_bandwidth = int(self.max_bandwidth * 1.5)
        
        print(f"[CHANNEL] Канал {self.id} восстановлен. Bandwidth: {old_bandwidth} → {self.max_bandwidth}")
    
    def get_health_report(self) -> Dict[str, Any]:
        """Отчет о здоровье канала"""
        success_rate = (
            self.successful_transmissions / self.total_transmissions 
            if self.total_transmissions > 0 else 0
        )
        
        # Анализ стабильности
        stability = 0.0
        if len(self.strength_history) > 5:
            recent_strengths = list(self.strength_history)[-5:]
            stability = 1.0 - statistics.stdev(recent_strengths)
        
        return {
            "channel_id": self.id,
            "hebrew_letter": self.hebrew_letter,
            "path": f"{self.from_sephira} → {self.to_sephira}",
            "is_active": self.is_active,
            "current_strength": self.current_strength,
            "resonance_factor": self.resonance_factor,
            "load_percentage": (self.current_load / self.max_bandwidth) * 100,
            "success_rate": success_rate,
            "total_transmissions": self.total_transmissions,
            "avg_latency": self.avg_latency,
            "avg_signal_strength": self.avg_signal_strength,
            "stability": stability,
            "health_score": self.calculate_health_score(),
            "last_used": self.last_used.isoformat() if self.last_used else None,
            "recommendations": self.generate_recommendations(),
            "timestamp": datetime.utcnow().isoformat()
        }
    
    def calculate_health_score(self) -> float:
        """Расчет общего показателя здоровья канала"""
        success_rate = (
            self.successful_transmissions / self.total_transmissions 
            if self.total_transmissions > 0 else 0.5
        )
        
        load_factor = 1.0 - (self.current_load / self.max_bandwidth)
        strength_factor = self.current_strength
        resonance_factor = min(1.0, self.resonance_factor)
        
        weights = {
            "success": 0.4,
            "load": 0.2,
            "strength": 0.25,
            "resonance": 0.15
        }
        
        score = (
            success_rate * weights["success"] +
            load_factor * weights["load"] +
            strength_factor * weights["strength"] +
            resonance_factor * weights["resonance"]
        )
        
        return min(max(score, 0.0), 1.0)
    
    def generate_recommendations(self) -> List[str]:
        """Генерация рекомендаций для канала"""
        recommendations = []
        
        health_score = self.calculate_health_score()
        
        if health_score < 0.3:
            recommendations.append("emergency_recovery_needed")
        elif health_score < 0.6:
            recommendations.append("optimization_recommended")
        
        if self.current_load > self.max_bandwidth * 0.8:
            recommendations.append("reduce_load_or_increase_bandwidth")
        
        if self.resonance_factor < 0.5:
            recommendations.append("improve_resonance_with_sync_signals")
        
        if self.successful_transmissions < 10 and self.total_transmissions > 50:
            recommendations.append("investigate_failure_patterns")
        
        return recommendations


class ChannelLoadBalancer:
    """Интеллектуальный балансировщик нагрузки каналов"""
    
    def __init__(self):
        self.selection_history: deque = deque(maxlen=1000)
        self.channel_performance: Dict[str, Dict[str, Any]] = {}
        self.last_rebalance: Optional[datetime] = None
    
    async def select_best_channel(self, available_channels: List[QuantumChannel], 
                                 signal_type: SignalType, signal_strength: float) -> Optional[QuantumChannel]:
        """
        Выбор лучшего канала для передачи с учетом множества факторов
        """
        if not available_channels:
            return None
        
        scored_channels = []
        
        for channel in available_channels:
            # Проверка возможности передачи
            can_transmit, reason, effective_strength = await channel.can_transmit(signal_strength)
            
            if not can_transmit:
                continue
            
            # Расчет скоринга
            score = await self.calculate_channel_score(
                channel, signal_type, effective_strength
            )
            
            scored_channels.append((score, channel, effective_strength, reason))
        
        if not scored_channels:
            return None
        
        # Сортировка по скору (высший скор = лучший канал)
        scored_channels.sort(key=lambda x: x[0], reverse=True)
        
        best_score, best_channel, best_strength, best_reason = scored_channels[0]
        
        # Запись в историю выбора
        self.selection_history.append({
            "timestamp": datetime.utcnow().isoformat(),
            "channel_id": best_channel.id,
            "signal_type": signal_type.value,
            "score": best_score,
            "strength": best_strength,
            "reason": best_reason,
            "alternatives": len(scored_channels) - 1
        })
        
        # Обновление статистики производительности
        await self.update_channel_performance(best_channel.id, best_score)
        
        return best_channel
    
    async def calculate_channel_score(self, channel: QuantumChannel, 
                                     signal_type: SignalType, effective_strength: float) -> float:
        """
        Расчет скоринга канала по множеству параметров
        """
        # Базовый скор на основе эффективной силы
        base_score = effective_strength * 0.3
        
        # Скор на основе истории успешности
        success_rate = (
            channel.successful_transmissions / channel.total_transmissions 
            if channel.total_transmissions > 0 else 0.5
        )
        success_score = success_rate * 0.25
        
        # Скор на основе резонанса
        resonance_score = min(1.0, channel.resonance_factor) * 0.2
        
        # Скор на основе нагрузки (меньше нагрузка = лучше)
        load_factor = 1.0 - (channel.current_load / channel.max_bandwidth)
        load_score = load_factor * 0.15
        
        # Скор на основе латенции (меньше латенция = лучше)
        latency_factor = 1.0 / (1.0 + channel.avg_latency) if channel.avg_latency > 0 else 0.5
        latency_score = latency_factor * 0.1
        
        # Бонусы/штрафы для типов сигналов
        type_bonus = 0.0
        
        if signal_type == SignalType.QUANTUM_SYNC and channel.resonance_factor > 1.2:
            type_bonus = 0.2  # Квантовые синхронизации любят высокий резонанс
        
        elif signal_type == SignalType.EMOTIONAL and channel.current_strength > 0.8:
            type_bonus = 0.15  # Эмоции любят сильные каналы
        
        elif signal_type == SignalType.INTENTION and success_rate > 0.8:
            type_bonus = 0.1  # Намерения любят надежные каналы
        
        # Итоговый скор
        total_score = (
            base_score + 
            success_score + 
            resonance_score + 
            load_score + 
            latency_score + 
            type_bonus
        )
        
        # Гарантируем диапазон 0-1
        return min(max(total_score, 0.0), 1.0)
    
    async def update_channel_performance(self, channel_id: str, score: float):
        """Обновление статистики производительности канала"""
        if channel_id not in self.channel_performance:
            self.channel_performance[channel_id] = {
                "scores": deque(maxlen=100),
                "selections": 0,
                "avg_score": 0.0,
                "last_selected": None
            }
        
        perf = self.channel_performance[channel_id]
        perf["scores"].append(score)
        perf["selections"] += 1
        perf["avg_score"] = statistics.mean(perf["scores"]) if perf["scores"] else 0.0
        perf["last_selected"] = datetime.utcnow().isoformat()
    
    async def rebalance_load(self, all_channels: List[QuantumChannel], 
                            target_utilization: float = 0.7):
        """
        Ребалансировка нагрузки между каналами
        """
        now = datetime.utcnow()
        
        # Не чаще чем раз в 5 минут
        if (self.last_rebalance and 
            (now - self.last_rebalance).total_seconds() < 300):
            return
        
        # Анализ текущей нагрузки
        channel_loads = []
        for channel in all_channels:
            utilization = channel.current_load / channel.max_bandwidth
            channel_loads.append((channel.id, utilization, channel.max_bandwidth))
        
        # Сортировка по утилизации
        channel_loads.sort(key=lambda x: x[1])
        
        # Если разница между самым загруженным и самым свободным > 30%
        if channel_loads:
            min_load = channel_loads[0][1]
            max_load = channel_loads[-1][1]
            
            if max_load - min_load > 0.3:
                # Ребалансировка: перераспределяем часть пропускной способности
                print(f"[BALANCER] Ребалансировка нагрузки: min={min_load:.2f}, max={max_load:.2f}")
                
                for i, (channel_id, utilization, bandwidth) in enumerate(channel_loads):
                    channel = next((c for c in all_channels if c.id == channel_id), None)
                    if channel:
                        # Увеличиваем пропускную способность перегруженных каналов
                        if utilization > target_utilization:
                            new_bandwidth = int(bandwidth * 1.1)
                            channel.max_bandwidth = min(200, new_bandwidth)
                            print(f"  ↑ {channel.id}: {bandwidth} → {channel.max_bandwidth}")
                        
                        # Уменьшаем у очень свободных
                        elif utilization < target_utilization * 0.5:
                            new_bandwidth = int(bandwidth * 0.9)
                            channel.max_bandwidth = max(50, new_bandwidth)
                            print(f"  ↓ {channel.id}: {bandwidth} → {channel.max_bandwidth}")
        
        self.last_rebalance = now
    
    def get_balancing_report(self) -> Dict[str, Any]:
        """Отчет о балансировке"""
        if not self.channel_performance:
            return {"total_channels": 0, "avg_selection_score": 0}
        
        avg_scores = [
            perf["avg_score"] 
            for perf in self.channel_performance.values() 
            if perf["avg_score"] > 0
        ]
        
        selection_counts = [
            perf["selections"] 
            for perf in self.channel_performance.values()
        ]
        
        return {
            "total_channels_tracked": len(self.channel_performance),
            "total_selections_recorded": len(self.selection_history),
            "avg_selection_score": statistics.mean(avg_scores) if avg_scores else 0,
            "min_selection_score": min(avg_scores) if avg_scores else 0,
            "max_selection_score": max(avg_scores) if avg_scores else 0,
            "most_selected_channels": sorted(
                [(cid, perf["selections"]) for cid, perf in self.channel_performance.items()],
                key=lambda x: x[1],
                reverse=True
            )[:5],
            "least_selected_channels": sorted(
                [(cid, perf["selections"]) for cid, perf in self.channel_performance.items()],
                key=lambda x: x[1]
            )[:5],
            "rebalance_last_performed": self.last_rebalance.isoformat() if self.last_rebalance else None,
            "timestamp": datetime.utcnow().isoformat()
        }


class SignalTracer:
    """Продвинутая система трассировки сигналов"""
    
    def __init__(self):
        self.traces: Dict[str, 'SignalTrace'] = {}
        self.trace_index: Dict[str, List[str]] = defaultdict(list)  # node -> trace_ids
        self.completed_traces: deque = deque(maxlen=1000)
        
    def create_trace(self, signal_package: SignalPackage, source_node: str) -> 'SignalTrace':
        """Создание новой трассировки"""
        trace_id = self._generate_trace_id(signal_package, source_node)
        
        trace = SignalTrace(
            id=trace_id,
            signal_package=signal_package,
            source_node=source_node,
            start_time=datetime.utcnow()
        )
        
        self.traces[trace_id] = trace
        self.trace_index[source_node].append(trace_id)
        
        return trace
    
    def _generate_trace_id(self, signal_package: SignalPackage, source_node: str) -> str:
        """Генерация уникального ID трассировки"""
        content = f"{source_node}_{signal_package.type}_{signal_package.id}_{datetime.utcnow().timestamp()}"
        return hashlib.sha256(content.encode()).hexdigest()[:16]
    
    async def add_hop(self, trace_id: str, channel: QuantumChannel, 
                     node: SephiroticNode, processing_time: float, 
                     output_strength: float):
        """Добавление шага в трассировку"""
        if trace_id not in self.traces:
            return
        
        trace = self.traces[trace_id]
        
        hop = {
            "timestamp": datetime.utcnow().isoformat(),
            "channel_id": channel.id,
            "channel_letter": channel.hebrew_letter,
            "from_node": channel.from_sephira,
            "to_node": channel.to_sephira,
            "node_status": node.status.value if node else "unknown",
            "node_resonance": node.resonance if hasattr(node, 'resonance') else 0.0,
            "processing_time": processing_time,
            "output_strength": output_strength,
            "channel_strength": channel.current_strength,
            "channel_resonance": channel.resonance_factor,
            "channel_load": channel.current_load
        }
        
        trace.hops.append(hop)
        
        # Индексация по узлу
        if channel.to_sephira:
            self.trace_index[channel.to_sephira].append(trace_id)
    
    def complete_trace(self, trace_id: str, success: bool, 
                      final_node: str = None, error: str = None):
        """Завершение трассировки"""
        if trace_id not in self.traces:
            return
        
        trace = self.traces[trace_id]
        trace.end_time = datetime.utcnow()
        trace.success = success
        trace.final_node = final_node
        trace.error = error
        
        # Расчет статистик
        if trace.hops:
            trace.total_hops = len(trace.hops)
            trace.total_duration = (trace.end_time - trace.start_time).total_seconds()
            trace.avg_processing_time = statistics.mean(
                [hop["processing_time"] for hop in trace.hops]
            )
            trace.min_strength = min([hop["output_strength"] for hop in trace.hops])
            trace.max_strength = max([hop["output_strength"] for hop in trace.hops])
            
            # Определение узких мест
            bottlenecks = []
            for hop in trace.hops:
                if hop["output_strength"] < 0.3:
                    bottlenecks.append({
                        "channel": hop["channel_letter"],
                        "strength": hop["output_strength"],
                        "reason": "low_strength"
                    })
                elif hop["processing_time"] > 1.0:
                    bottlenecks.append({
                        "channel": hop["channel_letter"],
                        "processing_time": hop["processing_time"],
                        "reason": "high_latency"
                    })
            
            trace.bottlenecks = bottlenecks
        
        # Перемещение в завершенные
        self.completed_traces.append(trace)
        
        # Удаление из активных (но храним по ID для быстрого доступа)
        # Не удаляем полностью, чтобы можно было запрашивать по ID
    
    def get_trace(self, trace_id: str) -> Optional[Dict[str, Any]]:
        """Получение трассировки по ID"""
        if trace_id in self.traces:
            return self.traces[trace_id].to_dict()
        return None
    
    def get_node_traces(self, node_name: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Получение трассировок связанных с узлом"""
        trace_ids = self.trace_index.get(node_name, [])[-limit:]
        traces = []
        
        for trace_id in trace_ids:
            if trace_id in self.traces:
                traces.append(self.traces[trace_id].to_dict())
        
        return traces
    
    def get_recent_traces(self, limit: int = 20) -> List[Dict[str, Any]]:
        """Получение последних трассировок"""
        recent = list(self.completed_traces)[-limit:]
        return [trace.to_dict() for trace in recent]
    
    def analyze_trace_patterns(self) -> Dict[str, Any]:
        """Анализ паттернов в трассировках"""
        if not self.completed_traces:
            return {"total_traces": 0}
        
        traces = list(self.completed_traces)
        
        # Статистики успешности
        successful = [t for t in traces if t.success]
        failed = [t for t in traces if not t.success]
        
        # Анализ по типам сигналов
        by_type = defaultdict(list)
        for trace in traces:
            by_type[trace.signal_package.type.value].append(trace)
        
        type_stats = {}
        for sig_type, type_traces in by_type.items():
            if type_traces:
                success_rate = len([t for t in type_traces if t.success]) / len(type_traces)
                avg_hops = statistics.mean([t.total_hops for t in type_traces]) if type_traces else 0
                avg_duration = statistics.mean([t.total_duration for t in type_traces]) if type_traces else 0
                
                type_stats[sig_type] = {
                    "count": len(type_traces),
                    "success_rate": success_rate,
                    "avg_hops": avg_hops,
                    "avg_duration": avg_duration
                }
        
        # Анализ узких мест
        all_bottlenecks = []
        for trace in traces:
            all_bottlenecks.extend(trace.bottlenecks)
        
        bottleneck_stats = defaultdict(int)
        for bottleneck in all_bottlenecks:
            key = f"{bottleneck.get('channel', 'unknown')}_{bottleneck.get('reason', 'unknown')}"
            bottleneck_stats[key] += 1
        
        return {
            "total_traces": len(traces),
            "successful_traces": len(successful),
            "failed_traces": len(failed),
            "overall_success_rate": len(successful) / len(traces) if traces else 0,
            "by_signal_type": type_stats,
            "common_bottlenecks": dict(sorted(bottleneck_stats.items(), key=lambda x: x[1], reverse=True)[:10]),
            "avg_hops_all": statistics.mean([t.total_hops for t in traces]) if traces else 0,
            "avg_duration_all": statistics.mean([t.total_duration for t in traces]) if traces else 0,
            "timestamp": datetime.utcnow().isoformat()
        }


@dataclass
class SignalTrace:
    """Структура трассировки сигнала"""
    id: str
    signal_package: SignalPackage
    source_node: str
    start_time: datetime
    end_time: Optional[datetime] = None
    success: bool = False
    final_node: Optional[str] = None
    error: Optional[str] = None
    hops: List[Dict[str, Any]] = field(default_factory=list)
    total_hops: int = 0
    total_duration: float = 0.0
    avg_processing_time: float = 0.0
    min_strength: float = 1.0
    max_strength: float = 1.0
    bottlenecks: List[Dict[str, Any]] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Конвертация в словарь"""
        return {
            "trace_id": self.id,
            "signal_id": self.signal_package.id,
            "signal_type": self.signal_package.type.value,
            "source_node": self.source_node,
            "target_node": self.signal_package.target,
            "start_time": self.start_time.isoformat(),
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "duration": self.total_duration,
            "success": self.success,
            "final_node": self.final_node,
            "error": self.error,
            "total_hops": self.total_hops,
            "avg_processing_time": self.avg_processing_time,
            "min_strength": self.min_strength,
            "max_strength": self.max_strength,
            "bottlenecks": self.bottlenecks,
            "hops": self.hops[-10:] if self.hops else [],  # Последние 10 шагов
            "hop_count": len(self.hops)
        }


class SephiroticBus:
    """СОВЕРШЕННАЯ сефиротическая шина с полной асинхронностью и интеллектом"""
    
    def __init__(self, config_file: str = "config/sephirot_channels.yaml"):
        # Ядро
        self.nodes: Dict[str, SephiroticNode] = {}
        self.channels: Dict[str, QuantumChannel] = {}
        self.channel_connections: Dict[str, List[str]] = defaultdict(list)  # node -> channel_ids
        
        # Подсистемы
        self.tracer = SignalTracer()
        self.load_balancer = ChannelLoadBalancer()
        self.feedback_processor = FeedbackProcessor(self)
        
        # Очереди
        self.signal_queue = asyncio.PriorityQueue(maxsize=10000)
        self.feedback_queue = asyncio.Queue(maxsize=5000)
        
        # Метрики
        self.metrics = BusMetrics()
        self.health_monitor = BusHealthMonitor(self)
        
        # Фоновые задачи
        self.background_tasks: List[asyncio.Task] = []
        self.is_running = False
        
        # Инициализация
        self._load_full_channel_config(config_file)
        self._init_background_services()
        
        print(f"[BUS] 🌳 Инициализирована совершенная сефиротическая шина")
        print(f"[BUS] 📊 Каналов: {len(self.channels)} | Макс. очередь: {self.signal_queue.maxsize}")
    
    def _load_full_channel_config(self, config_file: str):
        """Загрузка полной конфигурации 22 каналов"""
        try:
            # Попытка загрузки из YAML
            with open(config_file, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
                channels_config = config.get('channels', [])
                
                for chan_config in channels_config:
                    channel = QuantumChannel(**chan_config)
                    self.channels[channel.id] = channel
                    
                    # Создание связей
                    self.channel_connections[channel.from_sephira].append(channel.id)
                    if channel.direction in [ChannelDirection.BIDIRECTIONAL, ChannelDirection.REVERSE]:
                        self.channel_connections[channel.to_sephira].append(channel.id)
                
                print(f"[BUS] Загружено {len(self.channels)} каналов из конфига")              
