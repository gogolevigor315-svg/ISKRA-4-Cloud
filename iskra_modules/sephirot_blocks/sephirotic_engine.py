# sephirotic_engine.py - ПОЛНАЯ ОПТИМИЗИРОВАННАЯ ВЕРСИЯ В ОДНОМ ФАЙЛЕ

import asyncio
import importlib
import inspect
import json
import hashlib
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Set, Tuple
import statistics
from dataclasses import dataclass, field
from collections import deque, defaultdict
import random

from .sephirot_bus import SephiroticBus
from .sephirot_base import SephiroticNode, NodeStatus, SignalPackage, SignalType, NodeMetrics


# ============================================================================
# ВСПОМОГАТЕЛЬНЫЕ КЛАССЫ (ВСТРОЕННЫЕ В ОДИН ФАЙЛ)
# ============================================================================

@dataclass
class IntegrationLink:
    """Структура для связи сефирота с модулем ISKRA"""
    sephirot_name: str
    module_name: str
    link_type: str
    active: bool = True
    last_sync: Optional[datetime] = None
    sync_frequency: float = 5.0
    performance_score: float = 1.0


class EmotionalEnergyModel:
    """Модель эмоциональной энергии по типам сигналов"""
    
    def __init__(self):
        # Энергетические затраты на обработку разных типов сигналов
        self.energy_costs = {
            SignalType.INTENTION: 0.3,     # Намерения требуют воли
            SignalType.EMOTIONAL: 0.2,     # Эмоции требуют эмпатии
            SignalType.COMMAND: 0.4,       # Команды требуют власти
            SignalType.DATA: 0.1,          # Данные требуют внимания
            SignalType.HEARTBEAT: 0.05,    # Сердцебиение - базовые затраты
            SignalType.QUANTUM_SYNC: 0.25  # Квантовая синхронизация
        }
        
        # Восстановление энергии по типам деятельности
        self.energy_recovery = {
            "rest": 0.15,      # Пассивный отдых
            "meditation": 0.3, # Медитация/созерцание
            "harmony": 0.25,   # Гармоничное взаимодействие
            "creation": 0.2,   # Созидание
        }
        
        self.energy_history = deque(maxlen=1000)
    
    def calculate_cost(self, signal_type: SignalType, intensity: float = 1.0) -> float:
        """Расчет энергетических затрат на обработку сигнала"""
        base_cost = self.energy_costs.get(signal_type, 0.1)
        return base_cost * intensity
    
    def calculate_recovery(self, activity_type: str, duration: float = 1.0) -> float:
        """Расчет восстановления энергии"""
        recovery_rate = self.energy_recovery.get(activity_type, 0.1)
        return recovery_rate * duration
    
    def record_energy_flow(self, node_name: str, energy_change: float, 
                          reason: str, signal_type: Optional[SignalType] = None):
        """Запись энергетического потока"""
        self.energy_history.append({
            "timestamp": datetime.utcnow().isoformat(),
            "node": node_name,
            "energy_change": energy_change,
            "reason": reason,
            "signal_type": signal_type.value if signal_type else None,
            "total_energy": None  # Будет заполнено позже
        })


@dataclass
class NetworkFatigue:
    """Расширенная модель усталости с эмоциональной энергией"""
    
    fatigue_level: float = 0.0
    recovery_rate: float = 0.1
    fatigue_threshold: float = 0.8
    rest_mode: bool = False
    
    # Эмоциональная энергетика
    emotional_energy: Dict[str, float] = field(default_factory=lambda: {
        "will": 1.0,      # Воля (Kether)
        "wisdom": 1.0,    # Мудрость (Chokmah)
        "understanding": 1.0,  # Понимание (Binah)
        "mercy": 1.0,     # Милосердие (Chesed)
        "severity": 1.0,  # Строгость (Gevurah)
        "beauty": 1.0,    # Красота (Tiferet)
        "victory": 1.0,   # Победа (Netzach)
        "glory": 1.0,     # Слава (Hod)
        "foundation": 1.0, # Основание (Yesod)
        "kingdom": 1.0    # Царство (Malkuth)
    })
    
    fatigue_history: deque = field(default_factory=lambda: deque(maxlen=1000))
    signal_volume_window: deque = field(default_factory=lambda: deque(maxlen=60))
    emotional_energy_history: Dict[str, deque] = field(default_factory=lambda: {
        key: deque(maxlen=500) for key in [
            "will", "wisdom", "understanding", "mercy", "severity",
            "beauty", "victory", "glory", "foundation", "kingdom"
        ]
    })
    
    def update(self, signal_count: int, energy_model: EmotionalEnergyModel, 
               processed_signals: List[Tuple[str, SignalType]] = None,
               time_delta: float = 1.0) -> None:
        """Обновление усталости с учетом эмоциональной энергии"""
        
        # 1. Обновление базовой усталости
        self.signal_volume_window.append(signal_count)
        avg_signals = statistics.mean(self.signal_volume_window) if self.signal_volume_window else 0
        load_factor = min(avg_signals / 100.0, 1.0)
        
        # 2. Расчет эмоционального истощения
        emotional_drain = 0.0
        if processed_signals:
            for node_name, signal_type in processed_signals:
                # Определяем тип эмоциональной энергии для узла
                energy_type = self._map_node_to_energy(node_name)
                if energy_type in self.emotional_energy:
                    # Затраты на обработку сигнала
                    cost = energy_model.calculate_cost(signal_type)
                    self.emotional_energy[energy_type] = max(
                        0.0, self.emotional_energy[energy_type] - cost * 0.1
                    )
                    emotional_drain += cost * 0.05
                    
                    # Запись истории
                    self.emotional_energy_history[energy_type].append({
                        "timestamp": datetime.utcnow().isoformat(),
                        "energy": self.emotional_energy[energy_type],
                        "drain": cost * 0.1,
                        "signal_type": signal_type.value,
                        "node": node_name
                    })
        
        # 3. Восстановление в режиме отдыха
        if self.rest_mode:
            self.fatigue_level = max(0.0, self.fatigue_level - self.recovery_rate * 2 * time_delta)
            
            # Восстановление эмоциональной энергии
            for energy_type in self.emotional_energy:
                self.emotional_energy[energy_type] = min(
                    1.0, self.emotional_energy[energy_type] + 0.05 * time_delta
                )
            
            if self.fatigue_level < 0.3 and all(e > 0.5 for e in self.emotional_energy.values()):
                self.rest_mode = False
        
        else:
            # 4. Накопление усталости с учетом эмоционального истощения
            fatigue_increase = (load_factor * 0.05 + emotional_drain * 0.3) * time_delta
            self.fatigue_level = min(1.0, self.fatigue_level + fatigue_increase)
            
            # Автоматический переход в режим отдыха при высоком истощении
            if (self.fatigue_level > self.fatigue_threshold or 
                any(e < 0.2 for e in self.emotional_energy.values())):
                self.rest_mode = True
        
        # 5. Запись истории
        self.fatigue_history.append({
            "timestamp": datetime.utcnow().isoformat(),
            "fatigue": self.fatigue_level,
            "load": load_factor,
            "emotional_drain": emotional_drain,
            "rest_mode": self.rest_mode,
            "emotional_energy": self.emotional_energy.copy()
        })
    
    def _map_node_to_energy(self, node_name: str) -> str:
        """Сопоставление узла с типом эмоциональной энергии"""
        mapping = {
            "Kether": "will",
            "Chokmah": "wisdom",
            "Binah": "understanding",
            "Chesed": "mercy",
            "Gevurah": "severity",
            "Tiferet": "beauty",
            "Netzach": "victory",
            "Hod": "glory",
            "Yesod": "foundation",
            "Malkuth": "kingdom"
        }
        return mapping.get(node_name, "will")
    
    def get_emotional_balance_report(self) -> Dict[str, Any]:
        """Отчет об эмоциональном балансе системы"""
        return {
            "overall_fatigue": self.fatigue_level,
            "rest_mode": self.rest_mode,
            "emotional_energies": self.emotional_energy,
            "lowest_energy": min(self.emotional_energy.items(), key=lambda x: x[1]) if self.emotional_energy else None,
            "highest_energy": max(self.emotional_energy.items(), key=lambda x: x[1]) if self.emotional_energy else None,
            "energy_stability": statistics.stdev(self.emotional_energy.values()) if len(self.emotional_energy) > 1 else 0,
            "recommendation": self._generate_energy_recommendation(),
            "timestamp": datetime.utcnow().isoformat()
        }
    
    def _generate_energy_recommendation(self) -> str:
        """Генерация рекомендации по энергетическому балансу"""
        if self.rest_mode:
            return "continue_rest"
        
        low_energies = [(k, v) for k, v in self.emotional_energy.items() if v < 0.3]
        if low_energies:
            return f"focus_on_{low_energies[0][0]}_recovery"
        
        if self.fatigue_level > 0.7:
            return "reduce_workload"
        
        return "operational_normal"


class QuantumLinkValidator:
    """Валидатор квантовых связей с продвинутым мониторингом"""
    
    def __init__(self, inactive_threshold: int = 50):
        self.inactive_threshold = inactive_threshold
        self.validation_history: Dict[str, Dict[str, Any]] = {}
        self.link_metrics: Dict[str, Dict[str, Any]] = {}
        self.last_cleanup = datetime.utcnow()
    
    async def validate_link(self, link: Any, source_node: str, 
                           bus: Optional['SephiroticBus'] = None) -> Tuple[bool, str, Dict[str, Any]]:
        """Продвинутая валидация квантовой связи"""
        
        link_id = f"{source_node}->{link.target_node}"
        
        # 1. Проверка базовых атрибутов
        if not hasattr(link, 'target_node'):
            return False, "missing_target_attribute", {}
        
        # 2. Проверка времени последней активности
        current_time = datetime.utcnow()
        
        if hasattr(link, 'last_activity_time'):
            last_active = link.last_activity_time
            if isinstance(last_active, str):
                last_active = datetime.fromisoformat(last_active)
        elif hasattr(link, 'last_active_cycle'):
            # Конвертация циклов во время (предполагаем 1 цикл = 2 секунды)
            cycles_inactive = self.cycle_counter - link.last_active_cycle
            inactive_seconds = cycles_inactive * 2
            last_active = current_time - timedelta(seconds=inactive_seconds)
        else:
            last_active = current_time - timedelta(days=1)  # Помечаем как старую
        
        # 3. Расчет метрик
        inactive_duration = (current_time - last_active).total_seconds()
        cycles_inactive = int(inactive_duration / 2)  # Пример: 1 цикл = 2 секунды
        
        # 4. Проверка резонанса и качества связи
        resonance_ok = True
        quality_metrics = {}
        
        if hasattr(link, 'resonance_strength'):
            resonance_ok = link.resonance_strength > 0.1
            quality_metrics['resonance'] = link.resonance_strength
        
        if hasattr(link, 'latency'):
            quality_metrics['latency'] = link.latency
            resonance_ok = resonance_ok and link.latency < 5.0  # Макс 5 секунд задержки
        
        # 5. Проверка через шину (если доступна)
        if bus and hasattr(bus, 'check_link_health'):
            bus_health = await bus.check_link_health(source_node, link.target_node)
            quality_metrics['bus_health'] = bus_health
            resonance_ok = resonance_ok and bus_health.get('healthy', False)
        
        # 6. Определение статуса
        is_active = (cycles_inactive < self.inactive_threshold) and resonance_ok
        
        # 7. Сбор детальной диагностики
        diagnostics = {
            'link_id': link_id,
            'source': source_node,
            'target': link.target_node,
            'last_active': last_active.isoformat(),
            'inactive_seconds': inactive_duration,
            'cycles_inactive': cycles_inactive,
            'resonance_ok': resonance_ok,
            'quality_metrics': quality_metrics,
            'is_active': is_active,
            'validation_time': current_time.isoformat(),
            'recommendation': 'keep' if is_active else 'remove'
        }
        
        # 8. Обновление истории
        self.validation_history[link_id] = diagnostics
        
        if link_id not in self.link_metrics:
            self.link_metrics[link_id] = {
                'validations': [],
                'uptime_percentage': 100.0,
                'avg_resonance': quality_metrics.get('resonance', 0.5),
                'failure_count': 0,
                'success_count': 0
            }
        
        metrics = self.link_metrics[link_id]
        metrics['validations'].append({
            'time': current_time.isoformat(),
            'active': is_active,
            'resonance': quality_metrics.get('resonance', 0.5)
        })
        
        if is_active:
            metrics['success_count'] += 1
        else:
            metrics['failure_count'] += 1
        
        # Обновление uptime
        total_validations = metrics['success_count'] + metrics['failure_count']
        metrics['uptime_percentage'] = (
            metrics['success_count'] / total_validations * 100 
            if total_validations > 0 else 0
        )
        
        # 9. Очистка старых записей
        if (current_time - self.last_cleanup).total_seconds() > 3600:  # Каждый час
            await self._cleanup_old_records()
            self.last_cleanup = current_time
        
        return is_active, "active" if is_active else f"inactive_{cycles_inactive}_cycles", diagnostics
    
    async def _cleanup_old_records(self):
        """Очистка старых записей валидации"""
        cutoff_time = datetime.utcnow() - timedelta(hours=24)
        
        # Очистка истории валидации
        old_keys = [
            link_id for link_id, data in self.validation_history.items()
            if datetime.fromisoformat(data['validation_time']) < cutoff_time
        ]
        for key in old_keys:
            del self.validation_history[key]
        
        # Очистка метрик (храним только последние 1000 записей на связь)
        for link_id in self.link_metrics:
            if 'validations' in self.link_metrics[link_id]:
                validations = self.link_metrics[link_id]['validations']
                if len(validations) > 1000:
                    self.link_metrics[link_id]['validations'] = validations[-1000:]
    
    def get_link_health_report(self) -> Dict[str, Any]:
        """Отчет о здоровье всех связей"""
        active_links = []
        inactive_links = []
        warning_links = []
        
        for link_id, diagnostics in self.validation_history.items():
            if diagnostics['is_active']:
                if diagnostics['quality_metrics'].get('resonance', 1.0) < 0.3:
                    warning_links.append(diagnostics)
                else:
                    active_links.append(diagnostics)
            else:
                inactive_links.append(diagnostics)
        
        return {
            'total_links': len(self.validation_history),
            'active_links': len(active_links),
            'inactive_links': len(inactive_links),
            'warning_links': len(warning_links),
            'uptime_stats': self._calculate_uptime_stats(),
            'recommendations': self._generate_link_recommendations(active_links, warning_links, inactive_links),
            'timestamp': datetime.utcnow().isoformat()
        }
    
    def _calculate_uptime_stats(self) -> Dict[str, float]:
        """Расчет статистики uptime"""
        if not self.link_metrics:
            return {'average': 0, 'min': 0, 'max': 0}
        
        uptimes = [metrics['uptime_percentage'] for metrics in self.link_metrics.values()]
        return {
            'average': statistics.mean(uptimes) if uptimes else 0,
            'min': min(uptimes) if uptimes else 0,
            'max': max(uptimes) if uptimes else 0,
            'std_dev': statistics.stdev(uptimes) if len(uptimes) > 1 else 0
        }
    
    def _generate_link_recommendations(self, active: List, warning: List, inactive: List) -> List[str]:
        """Генерация рекомендаций по связям"""
        recommendations = []
        
        if inactive:
            recommendations.append(f"remove_{len(inactive)}_inactive_links")
        
        if warning:
            recommendations.append(f"investigate_{len(warning)}_low_resonance_links")
        
        if len(active) < 3 and len(self.validation_history) > 5:
            recommendations.append("create_more_connections_for_redundancy")
        
        return recommendations


class PrometheusMetricsExporter:
    """Экспортер метрик в формате Prometheus"""
    
    def __init__(self):
        self.metrics = {}
        self.last_export = datetime.utcnow()
    
    def register_metric(self, name: str, metric_type: str = "gauge", 
                       help_text: str = "", labels: Dict[str, str] = None):
        """Регистрация метрики"""
        if name not in self.metrics:
            self.metrics[name] = {
                'type': metric_type,
                'help': help_text,
                'labels': labels or {},
                'values': {},
                'history': deque(maxlen=1000)
            }
    
    def update_metric(self, name: str, value: float, 
                     labels: Dict[str, str] = None, timestamp: datetime = None):
        """Обновление значения метрики"""
        if name not in self.metrics:
            self.register_metric(name, "gauge", f"Auto-registered metric: {name}")
        
        label_key = self._labels_to_key(labels or {})
        self.metrics[name]['values'][label_key] = {
            'value': value,
            'labels': labels or {},
            'timestamp': timestamp or datetime.utcnow()
        }
        
        # Сохранение в историю
        self.metrics[name]['history'].append({
            'value': value,
            'labels': labels or {},
            'timestamp': timestamp or datetime.utcnow()
        })
    
    def _labels_to_key(self, labels: Dict[str, str]) -> str:
        """Конвертация лейблов в ключ"""
        return json.dumps(sorted(labels.items()), sort_keys=True)
    
    def generate_prometheus_output(self) -> str:
        """Генерация вывода в формате Prometheus"""
        output_lines = []
        
        for metric_name, metric_data in self.metrics.items():
            # HELP строка
            if metric_data['help']:
                output_lines.append(f"# HELP {metric_name} {metric_data['help']}")
            
            # TYPE строка
            output_lines.append(f"# TYPE {metric_name} {metric_data['type']}")
            
            # Значения
            for label_key, value_data in metric_data['values'].items():
                value = value_data['value']
                labels = value_data['labels']
                
                if labels:
                    label_str = ",".join([f'{k}="{v}"' for k, v in labels.items()])
                    output_lines.append(f'{metric_name}{{{label_str}}} {value}')
                else:
                    output_lines.append(f'{metric_name} {value}')
            
            output_lines.append("")  # Пустая строка между метриками
        
        self.last_export = datetime.utcnow()
        return "\n".join(output_lines)
    
    def get_metric_history(self, name: str, label_filter: Dict[str, str] = None, 
                          time_range: timedelta = None) -> List[Dict[str, Any]]:
        """Получение истории метрики"""
        if name not in self.metrics:
            return []
        
        history = list(self.metrics[name]['history'])
        
        # Фильтрация по времени
        if time_range:
            cutoff = datetime.utcnow() - time_range
            history = [h for h in history if h['timestamp'] >= cutoff]
        
        # Фильтрация по лейблам
        if label_filter:
            history = [
                h for h in history
                if all(h['labels'].get(k) == v for k, v in label_filter.items())
            ]
        
        return history
    
    def get_system_summary_metrics(self) -> Dict[str, Any]:
        """Генерация сводных метрик системы"""
        summary = {
            'metrics_registered': len(self.metrics),
            'total_data_points': sum(len(m['history']) for m in self.metrics.values()),
            'last_export': self.last_export.isoformat(),
            'metric_types': defaultdict(int)
        }
        
        for metric_data in self.metrics.values():
            summary['metric_types'][metric_data['type']] += 1
        
        return summary


class QuantumSyncManager:
    """Менеджер квантовой синхронизации состояний"""
    
    def __init__(self, bus: 'SephiroticBus'):
        self.bus = bus
        self.sync_groups: Dict[str, Set[str]] = defaultdict(set)
        self.sync_history: deque = deque(maxlen=1000)
        self.last_sync: Dict[str, datetime] = {}
        
        # Группы синхронизации по путям Древа
        self._init_sync_groups()
    
    def _init_sync_groups(self):
        """Инициализация групп синхронизации"""
        # Триада верхних сефирот
        self.sync_groups['supernal_triad'] = {'Kether', 'Chokmah', 'Binah'}
        
        # Триада эмоциональная
        self.sync_groups['emotional_triad'] = {'Chesed', 'Gevurah', 'Tiferet'}
        
        # Триада операционная
        self.sync_groups['operational_triad'] = {'Netzach', 'Hod', 'Yesod'}
        
        # Вертикальные колонны
        self.sync_groups['pillar_of_mercy'] = {'Kether', 'Chokmah', 'Chesed', 'Netzach'}
        self.sync_groups['pillar_of_severity'] = {'Kether', 'Binah', 'Gevurah', 'Hod'}
        self.sync_groups['pillar_of_mildness'] = {'Kether', 'Tiferet', 'Yesod', 'Malkuth'}
    
    async def broadcast_state_sync(self, source_node: str, 
                                  state_data: Dict[str, Any]) -> Dict[str, Any]:
        """Широковещательная синхронизация состояния"""
        
        # 1. Определение группы синхронизации
        sync_group = None
        for group_name, nodes in self.sync_groups.items():
            if source_node in nodes:
                sync_group = group_name
                break
        
        if not sync_group:
            sync_group = 'all'  # Синхронизация со всеми
        
        # 2. Подготовка синхро-сигнала
        sync_signal = SignalPackage(
            source=source_node,
            target=f"SYNC_GROUP:{sync_group}",
            type=SignalType.QUANTUM_SYNC,
            payload={
                "sync_type": "state_broadcast",
                "source_state": state_data,
                "sync_group": sync_group,
                "sync_timestamp": datetime.utcnow().isoformat(),
                "resonance_required": 0.7
            }
        )
        
        # 3. Отправка через шину
        results = await self.bus.broadcast_quantum_sync(sync_signal, sync_group)
        
        # 4. Анализ результатов
        successful_syncs = []
        failed_syncs = []
        
        for target_node, result in results.items():
            if result.get('sync_successful', False):
                successful_syncs.append(target_node)
                
                # Обновление резонанса на основе успешной синхронизации
                if 'resonance_gain' in result:
                    # Здесь можно обновить резонанс узла
                    pass
            else:
                failed_syncs.append({
                    'node': target_node,
                    'error': result.get('error', 'unknown'),
                    'resonance_level': result.get('current_resonance', 0)
                })
        
        # 5. Запись в историю
        sync_record = {
            'timestamp': datetime.utcnow().isoformat(),
            'source': source_node,
            'sync_group': sync_group,
            'successful': successful_syncs,
            'failed': failed_syncs,
            'total_nodes': len(results),
            'success_rate': len(successful_syncs) / max(len(results), 1),
            'state_summary': {
                k: str(type(v)).split("'")[1] if not isinstance(v, (int, float, str, bool)) else v
                for k, v in state_data.items()
            }
        }
        
        self.sync_history.append(sync_record)
        self.last_sync[source_node] = datetime.utcnow()
        
        return {
            'sync_completed': True,
            'sync_group': sync_group,
            'successful_nodes': successful_syncs,
            'failed_nodes': failed_syncs,
            'success_rate': sync_record['success_rate'],
            'resonance_impact': self._calculate_resonance_impact(successful_syncs, failed_syncs),
            'next_recommended_sync': self._calculate_next_sync_time(source_node),
            'timestamp': datetime.utcnow().isoformat()
        }
    
    def _calculate_resonance_impact(self, successful: List[str], failed: List[Dict]) -> float:
        """Расчет влияния синхронизации на резонанс"""
        total_nodes = len(successful) + len(failed)
        if total_nodes == 0:
            return 0.0
        
        base_gain = len(successful) / total_nodes * 0.1
        
        # Дополнительный бонус за синхронизацию внутри группы
        group_bonus = 0.0
        for node in successful:
            # Проверка, находится ли узел в той же группе, что и другие успешные
            for group_name, nodes in self.sync_groups.items():
                if node in nodes:
                    group_members = [n for n in successful if n in nodes]
                    if len(group_members) > 1:
                        group_bonus += 0.05 * (len(group_members) - 1)
        
        return min(base_gain + group_bonus, 0.3)
    
    def _calculate_next_sync_time(self, node_name: str) -> Optional[datetime]:
        """Расчет времени следующей синхронизации"""
        if node_name not in self.last_sync:
            return datetime.utcnow() + timedelta(seconds=30)
        
        last_time = self.last_sync[node_name]
        
        # Динамический интервал на основе истории
        recent_syncs = [
            record for record in self.sync_history
            if record['source'] == node_name
            and datetime.fromisoformat(record['timestamp']) > datetime.utcnow() - timedelta(hours=1)
        ]
        
        if not recent_syncs:
            return last_time + timedelta(seconds=60)
        
        avg_success_rate = statistics.mean([r['success_rate'] for r in recent_syncs])
        
        # Адаптивный интервал: лучше синхронизация -> чаще
        if avg_success_rate > 0.8:
            interval = 30  # 30 секунд
        elif avg_success_rate > 0.5:
            interval = 60  # 60 секунд
        else:
            interval = 120  # 120 секунд
        
        return last_time + timedelta(seconds=interval)
    
    async def perform_group_sync(self, group_name: str, 
                                initiator: str = None) -> Dict[str, Any]:
        """Синхронизация всей группы"""
        if group_name not in self.sync_groups:
            return {'error': f'Group {group_name} not found'}
        
        group_nodes = self.sync_groups[group_name]
        if not initiator:
            initiator = next(iter(group_nodes))
        
        # Сбор состояний всех узлов группы
        group_states = {}
        for node_name in group_nodes:
            # Здесь должен быть метод получения состояния узла
            group_states[node_name] = {'status': 'unknown', 'resonance': 0.5}
        
        # Создание общего состояния группы
        group_state = {
            'group_name': group_name,
            'nodes': list(group_nodes),
            'average_resonance': statistics.mean(
                [s.get('resonance', 0.5) for s in group_states.values()]
            ) if group_states else 0.5,
            'state_diversity': self._calculate_state_diversity(group_states),
            'sync_timestamp': datetime.utcnow().isoformat()
        }
        
        # Синхронизация группы с общим состоянием
        return await self.broadcast_state_sync(initiator, group_state)
    
    def _calculate_state_diversity(self, states: Dict[str, Dict]) -> float:
        """Расчет разнообразия состояний в группе"""
        if not states:
            return 0.0
        
        resonances = [s.get('resonance', 0.5) for s in states.values()]
        if len(resonances) < 2:
            return 0.0
        
        # Мера разнообразия = стандартное отклонение нормализованное
        std_dev = statistics.stdev(resonances)
        max_possible_std = 0.5  # Максимальное возможное std при диапазоне 0-1
        return min(std_dev / max_possible_std, 1.0)
    
    def get_sync_analytics(self) -> Dict[str, Any]:
        """Аналитика синхронизаций"""
        if not self.sync_history:
            return {'total_syncs': 0, 'average_success_rate': 0}
        
        recent_syncs = list(self.sync_history)
        
        # Группировка по источникам
        by_source = defaultdict(list)
        for sync in recent_syncs:
            by_source[sync['source']].append(sync)
        
        # Расчет статистик
        source_stats = {}
        for source, syncs in by_source.items():
            success_rates = [s['success_rate'] for s in syncs]
            source_stats[source] = {
                'total_syncs': len(syncs),
                'avg_success_rate': statistics.mean(success_rates) if success_rates else 0,
                'last_sync': max([datetime.fromisoformat(s['timestamp']) for s in syncs]).isoformat() if syncs else None,
                'preferred_groups': self._find_preferred_sync_groups(syncs)
            }
        
        # Общая статистика
        all_success_rates = [s['success_rate'] for s in recent_syncs]
        
        return {
            'total_syncs': len(recent_syncs),
            'time_range': {
                'first': recent_syncs[0]['timestamp'] if recent_syncs else None,
                'last': recent_syncs[-1]['timestamp'] if recent_syncs else None
            },
            'average_success_rate': statistics.mean(all_success_rates) if all_success_rates else 0,
            'by_source': source_stats,
            'sync_group_effectiveness': self._calculate_group_effectiveness(recent_syncs),
            'recommendations': self._generate_sync_recommendations(source_stats),
            'timestamp': datetime.utcnow().isoformat()
        }
    
    def _find_preferred_sync_groups(self, syncs: List[Dict]) -> List[str]:
        """Нахождение предпочтительных групп синхронизации"""
        group_counts = defaultdict(int)
        for sync in syncs:
            group_counts[sync['sync_group']] += 1
        
        # Сортировка по частоте использования
        return [group for group, count in sorted(
            group_counts.items(), key=lambda x: x[1], reverse=True
        )[:3]]  # Топ-3 группы
    
    def _calculate_group_effectiveness(self, syncs: List[Dict]) -> Dict[str, float]:
        """Расчет эффективности групп синхронизации"""
        group_success = defaultdict(list)
        
        for sync in syncs:
            group_success[sync['sync_group']].append(sync['success_rate'])
        
        return {
            group: statistics.mean(success_rates) if success_rates else 0
            for group, success_rates in group_success.items()
        }
    
    def _generate_sync_recommendations(self, source_stats: Dict[str, Dict]) -> List[str]:
        """Генерация рекомендаций по синхронизации"""
        recommendations = []
        
        for source, stats in source_stats.items():
            if stats['avg_success_rate'] < 0.5:
                recommendations.append(f"improve_{source}_sync_strategy")
            
            if stats['total_syncs'] < 5:
                recommendations.append(f"increase_{source}_sync_frequency")
        
        # Проверка баланса синхронизации
        sync_counts = [s['total_syncs'] for s in source_stats.values()]
        if sync_counts:
            max_sync = max(sync_counts)
            min_sync = min(sync_counts)
            if max_sync > min_sync * 3 and min_sync > 0:
                recommendations.append("balance_sync_distribution")
        
        return recommendations


# ============================================================================
