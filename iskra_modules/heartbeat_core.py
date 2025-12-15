# ==============================================================
# 🫀 HEARTBEAT_SYSTEM v2.0 — СИСТЕМА СЕРДЕЧНОГО РИТМА ISKRA-4
# ДЕТЕРМИНИРОВАННЫЙ СЕФИРОТИЧЕСКИЙ ПУЛЬС АРХИТЕКТУРЫ
# ==============================================================

import asyncio
import time
import numpy as np
import json
import logging
from datetime import datetime
from typing import Dict, List, Any, Optional, Union
from dataclasses import dataclass, field
from enum import Enum

# Настройка логгера модуля
logger = logging.getLogger(__name__)

# ==============================================================
# КОНФИГУРАЦИЯ МОДУЛЯ ДЛЯ ISKRA-4
# ==============================================================

MODULE_VERSION = "2.0-iskra-integrated"
MODULE_NAME = "heartbeat_system"

class HeartState(Enum):
    """Состояния сердечного ритма"""
    HARMONIOUS = "гармоничный"
    RESONANT = "резонансный"
    STABILIZING = "стабилизация"
    ARRHYTHMIC = "аритмия"
    COHERENT = "когерентный"
    TRANSCENDENT = "трансцендентный"

class PulsePhase(Enum):
    """Фазы сердечного цикла"""
    SYSTOLE = "систола"      # Сокращение, эмиссия
    DIASTOLE = "диастола"    # Расслабление, прием
    RESONANCE = "резонанс"   # Пик синхронизации
    REFRACTORY = "рефрактерный" # Восстановление

# ==============================================================
# СЕФИРОТИЧЕСКАЯ СИНХРОНИЗАЦИЯ
# ==============================================================

@dataclass
class SephiroticRhythm:
    """Ритмический паттерн для сефиры"""
    sephira: str
    frequency: float          # Базовая частота (Гц)
    amplitude: float          # Амплитуда влияния
    phase_offset: float       # Смещение фазы (радианы)
    emotional_weight: float   # Эмоциональный вес
    cognitive_weight: float   # Когнитивный вес
    ethical_weight: float     # Этический вес
    
    def get_contribution(self, time: float) -> float:
        """Вклад сефиры в момент времени"""
        return (
            self.amplitude * 
            np.sin(time * self.frequency * 2 * np.pi + self.phase_offset)
        )

class SephiroticSynchronizer:
    """Синхронизатор сефиротических ритмов"""
    
    SEPHIROTIC_RHYTHMS = {
        "Kether": SephiroticRhythm(
            sephira="Kether",
            frequency=0.618,      # Золотое сечение
            amplitude=0.95,
            phase_offset=0.0,
            emotional_weight=0.1,
            cognitive_weight=0.8,
            ethical_weight=0.9
        ),
        "Chokhmah": SephiroticRhythm(
            sephira="Chokhmah",
            frequency=1.0,
            amplitude=0.9,
            phase_offset=np.pi/6,
            emotional_weight=0.2,
            cognitive_weight=0.7,
            ethical_weight=0.8
        ),
        "Binah": SephiroticRhythm(
            sephira="Binah",
            frequency=1.618,      # Фи
            amplitude=0.85,
            phase_offset=np.pi/3,
            emotional_weight=0.3,
            cognitive_weight=0.9,
            ethical_weight=0.95
        ),
        "Chesed": SephiroticRhythm(
            sephira="Chesed",
            frequency=0.8,
            amplitude=0.75,
            phase_offset=np.pi/2,
            emotional_weight=0.7,
            cognitive_weight=0.6,
            ethical_weight=0.85
        ),
        "Gevurah": SephiroticRhythm(
            sephira="Gevurah",
            frequency=1.2,
            amplitude=0.7,
            phase_offset=2*np.pi/3,
            emotional_weight=0.4,
            cognitive_weight=0.7,
            ethical_weight=0.9
        ),
        "Tiphareth": SephiroticRhythm(
            sephira="Tiphareth",
            frequency=1.0,        # Центральный ритм
            amplitude=1.0,
            phase_offset=0.0,
            emotional_weight=0.5,
            cognitive_weight=0.8,
            ethical_weight=0.9
        ),
        "Netzach": SephiroticRhythm(
            sephira="Netzach",
            frequency=1.5,
            amplitude=0.8,
            phase_offset=5*np.pi/6,
            emotional_weight=0.9,
            cognitive_weight=0.5,
            ethical_weight=0.7
        ),
        "Hod": SephiroticRhythm(
            sephira="Hod",
            frequency=2.0,
            amplitude=0.65,
            phase_offset=np.pi,
            emotional_weight=0.3,
            cognitive_weight=0.9,
            ethical_weight=0.8
        ),
        "Yesod": SephiroticRhythm(
            sephira="Yesod",
            frequency=0.9,
            amplitude=0.85,
            phase_offset=7*np.pi/6,
            emotional_weight=0.6,
            cognitive_weight=0.7,
            ethical_weight=0.75
        ),
        "Malkuth": SephiroticRhythm(
            sephira="Malkuth",
            frequency=0.7,
            amplitude=0.9,
            phase_offset=4*np.pi/3,
            emotional_weight=0.8,
            cognitive_weight=0.6,
            ethical_weight=0.85
        )
    }
    
    def __init__(self):
        self.start_time = time.time()
        self.phase_history = []
        logger.info(f"[{MODULE_NAME}] SephiroticSynchronizer initialized")
    
    def calculate_combined_rhythm(self, time_offset: float = 0.0) -> Dict[str, Any]:
        """Расчет комбинированного ритма всех сефирот"""
        current_time = time.time() - self.start_time + time_offset
        
        contributions = {}
        total_amplitude = 0.0
        weighted_phase = 0.0
        
        for name, rhythm in self.SEPHIROTIC_RHYTHMS.items():
            contribution = rhythm.get_contribution(current_time)
            contributions[name] = {
                'value': contribution,
                'amplitude': rhythm.amplitude,
                'frequency': rhythm.frequency,
                'phase': (current_time * rhythm.frequency * 2 * np.pi + rhythm.phase_offset) % (2 * np.pi)
            }
            
            total_amplitude += abs(contribution)
            if abs(contribution) > 0:
                weighted_phase += contributions[name]['phase'] * rhythm.amplitude
        
        # Нормализация
        if total_amplitude > 0:
            for name in contributions:
                contributions[name]['value'] /= total_amplitude
        
        # Расчет когерентности
        coherence = self._calculate_coherence(contributions)
        
        return {
            'timestamp': current_time,
            'contributions': contributions,
            'combined_amplitude': total_amplitude / len(self.SEPHIROTIC_RHYTHMS),
            'weighted_phase': weighted_phase / total_amplitude if total_amplitude > 0 else 0,
            'coherence': coherence,
            'dominant_sephira': max(contributions.items(), key=lambda x: abs(x[1]['value']))[0]
        }
    
    def _calculate_coherence(self, contributions: Dict) -> float:
        """Расчет когерентности между сефиротическими ритмами"""
        values = [c['value'] for c in contributions.values()]
        phases = [c['phase'] for c in contributions.values()]
        
        if len(values) < 2:
            return 1.0
        
        # Когерентность по амплитуде
        amplitude_std = np.std(values)
        amplitude_coherence = 1.0 / (1.0 + amplitude_std)
        
        # Когерентность по фазе
        phase_vector_sum = sum(np.exp(1j * phase) for phase in phases)
        phase_coherence = abs(phase_vector_sum) / len(phases)
        
        return (amplitude_coherence + phase_coherence) / 2

# ==============================================================
# ОСНОВНОЙ КЛАСС СИСТЕМЫ СЕРДЕЧНОГО РИТМА
# ==============================================================

class HeartbeatSystem:
    """Детерминированная система сердечного ритма ISKRA-4"""
    
    def __init__(self, node_id: str = "ISKRA-4-CORE"):
        self.node_id = node_id
        self.sephirotic_sync = SephiroticSynchronizer()
        
        # Основные параметры ритма
        self.base_heart_rate = 1.0  # 60 BPM в единицах системы
        self.current_heart_rate = 1.0
        self.amplitude = 0.7
        self.phase = 0.0
        self.coherence = 0.5
        self.energy_level = 50.0
        
        # Состояния системы
        self.heart_state = HeartState.HARMONIOUS
        self.pulse_phase = PulsePhase.DIASTOLE
        self.is_active = False
        self.start_time = time.time()
        
        # Метрики и мониторинг
        self.metrics = {
            "heart_rate": self.current_heart_rate,
            "amplitude": self.amplitude,
            "coherence": self.coherence,
            "energy": self.energy_level,
            "state": self.heart_state.value,
            "pulse_phase": self.pulse_phase.value,
            "sephirotic_coherence": 0.0,
            "emotional_temperature": 0.5,
            "cognitive_load": 0.3,
            "ethical_integrity": 0.9
        }
        
        # История ритма
        self.rhythm_history = []
        self.max_history_size = 1000
        
        # Ссылки на другие модули (будут установлены при интеграции)
        self.linked_modules = {
            "emotional_weave": None,
            "data_bridge": None,
            "spinal_core": None,
            "sephirotic_mining": None,
            "immune_core": None
        }
        
        # Параметры регуляции
        self.regulation_params = {
            "min_heart_rate": 0.3,
            "max_heart_rate": 2.0,
            "coherence_threshold": 0.7,
            "energy_decay_rate": 0.01,
            "phase_correction_rate": 0.05,
            "amplitude_stabilization": 0.1
        }
        
        # Цикл регуляции
        self.regulation_task = None
        self.regulation_interval = 1.0  # секунды
        
        logger.info(f"🫀 HeartbeatSystem v{MODULE_VERSION} инициализирован для ноды {node_id}")
    
    # ========== ISKRA-4 ИНТЕРФЕЙС ==========
    
    def initialize(self) -> Dict:
        """Инициализация модуля для ISKRA-4"""
        logger.info(f"[{MODULE_NAME}] Module initialized for ISKRA-4")
        return {
            "status": "active",
            "version": MODULE_VERSION,
            "node_id": self.node_id,
            "heart_rate": self.current_heart_rate,
            "coherence": self.coherence,
            "energy": self.energy_level
        }
    
    def process_command(self, command: str, data: Dict = None) -> Dict:
        """Обработка команд ISKRA-4"""
        data = data or {}
        
        command_map = {
            "start": self.start_heartbeat,
            "stop": self.stop_heartbeat,
            "status": self.get_status,
            "pulse": self.single_pulse,
            "sync": self.synchronize,
            "coherence": self.get_coherence_report,
            "diagnostic": self.get_diagnostic_report,
            "visualize": self.visualize_rhythm,
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
    
    def start_heartbeat(self, data: Dict = None) -> Dict:
        """Запуск сердечного ритма"""
        if self.is_active:
            return {"status": "already_active", "heart_rate": self.current_heart_rate}
        
        self.is_active = True
        self.regulation_task = asyncio.create_task(self._regulation_cycle())
        
        logger.info(f"💓 Сердечный ритм запущен: {self.current_heart_rate:.2f} BPM")
        
        return {
            "status": "started",
            "heart_rate": self.current_heart_rate,
            "phase": self.pulse_phase.value,
            "state": self.heart_state.value
        }
    
    def stop_heartbeat(self, data: Dict = None) -> Dict:
        """Остановка сердечного ритма"""
        if not self.is_active:
            return {"status": "already_stopped"}
        
        self.is_active = False
        if self.regulation_task:
            self.regulation_task.cancel()
        
        logger.info("⏸️ Сердечный ритм остановлен")
        
        return {
            "status": "stopped",
            "final_heart_rate": self.current_heart_rate,
            "total_duration": time.time() - self.start_time
        }
    
    def get_status(self, data: Dict = None) -> Dict:
        """Получить статус системы"""
        return {
            "node": self.node_id,
            "active": self.is_active,
            "heart_rate": round(self.current_heart_rate, 3),
            "amplitude": round(self.amplitude, 3),
            "coherence": round(self.coherence, 3),
            "energy": round(self.energy_level, 2),
            "state": self.heart_state.value,
            "phase": self.pulse_phase.value,
            "uptime": round(time.time() - self.start_time, 1),
            "history_size": len(self.rhythm_history),
            "linked_modules": list(self.linked_modules.keys())
        }
    
    def single_pulse(self, data: Dict = None) -> Dict:
        """Выполнить одиночный импульс"""
        pulse_data = self._generate_pulse()
        
        # Отправка пульса связанным модулям
        if self.linked_modules:
            self._emit_pulse_to_modules(pulse_data)
        
        return pulse_data
    
    def synchronize(self, data: Dict) -> Dict:
        """Синхронизация с внешними данными"""
        external_coherence = data.get('coherence', 0.5)
        external_energy = data.get('energy', 0.0)
        
        # Адаптивная синхронизация
        sync_factor = 0.3
        self.coherence = self.coherence * (1 - sync_factor) + external_coherence * sync_factor
        self.energy_level += external_energy * 0.1
        
        # Обновление состояния
        self._update_heart_state()
        
        return {
            "sync_status": "completed",
            "new_coherence": round(self.coherence, 3),
            "new_energy": round(self.energy_level, 2)
        }
    
    def get_coherence_report(self, data: Dict = None) -> Dict:
        """Отчет о когерентности"""
        sephirotic_data = self.sephirotic_sync.calculate_combined_rhythm()
        
        return {
            "system_coherence": self.coherence,
            "sephirotic_coherence": sephirotic_data['coherence'],
            "dominant_sephira": sephirotic_data['dominant_sephira'],
            "combined_amplitude": sephirotic_data['combined_amplitude'],
            "phase_alignment": sephirotic_data['weighted_phase'],
            "contributions": {
                k: round(v['value'], 3) 
                for k, v in sephirotic_data['contributions'].items()
            }
        }
    
    def get_diagnostic_report(self, data: Dict = None) -> Dict:
        """Полный диагностический отчет"""
        # Анализ истории ритма
        rhythm_analysis = self._analyze_rhythm_history()
        
        # Проверка состояния
        health_checks = {
            "heart_rate_in_range": (
                self.regulation_params["min_heart_rate"] <= self.current_heart_rate <= 
                self.regulation_params["max_heart_rate"]
            ),
            "coherence_adequate": self.coherence >= 0.5,
            "energy_sufficient": self.energy_level > 20.0,
            "state_stable": self.heart_state not in [HeartState.ARRHYTHMIC],
            "history_consistent": len(self.rhythm_history) > 10
        }
        
        # Предупреждения
        warnings = []
        if self.current_heart_rate > 1.5:
            warnings.append("Высокая частота сердечного ритма")
        if self.coherence < 0.4:
            warnings.append("Низкая когерентность ритма")
        if self.energy_level < 30.0:
            warnings.append("Низкий уровень энергии")
        
        return {
            "timestamp": datetime.utcnow().isoformat(),
            "module": MODULE_NAME,
            "version": MODULE_VERSION,
            "node": self.node_id,
            "health_checks": health_checks,
            "current_state": self.heart_state.value,
            "metrics": self.metrics,
            "rhythm_analysis": rhythm_analysis,
            "warnings": warnings,
            "active_modules": [
                name for name, module in self.linked_modules.items() 
                if module is not None
            ],
            "regulation_active": self.is_active
        }
    
    def visualize_rhythm(self, data: Dict = None) -> Dict:
        """Данные для визуализации ритма"""
        window = data.get('window', 100)
        recent_history = self.rhythm_history[-min(window, len(self.rhythm_history)):]
        
        if not recent_history:
            return {"error": "Нет данных для визуализации"}
        
        timestamps = [h['timestamp'] for h in recent_history]
        heart_rates = [h['heart_rate'] for h in recent_history]
        amplitudes = [h['amplitude'] for h in recent_history]
        phases = [h['phase'] for h in recent_history]
        
        # Цвета на основе когерентности
        colors = []
        for h in recent_history:
            coherence = h['coherence']
            if coherence > 0.8:
                colors.append('#00ff88')  # Ярко-зеленый
            elif coherence > 0.6:
                colors.append('#88ff00')  # Лаймовый
            elif coherence > 0.4:
                colors.append('#ffff00')  # Желтый
            elif coherence > 0.2:
                colors.append('#ff8800')  # Оранжевый
            else:
                colors.append('#ff0000')  # Красный
        
        return {
            "time_series": {
                "timestamps": timestamps,
                "heart_rates": heart_rates,
                "amplitudes": amplitudes,
                "phases": phases,
                "colors": colors
            },
            "current": {
                "heart_rate": self.current_heart_rate,
                "amplitude": self.amplitude,
                "phase": self.phase,
                "coherence": self.coherence
            },
            "statistics": {
                "avg_heart_rate": np.mean(heart_rates) if heart_rates else 0,
                "std_heart_rate": np.std(heart_rates) if len(heart_rates) > 1 else 0,
                "min_heart_rate": min(heart_rates) if heart_rates else 0,
                "max_heart_rate": max(heart_rates) if heart_rates else 0,
                "trend": self._calculate_trend(heart_rates)
            }
        }
    
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
        old_params = self.regulation_params.copy()
        
        for key, value in data.items():
            if key in self.regulation_params:
                self.regulation_params[key] = float(value)
                logger.info(f"⚙️ Параметр {key} изменен: {old_params[key]} → {value}")
        
        return {
            "status": "adjusted",
            "old_parameters": old_params,
            "new_parameters": self.regulation_params
        }
    
    # ========== ВНУТРЕННИЕ МЕТОДЫ ==========
    
    async def _regulation_cycle(self):
        """Цикл регуляции сердечного ритма"""
        logger.info("🔄 Цикл регуляции сердечного ритма запущен")
        
        cycle_count = 0
        while self.is_active:
            try:
                cycle_count += 1
                
                # 1. Обновление сефиротической синхронизации
                sephirotic_data = self.sephirotic_sync.calculate_combined_rhythm()
                
                # 2. Коррекция частоты на основе сефиротического ритма
                rhythm_influence = sephirotic_data['combined_amplitude']
                frequency_correction = (rhythm_influence - 0.5) * 0.2
                self.current_heart_rate = max(
                    self.regulation_params["min_heart_rate"],
                    min(
                        self.regulation_params["max_heart_rate"],
                        self.current_heart_rate + frequency_correction
                    )
                )
                
                # 3. Обновление когерентности
                self.coherence = sephirotic_data['coherence']
                self.metrics['sephirotic_coherence'] = self.coherence
                
                # 4. Обновление фазы
                self.phase = (self.phase + self.current_heart_rate * 2 * np.pi * 
                            self.regulation_interval) % (2 * np.pi)
                
                # 5. Определение фазы пульса
                self._update_pulse_phase()
                
                # 6. Генерация пульса
                pulse_data = self._generate_pulse()
                
                # 7. Отправка пульса модулям (каждые 5 циклов)
                if cycle_count % 5 == 0 and self.linked_modules:
                    self._emit_pulse_to_modules(pulse_data)
                
                # 8. Логирование состояния
                self._log_rhythm_state(pulse_data)
                
                # 9. Обновление метрик
                self._update_metrics()
                
                # 10. Обновление состояния
                self._update_heart_state()
                
                # 11. Ожидание следующего цикла
                await asyncio.sleep(self.regulation_interval)
                
            except asyncio.CancelledError:
                logger.info("Цикл регуляции остановлен")
                break
            except Exception as e:
                logger.error(f"Ошибка в цикле регуляции: {e}")
                await asyncio.sleep(self.regulation_interval * 2)
    
    def _generate_pulse(self) -> Dict:
        """Генерация данных пульса"""
        pulse_strength = self.amplitude * self.coherence
        
        # Энергетический баланс
        energy_consumption = pulse_strength * 0.1
        energy_recovery = (1.0 - self.coherence) * 0.05
        self.energy_level = max(0.0, self.energy_level - energy_consumption + energy_recovery)
        
        pulse_data = {
            "timestamp": time.time(),
            "node": self.node_id,
            "heart_rate": self.current_heart_rate,
            "amplitude": self.amplitude,
            "strength": pulse_strength,
            "phase": self.phase,
            "coherence": self.coherence,
            "energy": self.energy_level,
            "state": self.heart_state.value,
            "pulse_phase": self.pulse_phase.value
        }
        
        return pulse_data
    
    def _update_pulse_phase(self):
        """Обновление фазы сердечного цикла на основе текущей фазы"""
        phase_normalized = self.phase / (2 * np.pi)
        
        if phase_normalized < 0.25:
            self.pulse_phase = PulsePhase.SYSTOLE
        elif phase_normalized < 0.5:
            self.pulse_phase = PulsePhase.RESONANCE
        elif phase_normalized < 0.75:
            self.pulse_phase = PulsePhase.DIASTOLE
        else:
            self.pulse_phase = PulsePhase.REFRACTORY
    
    def _update_heart_state(self):
        """Обновление состояния сердца на основе метрик"""
        if self.coherence > 0.8 and self.energy_level > 70:
            self.heart_state = HeartState.TRANSCENDENT
        elif self.coherence > 0.7:
            self.heart_state = HeartState.COHERENT
        elif self.coherence > 0.5:
            self.heart_state = HeartState.HARMONIOUS
        elif abs(self.current_heart_rate - self.base_heart_rate) > 0.5:
            self.heart_state = HeartState.ARRHYTHMIC
        else:
            self.heart_state = HeartState.STABILIZING
    
    def _update_metrics(self):
        """Обновление всех метрик"""
        self.metrics.update({
            "heart_rate": self.current_heart_rate,
            "amplitude": self.amplitude,
            "coherence": self.coherence,
            "energy": self.energy_level,
            "state": self.heart_state.value,
            "pulse_phase": self.pulse_phase.value,
            "emotional_temperature": 0.5 + (self.coherence - 0.5) * 0.3,
            "cognitive_load": 0.3 + (1.0 - self.coherence) * 0.4,
            "ethical_integrity": 0.9 - (1.0 - self.coherence) * 0.2
        })
    
    def _log_rhythm_state(self, pulse_data: Dict):
        """Логирование состояния ритма"""
        self.rhythm_history.append(pulse_data)
        
        # Ограничение размера истории
        if len(self.rhythm_history) > self.max_history_size:
            self.rhythm_history.pop(0)
        
        # Периодическое логирование
        if len(self.rhythm_history) % 50 == 0:
            logger.debug(
                f"💓 Ритм: {self.current_heart_rate:.2f} BPM, "
                f"Когерентность: {self.coherence:.3f}, "
                f"Энергия: {self.energy_level:.1f}"
            )
    
    def _emit_pulse_to_modules(self, pulse_data: Dict):
        """Отправка пульса связанным модулям"""
        for module_name, module_ref in self.linked_modules.items():
            if module_ref:
                try:
                    # Эмуляция отправки - в реальной системе будет прямой вызов
                    if hasattr(module_ref, 'receive_heartbeat'):
                        module_ref.receive_heartbeat(pulse_data)
                        logger.debug(f"Пульс отправлен в {module_name}")
                except Exception as e:
                    logger.error(f"Ошибка отправки пульса в {module_name}: {e}")
    
    def _analyze_rhythm_history(self) -> Dict:
        """Анализ истории ритма"""
        if len(self.rhythm_history) < 10:
            return {"status": "insufficient_data", "history_size": len(self.rhythm_history)}
        
        heart_rates = [h['heart_rate'] for h in self.rhythm_history[-100:]]
        coherences = [h['coherence'] for h in self.rhythm_history[-100:]]
        
        return {
            "history_size": len(self.rhythm_history),
            "avg_heart_rate": round(np.mean(heart_rates), 3),
            "std_heart_rate": round(np.std(heart_rates), 3),
            "avg_coherence": round(np.mean(coherences), 3),
            "heart_rate_trend": self._calculate_trend(heart_rates),
            "coherence_trend": self._calculate_trend(coherences),
            "stability_index": round(1.0 / (1.0 + np.std(heart_rates)), 3)
        }
    
    def _calculate_trend(self, data: List[float]) -> str:
        """Расчет тренда данных"""
        if len(data) < 2:
            return "недостаточно данных"
        
        try:
            x = np.arange(len(data))
            slope = np.polyfit(x, data, 1)[0]
            
            if slope > 0.01:
                return "растет ↗️"
            elif slope < -0.01:
                return "падает ↘️"
            else:
                return "стабильно ➡️"
        except Exception:
            return "не определено"

# ==============================================================
# ТЕСТИРОВАНИЕ МОДУЛЯ
# ==============================================================

async def test_heartbeat_system():
    """Тестирование системы сердечного ритма"""
    print("🧪 ТЕСТИРОВАНИЕ HEARTBEAT_SYSTEM v2.0")
    print("=" * 60)
    
    # Инициализация
    heartbeat = HeartbeatSystem("Test-Node-Alpha")
    
    # 1. Инициализация ISKRA-4
    print("\n1. 🔧 Инициализация модуля...")
    init_result = heartbeat.initialize()
    print(f"   Статус: {init_result['status']}")
    print(f"   Версия: {init_result['version']}")
    
    # 2. Запуск сердечного ритма
    print("\n2. 💓 Запуск сердечного ритма...")
    start_result = heartbeat.start_heartbeat()
    print(f"   Статус: {start_result['status']}")
    print(f"   Частота: {start_result['heart_rate']:.2f} BPM")
    print(f"   Фаза: {start_result['phase']}")
    
    # 3. Работа в течение нескольких секунд
    print("\n3. ⏱️ Работа системы (5 секунд)...")
    await asyncio.sleep(5)
    
    # 4. Проверка статуса
    print("\n4. 📊 Проверка статуса...")
    status = heartbeat.get_status()
    print(f"   Активен: {status['active']}")
    print(f"   Частота: {status['heart_rate']} BPM")
    print(f"   Когерентность: {status['coherence']}")
    print(f"   Энергия: {status['energy']}")
    print(f"   Состояние: {status['state']}")
    
    # 5. Отчет о когерентности
    print("\n5. 🔍 Отчет о когерентности...")
    coherence_report = heartbeat.get_coherence_report()
    print(f"   Системная когерентность: {coherence_report['system_coherence']:.3f}")
    print(f"   Сефиротическая когерентность: {coherence_report['sephirotic_coherence']:.3f}")
    print(f"   Доминирующая сефира: {coherence_report['dominant_sephira']}")
    
    # 6. Визуализация ритма
    print("\n6. 📈 Визуализация ритма...")
    viz_data = heartbeat.visualize_rhythm({"window": 50})
    if "error" not in viz_data:
        stats = viz_data['statistics']
        print(f"   Средняя частота: {stats['avg_heart_rate']:.3f} BPM")
        print(f"   Стабильность: {stats['std_heart_rate']:.3f}")
        print(f"   Тренд: {stats['trend']}")
    
    # 7. Диагностический отчет
    print("\n7. 🩺 Диагностический отчет...")
    diagnostic = heartbeat.get_diagnostic_report()
    print(f"   Проверки здоровья: {diagnostic['health_checks']}")
    print(f"   Предупреждения: {len(diagnostic['warnings'])}")
    if diagnostic['warnings']:
        for warning in diagnostic['warnings']:
            print(f"     ⚠️ {warning}")
    
    # 8. Остановка системы
    print("\n8. ⏸️ Остановка системы...")
    stop_result = heartbeat.stop_heartbeat()
    print(f"   Статус: {stop_result['status']}")
    print(f"   Финальная частота: {stop_result['final_heart_rate']:.2f} BPM")
    
    print("\n✅ ТЕСТ ЗАВЕРШЕН УСПЕШНО")
    
    return {
        "heartbeat": heartbeat,
        "init_result": init_result,
        "status": status,
        "coherence_report": coherence_report,
        "diagnostic": diagnostic
    }

# ==============================================================
# ТОЧКА ВХОДА ДЛЯ ТЕСТИРОВАНИЯ
# ==============================================================

if __name__ == "__main__":
    print("🫀 ЗАПУСК СИСТЕМЫ СЕРДЕЧНОГО РИТМА ISKRA-4")
    print("⚡ Детерминированный сефиротический пульс активирован")
    print("=" * 60)
    
    # Запуск теста
    results = asyncio.run(test_heartbeat_system())
    
    print("\n" + "=" * 60)
    print("📊 ИТОГОВАЯ СТАТИСТИКА:")
    print(f"   Модуль: {MODULE_NAME} v{MODULE_VERSION}")
    print(f"   Частота ритма: {results['status']['heart_rate']} BPM")
    print(f"   Когерентность: {results['co
