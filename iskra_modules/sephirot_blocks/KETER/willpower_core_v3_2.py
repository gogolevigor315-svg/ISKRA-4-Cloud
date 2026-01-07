"""
ISKRA-4 · WILLPOWER-CORE v3.2 (Sephirotic Hybrid Will Engine) · KETHERIC BLOCK
Ядро божественной воли Keter - управление энергией, намерением и автономией
"""

import asyncio
import math
import time
import statistics
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Protocol
import logging

# Настройка логирования
logger = logging.getLogger("keter.willpower_core")

# ===============================================================
# I. ИНТЕРФЕЙСЫ ДЛЯ ИНТЕГРАЦИИ
# ===============================================================

class IMoralMemoryLink(Protocol):
    """Связь с MORAL-MEMORY 3.1"""
    async def get_alignment_score(self) -> float: ...
    async def get_ethical_coherence(self) -> float: ...
    async def register_willpower_pattern(self, pattern: Dict) -> bool: ...

class ISpiritCoreLink(Protocol):
    """Связь с духовным ядром"""
    async def get_spiritual_resonance(self) -> float: ...
    async def receive_willpower_boost(self, boost_amount: float) -> bool: ...

class IPolicyGovernorLink(Protocol):
    """Связь с Policy Governor"""
    async def get_willpower_constraints(self) -> Dict[str, float]: ...
    async def report_willpower_metrics(self, metrics: Dict) -> bool: ...

class IKeterIntegration(Protocol):
    """Интеграция с ядром Keter"""
    async def register_willpower_core(self, willpower_instance: Any) -> None: ...
    async def distribute_will_energy(self, target: str, amount: float) -> bool: ...
    async def broadcast_will_state(self, state: Dict) -> bool: ...

# ===============================================================
# II. ВСПОМОГАТЕЛЬНЫЕ МОДУЛИ (адаптированные для Keter)
# ===============================================================

@dataclass
class KeterTemporalDecay:
    """
    Божественное затухание воли Keter
    Учитывает космическую усталость и временные циклы
    """
    cosmic_half_life: float = 120.0  # секунд (увеличен для Keter)
    divine_half_life: float = 300.0   # божественное затухание
    last_update: float = field(default_factory=time.time)
    cosmic_value: float = 1.0
    divine_value: float = 1.0
    decay_history: List[Dict] = field(default_factory=list)
    
    async def calculate_divine_decay(self) -> float:
        """Вычисление божественного затухания воли Keter"""
        now = time.time()
        dt = now - self.last_update
        self.last_update = now
        
        # Космическое затухание (быстрое)
        self.cosmic_value *= 0.5 ** (dt / self.cosmic_half_life)
        
        # Божественное затухание (медленное)
        self.divine_value *= 0.5 ** (dt / self.divine_half_life)
        
        # Комбинированный коэффициент
        decay_factor = (self.cosmic_value * 0.6 + self.divine_value * 0.4)
        
        # Записываем историю
        decay_record = {
            "timestamp": now,
            "cosmic_decay": self.cosmic_value,
            "divine_decay": self.divine_value,
            "combined": decay_factor,
            "time_delta": dt
        }
        self.decay_history.append(decay_record)
        self.decay_history[:] = self.decay_history[-500:]
        
        return max(0.1, min(1.0, decay_factor))
    
    async def reset_divine_will(self):
        """Сброс божественной воли (полное восстановление)"""
        self.cosmic_value = 1.0
        self.divine_value = 1.0
        self.last_update = time.time()
        logger.info("[DECAY] Божественная воля восстановлена")
    
    async def get_decay_statistics(self) -> Dict:
        """Статистика затухания"""
        if not self.decay_history:
            return {"average_decay": 1.0, "stability": 1.0}
        
        recent_decays = [d["combined"] for d in self.decay_history[-20:]]
        avg_decay = statistics.mean(recent_decays)
        
        if len(recent_decays) > 1:
            stability = 1.0 - statistics.stdev(recent_decays)
        else:
            stability = 1.0
        
        return {
            "average_decay": round(avg_decay, 4),
            "decay_stability": round(stability, 4),
            "cosmic_decay": round(self.cosmic_value, 4),
            "divine_decay": round(self.divine_value, 4)
        }

@dataclass
class KeterMoralFilter:
    """
    Фильтр божественного морального выравнивания Keter
    Учитывает этическую когерентность и космическую справедливость
    """
    divine_sensitivity: float = 0.85
    cosmic_justice_factor: float = 0.9
    last_alignment: float = 1.0
    ethical_coherence: float = 0.88
    filter_history: List[Dict] = field(default_factory=list)
    
    async def adjust_divine_alignment(self, new_value: float, moral_source: str = "unknown") -> float:
        """Корректировка божественного морального выравнивания"""
        # Вес в зависимости от источника
        source_weight = {
            "moral_memory": 0.4,
            "policy_governor": 0.3,
            "sephirotic_engine": 0.2,
            "operator": 0.1
        }.get(moral_source, 0.2)
        
        # Плавное обновление с учётом чувствительности
        adjustment = new_value * self.divine_sensitivity * source_weight
        self.last_alignment = (
            0.6 * self.last_alignment + 
            0.4 * adjustment
        )
        
        # Обновление этической когерентности
        alignment_delta = abs(new_value - self.last_alignment)
        self.ethical_coherence = max(0.1, 1.0 - alignment_delta * 0.5)
        
        # Запись в историю
        filter_record = {
            "timestamp": time.time(),
            "new_value": new_value,
            "adjusted_alignment": self.last_alignment,
            "source": moral_source,
            "ethical_coherence": self.ethical_coherence
        }
        self.filter_history.append(filter_record)
        self.filter_history[:] = self.filter_history[-300:]
        
        logger.debug(f"[MORAL-FILTER] Выравнивание: {self.last_alignment:.3f} (источник: {moral_source})")
        return self.last_alignment
    
    async def apply_cosmic_justice(self, justice_level: float):
        """Применение космической справедливости"""
        self.cosmic_justice_factor = justice_level
        self.divine_sensitivity = max(0.5, min(1.0, self.divine_sensitivity * justice_level))
        logger.info(f"[MORAL-FILTER] Космическая справедливость: {justice_level:.3f}")
    
    async def get_moral_statistics(self) -> Dict:
        """Статистика морального фильтра"""
        return {
            "current_alignment": round(self.last_alignment, 4),
            "ethical_coherence": round(self.ethical_coherence, 4),
            "divine_sensitivity": round(self.divine_sensitivity, 4),
            "cosmic_justice": round(self.cosmic_justice_factor, 4),
            "history_size": len(self.filter_history)
        }

# ===============================================================
# III. ГЛАВНОЕ ЯДРО ВОЛИ KETER
# ===============================================================

@dataclass
class WILLPOWER_CORE_v32_KETER:
    """
    Гибридное ядро божественной воли Keter v3.2
    Управление энергией, намерением, фокусом и автономией
    """
    
    def __init__(
        self,
        moral_memory_link: Optional[IMoralMemoryLink] = None,
        spirit_core_link: Optional[ISpiritCoreLink] = None,
        policy_governor_link: Optional[IPolicyGovernorLink] = None,
        keter_integration: Optional[IKeterIntegration] = None
    ):
        self.name = "WILLPOWER-CORE-v3.2"
        self.version = "3.2.0"
        self.role = "divine_will_engine"
        
        # Внешние связи
        self.moral_memory = moral_memory_link
        self.spirit_core = spirit_core_link
        self.policy_governor = policy_governor_link
        self.keter_integration = keter_integration
        
        # Основные компоненты воли Keter
        self.divine_essence: float = 0.85      # Внутренняя божественная сила
        self.cosmic_focus: float = 0.9         # Космическая направленность
        self.sephirotic_autonomy: float = 0.8  # Сефиротическая автономия
        self.operator_trust_link: float = 0.88 # Связь с оператором
        
        # Вспомогательные модули
        self.temporal_decay = KeterTemporalDecay()
        self.moral_filter = KeterMoralFilter()
        
        # Состояние
        self.will_history: List[Dict] = []
        self.last_impulse: float = 0.0
        self.activation_time = time.time()
        self.is_active = False
        self.impulse_count = 0
        
        logger.info(f"[{self.name}] Инициализирован v{self.version}")
    
    async def activate(self) -> bool:
        """Активация ядра божественной воли"""
        try:
            # Регистрация в Keter
            if self.keter_integration:
                await self.keter_integration.register_willpower_core(self)
            
            # Получение начального морального выравнивания
            if self.moral_memory:
                initial_alignment = await self.moral_memory.get_alignment_score()
                await self.moral_filter.adjust_divine_alignment(
                    initial_alignment, 
                    moral_source="moral_memory"
                )
            
            # Получение ограничений от Policy Governor
            if self.policy_governor:
                constraints = await self.policy_governor.get_willpower_constraints()
                await self._apply_constraints(constraints)
            
            self.is_active = True
            self.activation_time = time.time()
            
            # Сброс затухания при активации
            await self.temporal_decay.reset_divine_will()
            
            logger.info(f"[{self.name}] ✅ Ядро божественной воли активировано")
            return True
            
        except Exception as e:
            logger.error(f"[{self.name}] ❌ Ошибка активации: {e}")
            return False
    
    async def _apply_constraints(self, constraints: Dict):
        """Применение ограничений от Policy Governor"""
        if "max_will_strength" in constraints:
            max_strength = constraints["max_will_strength"]
            self.divine_essence = min(self.divine_essence, max_strength)
        
        if "focus_limits" in constraints:
            focus_limits = constraints["focus_limits"]
            self.cosmic_focus = max(
                focus_limits.get("min", 0.1),
                min(self.cosmic_focus, focus_limits.get("max", 1.0))
            )
    
    async def generate_divine_impulse(self, divine_intent: Dict[str, float]) -> Dict:
        """
        Генерация божественного импульса воли Keter
        divine_intent: {
            "cosmic_clarity": 0.0-1.0,
            "divine_purpose": 0.0-1.0,
            "sephirotic_alignment": 0.0-1.0
        }
        """
        if not self.is_active:
            return {"error": "Ядро воли не активировано"}
        
        self.impulse_count += 1
        start_time = time.time()
        
        try:
            # 1. Извлечение компонентов намерения
            cosmic_clarity = divine_intent.get("cosmic_clarity", 0.8)
            divine_purpose = divine_intent.get("divine_purpose", 0.9)
            sephirotic_alignment = divine_intent.get("sephirotic_alignment", 0.85)
            
            # 2. Получение текущего затухания
            decay_factor = await self.temporal_decay.calculate_divine_decay()
            
            # 3. Получение морального выравнивания
            moral_alignment = self.moral_filter.last_alignment
            
            # 4. Получение духовного резонанса (если есть связь)
            spiritual_resonance = 1.0
            if self.spirit_core:
                spiritual_resonance = await self.spirit_core.get_spiritual_resonance()
            
            # 5. Вычисление божественного импульса
            divine_impulse = (
                self.divine_essence * 
                cosmic_clarity * 
                divine_purpose *
                self.cosmic_focus *
                decay_factor *
                moral_alignment *
                spiritual_resonance *
                sephirotic_alignment
            )
            
            # 6. Ограничение и нормализация
            divine_impulse = max(0.01, min(1.0, divine_impulse))
            self.last_impulse = divine_impulse
            
            # 7. Запись состояния
            impulse_record = {
                "timestamp": time.time(),
                "impulse_id": self.impulse_count,
                "divine_impulse": round(divine_impulse, 4),
                "duration": round(time.time() - start_time, 4),
                "components": {
                    "divine_essence": self.divine_essence,
                    "cosmic_focus": self.cosmic_focus,
                    "cosmic_clarity": cosmic_clarity,
                    "divine_purpose": divine_purpose,
                    "decay_factor": round(decay_factor, 4),
                    "moral_alignment": round(moral_alignment, 4),
                    "spiritual_resonance": round(spiritual_resonance, 4),
                    "sephirotic_alignment": sephirotic_alignment
                },
                "intent": divine_intent
            }
            
            self.will_history.append(impulse_record)
            self.will_history[:] = self.will_history[-200:]
            
            # 8. Отправка событий
            if self.keter_integration:
                await self.keter_integration.broadcast_will_state({
                    "impulse": divine_impulse,
                    "impulse_id": self.impulse_count,
                    "timestamp": time.time()
                })
            
            # 9. Отправка буста духовному ядру
            if self.spirit_core and divine_impulse > 0.7:
                boost_amount = divine_impulse * 0.3
                await self.spirit_core.receive_willpower_boost(boost_amount)
            
            # 10. Регистрация паттерна в MORAL-MEMORY
            if self.moral_memory:
                pattern = {
                    "willpower_pattern": "divine_impulse",
                    "impulse_strength": divine_impulse,
                    "moral_context": moral_alignment,
                    "source": self.name
                }
                await self.moral_memory.register_willpower_pattern(pattern)
            
            logger.info(f"[{self.name}] ⚡ Божественный импульс: {divine_impulse:.3f} (ID: {self.impulse_count})")
            
            return impulse_record
            
        except Exception as e:
            logger.error(f"[{self.name}] Ошибка генерации импульса: {e}")
            return {
                "error": str(e),
                "impulse_id": self.impulse_count,
                "timestamp": time.time()
            }
    
    async def adjust_divine_will(self, moral_factor: float, source: str = "unknown") -> Dict:
        """
        Корректировка божественной воли через моральное выравнивание
        Возвращает обновлённое состояние
        """
        # 1. Обновление морального фильтра
        adjusted_alignment = await self.moral_filter.adjust_divine_alignment(
            moral_factor, 
            moral_source=source
        )
        
        # 2. Корректировка фокуса
        focus_adjustment = moral_factor * 0.02
        self.cosmic_focus = max(0.1, min(1.0, 
            self.cosmic_focus * 0.98 + focus_adjustment
        ))
        
        # 3. Корректировка связи с оператором
        trust_adjustment = (self.operator_trust_link + moral_factor) / 2
        self.operator_trust_link = max(0.3, min(1.0, trust_adjustment))
        
        # 4. Обновление божественной сущности (медленно)
        essence_adjustment = moral_factor * 0.005
        self.divine_essence = max(0.5, min(1.0,
            self.divine_essence * 0.995 + essence_adjustment
        ))
        
        adjustment_record = {
            "timestamp": time.time(),
            "moral_factor": moral_factor,
            "source": source,
            "adjusted_alignment": adjusted_alignment,
            "resulting_state": {
                "cosmic_focus": round(self.cosmic_focus, 4),
                "operator_trust": round(self.operator_trust_link, 4),
                "divine_essence": round(self.divine_essence, 4)
            }
        }
        
        logger.info(f"[{self.name}] Корректировка воли (источник: {source}): фокус={self.cosmic_focus:.3f}")
        
        return adjustment_record
    
    async def get_current_divine_strength(self) -> float:
        """Получение текущей силы божественной воли Keter"""
        # 1. Текущее затухание
        decay_factor = await self.temporal_decay.calculate_divine_decay()
        
        # 2. Базовая сила
        base_strength = statistics.mean([
            self.divine_essence,
            self.cosmic_focus,
            self.sephirotic_autonomy,
            self.operator_trust_link
        ])
        
        # 3. Учёт морального выравнивания
        moral_influence = self.moral_filter.last_alignment * 0.3 + 0.7
        
        # 4. Итоговая сила
        divine_strength = base_strength * decay_factor * moral_influence
        
        # 5. Ограничение
        divine_strength = max(0.01, min(1.0, divine_strength))
        
        return round(divine_strength, 4)
    
    async def receive_priority_boost(self, boost_amount: float) -> bool:
        """Получение приоритетного буста от оркестратора"""
        try:
            # Усиление фокуса
            self.cosmic_focus = min(1.0, self.cosmic_focus + boost_amount * 0.2)
            
            # Усиление сущности
            self.divine_essence = min(1.0, self.divine_essence + boost_amount * 0.1)
            
            # Сброс части затухания
            if boost_amount > 0.5:
                self.temporal_decay.cosmic_value = min(1.0, 
                    self.temporal_decay.cosmic_value + 0.3
                )
            
            logger.info(f"[{self.name}] Получен приоритетный буст: {boost_amount:.3f}")
            return True
            
        except Exception as e:
            logger.error(f"[{self.name}] Ошибка обработки буста: {e}")
            return False
    
    async def get_willpower_statistics(self) -> Dict:
        """Полная статистика ядра воли"""
        # Основные метрики
        current_strength = await self.get_current_divine_strength()
        decay_stats = await self.temporal_decay.get_decay_statistics()
        moral_stats = await self.moral_filter.get_moral_statistics()
        
        # Анализ истории импульсов
        impulse_analysis = {"total_impulses": 0, "average_impulse": 0.0}
        if self.will_history:
            impulse_strengths = [i["divine_impulse"] for i in self.will_history[-20:]]
            impulse_analysis = {
                "total_impulses": len(self.will_history),
                "average_impulse": round(statistics.mean(impulse_strengths), 4),
                "recent_trend": "stable"
            }
            
            if len(impulse_strengths) >= 10:
                first_half = statistics.mean(impulse_strengths[:5])
                second_half = statistics.mean(impulse_strengths[-5:])
                impulse_analysis["recent_trend"] = (
                    "increasing" if second_half > first_half * 1.1 else
                    "decreasing" if second_half < first_half * 0.9 else "stable"
                )
        
        # Вычисление здоровья воли
        will_health = (
            current_strength * 0.4 +
            decay_stats["average_decay"] * 0.3 +
            moral_stats["ethical_coherence"] * 0.3
        )
        
        return {
            "name": self.name,
            "version": self.version,
            "active": self.is_active,
            "uptime": round(time.time() - self.activation_time, 2),
            "current_strength": current_strength,
            "will_health": round(will_health, 4),
            "component_states": {
                "divine_essence": round(self.divine_essence, 4),
                "cosmic_focus": round(self.cosmic_focus, 4),
                "sephirotic_autonomy": round(self.sephirotic_autonomy, 4),
                "operator_trust": round(self.operator_trust_link, 4)
            },
            "decay_statistics": decay_stats,
            "moral_statistics": moral_stats,
            "impulse_analysis": impulse_analysis,
            "last_impulse": round(self.last_impulse, 4),
            "impulse_count": self.impulse_count,
            "connections": {
                "has_moral_memory": self.moral_memory is not None,
                "has_spirit_core": self.spirit_core is not None,
                "has_policy_governor": self.policy_governor is not None,
                "has_keter_integration": self.keter_integration is not None
            }
        }
    
    async def apply_cosmic_justice(self, justice_level: float):
        """Применение космической справедливости к ядру воли"""
        await self.moral_filter.apply_cosmic_justice(justice_level)
        
        # Корректировка воли на основе справедливости
        self.divine_essence *= justice_level
        self.cosmic_focus = max(0.5, self.cosmic_focus * (0.8 + justice_level * 0.2))
        
        logger.info(f"[{self.name}] Космическая справедливость применена: {justice_level:.3f}")
    
    async def shutdown(self):
        """Корректное выключение ядра воли"""
        self.is_active = False
        
        # Финальный отчёт в Policy Governor
        if self.policy_governor:
            final_metrics = {
                "total_impulses": self.impulse_count,
                "final_strength": await self.get_current_divine_strength(),
                "total_uptime": round(time.time() - self.activation_time, 2),
                "average_moral_alignment": self.moral_filter.last_alignment
            }
            await self.policy_governor.report_willpower_metrics(final_metrics)
        
        logger.info(f"[{self.name}] Выключено (импульсов: {self.impulse_count})")

# ===============================================================
# IV. ФАБРИЧНАЯ ФУНКЦИЯ ДЛЯ ИНТЕГРАЦИИ
# ===============================================================

async def create_willpower_core_v32_module(
    moral_memory: Optional[IMoralMemoryLink] = None,
    spirit_core: Optional[ISpiritCoreLink] = None,
    policy_governor: Optional[IPolicyGovernorLink] = None,
    keter_core: Optional[IKeterIntegration] = None
) -> WILLPOWER_CORE_v32_KETER:
    """
    Фабричная функция для создания WILLPOWER-CORE v3.2
    Используется в keter_core.py для интеграции
    """
    module = WILLPOWER_CORE_v32_KETER(
        moral_memory_link=moral_memory,
        spirit_core_link=spirit_core,
        policy_governor_link=policy_governor,
        keter_integration=keter_core
    )
    
    # Автоматическая активация
    await module.activate()
    
    return module

# ===============================================================
# VI. АЛИАС ДЛЯ ОБРАТНОЙ СОВМЕСТИМОСТИ
# ===============================================================

# Алиас для совместимости с существующим кодом
WillpowerCoreV3_2 = WILLPOWER_CORE_v32_KETER

# ===============================================================
# V. ТЕСТОВЫЙ ЗАПУСК
# ===============================================================

async def _test_willpower_core_v32():
    """Тестовый запуск ядра божественной воли"""
    print("🧪 Тест WILLPOWER-CORE v3.2 (Divine Will Engine)")
    
    # Мок-объекты
    class MockMoralMemory:
        async def get_alignment_score(self): return 0.92
        async def get_ethical_coherence(self): return 0.88
        async def register_willpower_pattern(self, pattern):
            print(f"[MOCK-MORAL] Паттерн зарегистрирован: {pattern['willpower_pattern']}")
            return True
    
    class MockSpiritCore:
        async def get_spiritual_resonance(self): return 0.95
        async def receive_willpower_boost(self, boost):
            print(f"[MOCK-SPIRIT] Получен буст воли: {boost:.3f}")
            return True
    
    class MockPolicyGovernor:
        async def get_willpower_constraints(self):
            return {"max_will_strength": 0.95, "focus_limits": {"min": 0.7, "max": 1.0}}
        async def report_willpower_metrics(self, metrics):
            print(f"[MOCK-POLICY] Метрики получены: {metrics['total_impulses']} импульсов")
            return True
    
    # Создание модуля
    module = WILLPOWER_CORE_v32_KETER(
        moral_memory_link=MockMoralMemory(),
        spirit_core_link=MockSpiritCore(),
        policy_governor_link=MockPolicyGovernor()
    )
    
    # Активация
    success = await module.activate()
    print(f"Активация: {'✅' if success else '❌'}")
    
    if success:
        # Генерация нескольких импульсов
        for i in range(3):
            divine_intent = {
                "cosmic_clarity": 0.85 + (i * 0.05),
                "divine_purpose": 0.9,
                "sephirotic_alignment": 0.87
            }
            
            result = await module.generate_divine_impulse(divine_intent)
            impulse = result.get("divine_impulse", 0.0)
            print(f"Импульс {i+1}: {impulse:.3f}")
            
            # Корректировка морали
            await module.adjust_divine_will(0.91, source="test")
            
            await asyncio.sleep(0.3)
        
        # Получение статистики
        stats = await module.get_willpower_statistics()
        print(f"Сила воли: {stats['current_strength']:.3f}")
        print(f"Здоровье воли: {stats['will_health']:.3f}")
        print(f"Всего импульсов: {stats['impulse_count']}")
        
        # Применение космической справедливости
        await module.apply_cosmic_justice(0.85)
        
        # Выключение
        await module.shutdown()

if __name__ == "__main__":
    import sys
    if "--test" in sys.argv:
        asyncio.run(_test_willpower_core_v32())
    else:
        print("ISKRA-4 · WILLPOWER-CORE v3.2 (Sephirotic Hybrid Will Engine)")
        print("Ядро божественной воли Keter")
        print("Используйте --test для запуска теста")
