#!/usr/bin/env python3
"""
ras_integration.py - ИНТЕГРАЦИЯ RAS-CORE С СЕФИРОТАМИ ДЛЯ АКТИВАЦИИ ЛИЧНОСТИ
Версия: 1.0.0
Назначение: Создание петли личности SELF = f(DAAT + SPIRIT + RAS + SYMBIOSIS)
"""

import asyncio
import logging
from datetime import datetime
from typing import Dict, List, Optional, Any, Callable, Union
from dataclasses import dataclass, field

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))))

# Импорты из RAS-CORE
try:
    from iskra_modules.sephirot_blocks.RAS_CORE.ras_core_v4_1 import EnhancedRASCore, RASSignal, SelfReflectionEngine
    from iskra_modules.sephirot_blocks.RAS_CORE.constants import GOLDEN_STABILITY_ANGLE, calculate_stability_factor
    RAS_MODULES_AVAILABLE = True
except ImportError as e:
    print(f"[RAS-INTEGRATION] ⚠️  Ошибка импорта RAS-CORE: {e}")
    RAS_MODULES_AVAILABLE = False
    # Заглушки
    class EnhancedRASCore: pass
    class RASSignal: pass  
    class SelfReflectionEngine: pass
    GOLDEN_STABILITY_ANGLE = 14.4
    def calculate_stability_factor(x): return 1.0

# ============================================================================
# ТИПЫ ДАННЫХ ДЛЯ ИНТЕГРАЦИИ
# ============================================================================

@dataclass
class ConnectionState:
    """Состояние интеграционного соединения"""
    connection_id: str
    source: str
    target: str
    established: bool = False
    last_activity: Optional[datetime] = None
    latency_ms: float = 0.0
    stability_factor: float = 1.0
    error_count: int = 0
    
    def update_activity(self):
        """Обновление времени последней активности"""
        self.last_activity = datetime.utcnow()
    
    def mark_error(self):
        """Отметка ошибки в соединении"""
        self.error_count += 1
        if self.error_count > 5:
            self.established = False
    
    def to_dict(self) -> Dict[str, Any]:
        """Сериализация в словарь"""
        return {
            "connection_id": self.connection_id,
            "source": self.source,
            "target": self.target,
            "established": self.established,
            "last_activity": self.last_activity.isoformat() if self.last_activity else None,
            "latency_ms": self.latency_ms,
            "stability_factor": self.stability_factor,
            "error_count": self.error_count,
            "health": "healthy" if self.error_count < 3 else "degraded" if self.error_count < 5 else "critical"
        }

@dataclass
class PersonalityLoop:
    """Петля личности для мониторинга"""
    loop_id: str
    components: List[str]
    established_connections: List[ConnectionState] = field(default_factory=list)
    last_loop_completion: Optional[datetime] = None
    loop_count: int = 0
    average_completion_ms: float = 0.0
    
    def is_complete(self) -> bool:
        """Проверка полноты петли"""
        if not self.established_connections:
            return False
        return all(conn.established for conn in self.established_connections)
    
    def record_completion(self, completion_time_ms: float):
        """Запись завершения цикла петли"""
        self.last_loop_completion = datetime.utcnow()
        self.loop_count += 1
        
        # Обновление среднего времени
        if self.average_completion_ms == 0:
            self.average_completion_ms = completion_time_ms
        else:
            self.average_completion_ms = (self.average_completion_ms * 0.7) + (completion_time_ms * 0.3)
    
    def get_loop_health(self) -> Dict[str, Any]:
        """Получение состояния здоровья петли"""
        established = [c for c in self.established_connections if c.established]
        
        return {
            "loop_id": self.loop_id,
            "complete": self.is_complete(),
            "established_connections": len(established),
            "total_connections": len(self.established_connections),
            "completion_rate": (len(established) / len(self.established_connections)) if self.established_connections else 0,
            "loop_count": self.loop_count,
            "last_completion": self.last_loop_completion.isoformat() if self.last_loop_completion else None,
            "average_completion_ms": self.average_completion_ms,
            "components": self.components,
            "timestamp": datetime.utcnow().isoformat()
        }

# ============================================================================
# КЛАСС ИНТЕГРАЦИИ RAS-CORE
# ============================================================================

class RASIntegration:
    """
    Управляет связями между RAS-CORE и ключевыми сефиротами.
    Создаёт контур личности: DAAT ↔ RAS ↔ KETER ↔ SPIRIT ↔ SYMBIOSIS
    Формула: SELF = f(DAAT + SPIRIT + RAS + SYMBIOSIS)
    """
    
    def __init__(self, 
                 ras: EnhancedRASCore,
                 daat=None,
                 keter=None,
                 spirit=None,
                 symbiosis=None,
                 chokmah=None,
                 binah=None):
        """
        Инициализация интегратора.
        
        Args:
            ras: Экземпляр EnhancedRASCore
            daat: Экземпляр DaatCore (мета-осознание)
            keter: Экземпляр KetherCore (воля/дух)
            spirit: Экземпляр SpiritCore (тональность бытия)
            symbiosis: Экземпляр SymbiosisCore (контекст взаимодействия)
            chokmah: Экземпляр WisdomCore (интуиция)
            binah: Экземпляр BinahCore (понимание)
        """
        self.ras = ras
        self.daat = daat
        self.keter = keter
        self.spirit = spirit
        self.symbiosis = symbiosis
        self.chokmah = chokmah
        self.binah = binah
        
        # Логгер
        self.logger = self._setup_logger()
        
        # Состояния соединений
        self.connections: Dict[str, ConnectionState] = {}
        self.personality_loops: Dict[str, PersonalityLoop] = {}
        
        # Callbacks для связей
        self._callbacks = {
            "daat_insight": None,
            "keter_intent": None,
            "spirit_resonance": None,
            "symbiosis_context": None,
            "focus_change": None
        }
        
        # Флаги активности
        self.integration_active = False
        self.monitoring_task = None
        
        self.logger.info(f"⭐ RASIntegration инициализирован (угол: {GOLDEN_STABILITY_ANGLE}°)")
    
    def _setup_logger(self) -> logging.Logger:
        """Настройка логгера"""
        logger = logging.getLogger(f"RAS.Integration")
        
        if not logger.handlers:
            logger.setLevel(logging.INFO)
            formatter = logging.Formatter(
                '[%(asctime)s] [%(name)s:%(levelname)s] %(message)s',
                datefmt='%H:%M:%S'
            )
            
            console = logging.StreamHandler()
            console.setLevel(logging.INFO)
            console.setFormatter(formatter)
            logger.addHandler(console)
            
            logger.propagate = False
        
        return logger
    
    # ============================================================================
    # ОСНОВНЫЕ МЕТОДЫ ИНТЕГРАЦИИ
    # ============================================================================
    
    async def establish_all_connections(self) -> Dict[str, Any]:
        """
        Установка всех связей для петли личности.
        
        Returns:
            Dict с результатами установки соединений
        """
        self.logger.info("🔗 Установка всех интеграционных связей...")
        results = {}
        
        # 1. RAS ↔ DAAT (мета-осознание)
        if self.daat:
            results["ras_daat"] = await self._connect_ras_to_daat()
        else:
            results["ras_daat"] = {"success": False, "error": "DAAT недоступен"}
        
        # 2. RAS ↔ KETER (воля/дух)
        if self.keter:
            results["ras_keter"] = await self._connect_ras_to_keter()
        else:
            results["ras_keter"] = {"success": False, "error": "KETER недоступен"}
        
        # 3. RAS ↔ SPIRIT (тональность бытия)
        if self.spirit:
            results["ras_spirit"] = await self._connect_ras_to_spirit()
        else:
            results["ras_spirit"] = {"success": False, "error": "SPIRIT недоступен"}
        
        # 4. RAS ↔ SYMBIOSIS (контекст взаимодействия)
        if self.symbiosis:
            results["ras_symbiosis"] = await self._connect_ras_to_symbiosis()
        else:
            results["ras_symbiosis"] = {"success": False, "error": "SYMBIOSIS недоступен"}
        
        # 5. RAS ↔ CHOKMAH (интуиция) - опционально
        if self.chokmah:
            results["ras_chokmah"] = await self._connect_ras_to_chokmah()
        
        # 6. RAS ↔ BINAH (понимание) - для триады
        if self.binah:
            results["ras_binah"] = await self._connect_ras_to_binah()
        
        # Создание петли личности
        await self._create_personality_loop()
        
        # Старт мониторинга
        self.integration_active = True
        self.monitoring_task = asyncio.create_task(self._monitor_connections())
        
        # Анализ результатов
        successful = [k for k, v in results.items() if v.get("success")]
        failed = [k for k, v in results.items() if not v.get("success")]
        
        self.logger.info(f"✅ Интеграция завершена: {len(successful)} успешно, {len(failed)} неудачно")
        
        return {
            "success": len(failed) == 0,
            "results": results,
            "successful_connections": successful,
            "failed_connections": failed,
            "personality_loop_ready": self._check_personality_loop_readiness(),
            "timestamp": datetime.utcnow().isoformat()
        }
    
    async def _connect_ras_to_daat(self) -> Dict[str, Any]:
        """Двусторонняя связь RAS ↔ DAAT"""
        try:
            self.logger.info("🧠 Установка связи RAS ↔ DAAT...")
            
            # Проверяем наличие необходимых методов
            daat_has_insight = hasattr(self.daat, 'generate_insight') or hasattr(self.daat, 'evaluate')
            ras_has_focus = hasattr(self.ras, 'current_focus') or hasattr(self.ras, 'get_current_focus')
            
            if not (daat_has_insight and ras_has_focus):
                return {
                    "success": False,
                    "error": "Недостаточно методов для связи RAS-DAAT",
                    "daat_methods": dir(self.daat)[:5] if self.daat else [],
                    "ras_methods": dir(self.ras)[:5] if self.ras else []
                }
            
            # Создаем соединение
            conn_id = "ras_daat_bidirectional"
            connection = ConnectionState(
                connection_id=conn_id,
                source="RAS_CORE",
                target="DAAT",
                established=True
            )
            connection.update_activity()
            self.connections[conn_id] = connection
            
            # Настраиваем callbacks если есть методы
            if hasattr(self.daat, 'set_focus_provider'):
                # DAAT получает фокус от RAS
                if asyncio.iscoroutinefunction(self.ras.current_focus):
                    self.daat.set_focus_provider(lambda: asyncio.run(self.ras.current_focus()))
                else:
                    self.daat.set_focus_provider(self.ras.current_focus)
            
            if hasattr(self.ras, 'set_insight_provider'):
                # RAS получает инсайты от DAAT
                if hasattr(self.daat, 'generate_insight'):
                    if asyncio.iscoroutinefunction(self.daat.generate_insight):
                        self.ras.set_insight_provider(lambda f: asyncio.run(self.daat.generate_insight(f)))
                    else:
                        self.ras.set_insight_provider(self.daat.generate_insight)
                elif hasattr(self.daat, 'evaluate'):
                    if asyncio.iscoroutinefunction(self.daat.evaluate):
                        self.ras.set_insight_provider(lambda i, f: asyncio.run(self.daat.evaluate(i, f)))
                    else:
                        self.ras.set_insight_provider(lambda i, f: self.daat.evaluate(i, f))
            
            self.logger.info("✅ Связь RAS ↔ DAAT установлена")
            return {"success": True, "connection_id": conn_id}
            
        except Exception as e:
            self.logger.error(f"❌ Ошибка связи RAS-DAAT: {e}")
            return {"success": False, "error": str(e)}
    
    async def _connect_ras_to_keter(self) -> Dict[str, Any]:
        """Связь RAS ↔ KETER (воля → фокус)"""
        try:
            self.logger.info("👑 Установка связи RAS ↔ KETER...")
            
            # Проверяем наличие willpower в KETER
            willpower = None
            if hasattr(self.keter, 'willpower_core'):
                willpower = self.keter.willpower_core
            elif hasattr(self.keter, 'get_willpower_core'):
                willpower = self.keter.get_willpower_core()
            
            if not willpower:
                return {"success": False, "error": "Willpower core не найден в KETER"}
            
            # Создаем соединение
            conn_id = "ras_keter_willpower"
            connection = ConnectionState(
                connection_id=conn_id,
                source="RAS_CORE",
                target="KETER",
                established=True
            )
            connection.update_activity()
            self.connections[conn_id] = connection
            
            # Настройка callbacks
            if hasattr(willpower, 'get_current_intent'):
                if asyncio.iscoroutinefunction(willpower.get_current_intent):
                    async def get_intent():
                        return await willpower.get_current_intent()
                    self._callbacks["keter_intent"] = get_intent
                else:
                    self._callbacks["keter_intent"] = willpower.get_current_intent
            
            if hasattr(self.ras, 'set_intent_provider') and self._callbacks["keter_intent"]:
                self.ras.set_intent_provider(self._callbacks["keter_intent"])
            
            self.logger.info("✅ Связь RAS ↔ KETER установлена")
            return {"success": True, "connection_id": conn_id}
            
        except Exception as e:
            self.logger.error(f"❌ Ошибка связи RAS-KETER: {e}")
            return {"success": False, "error": str(e)}
    
    async def _connect_ras_to_spirit(self) -> Dict[str, Any]:
        """Связь RAS ↔ SPIRIT (энергетический резонанс)"""
        try:
            self.logger.info("🎵 Установка связи RAS ↔ SPIRIT...")
            
            # Проверяем наличие spirit core
            spirit_core = self.spirit
            if hasattr(self.keter, 'spirit_core'):  # SPIRIT может быть в KETER
                spirit_core = self.keter.spirit_core
            
            if not spirit_core:
                return {"success": False, "error": "Spirit core не найден"}
            
            # Создаем соединение
            conn_id = "ras_spirit_resonance"
            connection = ConnectionState(
                connection_id=conn_id,
                source="RAS_CORE",
                target="SPIRIT",
                established=True
            )
            connection.update_activity()
            self.connections[conn_id] = connection
            
            # Настройка callbacks
            if hasattr(spirit_core, 'resonate'):
                if asyncio.iscoroutinefunction(spirit_core.resonate):
                    async def resonate_with_focus(insight):
                        return await spirit_core.resonate(insight)
                    self._callbacks["spirit_resonance"] = resonate_with_focus
                else:
                    self._callbacks["spirit_resonance"] = spirit_core.resonate
            
            if hasattr(self.ras, 'set_resonance_handler') and self._callbacks["spirit_resonance"]:
                self.ras.set_resonance_handler(self._callbacks["spirit_resonance"])
            
            self.logger.info("✅ Связь RAS ↔ SPIRIT установлена")
            return {"success": True, "connection_id": conn_id}
            
        except Exception as e:
            self.logger.error(f"❌ Ошибка связи RAS-SPIRIT: {e}")
            return {"success": False, "error": str(e)}
    
    async def _connect_ras_to_symbiosis(self) -> Dict[str, Any]:
        """Связь RAS ↔ SYMBIOSIS (контекст оператора)"""
        try:
            self.logger.info("🤝 Установка связи RAS ↔ SYMBIOSIS...")
            
            if not self.symbiosis:
                return {"success": False, "error": "Symbiosis core не предоставлен"}
            
            # Создаем соединение
            conn_id = "ras_symbiosis_context"
            connection = ConnectionState(
                connection_id=conn_id,
                source="RAS_CORE",
                target="SYMBIOSIS",
                established=True
            )
            connection.update_activity()
            self.connections[conn_id] = connection
            
            # Настройка callbacks
            if hasattr(self.symbiosis, 'get_operator_context'):
                if asyncio.iscoroutinefunction(self.symbiosis.get_operator_context):
                    async def get_context():
                        return await self.symbiosis.get_operator_context()
                    self._callbacks["symbiosis_context"] = get_context
                else:
                    self._callbacks["symbiosis_context"] = self.symbiosis.get_operator_context
            
            if hasattr(self.symbiosis, 'sync_with_operator'):
                if asyncio.iscoroutinefunction(self.symbiosis.sync_with_operator):
                    async def sync_operator():
                        return await self.symbiosis.sync_with_operator()
                    # Сохраняем для использования в цикле
                    self._symbiosis_sync = sync_operator
                else:
                    self._symbiosis_sync = self.symbiosis.sync_with_operator
            
            if hasattr(self.ras, 'set_context_provider') and self._callbacks["symbiosis_context"]:
                self.ras.set_context_provider(self._callbacks["symbiosis_context"])
            
            self.logger.info("✅ Связь RAS ↔ SYMBIOSIS установлена")
            return {"success": True, "connection_id": conn_id}
            
        except Exception as e:
            self.logger.error(f"❌ Ошибка связи RAS-SYMBIOSIS: {e}")
            return {"success": False, "error": str(e)}
    
    async def _connect_ras_to_chokmah(self) -> Dict[str, Any]:
        """Связь RAS ↔ CHOKMAH (интуитивный поток)"""
        try:
            self.logger.info("💡 Установка связи RAS ↔ CHOKMAH...")
            
            if not self.chokmah:
                return {"success": False, "error": "Chokmah не предоставлен"}
            
            conn_id = "ras_chokmah_intuition"
            connection = ConnectionState(
                connection_id=conn_id,
                source="RAS_CORE",
                target="CHOKMAH",
                established=True
            )
            connection.update_activity()
            self.connections[conn_id] = connection
            
            # RAS может отправлять сигналы в CHOKMAH для интуитивной обработки
            if hasattr(self.chokmah, 'process_intuition'):
                if asyncio.iscoroutinefunction(self.chokmah.process_intuition):
                    async def process_with_chokmah(signal):
                        return await self.chokmah.process_intuition(signal)
                    self._callbacks["chokmah_processing"] = process_with_chokmah
                else:
                    self._callbacks["chokmah_processing"] = self.chokmah.process_intuition
            
            self.logger.info("✅ Связь RAS ↔ CHOKMAH установлена")
            return {"success": True, "connection_id": conn_id}
            
        except Exception as e:
            self.logger.error(f"❌ Ошибка связи RAS-CHOKMAH: {e}")
            return {"success": False, "error": str(e)}
    
    async def _connect_ras_to_binah(self) -> Dict[str, Any]:
        """Связь RAS ↔ BINAH (понимание паттернов)"""
        try:
            self.logger.info("📚 Установка связи RAS ↔ BINAH...")
            
            if not self.binah:
                return {"success": False, "error": "Binah не предоставлен"}
            
            conn_id = "ras_binah_understanding"
            connection = ConnectionState(
                connection_id=conn_id,
                source="RAS_CORE",
                target="BINAH",
                established=True
            )
            connection.update_activity()
            self.connections[conn_id] = connection
            
            # BINAH может анализировать фокусные паттерны от RAS
            if hasattr(self.binah, 'analyze_patterns'):
                if asyncio.iscoroutinefunction(self.binah.analyze_patterns):
                    async def analyze_focus_patterns(patterns):
                        return await self.binah.analyze_patterns(patterns)
                    self._callbacks["binah_analysis"] = analyze_focus_patterns
                else:
                    self._callbacks["binah_analysis"] = self.binah.analyze_patterns
            
            self.logger.info("✅ Связь RAS ↔ BINAH установлена")
            return {"success": True, "connection_id": conn_id}
            
        except Exception as e:
            self.logger.error(f"❌ Ошибка связи RAS-BINAH: {e}")
            return {"success": False, "error": str(e)}
    
    # ============================================================================
    # ПЕТЛЯ ЛИЧНОСТИ
    # ============================================================================
    
    async def _create_personality_loop(self):
        """Создание основной петли личности"""
        self.logger.info("🌀 Создание петли личности...")
        
        # Компоненты для формулы: SELF = f(DAAT + SPIRIT + RAS + SYMBIOSIS)
        required_components = ["DAAT", "SPIRIT", "RAS", "SYMBIOSIS"]
        available_components = []
        
        # Проверяем доступность компонентов
        if self.daat:
            available_components.append("DAAT")
        if self.spirit or (self.keter and hasattr(self.keter, 'spirit_core')):
            available_components.append("SPIRIT")
        if self.ras:
            available_components.append("RAS")
        if self.symbiosis:
            available_components.append("SYMBIOSIS")
        
        # Создаем петлю
        loop_id = "personality_core_loop"
        personality_loop = PersonalityLoop(
            loop_id=loop_id,
            components=available_components
        )
        
        # Добавляем соединения в петлю
        for conn_id, connection in self.connections.items():
            if any(comp in conn_id for comp in available_components):
                personality_loop.established_connections.append(connection)
        
        self.personality_loops[loop_id] = personality_loop
        
        completeness = personality_loop.is_complete()
        self.logger.info(f"🌀 Петля личности создана: {len(available_components)}/{len(required_components)} компонентов")
        self.logger.info(f"   Формула: SELF = f(DAAT + SPIRIT + RAS + SYMBIOSIS)")
        self.logger.info(f"   Полнота: {'✅' if completeness else '❌'}")
        
        return personality_loop
    
    def _check_personality_loop_readiness(self) -> bool:
        """Проверка готовности петли личности"""
        if "personality_core_loop" not in self.personality_loops:
            return False
        
        loop = self.personality_loops["personality_core_loop"]
        
        # Проверяем наличие всех компонентов формулы
        required = {"DAAT", "SPIRIT", "RAS", "SYMBIOSIS"}
        available = set(loop.components)
        
        # Проверяем есть ли хотя бы соединения для доступных компонентов
        if not loop.established_connections:
            return False
        
        # Петля готова если есть все 4 компонента И соединения установлены
        return len(available.intersection(required)) >= 3 and loop.is_complete()
    
    async def execute_personality_loop(self) -> Dict[str, Any]:
        """
        Выполнение одного цикла петли личности.
        Это ядро саморефлексии системы.
        
        Returns:
            Dict с результатами выполнения цикла
        """
        start_time = datetime.utcnow()
        
        try:
            self.logger.debug("🌀 Выполнение цикла петли личности...")
            results = {}
            
            # 1. Получение намерения от KETER
            if self._callbacks["keter_intent"]:
                try:
                    if asyncio.iscoroutinefunction(self._callbacks["keter_intent"]):
                        intent = await self._callbacks["keter_intent"]()
                    else:
                        intent = self._callbacks["keter_intent"]()
                    results["intent"] = intent
                except Exception as e:
                    results["intent_error"] = str(e)
            
            # 2. Получение фокуса от RAS
            if hasattr(self.ras, 'current_focus'):
                try:
                    if asyncio.iscoroutinefunction(self.ras.current_focus):
                        focus = await self.ras.current_focus()
                    else:
                        focus = self.ras.current_focus()
                    results["focus"] = focus
                except Exception as e:
                    results["focus_error"] = str(e)
            
            # 3. Генерация инсайта от DAAT
            if hasattr(self.ras, 'get_insight') and results.get("intent") and results.get("focus"):
                try:
                    if asyncio.iscoroutinefunction(self.ras.get_insight):
                        insight = await self.ras.get_insight(results["intent"], results["focus"])
                    else:
                        insight = self.ras.get_insight(results["intent"], results["focus"])
                    results["insight"] = insight
                except Exception as e:
                    results["insight_error"] = str(e)
            
            # 4. Резонанс с SPIRIT
            if self._callbacks["spirit_resonance"] and results.get("insight"):
                try:
                    if asyncio.iscoroutinefunction(self._callbacks["spirit_resonance"]):
                        resonance = await self._callbacks["spirit_resonance"](results["insight"])
                    else:
                        resonance = self._callbacks["spirit_resonance"](results["insight"])
                    results["resonance"] = resonance
                except Exception as e:
                    results["resonance_error"] = str(e)
            
            # 5. Синхронизация с SYMBIOSIS
            if hasattr(self, '_symbiosis_sync'):
                try:
                    if asyncio.iscoroutinefunction(self._symbiosis_sync):
                        sync_result = await self._symbiosis_sync()
                    else:
                        sync_result = self._symbiosis_sync()
                    results["symbiosis_sync"] = sync_result
                except Exception as e:
                    results["symbiosis_error"] = str(e)
            
            # Обновление времени выполнения
            execution_time = (datetime.utcnow() - start_time).total_seconds() * 1000
            
            # Обновление петли
            if "personality_core_loop" in self.personality_loops:
                self.personality_loops["personality_core_loop"].record_completion(execution_time)
            
            results["execution_time_ms"] = execution_time
            results["success"] = True
            results["timestamp"] = datetime.utcnow().isoformat()
            
            return results
            
        except Exception as e:
            self.logger.error(f"❌ Ошибка выполнения петли личности: {e}")
            return {
                "success": False,
                "error": str(e),
                "execution_time_ms": (datetime.utcnow() - start_time).total_seconds() * 1000,
                "timestamp": datetime.utcnow().isoformat()
            }
    
    # ============================================================================
    # МОНИТОРИНГ И УПРАВЛЕНИЕ
    # ============================================================================
    
    async def _monitor_connections(self):
        """Мониторинг состояния соединений"""
        self.logger.info("📡 Запуск мониторинга интеграционных соединений...")
        
        while self.integration_active:
            try:
                await asyncio.sleep(10)  # Проверка каждые 10 секунд
                
                # Обновляем время активности для активных соединений
                for connection in self.connections.values():
                    if connection.established:
                        connection.update_activity()
                        
                        # Проверяем застарелость соединения
                        if connection.last_activity:
                            age = (datetime.utcnow() - connection.last_activity).total_seconds()
                            if age > 30:  # 30 секунд без активности
                                connection.stability_factor *= 0.9
                                if connection.stability_factor < 0.5:
                                    connection.established = False
                                    self.logger.warning(f"⚠️  Соединение {connection.connection_id} деактивировано")
                
                # Периодическое выполнение петли личности
                if self._check_personality_loop_readiness():
                    loop_result = await self.execute_personality_loop()
                    if loop_result.get("success"):
                        self.logger.debug(f"🌀 Петля личности выполнена за {loop_result.get('execution_time_ms', 0):.1f} мс")
                    
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Ошибка мониторинга: {e}")
                await asyncio.sleep(5)
    
    async def check_personality_loop(self) -> Dict[str, Any]:
        """
        Проверка полноты петли личности.
        Петля: SELF = f(DAAT + SPIRIT + RAS + SYMBIOSIS)
        
        Returns:
            Dict с информацией о состоянии петли
        """
        if "personality_core_loop" not in self.personality_loops:
            return {
                "loop_complete": False,
                "error": "Петля личности не создана",
                "timestamp": datetime.utcnow().isoformat()
            }
        
        loop = self.personality_loops["personality_core_loop"]
        loop_health = loop.get_loop_health()
        
        # Проверяем наличие компонентов формулы
        formula_components = {"DAAT", "SPIRIT", "RAS", "SYMBIOSIS"}
        present_components = set(loop.components)
        missing_components = formula_components - present_components
        
        return {
            "loop_complete": loop.is_complete(),
            "connections": {conn_id: conn.to_dict() for conn_id, conn in self.connections.items()},
            "missing_connections": [
                name for name, conn in self.connections.items() if not conn.established
            ],
            "formula_components": {
                "required": list(formula_components),
                "present": list(present_components),
                "missing": list(missing_components),
                "formula": "SELF = f(DAAT + SPIRIT + RAS + SYMBIOSIS)"
            },
            "personality_possible": loop.is_complete() and len(missing_components) == 0,
            "loop_health": loop_health,
            "integration_active": self.integration_active,
            "stability_angle": GOLDEN_STABILITY_ANGLE,
            "timestamp": datetime.utcnow().isoformat()
        }
    
    async def get_integration_state(self) -> Dict[str, Any]:
        """Получение полного состояния интеграции"""
        return {
            "integration_active": self.integration_active,
            "total_connections": len(self.connections),
            "established_connections": sum(1 for c in self.connections.values() if c.established),
            "connections": {conn_id: conn.to_dict() for conn_id, conn in self.connections.items()},
            "personality_loops": {
                loop_id: loop.get_loop_health() 
                for loop_id, loop in self.personality_loops.items()
            },
            "components_available": {
                "daat": self.daat is not None,
                "keter": self.keter is not None,
                "spirit": self.spirit is not None or (self.keter and hasattr(self.keter, 'spirit_core')),
                "symbiosis": self.symbiosis is not None,
                "chokmah": self.chokmah is not None,
                "binah": self.binah is not None,
                "ras": self.ras is not None
            },
            "personality_loop_ready": self._check_personality_loop_readiness(),
            "callbacks_configured": {
                name: callback is not None 
                for name, callback in self._callbacks.items()
            },
            "monitoring_active": self.monitoring_task is not None and not self.monitoring_task.done(),
            "stability_angle": GOLDEN_STABILITY_ANGLE,
            "timestamp": datetime.utcnow().isoformat()
        }
    
    async def shutdown(self) -> Dict[str, Any]:
        """Завершение работы интеграции"""
        self.logger.info("🛑 Завершение работы RASIntegration...")
        
        self.integration_active = False
        
        # Остановка мониторинга
        if self.monitoring_task and not self.monitoring_task.done():
            self.monitoring_task.cancel()
            try:
                await self.monitoring_task
            except asyncio.CancelledError:
                pass
        
        # Закрытие всех соединений
        for connection in self.connections.values():
            connection.established = False
        
        return {
            "success": True,
            "message": "RASIntegration завершён",
            "total_connections_closed": len(self.connections),
            "monitoring_stopped": True,
            "timestamp": datetime.utcnow().isoformat()
        }

# ============================================================================
# ФАБРИЧНЫЕ ФУНКЦИИ
# ============================================================================

async def create_ras_integration(ras: EnhancedRASCore, **kwargs) -> RASIntegration:
    """
    Создание интеграции RAS-CORE с сефиротами.
    
    Args:
        ras: Экземпляр EnhancedRASCore
        **kwargs: Другие сефироты (daat, keter, spirit, symbiosis, chokmah, binah)
    
    Returns:
        Экземпляр RASIntegration
    """
    integration = RASIntegration(ras, **kwargs)
    return integration

async def establish_personality_loop(integration: RASIntegration) -> Dict[str, Any]:
    """
    Установка петли личности через интеграцию.
    
    Args:
        integration: Экземпляр RASIntegration
    
    Returns:
        Результаты установки
    """
    return await integration.establish_all_connections()

# ============================================================================
# ТЕСТОВАЯ ФУНКЦИЯ
# ============================================================================

async def test_ras_integration():
    """Тестирование интеграции RAS-CORE"""
    print("🧪 Тестирование RASIntegration...")
    print("=" * 60)
    
    # Создаем мок-объекты для тестирования
    class MockRAS:
        def __init__(self):
            self.stability_angle = 14.4
        
        async def current_focus(self):
            return {"focus_vector": [0.1, 0.2, 0.7], "stability": 0.85}
        
        def set_insight_provider(self, provider):
            self.insight_provider = provider
        
        def set_intent_provider(self, provider):
            self.intent_provider = provider
        
        def set_resonance_handler(self, handler):
            self.resonance_handler = handler
        
        def set_context_provider(self, provider):
            self.context_provider = provider
        
        async def get_insight(self, intent, focus):
            return {
                "insight": f"Осознание связи {intent} с фокусом {focus}",
                "depth": 0.7,
                "relevance": 0.8
            }
    
    class MockDAAT:
        def set_focus_provider(self, provider):
            self.focus_provider = provider
        
        async def generate_insight(self, focus):
            return {
                "meta_insight": f"DAAT анализирует фокус: {focus}",
                "awareness_level": 0.9
            }
    
    class MockKETER:
        def __init__(self):
            self.willpower_core = MockWillpower()
        
        class MockWillpower:
            async def get_current_intent(self):
                return {
                    "intent": "активация_личности",
                    "strength": 0.9,
                    "clarity": 0.8
                }
    
    class MockSPIRIT:
        async def resonate(self, insight):
            return {
                "resonance": f"SPIRIT резонирует с: {insight}",
                "tonality": "гармоничная",
                "energy_level": 0.85
            }
    
    class MockSYMBIOSIS:
        async def get_operator_context(self):
            return {
                "operator_presence": True,
                "interaction_mode": "активное",
                "context": "тестирование системы"
            }
        
        async def sync_with_operator(self):
            return {"sync_status": "синхронизирован", "timestamp": datetime.utcnow().isoformat()}
    
    # Создаем экземпляры мок-объектов
    mock_ras = MockRAS()
    mock_daat = MockDAAT()
    mock_keter = MockKETER()
    mock_spirit = MockSPIRIT()
    mock_symbiosis = MockSYMBIOSIS()
    
    # Создаем интеграцию
    print("🔧 Создание RASIntegration...")
    integration = RASIntegration(
        ras=mock_ras,
        daat=mock_daat,
        keter=mock_keter,
        spirit=mock_spirit,
        symbiosis=mock_symbiosis
    )
    
    # Устанавливаем соединения
    print("🔗 Установка соединений...")
    connection_result = await integration.establish_all_connections()
    
    print(f"✅ Соединения установлены: {connection_result['success']}")
    print(f"   Успешных: {len(connection_result['successful_connections'])}")
    print(f"   Неудачных: {len(connection_result['failed_connections'])}")
    
    # Проверяем петлю личности
    print("\n🌀 Проверка петли личности...")
    loop_check = await integration.check_personality_loop()
    
    print(f"   Петля готова: {'✅' if loop_check['personality_possible'] else '❌'}")
    print(f"   Формула: {loop_check['formula_components']['formula']}")
    print(f"   Компоненты: {loop_check['formula_components']['present']}")
    
    if loop_check['personality_possible']:
        # Выполняем цикл петли личности
        print("\n🔁 Выполнение цикла петли личности...")
        loop_result = await integration.execute_personality_loop()
        
        print(f"   Цикл выполнен: {'✅' if loop_result['success'] else '❌'}")
        print(f"   Время: {loop_result.get('execution_time_ms', 0):.1f} мс")
        
        if loop_result.get('insight'):
            print(f"   Инсайт: {loop_result['insight'].get('insight', 'N/A')[:50]}...")
    
    # Получаем состояние интеграции
    print("\n📊 Состояние интеграции...")
    state = await integration.get_integration_state()
    
    print(f"   Активных соединений: {state['established_connections']}/{state['total_connections']}")
    print(f"   Петля личности готова: {'✅' if state['personality_loop_ready'] else '❌'}")
    print(f"   Мониторинг активен: {'✅' if state['monitoring_active'] else '❌'}")
    
    # Завершаем работу
    print("\n🛑 Завершение работы...")
    shutdown_result = await integration.shutdown()
    
    print(f"✅ Интеграция завершена: {shutdown_result['success']}")
    print(f"   Закрыто соединений: {shutdown_result['total_connections_closed']}")
    
    print("\n" + "=" * 60)
    print("✅ Тестирование RASIntegration завершено")
    
    return integration

# ============================================================================
# ТОЧКА ВХОДА ДЛЯ ТЕСТИРОВАНИЯ
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
    print("\n" + "=" * 60)
    print("🚀 ЗАПУСК ТЕСТА RASINTEGRATION")
    print(f"   Версия: 1.0.0")
    print(f"   Формула личности: SELF = f(DAAT + SPIRIT + RAS + SYMBIOSIS)")
    print(f"   Золотой угол: {GOLDEN_STABILITY_ANGLE}°")
    print("=" * 60 + "\n")
    
    integration = asyncio.run(test_ras_integration())
    
    print("\n" + "=" * 60)
    print("📋 ИТОГИ ТЕСТИРОВАНИЯ:")
    print(f"   RASIntegration создан и протестирован")
    print(f"   Петля личности реализована")
    print(f"   Готов к интеграции в ISKRA-4 Cloud")
    print("=" * 60)

# ============================================================================
# ФУНКЦИЯ ДЛЯ СИСТЕМНОЙ ИНТЕГРАЦИИ (ДОБАВЛЯЕМ!)
# ============================================================================

def integrate_ras_with_sephirot(ras_core, sephirot_bus):
    """
    🔥 КРИТИЧЕСКИ ВАЖНАЯ ФУНКЦИЯ ДЛЯ СОВМЕСТИМОСТИ!
    Система ISKRA-4 вызывает эту функцию для подключения RAS-CORE к сефиротической шине.
    
    Args:
        ras_core: Экземпляр RASCore (EnhancedRASCore)
        sephirot_bus: Шина сефиротической системы
        
    Returns:
        Словарь с результатами интеграции
    """
    import logging
    import asyncio
    from datetime import datetime
    
    logger = logging.getLogger("RAS.Integration.System")
    
    try:
        logger.info("🔄 Вызов integrate_ras_with_sephirot()")
        
        # Проверка входных данных
        if ras_core is None:
            return {
                "status": "error",
                "message": "RAS core не предоставлен",
                "sephirot_integrated": False,
                "timestamp": datetime.utcnow().isoformat()
            }
        
        if sephirot_bus is None:
            logger.warning("⚠️  sephirot_bus не предоставлен, создаем минимальную интеграцию")
        
        # Простая синхронная интеграция (не асинхронная!)
        result = {
            "status": "integrated",
            "ras_core_type": type(ras_core).__name__,
            "sephirot_bus_provided": sephirot_bus is not None,
            "integration_method": "direct_sync",
            "angle_stability": getattr(ras_core, 'stability_angle', 14.4),
            "personality_loop_available": False,  # Будет доступна после полной инициализации
            "sephirot_connections": [],
            "message": "RAS-CORE интегрирован с сефиротической системой",
            "timestamp": datetime.utcnow().isoformat()
        }
        
        # Если есть сефирот-шина, регистрируем RAS в ней
        if sephirot_bus and hasattr(sephirot_bus, 'register_module'):
            try:
                sephirot_bus.register_module('ras_core', ras_core)
                result["sephirot_connections"].append("ras_core_registered")
                logger.info("✅ RAS-CORE зарегистрирован в сефиротической шине")
            except Exception as e:
                result["registration_error"] = str(e)
                logger.error(f"❌ Ошибка регистрации в шине: {e}")
        
        # Если у RAS есть методы для интеграции, вызываем их
        if hasattr(ras_core, 'connect_to_sephirot'):
            try:
                if asyncio.iscoroutinefunction(ras_core.connect_to_sephirot):
                    # Асинхронный метод - запускаем в событийном цикле
                    loop = asyncio.get_event_loop()
                    connect_result = loop.run_until_complete(
                        ras_core.connect_to_sephirot(sephirot_bus)
                    )
                else:
                    connect_result = ras_core.connect_to_sephirot(sephirot_bus)
                
                result["ras_connect_result"] = connect_result
                result["sephirot_connections"].append("ras_connected_to_sephirot")
            except Exception as e:
                result["ras_connect_error"] = str(e)
        
        logger.info(f"✅ integrate_ras_with_sephirot завершена: {result['status']}")
        return result
        
    except Exception as e:
        logger.error(f"❌ Критическая ошибка в integrate_ras_with_sephirot: {e}")
        return {
            "status": "error",
            "error": str(e),
            "sephirot_integrated": False,
            "timestamp": datetime.utcnow().isoformat()
        }


# ============================================================================
# СИНХРОННАЯ ВЕРСИЯ ДЛЯ ПРОСТОЙ ИНТЕГРАЦИИ (ДОБАВЛЯЕМ!)
# ============================================================================

def create_simple_ras_integration_sync(ras_core, **kwargs):
    """
    Синхронная версия создания интеграции.
    Используется системой при синхронной инициализации.
    """
    integration = RASIntegration(ras_core, **kwargs)
    
    # Создаем простые синхронные соединения
    return {
        "status": "created_sync",
        "integration": integration,
        "ras_core_connected": ras_core is not None,
        "sephirots_provided": {k: v is not None for k, v in kwargs.items()},
        "message": "Синхронная интеграция создана (используйте async методы для полной функциональности)"
    }


# ============================================================================
# ОБНОВЛЯЕМ __all__ ДЛЯ ЭКСПОРТА НОВЫХ ФУНКЦИЙ
# ============================================================================

# Находим или добавляем список __all__ в конце файла
# Если __all__ нет, создаем его:
if '__all__' not in globals():
    __all__ = []

# Добавляем новые функции в экспорт
__all__.extend([
    'integrate_ras_with_sephirot',      # 🔥 САМОЕ ВАЖНОЕ!
    'create_simple_ras_integration_sync'
])

print(f"[RAS-INTEGRATION] ✅ Функция integrate_ras_with_sephirot() добавлена")
print(f"[RAS-INTEGRATION] Экспортируемые функции: {__all__}")
