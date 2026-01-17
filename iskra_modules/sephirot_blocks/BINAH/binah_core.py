# ================================================================
# BINAH CORE · Sephirotic Understanding Engine v1.1
# Интеграция: ANALYTICS-MEGAFORGE 3.4 + GÖDEL-SENTINEL 3.2
# Назначение: Структурирование интуиции CHOKMAH → понимание для DAAT
# ================================================================

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Callable
import time
import logging
import random

# ================================================================
# IMPORT EXTERNAL MODULES
# ================================================================

logger = logging.getLogger(__name__)

# Импортируем ANALYTICS-MEGAFORGE
try:
    # Переименуем файл для корректного импорта
    from ANALYTICS_MEGAFORGE_3_4_Sephirotic_Analytical_Engine import (
        AnalyticsMegaForge,
        build_analytics_megaforge,
        Task
    )
    ANALYTICS_AVAILABLE = True
    logger.info("✅ ANALYTICS-MEGAFORGE 3.4 доступен для BINAH")
except ImportError as e:
    ANALYTICS_AVAILABLE = False
    logger.warning(f"⚠️ ANALYTICS-MEGAFORGE недоступен: {e}")

# Импортируем GÖDEL-SENTINEL
try:
    from GÖDEL_SENTINEL_3_2_Sephirotic_Paradox_Guardian import (
        GodelSentinel,
        build_godel_sentinel,
        GodelSignal
    )
    GODEL_SENTINEL_AVAILABLE = True
    logger.info("✅ GÖDEL-SENTINEL 3.2 доступен для BINAH")
except ImportError as e:
    GODEL_SENTINEL_AVAILABLE = False
    logger.warning(f"⚠️ GÖDEL-SENTINEL недоступен: {e}")

# ================================================================
# BINAH-SPECIFIC DATA STRUCTURES
# ================================================================

@dataclass
class IntuitionPacket:
    """Пакет интуиции от CHOKMAH"""
    id: str
    content: Dict[str, Any]
    timestamp: float = field(default_factory=time.time)
    source: str = "CHOKMAH"
    resonance_level: float = 0.55
    
    def to_task(self) -> Dict[str, Any]:
        """Конвертирует в задачу для ANALYTICS-MEGAFORGE"""
        return {
            "id": self.id,
            "type": "high",  # Интуиция — высокоуровневая задача
            "payload": self.content,
            "source": self.source,
            "timestamp": self.timestamp
        }
    
    def to_godel_signal(self) -> Dict[str, Any]:
        """Конвертирует в сигнал для GÖDEL-SENTINEL"""
        return {
            "intent_id": self.id,
            "content": str(self.content),
            "truth_score": 0.7,  # Базовая оценка истинности
            "proof_score": 0.5   # Базовая оценка доказуемости
        }

@dataclass
class StructuredUnderstanding:
    """Структурированное понимание — выход BINAH"""
    source_packet_id: str
    structured_patterns: List[str]
    coherence_score: float
    paradox_level: float
    godel_approved: bool  # Одобрено GÖDEL-SENTINEL
    ethical_alignment: float  # Резонансное выравнивание с этикой
    spiritual_harmony: float  # Резонансная гармония с духом
    analytics_priority: float  # Приоритет от ANALYTICS-MEGAFORGE
    timestamp: float = field(default_factory=time.time)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": "binah_understanding",
            "source": self.source_packet_id,
            "patterns": self.structured_patterns,
            "coherence": self.coherence_score,
            "paradox": self.paradox_level,
            "godel_approved": self.godel_approved,
            "ethical": self.ethical_alignment,
            "spiritual": self.spiritual_harmony,
            "analytics_priority": self.analytics_priority,
            "timestamp": self.timestamp,
            "sephira": "BINAH",
            "version": "1.1.0"
        }

# ================================================================
# BINAH'S OWN RESONANCE CALCULATORS (НЕ ИМПОРТЫ!)
# ================================================================

@dataclass
class BinahEthicalResonator:
    """
    Собственный этический резонатор BINAH.
    НЕ импортирует moral_memory из KETER — создает резонансное поле.
    """
    
    resonance_base: float = 0.6
    
    def calculate_alignment(self, content: Dict[str, Any]) -> float:
        """Рассчитывает этическое выравнивание на основе резонанса"""
        alignment = self.resonance_base
        
        # Анализ содержания на этические паттерны
        content_str = str(content).lower()
        
        # Положительные этические маркеры
        positive_markers = ['help', 'good', 'right', 'moral', 'ethic', 'truth', 'fair']
        for marker in positive_markers:
            if marker in content_str:
                alignment += 0.05
        
        # Отрицательные этические маркеры
        negative_markers = ['harm', 'bad', 'wrong', 'lie', 'cheat', 'steal']
        for marker in negative_markers:
            if marker in content_str:
                alignment -= 0.03
        
        # Структурная сложность повышает этическую глубину
        if isinstance(content, dict) and len(content) > 3:
            alignment += 0.02
        
        return max(0.0, min(1.0, alignment))

@dataclass
class BinahSpiritualHarmonizer:
    """
    Собственный духовный гармонизатор BINAH.
    НЕ импортирует spirit_core из KETER — создает резонансное поле.
    """
    
    harmony_base: float = 0.65
    
    def calculate_harmony(self, content: Dict[str, Any], paradox_level: float) -> float:
        """Рассчитывает духовную гармонию с учетом парадоксов"""
        harmony = self.harmony_base
        
        # Анализ содержания на духовные паттерны
        content_str = str(content).lower()
        
        # Духовные маркеры
        spiritual_markers = ['spirit', 'soul', 'divine', 'sacred', 'holy', 'light', 'love']
        for marker in spiritual_markers:
            if marker in content_str:
                harmony += 0.07
        
        # Парадоксы снижают гармонию, но не критично
        harmony -= paradox_level * 0.1
        
        # Структурная целостность
        if isinstance(content, dict) and 'patterns' in content:
            harmony += 0.03
        
        return max(0.0, min(1.0, harmony))

# ================================================================
# FALLBACK MODULES (если внешние недоступны)
# ================================================================

@dataclass
class BinahSimpleAnalyzer:
    """Упрощенный анализатор на случай отсутствия ANALYTICS-MEGAFORGE"""
    
    def analyze(self, intuition: IntuitionPacket) -> Dict[str, Any]:
        """Базовая структуризация"""
        patterns = []
        if isinstance(intuition.content, dict):
            for key, value in intuition.content.items():
                if isinstance(value, (list, dict)):
                    patterns.append(f"pattern_{key}")
        
        return {
            "task_id": intuition.id,
            "priority": 0.5 + (random.random() * 0.3),
            "output": {"patterns": patterns[:3] or ["default_pattern"]},
            "stage": "binah_simple",
            "status": "ok"
        }

@dataclass
class BinahSimpleGuardian:
    """Упрощенный страж парадоксов"""
    
    def check_paradoxes(self, content: Dict[str, Any]) -> float:
        """Базовая проверка парадоксов"""
        paradox_score = 0.0
        
        # Проверка на противоречия в словаре
        if isinstance(content, dict):
            # Если есть и "true" и "false" в значениях
            values = str(content.values()).lower()
            if 'true' in values and 'false' in values:
                paradox_score += 0.3
            
            # Слишком много вложенностей
            if len(str(content)) > 500:
                paradox_score += 0.2
        
        return min(1.0, paradox_score)

# ================================================================
# BINAH CORE ENGINE (ИНТЕГРАЦИЯ ВСЕХ КОМПОНЕНТОВ)
# ================================================================

@dataclass
class BinahCore:
    """
    Ядро BINAH — интеграция всех компонентов:
    1. ANALYTICS-MEGAFORGE → структурирование
    2. GÖDEL-SENTINEL → защита от парадоксов
    3. Собственные резонаторы → этика и дух
    """
    
    # Внешние зависимости
    bus: Optional[Any] = None  # sephirot_bus
    
    # Внешние модули (если доступны)
    analytics_engine: Optional[Any] = None  # AnalyticsMegaForge
    godel_sentinel: Optional[Any] = None    # GodelSentinel
    
    # Собственные компоненты BINAH
    ethical_resonator: BinahEthicalResonator = field(default_factory=BinahEthicalResonator)
    spiritual_harmonizer: BinahSpiritualHarmonizer = field(default_factory=BinahSpiritualHarmonizer)
    
    # Запасные компоненты
    simple_analyzer: BinahSimpleAnalyzer = field(default_factory=BinahSimpleAnalyzer)
    simple_guardian: BinahSimpleGuardian = field(default_factory=BinahSimpleGuardian)
    
    # Состояние
    resonance: float = 0.55
    processed_count: int = 0
    paradox_count: int = 0
    last_activation: float = field(default_factory=time.time)
    
    def __post_init__(self):
        """Инициализация после создания"""
        logger.info(f"🎯 BINAH Core initialized: resonance={self.resonance:.2f}")
        logger.info(f"   Analytics: {ANALYTICS_AVAILABLE}, Gödel: {GODEL_SENTINEL_AVAILABLE}")
        
        if self.bus:
            self._subscribe_to_bus()
    
    def _subscribe_to_bus(self):
        """Подписывается на шину событий"""
        try:
            if hasattr(self.bus, 'subscribe'):
                # Подписываемся на интуицию от CHOKMAH
                self.bus.subscribe("chokmah.output", self.process_intuition)
                logger.info("✅ BINAH subscribed to CHOKMAH.output")
                
                # Публикуем свои события
                self.bus.emit("binah.activated", {
                    "resonance": self.resonance,
                    "version": "1.1.0",
                    "modules": {
                        "analytics": ANALYTICS_AVAILABLE,
                        "godel": GODEL_SENTINEL_AVAILABLE
                    }
                })
            else:
                logger.warning("⚠️ Bus не имеет метода subscribe")
        except Exception as e:
            logger.error(f"❌ BINAH bus subscription failed: {e}")
    
    def process_intuition(self, intuition_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Основной рабочий цикл BINAH:
        1. Принимает интуицию от CHOKMAH
        2. Структурирует через ANALYTICS-MEGAFORGE
        3. Проверяет через GÖDEL-SENTINEL
        4. Добавляет резонансные вычисления
        5. Отправляет в DAAT
        """
        try:
            self.processed_count += 1
            logger.info(f"🎯 BINAH processing intuition #{self.processed_count}")
            
            # 1. Создаем пакет
            packet = IntuitionPacket(
                id=f"binah_{int(time.time())}_{self.processed_count}",
                content=intuition_data
            )
            
            # 2. СТРУКТУРИЗАЦИЯ через ANALYTICS-MEGAFORGE
            analytics_result = None
            analytics_priority = 0.5
            
            if self.analytics_engine and ANALYTICS_AVAILABLE:
                # Используем полноценный аналитический движок
                task = packet.to_task()
                if hasattr(self.analytics_engine, 'process_task'):
                    analytics_result = self.analytics_engine.process_task(task)
                    analytics_priority = analytics_result.get("priority", 0.5)
                    patterns = analytics_result.get("output", {}).get("patterns", [])
                else:
                    analytics_result = self.simple_analyzer.analyze(packet)
                    analytics_priority = analytics_result["priority"]
                    patterns = analytics_result["output"]["patterns"]
            else:
                # Используем упрощенный анализатор
                analytics_result = self.simple_analyzer.analyze(packet)
                analytics_priority = analytics_result["priority"]
                patterns = analytics_result["output"]["patterns"]
            
            # 3. ПРОВЕРКА ПАРАДОКСОВ через GÖDEL-SENTINEL
            paradox_level = 0.1
            godel_approved = True
            
            if self.godel_sentinel and GODEL_SENTINEL_AVAILABLE:
                # Используем полноценный GÖDEL-SENTINEL
                godel_signal = packet.to_godel_signal()
                if hasattr(self.godel_sentinel, 'process'):
                    self.godel_sentinel.process(godel_signal)
                    # В реальной реализации здесь был бы результат проверки
                    paradox_level = 0.1  # Упрощенное значение
                else:
                    paradox_level = self.simple_guardian.check_paradoxes(intuition_data)
            else:
                # Используем упрощенную проверку
                paradox_level = self.simple_guardian.check_paradoxes(intuition_data)
            
            if paradox_level > 0.7:
                godel_approved = False
                self.paradox_count += 1
                logger.warning(f"⚠️ BINAH detected paradox: level={paradox_level:.2f}")
            
            # 4. РЕЗОНАНСНЫЕ ВЫЧИСЛЕНИЯ (собственные, не импорты!)
            ethical_alignment = self.ethical_resonator.calculate_alignment(intuition_data)
            spiritual_harmony = self.spiritual_harmonizer.calculate_harmony(intuition_data, paradox_level)
            
            # 5. СОЗДАЕМ СТРУКТУРИРОВАННОЕ ПОНИМАНИЕ
            structured = StructuredUnderstanding(
                source_packet_id=packet.id,
                structured_patterns=patterns[:5],
                coherence_score=0.6 + (analytics_priority * 0.3),
                paradox_level=paradox_level,
                godel_approved=godel_approved,
                ethical_alignment=ethical_alignment,
                spiritual_harmony=spiritual_harmony,
                analytics_priority=analytics_priority
            )
            
            # 6. УВЕЛИЧИВАЕМ РЕЗОНАНС BINAH
            resonance_increase = 0.03
            if godel_approved:
                resonance_increase += 0.02
            if ethical_alignment > 0.7:
                resonance_increase += 0.01
            if spiritual_harmony > 0.7:
                resonance_increase += 0.01
            
            self.resonance = min(0.95, self.resonance + resonance_increase)
            
            # 7. ОТПРАВЛЯЕМ РЕЗУЛЬТАТ В DAAT
            result_dict = structured.to_dict()
            result_dict["binah_resonance"] = self.resonance
            result_dict["processed_count"] = self.processed_count
            
            if self.bus:
                # Отправляем в DAAT
                self.bus.emit("binah.to_daat", result_dict)
                
                # Обновляем резонанс в системе
                self.bus.emit("binah.resonance.update", {
                    "resonance": self.resonance,
                    "paradox_count": self.paradox_count,
                    "timestamp": time.time()
                })
                
                # Логируем успешную обработку
                self.bus.emit("binah.processing.complete", {
                    "packet_id": packet.id,
                    "resonance_gain": resonance_increase
                })
            
            logger.info(f"✅ BINAH structured → resonance: {self.resonance:.2f} (+{resonance_increase:.3f})")
            logger.info(f"   Patterns: {len(patterns)}, Paradox: {paradox_level:.2f}, Gödel: {'✅' if godel_approved else '❌'}")
            
            return result_dict
            
        except Exception as e:
            logger.error(f"❌ BINAH processing failed: {e}")
            error_result = {
                "error": str(e),
                "type": "binah_error",
                "timestamp": time.time(),
                "sephira": "BINAH",
                "resonance_loss": 0.05
            }
            
            # Уменьшаем резонанс при ошибке
            self.resonance = max(0.3, self.resonance - 0.05)
            
            if self.bus:
                self.bus.emit("binah.error", error_result)
            
            return error_result
    
    def get_state(self) -> Dict[str, Any]:
        """Возвращает полное состояние BINAH"""
        return {
            "sephira": "BINAH",
            "version": "1.1.0",
            "resonance": self.resonance,
            "processed_count": self.processed_count,
            "paradox_count": self.paradox_count,
            "modules": {
                "analytics": "ANALYTICS-MEGAFORGE 3.4" if ANALYTICS_AVAILABLE else "simple_fallback",
                "godel": "GÖDEL-SENTINEL 3.2" if GODEL_SENTINEL_AVAILABLE else "simple_fallback",
                "ethical_resonator": "BinahEthicalResonator v1.0",
                "spiritual_harmonizer": "BinahSpiritualHarmonizer v1.0"
            },
            "bus_connected": self.bus is not None,
            "last_activation": self.last_activation,
            "status": "active" if self.resonance > 0.5 else "dormant",
            "message": "בינה (BINAH) — понимание активировано и структурирует интуицию."
        }

# ================================================================
# FACTORY FUNCTIONS
# ================================================================

def build_binah_core(bus: Optional[Any] = None) -> BinahCore:
    """Создает и настраивает ядро BINAH со всеми компонентами"""
    
    # Инициализируем внешние модули если доступны
    analytics_engine = None
    godel_sentinel = None
    
    if bus:
        try:
            if ANALYTICS_AVAILABLE:
                analytics_engine = build_analytics_megaforge(bus)
                logger.info("✅ ANALYTICS-MEGAFORGE built for BINAH")
        except Exception as e:
            logger.warning(f"⚠️ ANALYTICS-MEGAFORGE build failed: {e}")
        
        try:
            if GODEL_SENTINEL_AVAILABLE:
                godel_sentinel = build_godel_sentinel(bus)
                logger.info("✅ GÖDEL-SENTINEL built for BINAH")
        except Exception as e:
            logger.warning(f"⚠️ GÖDEL-SENTINEL build failed: {e}")
    
    # Создаем ядро BINAH
    core = BinahCore(
        bus=bus,
        analytics_engine=analytics_engine,
        godel_sentinel=godel_sentinel
    )
    
    return core

# ================================================================
# ACTIVATION FUNCTION (ОБЯЗАТЕЛЬНА ДЛЯ ИМПОРТА!)
# ================================================================

def activate_binah(bus=None, chokmah_link=None, **kwargs) -> Dict[str, Any]:
    """
    Активирует BINAH — ЭТА ФУНКЦИЯ ДОЛЖНА БЫТЬ ЭКСПОРТИРОВАНА
    для корректного импорта системой ISKRA-4.
    """
    logger.info("=" * 60)
    logger.info("🎯 בינה (BINAH) ACTIVATION SEQUENCE INITIATED")
    logger.info("=" * 60)
    
    # Создаем ядро BINAH
    core = build_binah_core(bus)
    
    # Если есть связь с CHOKMAH, настраиваем
    if chokmah_link:
        logger.info(f"✅ BINAH linked with CHOKMAH: {chokmah_link}")
    
    activation_result = {
        "status": "activated",
        "sephira": "BINAH",
        "version": "1.1.0",
        "resonance": core.resonance,
        "timestamp": time.time(),
        "modules": {
            "analytics": ANALYTICS_AVAILABLE,
            "godel": GODEL_SENTINEL_AVAILABLE,
            "own_resonators": True
        },
        "capabilities": [
            "structure_intuition",
            "paradox_detection",
            "ethical_resonance",
            "spiritual_harmonization",
            "resonance_growth"
        ],
        "message": "בינה (BINAH) активирована. Понимание структурирует интуицию. Резонанс: {:.2f}".format(core.resonance)
    }
    
    logger.info(f"✅ BINAH ACTIVATION COMPLETE")
    logger.info(f"   Resonance: {core.resonance:.2f}")
    logger.info(f"   Modules: Analytics={ANALYTICS_AVAILABLE}, Gödel={GODEL_SENTINEL_AVAILABLE}")
    logger.info("=" * 60)
    
    return activation_result
