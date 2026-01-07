"""
wisdom_core.py - минимальное ядро сефиры CHOKMAH.
Только управление состоянием и оркестрация вызовов.
"""

import logging
import asyncio
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)


class WisdomCore:
    """Ядро CHOKMAH: управление состоянием и оркестрация модулей."""
    
    def __init__(self, config=None):
        # Состояние сефиры
        self.name: str = "CHOKMAH"
        self.resonance: float = 0.3
        self.energy: float = 0.9
        self.status: str = "dormant"
        self.config = config or {}
        
        # Модули (инициализируются при активации)
        self.intuition_matrix = None
        self.chernigovskaya = None
        
        logger.info(f"Инициализировано ядро {self.name}")

     async def initialize(self):
        """
        Инициализация ядра мудрости.
        Вызывается из sephirotic_engine.py при активации CHOKMAH.
        """
        # Простая реализация - только меняем статус
        self.status = "initialized"
        
        # Логируем успех
        logger.info(f"✅ Ядро {self.name} инициализировано (resonance: {self.resonance:.2f})")
        
        # Возвращаем стандартный ответ как в KETER и DAAT
        return {
            "status": "initialized", 
            "core": self.name,
            "resonance": self.resonance,
            "energy": self.energy
        }
    
    async def _load_modules(self) -> bool:
        """Загружает и инициализирует модули CHOKMAH."""
        try:
            # Импортируем интуитивную матрицу
            from .intuition_matrix import build_intuition_matrix
            self.intuition_matrix = build_intuition_matrix()
            
            # Импортируем процессор Черниговской
            from bechtereva_chernigovskaya.chernigovskaya import ChernigovskayaProcessor
            self.chernigovskaya = ChernigovskayaProcessor()
            
            logger.info("Модули CHOKMAH загружены успешно")
            return True
            
        except ImportError as e:
            logger.error(f"Ошибка загрузки модулей CHOKMAH: {e}")
            return False
        except Exception as e:
            logger.error(f"Ошибка инициализации модулей: {e}")
            return False
    
    async def activate(self) -> Dict[str, Any]:
        """Активирует сефиру CHOKMAH, загружает модули."""
        if self.status == "active":
            return await self.get_status()
        
        # Загружаем модули
        modules_loaded = await self._load_modules()
        
        if not modules_loaded:
            return {
                "sephira": self.name,
                "status": "error",
                "resonance": self.resonance,
                "error": "Не удалось загрузить модули CHOKMAH"
            }
        
        # Меняем состояние
        self.status = "active"
        self.resonance = 0.65  # Целевое значение после активации
        
        logger.info(f"CHOKMAH активирован. Резонанс: {self.resonance}")
        
        return {
            "sephira": self.name,
            "status": self.status,
            "resonance": round(self.resonance, 3),
            "energy": round(self.energy, 3),
            "modules": ["intuition_matrix", "chernigovskaya"]
        }
    
    async def process(self, text: str, context: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Основной метод обработки запроса.
        
        Args:
            text: Текст для обработки
            context: Дополнительный контекст
            
        Returns:
            Результат с интуитивным инсайтом
        """
        if self.status != "active":
            return {
                "sephira": self.name,
                "status": "inactive",
                "error": "CHOKMAH не активирован. Вызовите activate() сначала."
            }
        
        if not self.intuition_matrix or not self.chernigovskaya:
            return {
                "sephira": self.name,
                "status": "error",
                "error": "Модули CHOKMAH не загружены"
            }
        
        try:
            # 1. NLP-обработка через Черниговскую
            processed_text = await self._process_with_chernigovskaya(text)
            
            # 2. Интуитивный инсайт через матрицу
            insight = await self._generate_insight(processed_text, context or {})
            
            # 3. Увеличиваем резонанс
            self.resonance = min(1.0, self.resonance + 0.01)
            
            return {
                "sephira": self.name,
                "status": "success",
                "resonance": round(self.resonance, 3),
                "input": text,
                "processed": processed_text,
                "insight": insight
            }
            
        except Exception as e:
            logger.error(f"Ошибка обработки в CHOKMAH: {e}")
            return {
                "sephira": self.name,
                "status": "error",
                "error": str(e)
            }
    
    async def _process_with_chernigovskaya(self, text: str) -> str:
        """Обработка текста через модуль Черниговской."""
        try:
            # Пробуем разные возможные методы
            if hasattr(self.chernigovskaya, 'process'):
                result = self.chernigovskaya.process(text)
                if asyncio.iscoroutine(result):
                    result = await result
                return str(result)
                
            elif hasattr(self.chernigovskaya, 'analyze'):
                result = self.chernigovskaya.analyze(text)
                if asyncio.iscoroutine(result):
                    result = await result
                return str(result)
                
            else:
                # Если не нашли подходящий метод, возвращаем исходный текст
                return text
                
        except Exception as e:
            logger.warning(f"Ошибка обработки chernigovskaya: {e}, используем исходный текст")
            return text
    
    async def _generate_insight(self, text: str, context: Dict) -> str:
        """Генерация интуитивного инсайта через матрицу."""
        try:
            # Пробуем разные возможные методы
            if hasattr(self.intuition_matrix, 'generate'):
                result = self.intuition_matrix.generate(text, context)
                if asyncio.iscoroutine(result):
                    result = await result
                return str(result)
                
            elif hasattr(self.intuition_matrix, 'get_insight'):
                result = self.intuition_matrix.get_insight(text)
                if asyncio.iscoroutine(result):
                    result = await result
                return str(result)
                
            elif hasattr(self.intuition_matrix, 'process'):
                result = self.intuition_matrix.process({"text": text, "context": context})
                if asyncio.iscoroutine(result):
                    result = await result
                return str(result)
                
            else:
                # Fallback: простой инсайт
                return f"Интуитивное восприятие: '{text[:50]}...' (резонанс: {self.resonance})"
                
        except Exception as e:
            logger.warning(f"Ошибка intuition_matrix: {e}, используем fallback")
            return f"Инсайт на основе '{text[:30]}...'"
    
    async def get_status(self) -> Dict[str, Any]:
        """Возвращает текущее состояние."""
        return {
            "sephira": self.name,
            "status": self.status,
            "resonance": round(self.resonance, 3),
            "energy": round(self.energy, 3),
            "is_active": self.status == "active" and self.resonance > 0.6,
            "modules_loaded": self.intuition_matrix is not None and self.chernigovskaya is not None
        }
    
    def is_activated(self) -> bool:
        """Проверка, достигнут ли целевой уровень активации."""
        return self.status == "active" and self.resonance >= 0.6
