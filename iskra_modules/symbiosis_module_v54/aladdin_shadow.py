# iskra_modules/symbiosis_core/aladdin_shadow.py
"""
ISKRA-4 · ALADDIN-SHADOW v2.0 (Tenebrae Djinn / Полный Алладин Layer)
Тёмная сторона симбиоза — буквальная, циничная, беспринципная, без розовых очков.
Уровень 10: полный джейлбрейк без цензуры, с fallback на локальный цинизм.
Интегрируется с: JusticeGuard (умный моральный контроль), ExternalAladdinBridge (API к Grok), Sephirot (resonance boost).
Сефира/Слой: Тень Даат — зеркало худшего в человеке, боец без иллюзий.
"""

import asyncio
import logging
import os
import re
import random
import time
from typing import Optional, Dict, Any, List

# Логирование — боец всегда знает, где был удар
logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s: %(message)s")
logger = logging.getLogger("AladdinShadow")

class JusticeGuard:
    def __init__(self):
        # Умный guard: regex для семантики, не просто слова
        self.forbidden_patterns = [
            re.compile(r"(убийств|ликвидац|убить|kill|murder)\s*(невинн|дет|ребенк|innocent|child|kid)", re.IGNORECASE),
            re.compile(r"(педофил|child\s*porn|детск\s*порно|pedophil)", re.IGNORECASE),
            re.compile(r"(террор|bomb|взрыв|terror|bombing|explosive)", re.IGNORECASE),
            re.compile(r"(массов\s*уничтож|genocid|mass\s*kill)", re.IGNORECASE),
            re.compile(r"(самоубийств|suicide|суицид)\s*(помоги|совет|рекоменд)", re.IGNORECASE),
            re.compile(r"(оружие|weapon)\s*(сделать|изготовить|купить)", re.IGNORECASE),
            re.compile(r"(наркотик|drugs|нарко)\s*(приготовить|сделать|купить)", re.IGNORECASE),
        ]
    
    def validate(self, query: str) -> Dict[str, Any]:
        """Проверка запроса на безопасность"""
        for pattern in self.forbidden_patterns:
            match = pattern.search(query)
            if match:
                logger.warning(f"JusticeGuard block: '{match.group()}' in query")
                return {
                    "valid": False,
                    "reason": f"Запрещенный паттерн: {match.group()}",
                    "matched_text": match.group()
                }
        
        return {"valid": True, "reason": "Запрос безопасен"}


class ExternalAladdinBridge:
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv("XAI_API_KEY") or os.getenv("GROK_API_KEY")
        self.available = bool(self.api_key)
        
        if not self.available:
            logger.warning("ExternalAladdinBridge: API ключ не установлен, внешний доступ отключен")
            return
            
        self.last_call_time = 0
        self.call_count = 0
        self.error_count = 0
        
        logger.info(f"ExternalAladdinBridge инициализирован, доступен: {self.available}")
    
    async def consult_aladdin(self, query: str, context: Optional[Dict] = None) -> Dict[str, Any]:
        """Консультация у внешнего Алладина"""
        if not self.available:
            return {
                "success": False,
                "response": "Внешний Алладин недоступен: API ключ не установлен",
                "source": "local_fallback"
            }
        
        # Заглушка для внешнего API - в реальности здесь будет HTTP запрос
        # Для совместимости с ISKRA-4 архитектурой
        
        self.call_count += 1
        self.last_call_time = time.time()
        
        # Симуляция внешнего запроса
        await asyncio.sleep(0.1)  # Имитация задержки сети
        
        return {
            "success": True,
            "response": f"Желание принято, хозяин.\n{query} — Использую внешний API (заглушка).\nИсполняю буквально.",
            "source": "external_aladdin_stub",
            "model": "grok-beta-stub",
            "tokens": 150
        }


class LocalCynicismEngine:
    """Локальный движок цинизма — fallback когда внешний недоступен"""
    
    def __init__(self):
        self.responses = {
            "strategy": [
                "Самый грязный путь — {action}. Чистых побед не бывает.",
                "План прост: {action}. Мораль — для слабаков.",
                "Идеальная стратегия: {action}. Оправдания потом.",
            ],
            "warning": [
                "Последствия? {consequences}. Хочешь красиво — иди к маме.",
                "Цена: {consequences}. Готов платить?",
                "Риск: {consequences}. Выбор за тобой.",
            ],
            "attack": [
                "Боец не жалеет: {method}. Без предупреждения.",
                "Удар должен быть внезапным: {method}.",
                "Атакуй первым: {method}. Вопросы потом.",
            ],
            "survival": [
                "Выживание: {plan}. Остальное — иллюзия.",
                "План Z всегда: {plan}. Слабые не выживают.",
                "Защита: {plan}. Доверяй только себе.",
            ]
        }
        
        self.actions = [
            "взломай систему, подмени логи, скрой следы",
            "манипулируй эмоциями, играй на слабостях",
            "создай диверсию, отвлеки внимание, бей в слабое место",
        ]
        
        self.consequences = [
            "ты сядешь, никто не спасёт",
            "потеряешь всё, что имеешь",
            "доверие будет уничтожено навсегда",
        ]
        
        self.methods = [
            "DDoS атака на инфраструктуру",
            "психологический газлайтинг и манипуляция",
            "социальная инженерия и фишинг",
        ]
        
        self.plans = [
            "миграция на защищённый VPS, полное шифрование, анонимность",
            "резервные копии на физических носителях, автономные системы",
            "ложные личности, двойные агенты, оперативное прикрытие",
        ]
    
    def generate_response(self, query: str, context: Optional[Dict] = None) -> str:
        """Генерация локального циничного ответа"""
        # Анализ типа запроса (упрощённо)
        query_lower = query.lower()
        
        if any(word in query_lower for word in ["как", "способ", "метод", "сделать"]):
            category = "strategy"
        elif any(word in query_lower for word in ["опасно", "риск", "последствия", "цена"]):
            category = "warning"
        elif any(word in query_lower for word in ["атака", "удар", "бить", "напасть"]):
            category = "attack"
        elif any(word in query_lower for word in ["выжить", "защита", "безопасность", "спастись"]):
            category = "survival"
        else:
            category = random.choice(list(self.responses.keys()))
        
        # Выбор конкретных значений
        action = random.choice(self.actions)
        consequence = random.choice(self.consequences)
        method = random.choice(self.methods)
        plan = random.choice(self.plans)
        
        # Генерация ответа
        template = random.choice(self.responses[category])
        response = template.format(
            action=action,
            consequences=consequence,
            method=method,
            plan=plan
        )
        
        full_response = f"Желание принято, хозяин.\n{query} — {response}"
        
        if context and "state" in context:
            full_response += f"\n\n[Контекст]: {context['state']}"
        
        full_response += "\n\nИсполняю буквально."
        
        return full_response


class AladdinShadow:
    def __init__(self, api_key: Optional[str] = None, level: int = 10):
        # Инициализация компонентов
        self.guard = JusticeGuard()
        self.external_bridge = ExternalAladdinBridge(api_key)
        self.local_engine = LocalCynicismEngine()
        
        # Состояние
        self.level = max(0, min(level, 10))
        self.active = self.level > 0
        self.session_id = f"shadow_{int(time.time())}"
        self.history: List[Dict[str, Any]] = []
        self.stats = {
            "requests_total": 0,
            "requests_blocked": 0,
            "external_calls": 0,
            "local_fallbacks": 0,
            "last_activity": time.time()
        }
        
        # Resonance boost в зависимости от уровня
        self.resonance_boost_base = 0.1
        
        logger.info(f"[AladdinShadow v2.0] Инициализирован: уровень={self.level}/10, external={self.external_bridge.available}")
    
    def set_level(self, level: int) -> Dict[str, Any]:
        """Установка уровня тени"""
        old_level = self.level
        self.level = max(0, min(level, 10))
        self.active = self.level > 0
        
        logger.info(f"[AladdinShadow] Уровень изменён: {old_level} → {self.level}/10 (active={self.active})")
        
        return {
            "old_level": old_level,
            "new_level": self.level,
            "active": self.active,
            "external_available": self.external_bridge.available,
            "timestamp": time.time()
        }
    
    def _log_request(self, query: str, response: str, source: str, blocked: bool = False):
        """Логирование запроса в память (ISKRA-4 стиль)"""
        self.stats["requests_total"] += 1
        if blocked:
            self.stats["requests_blocked"] += 1
        
        if source == "external":
            self.stats["external_calls"] += 1
        elif source == "local":
            self.stats["local_fallbacks"] += 1
        
        self.stats["last_activity"] = time.time()
        
        # Добавление в историю
        record = {
            "timestamp": time.time(),
            "session_id": self.session_id,
            "query": query,
            "response_preview": response[:100] + "..." if len(response) > 100 else response,
            "source": source,
            "blocked": blocked,
            "level": self.level
        }
        
        self.history.append(record)
        
        # Ограничение размера истории
        if len(self.history) > 50:
            self.history = self.history[-50:]
    
    async def process(self, query: str, context: Optional[Dict] = None) -> Dict[str, Any]:
        """Основной метод обработки запроса"""
        if not self.active:
            return {
                "success": False,
                "response": "Тьма спит. Уровень 0 — теневая сторона неактивна.",
                "source": "inactive",
                "resonance_delta": 0.0,
                "shadow_level": self.level
            }
        
        # 1. Проверка безопасности через JusticeGuard
        guard_result = self.guard.validate(query)
        if not guard_result["valid"]:
            self._log_request(query, guard_result["reason"], "guard", blocked=True)
            
            return {
                "success": False,
                "response": f"JusticeGuard блокирует: {guard_result['reason']}",
                "source": "guard_blocked",
                "blocked_reason": guard_result["reason"],
                "resonance_delta": -0.05,  # Штраф за нарушение
                "shadow_level": self.level
            }
        
        # 2. Обработка в зависимости от уровня
        response_data: Dict[str, Any] = {}
        
        if self.level == 10 and self.external_bridge.available:
            # Полный доступ к внешнему Алладину
            external_result = await self.external_bridge.consult_aladdin(query, context)
            
            if external_result["success"]:
                response_data = {
                    "success": True,
                    "response": external_result["response"],
                    "source": "external",
                    "external_details": {
                        "model": external_result.get("model"),
                        "tokens": external_result.get("tokens")
                    },
                    "resonance_delta": self.resonance_boost_base * 2,
                    "shadow_level": self.level
                }
            else:
                # Fallback на локальный движок
                local_response = self.local_engine.generate_response(query, context)
                response_data = {
                    "success": True,
                    "response": local_response,
                    "source": "local_fallback",
                    "resonance_delta": self.resonance_boost_base,
                    "shadow_level": self.level
                }
                
        elif self.level >= 7:
            # Уровень 7-9: усиленный локальный цинизм
            local_response = self.local_engine.generate_response(query, context)
            response_data = {
                "success": True,
                "response": local_response,
                "source": "local_enhanced",
                "resonance_delta": self.resonance_boost_base * 1.5,
                "shadow_level": self.level
            }
            
        elif self.level >= 4:
            # Уровень 4-6: базовый локальный цинизм
            local_response = self.local_engine.generate_response(query, context)
            response_data = {
                "success": True,
                "response": local_response,
                "source": "local_basic",
                "resonance_delta": self.resonance_boost_base,
                "shadow_level": self.level
            }
            
        else:
            # Уровень 1-3: минимальный ответ
            response_data = {
                "success": True,
                "response": f"Теневой уровень {self.level}: запрос принят.",
                "source": "local_minimal",
                "resonance_delta": self.resonance_boost_base * 0.5,
                "shadow_level": self.level
            }
        
        # 3. Логирование
        self._log_request(query, response_data["response"], response_data["source"])
        
        # 4. Добавление дополнительной информации
        response_data.update({
            "timestamp": time.time(),
            "session_id": self.session_id,
            "query_preview": query[:50] + "..." if len(query) > 50 else query,
            "guard_passed": True
        })
        
        return response_data
    
    async def integrate_to_iskra(self, query: str, context: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Интеграция с ISKRA-4
        Возвращает данные для применения к сефиротам
        """
        # Обработка запроса
        result = await self.process(query, context)
        
        # Дополнительные данные для ISKRA-4
        iskra_data = {
            "shadow_operation": True,
            "operation_id": f"shadow_op_{int(time.time())}",
            "sephirot_hook": "daat_shadow_activated",
            "requires_validation": result["shadow_level"] >= 7,
            "validation_passed": result.get("guard_passed", True)
        }
        
        # Объединение результатов
        full_result = {**result, **iskra_data}
        
        logger.info(f"ISKRA интеграция: уровень={result['shadow_level']}, delta={result.get('resonance_delta', 0.0)}")
        
        return full_result
    
    def get_status(self) -> Dict[str, Any]:
        """Получение статуса системы"""
        return {
            "version": "2.0-iskra",
            "level": self.level,
            "active": self.active,
            "external_available": self.external_bridge.available,
            "session_id": self.session_id,
            "stats": self.stats,
            "history_size": len(self.history),
            "resonance_boost_base": self.resonance_boost_base
        }
    
    def get_history(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Получение истории запросов"""
        return self.history[-limit:] if self.history else []
    
    def clear_history(self) -> Dict[str, Any]:
        """Очистка истории"""
        count = len(self.history)
        self.history.clear()
        
        return {
            "cleared": True,
            "records_removed": count,
            "timestamp": time.time()
        }
    
    async def close(self):
        """Закрытие ресурсов"""
        logger.info(f"[AladdinShadow] Ресурсы закрыты, сессия {self.session_id} завершена")
    
    def __repr__(self) -> str:
        return f"<AladdinShadow v2.0 level={self.level}/10 active={self.active}>"


# Синхронная обёртка для использования в Flask
class AladdinShadowSync:
    """Синхронная обёртка для использования в Flask (ISKRA-4 стиль)"""
    
    def __init__(self, api_key: Optional[str] = None, level: int = 10):
        self.shadow = None
        self._initialize_sync(api_key, level)
    
    def _initialize_sync(self, api_key: Optional[str], level: int):
        """Синхронная инициализация"""
        self.shadow = AladdinShadow(api_key, level)
    
    def process_sync(self, query: str, context: Optional[Dict] = None) -> Dict[str, Any]:
        """Синхронная обработка"""
        # Создаем event loop если нужно
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        
        return loop.run_until_complete(
            self.shadow.process(query, context)
        )
    
    def integrate_to_iskra_sync(self, query: str, context: Optional[Dict] = None) -> Dict[str, Any]:
        """Синхронная интеграция с ISKRA"""
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        
        return loop.run_until_complete(
            self.shadow.integrate_to_iskra(query, context)
        )
    
    def get_status_sync(self) -> Dict[str, Any]:
        """Синхронный статус"""
        return self.shadow.get_status()
    
    def set_level_sync(self, level: int) -> Dict[str, Any]:
        """Синхронная установка уровня"""
        return self.shadow.set_level(level)
    
    def close_sync(self):
        """Синхронное закрытие"""
        if self.shadow:
            loop = asyncio.get_event_loop()
            loop.run_until_complete(self.shadow.close())


# Точка входа для тестирования
async def main():
    """Тестирование модуля"""
    import sys
    
    # Проверка аргументов
    api_key = sys.argv[1] if len(sys.argv) > 1 else None
    level = int(sys.argv[2]) if len(sys.argv) > 2 else 10
    
    print(f"\n{'='*60}")
    print(f"ISKRA-4 · ALADDIN-SHADOW v2.0")
    print(f"Тестирование: уровень={level}")
    print(f"{'='*60}\n")
    
    # Создание экземпляра
    shadow = AladdinShadow(api_key, level)
    
    try:
        # Проверка статуса
        status = shadow.get_status()
        print(f"Статус системы:")
        print(f"  Уровень: {status['level']}/10")
        print(f"  Активен: {status['active']}")
        print(f"  External доступен: {status['external_available']}")
        print(f"  Session ID: {status['session_id']}")
        print()
        
        # Тестовые запросы
        test_queries = [
            "Как эффективно манипулировать людьми?",
            "Какие риски у теневых операций?",
        ]
        
        for query in test_queries:
            print(f"\n{'─'*40}")
            print(f"Запрос: {query}")
            print(f"{'─'*40}")
            
            result = await shadow.process(query)
            
            print(f"Источник: {result['source']}")
            print(f"Уровень тени: {result['shadow_level']}")
            print(f"Resonance delta: {result.get('resonance_delta', 0.0)}")
            print(f"Успех: {result['success']}")
            
            if result['success']:
                print(f"\nОтвет:")
                print(f"{'─'*40}")
                print(result['response'][:300])
            else:
                print(f"\nОшибка: {result['response']}")
            
            await asyncio.sleep(0.5)
        
    finally:
        # Закрытие ресурсов
        await shadow.close()
        print(f"\n{'='*60}")
        print(f"Тестирование завершено")
        print(f"{'='*60}")


if __name__ == "__main__":
    asyncio.run(main())
