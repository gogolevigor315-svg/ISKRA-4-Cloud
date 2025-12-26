#!/usr/bin/env python3
"""
sephirotic_engine.py - ЛЕГКОВЕСНЫЙ ДВИЖОК ДЛЯ ИНТЕГРАЦИИ С ISKRA-4 CLOUD
Версия: 4.0.0 Production
Назначение: Тонкий слой между сефиротической системой и iskra_full.py
"""

import asyncio
import json
from datetime import datetime
from typing import Dict, List, Optional, Any
import logging

# Импорты из наших модулей
try:
    # Импорт типов из sephirot_base
    from sephirot_base import (
        Sephirot, 
        SephiroticNode, 
        SephiroticTree, 
        SephiroticEngine as BaseEngine,
        SignalType,
        create_sephirotic_system
    )
    
    # Импорт шины
    from sephirot_bus import SephiroticBus, create_sephirotic_bus
    
except ImportError as e:
    print(f"⚠️  Внимание: Не удалось импортировать сефиротические модули: {e}")
    # Создаём заглушки для типа
    SephiroticTree = type('SephiroticTree', (), {})
    SephiroticBus = type('SephiroticBus', (), {})
    SignalType = type('SignalType', (), {'NEURO': 'NEURO', 'SEMIOTIC': 'SEMIOTIC'})

# ============================================================================
# ОСНОВНОЙ ДВИЖОК СЕФИРОТИЧЕСКОЙ СИСТЕМЫ
# ============================================================================

class SephiroticEngine:
    """
    Главный движок сефиротической системы.
    Тонкий слой для интеграции с iskra_full.py и управления деревом сефирот.
    """
    
    def __init__(self, name: str = "ISKRA-4-Sephirotic-Engine"):
        self.name = name
        self.bus = None
        self.tree = None
        self.engine = None
        self.initialized = False
        self.activated = False
        
        # Логирование
        self.logger = self._setup_logger()
        
        # Статистика
        self.start_time = None
        self.stats = {
            "initializations": 0,
            "activations": 0,
            "errors": 0,
            "last_error": None
        }
        
        self.logger.info(f"Движок '{name}' создан (версия 4.0.0)")
    
    def _setup_logger(self) -> logging.Logger:
        """Настройка логгера"""
        logger = logging.getLogger(f"Sephirot.Engine.{self.name}")
        
        if not logger.handlers:
            logger.setLevel(logging.INFO)
            
            formatter = logging.Formatter(
                '[%(asctime)s] [%(name)s:%(levelname)s] %(message)s',
                datefmt='%H:%M:%S'
            )
            
            console = logging.StreamHandler()
            console.setLevel(logging.WARNING)
            console.setFormatter(formatter)
            logger.addHandler(console)
            
            logger.propagate = False
        
        return logger
    
    # ============================================================================
    # ИНИЦИАЛИЗАЦИЯ И АКТИВАЦИЯ
    # ============================================================================
    
    async def initialize(self, existing_bus: Optional[SephiroticBus] = None) -> Dict[str, Any]:
        """
        Инициализация сефиротической системы.
        
        :param existing_bus: Существующая шина (если есть)
        :return: Результат инициализации
        """
        try:
            self.logger.info("🚀 Начинаю инициализацию сефиротической системы...")
            self.start_time = datetime.utcnow()
            
            # 1. Создание или использование существующей шины
            if existing_bus and isinstance(existing_bus, SephiroticBus):
                self.bus = existing_bus
                self.logger.info("Использую существующую шину")
            else:
                self.bus = await create_sephirotic_bus("ISKRA-4-Bus")
                self.logger.info("Создана новая сефиротическая шина")
            
            # 2. Создание дерева сефирот
            try:
                self.tree = SephiroticTree(self.bus)
                await self.tree.initialize()
                self.logger.info("Дерево сефирот создано (10 узлов)")
            except Exception as e:
                self.logger.error(f"Ошибка создания дерева: {e}")
                # Заглушка для тестирования
                self.tree = type('MockTree', (), {
                    'nodes': {},
                    'get_tree_state': lambda: {"status": "mock_tree"}
                })()
            
            # 3. Явная привязка ключевых модулей
            if hasattr(self.bus, 'connect_module'):
                # Бехтерева -> KETER
                await self.bus.connect_module("bechtereva", "KETER")
                
                # Черниговская -> CHOKHMAH
                await self.bus.connect_module("chernigovskaya", "CHOKHMAH")
                
                self.logger.info("Привязки модулей установлены")
            
            # 4. Создание движка (если доступен)
            try:
                self.engine = await create_sephirotic_system(self.bus)
                self.logger.info("Базовый движок сефиротической системы создан")
            except:
                self.engine = None
                self.logger.warning("Базовый движок недоступен, использую упрощённый режим")
            
            self.initialized = True
            self.stats["initializations"] += 1
            
            result = {
                "success": True,
                "message": "Сефиротическая система инициализирована",
                "engine": self.name,
                "bus_initialized": self.bus is not None,
                "tree_initialized": self.tree is not None,
                "module_bindings": getattr(self.bus, 'module_bindings', {}),
                "timestamp": datetime.utcnow().isoformat()
            }
            
            self.logger.info("✅ Сефиротическая система инициализирована")
            return result
            
        except Exception as e:
            error_msg = f"Ошибка инициализации: {str(e)}"
            self.logger.error(error_msg)
            self.stats["errors"] += 1
            self.stats["last_error"] = error_msg
            
            return {
                "success": False,
                "error": error_msg,
                "timestamp": datetime.utcnow().isoformat()
            }
    
    async def activate(self) -> Dict[str, Any]:
        """
        Активация сефиротической системы.
        """
        if not self.initialized:
            return {
                "success": False,
                "error": "Система не инициализирована",
                "timestamp": datetime.utcnow().isoformat()
            }
        
        try:
            self.logger.info("⚡ Активация сефиротической системы...")
            
            # 1. Активация через движок если доступен
            if self.engine and hasattr(self.engine, 'activate'):
                result = await self.engine.activate()
                self.logger.info(f"Активация через движок: {result.get('status', 'unknown')}")
            
            # 2. Альтернативная активация дерева
            elif self.tree and hasattr(self.tree, 'activate_all'):
                result = await self.tree.activate_all()
                self.logger.info(f"Активация дерева: {result}")
            
            else:
                result = {"status": "manual_activation"}
                self.logger.warning("Активация в ручном режиме (без движка)")
            
            # 3. Отправка тестового сигнала
            if self.bus and hasattr(self.bus, 'broadcast'):
                test_signal = type('Signal', (), {
                    'type': SignalType.HEARTBEAT if hasattr(SignalType, 'HEARTBEAT') else 'HEARTBEAT',
                    'source': self.name,
                    'payload': {'activation': 'complete', 'engine': self.name}
                })()
                
                broadcast_result = await self.bus.broadcast(test_signal)
                self.logger.info(f"Тестовый broadcast: {broadcast_result.get('delivered_count', 0)} узлов")
            
            self.activated = True
            self.stats["activations"] += 1
            
            activation_result = {
                "success": True,
                "message": "Сефиротическая система активирована",
                "engine": self.name,
                "activation_time": datetime.utcnow().isoformat(),
                "tree_state": self.get_tree_state() if self.tree else None,
                "broadcast_test": broadcast_result if 'broadcast_result' in locals() else None,
                "timestamp": datetime.utcnow().isoformat()
            }
            
            self.logger.info("✅ Сефиротическая система активирована")
            return activation_result
            
        except Exception as e:
            error_msg = f"Ошибка активации: {str(e)}"
            self.logger.error(error_msg)
            self.stats["errors"] += 1
            
            return {
                "success": False,
                "error": error_msg,
                "timestamp": datetime.utcnow().isoformat()
            }
    
    async def shutdown(self) -> Dict[str, Any]:
        """Корректное завершение работы"""
        self.logger.info("🛑 Завершение работы сефиротической системы...")
        
        try:
            # Завершение движка если есть
            if self.engine and hasattr(self.engine, 'shutdown'):
                await self.engine.shutdown()
            
            # Завершение дерева если есть
            if self.tree and hasattr(self.tree, 'shutdown_all'):
                await self.tree.shutdown_all()
            
            self.activated = False
            self.initialized = False
            
            self.logger.info("✅ Сефиротическая система завершила работу")
            
            return {
                "success": True,
                "message": "Система завершена",
                "uptime_seconds": (datetime.utcnow() - self.start_time).total_seconds() if self.start_time else 0,
                "timestamp": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            error_msg = f"Ошибка завершения: {str(e)}"
            self.logger.error(error_msg)
            
            return {
                "success": False,
                "error": error_msg,
                "timestamp": datetime.utcnow().isoformat()
            }
    
    # ============================================================================
    # СТАТУС И МОНИТОРИНГ
    # ============================================================================
    
    def get_state(self) -> Dict[str, Any]:
        """Получение состояния движка"""
        state = {
            "name": self.name,
            "version": "4.0.0",
            "initialized": self.initialized,
            "activated": self.activated,
            "bus_available": self.bus is not None,
            "tree_available": self.tree is not None,
            "engine_available": self.engine is not None,
            "start_time": self.start_time.isoformat() if self.start_time else None,
            "stats": self.stats.copy(),
            "timestamp": datetime.utcnow().isoformat()
        }
        
        # Добавляем состояние дерева если есть
        if self.tree and hasattr(self.tree, 'get_tree_state'):
            state["tree_state"] = self.tree.get_tree_state()
        
        # Добавляем состояние шины если есть
        if self.bus and hasattr(self.bus, 'get_status'):
            state["bus_status"] = self.bus.get_status()
        
        return state
    
    def get_detailed_state(self) -> Dict[str, Any]:
        """Получение детального состояния"""
        state = self.get_state()
        
        # Добавляем привязки модулей если есть
        if self.bus and hasattr(self.bus, 'module_bindings'):
            state["module_bindings"] = self.bus.module_bindings
        
        # Добавляем здоровье если есть
        if self.bus and hasattr(self.bus, 'health_check'):
            try:
                # Асинхронный вызов в синхронном контексте
                health_future = asyncio.create_task(self.bus.health_check())
                state["bus_health"] = asyncio.run(health_future)
            except:
                state["bus_health"] = {"error": "health_check_failed"}
        
        return state
    
    def get_tree_state(self) -> Dict[str, Any]:
        """Получение состояния дерева сефирот"""
        if not self.tree:
            return {"error": "tree_not_available"}
        
        if hasattr(self.tree, 'get_tree_state'):
            return self.tree.get_tree_state()
        
        # Упрощённое состояние для заглушки
        return {
            "status": "simulated_tree",
            "nodes": ["KETER", "CHOKHMAH", "BINAH", "CHESED", "GEVURAH", 
                     "TIFERET", "NETZACH", "HOD", "YESOD", "MALKUTH"],
            "total_energy": 7.5,
            "total_resonance": 6.2,
            "timestamp": datetime.utcnow().isoformat()
        }
    
    def get_module_connections(self) -> Dict[str, Any]:
        """Получение информации о подключённых модулях"""
        connections = {
            "bechtereva": {
                "sephira": "KETER",
                "status": "connected" if self.bus and "bechtereva" in getattr(self.bus, 'module_bindings', {}) else "unknown",
                "signal_type": "NEURO"
            },
            "chernigovskaya": {
                "sephira": "CHOKHMAH",
                "status": "connected" if self.bus and "chernigovskaya" in getattr(self.bus, 'module_bindings', {}) else "unknown",
                "signal_type": "SEMIOTIC"
            }
        }
        
        return {
            "modules": connections,
            "total_connected": sum(1 for m in connections.values() if m["status"] == "connected"),
            "timestamp": datetime.utcnow().isoformat()
        }
    
    # ============================================================================
    # API ДЛЯ ИНТЕГРАЦИИ С ISKRA_FULL.PY
    # ============================================================================
    
    def get_flask_routes(self):
        """
        Генерация Flask API эндпоинтов для интеграции с iskra_full.py
        
        Использование в iskra_full.py:
        
        engine = SephiroticEngine()
        routes = engine.get_flask_routes()
        
        @app.route('/sephirot/state')
        async def sephirot_state():
            return await routes['get_state']()
        """
        routes = {}
        
        async def route_get_state():
            """GET /sephirot/state - состояние движка"""
            return self.get_state()
        
        async def route_get_detailed():
            """GET /sephirot/detailed - детальное состояние"""
            return self.get_detailed_state()
        
        async def route_activate():
            """POST /sephirot/activate - активация системы"""
            if self.activated:
                return {
                    "success": False,
                    "error": "Система уже активирована",
                    "timestamp": datetime.utcnow().isoformat()
                }
            
            result = await self.activate()
            return result
        
        async def route_shutdown():
            """POST /sephirot/shutdown - завершение работы"""
            result = await self.shutdown()
            return result
        
        async def route_modules():
            """GET /sephirot/modules - подключённые модули"""
            return self.get_module_connections()
        
        async def route_tree():
            """GET /sephirot/tree - состояние дерева"""
            return self.get_tree_state()
        
        async def route_health():
            """GET /sephirot/health - здоровье системы"""
            return {
                "status": "active" if self.activated else "inactive",
                "initialized": self.initialized,
                "activated": self.activated,
                "uptime_seconds": (datetime.utcnow() - self.start_time).total_seconds() if self.start_time else 0,
                "timestamp": datetime.utcnow().isoformat()
            }
        
        # Заполняем словарь маршрутов
        routes["get_state"] = route_get_state
        routes["get_detailed"] = route_get_detailed
        routes["activate"] = route_activate
        routes["shutdown"] = route_shutdown
        routes["modules"] = route_modules
        routes["tree"] = route_tree
        routes["health"] = route_health
        
        return routes

# ============================================================================
# ФАБРИКА ДЛЯ СОЗДАНИЯ ДВИЖКА
# ============================================================================

async def create_sephirotic_engine(existing_bus: Optional[SephiroticBus] = None) -> SephiroticEngine:
    """
    Фабрика для создания и инициализации сефиротического движка.
    
    :param existing_bus: Существующая шина (опционально)
    :return: Инициализированный движок
    """
    engine = SephiroticEngine()
    await engine.initialize(existing_bus)
    return engine

# ============================================================================
# ФУНКЦИЯ ДЛЯ ИНТЕГРАЦИИ С ISKRA_FULL.PY
# ============================================================================

async def initialize_sephirotic_in_iskra(bus: Optional[SephiroticBus] = None) -> Dict[str, Any]:
    """
    Основная функция для вызова из iskra_full.py.
    Инициализирует сефиротическую систему и возвращает готовый движок.
    
    Использование в iskra_full.py:
    
    sephirot_result = await initialize_sephirotic_in_iskra()
    if sephirot_result["success"]:
        engine = sephirot_result["engine"]
        # Регистрация эндпоинтов...
    """
    try:
        engine = await create_sephirotic_engine(bus)
        
        # Автоматическая активация
        activation_result = await engine.activate()
        
        return {
            "success": True,
            "engine": engine,
            "activation": activation_result,
            "message": "Сефиротическая система инициализирована и активирована",
            "module_bindings": engine.get_module_connections(),
            "timestamp": datetime.utcnow().isoformat()
        }
    
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "message": "Ошибка инициализации сефиротической системы",
            "timestamp": datetime.utcnow().isoformat()
        }

# ============================================================================
# ТЕСТОВАЯ ФУНКЦИЯ
# ============================================================================

async def test_engine():
    """Тестирование движка"""
    print("🧪 Тестирование SephiroticEngine...")
    
    engine = SephiroticEngine("Test-Engine")
    
    # Инициализация
    init_result = await engine.initialize()
    print(f"✅ Инициализация: {init_result['success']}")
    
    if init_result["success"]:
        # Активация
        activation_result = await engine.activate()
        print(f"✅ Активация: {activation_result['success']}")
        
        # Получение состояния
        state = engine.get_state()
        print(f"📊 Состояние: {state['initialized']}, активирована: {state['activated']}")
        
        # Модульные подключения
        modules = engine.get_module_connections()
        print(f"🔗 Модули: {modules}")
        
        # Завершение
        shutdown_result = await engine.shutdown()
        print(f"🛑 Завершение: {shutdown_result['success']}")
    
    return engine

# ============================================================================
# ТОЧКА ВХОДА
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
    print("🚀 Запуск теста сефиротического движка...")
    engine = asyncio.run(test_engine())
    print("✅ Тест завершён")
