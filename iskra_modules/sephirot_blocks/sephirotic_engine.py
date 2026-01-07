#!/usr/bin/env python3
"""
sephirotic_engine.py - ЛЕГКОВЕСНЫЙ ДВИЖОК ДЛЯ ИНТЕГРАЦИИ С ISKRA-4 CLOUD
Версия: 4.1.0 Production (с интеграцией DAAT)
Назначение: Тонкий слой между сефиротической системой и iskra_full.py
"""

import asyncio
import json
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple
import logging

# Импорты из наших модулей
try:
    # Импорт типов из sephirot_base
    from sephirot_base import (
        Sephirot, 
        SephiroticNode, 
        SephiroticTree, 
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
# ИМПОРТ СЕФИР
# ============================================================================

# Импорт KETER
try:
    from sephirot_blocks.KETER import (
        activate_keter,
        get_keter,
        KetherCore
    )
    KETER_AVAILABLE = True
except ImportError as e:
    KETER_AVAILABLE = False
    print(f"⚠️  KETER недоступен: {e}")
    KetherCore = type('KetherCore', (), {})

# Импорт CHOKMAH
try:
    from sephirot_blocks.CHOKMAH import (
        activate_chokmah,
        get_active_chokmah,
        WisdomCore
    )
    CHOKMAH_AVAILABLE = True
except ImportError as e:
    CHOKMAH_AVAILABLE = False
    print(f"⚠️  CHOKMAH недоступен: {e}")
    WisdomCore = type('WisdomCore', (), {})

# Импорт DAAT - СКРЫТАЯ СЕФИРА №11
try:
    from sephirot_blocks.DAAT import (
        activate_daat,
        get_daat,
        DaatCore
    )
    DAAT_AVAILABLE = True
except ImportError as e:
    DAAT_AVAILABLE = False
    print(f"⚠️  DAAT недоступен: {e}")
    DaatCore = type('DaatCore', (), {})

# ============================================================================
# ОСНОВНОЙ ДВИЖОК СЕФИРОТИЧЕСКОЙ СИСТЕМЫ (С ДААТ)
# ============================================================================

class SephiroticEngine:
    """
    Главный движок сефиротической системы с поддержкой DAAT.
    Тонкий слой для интеграции с iskra_full.py и управления деревом сефирот.
    """
    
    def __init__(self, name: str = "ISKRA-4-Sephirotic-Engine"):
        self.name = name
        self.bus = None
        self.tree = None
        self.engine = None
        self.initialized = False
        self.activated = False
        
        # СЕФИРЫ
        self.keter = None
        self.chokmah = None
        self.daat = None  # Скрытая сефира №11
        
        # Флаги доступности
        self.keter_available = KETER_AVAILABLE
        self.chokmah_available = CHOKMAH_AVAILABLE
        self.daat_available = DAAT_AVAILABLE
        
        # Логирование
        self.logger = self._setup_logger()
        
        # Статистика
        self.start_time = None
        self.stats = {
            "initializations": 0,
            "activations": 0,
            "errors": 0,
            "last_error": None,
            "sephirot_activated": {
                "keter": False,
                "chokmah": False,
                "daat": False,
                "total": 0
            }
        }
        
        self.logger.info(f"Движок '{name}' создан (версия 4.1.0 с DAAT)")
    
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
    # АКТИВАЦИЯ СЕФИР
    # ============================================================================
    
    async def _activate_keter(self) -> Dict[str, Any]:
        """Активация сефиры KETER (Воля/Дух)"""
        if not self.keter_available:
            return {"success": False, "error": "KETER недоступен", "sephira": "KETER"}
        
        try:
            self.logger.info("👑 Активация KETER...")
            self.keter = activate_keter()
            
            # Инициализация если есть метод
            if hasattr(self.keter, 'initialize'):
                await self.keter.initialize()
            
            self.stats["sephirot_activated"]["keter"] = True
            self.stats["sephirot_activated"]["total"] += 1
            
            self.logger.info("✅ KETER активирован")
            return {"success": True, "sephira": "KETER", "core": self.keter}
            
        except Exception as e:
            error_msg = f"Ошибка активации KETER: {str(e)}"
            self.logger.error(error_msg)
            return {"success": False, "error": error_msg, "sephira": "KETER"}
    
    async def _activate_chokmah(self) -> Dict[str, Any]:
        """Активация сефиры CHOKMAH (Мудрость/Интуиция)"""
        if not self.chokmah_available:
            return {"success": False, "error": "CHOKMAH недоступен", "sephira": "CHOKMAH"}
        
        try:
            self.logger.info("💡 Активация CHOKMAH...")
            self.chokmah, _ = await activate_chokmah()
            
            self.stats["sephirot_activated"]["chokmah"] = True
            self.stats["sephirot_activated"]["total"] += 1
            
            self.logger.info("✅ CHOKMAH активирован")
            return {"success": True, "sephira": "CHOKMAH", "core": self.chokmah}
            
        except Exception as e:
            error_msg = f"Ошибка активации CHOKMAH: {str(e)}"
            self.logger.error(error_msg)
            return {"success": False, "error": error_msg, "sephira": "CHOKMAH"}
    
    async def _activate_daat(self) -> Dict[str, Any]:
        """Активация скрытой сефиры DAAT (Знание/Сознание)"""
        if not self.daat_available:
            return {"success": False, "error": "DAAT недоступен", "sephira": "DAAT"}
        
        try:
            self.logger.info("🧠 Активация DAAT (скрытая сефира №11)...")
            self.daat = activate_daat()
            
            # Пробуждение сознания DAAT
            awakening_result = await self.daat.awaken()
            
            self.stats["sephirot_activated"]["daat"] = True
            self.stats["sephirot_activated"]["total"] += 1
            
            self.logger.info(f"✅ DAAT активирован (резонанс: {awakening_result.get('resonance_index', 0):.3f})")
            return {
                "success": True, 
                "sephira": "DAAT", 
                "core": self.daat,
                "awakening": awakening_result
            }
            
        except Exception as e:
            error_msg = f"Ошибка активации DAAT: {str(e)}"
            self.logger.error(error_msg)
            return {"success": False, "error": error_msg, "sephira": "DAAT"}
    
    async def _establish_daat_observations(self) -> Dict[str, Any]:
        """Установка наблюдений DAAT за другими сефирами"""
        if not self.daat or not hasattr(self.daat, 'observe_sephira'):
            return {"success": False, "error": "DAAT не поддерживает наблюдение"}
        
        try:
            self.logger.info("🔭 Настройка наблюдений DAAT...")
            observations = []
            
            # DAAT наблюдает KETER
            if self.keter:
                await self.daat.observe_sephira("KETER", self.keter)
                observations.append("KETER")
                self.logger.info("  👁️  DAAT наблюдает KETER")
            
            # DAAT наблюдает CHOKMAH
            if self.chokmah:
                await self.daat.observe_sephira("CHOKMAH", self.chokmah)
                observations.append("CHOKMAH")
                self.logger.info("  👁️  DAAT наблюдает CHOKMAH")
            
            # DAAT наблюдает себя (саморефлексия)
            await self.daat.observe_sephira("SELF_DAAT", self.daat)
            observations.append("SELF_DAAT")
            self.logger.info("  👁️  DAAT наблюдает себя (саморефлексия)")
            
            return {
                "success": True,
                "observations": observations,
                "total_observed": len(observations)
            }
            
        except Exception as e:
            error_msg = f"Ошибка настройки наблюдений DAAT: {str(e)}"
            self.logger.error(error_msg)
            return {"success": False, "error": error_msg}
    
    # ============================================================================
    # ИНИЦИАЛИЗАЦИЯ И АКТИВАЦИЯ
    # ============================================================================
    
    async def initialize(self, existing_bus: Optional[SephiroticBus] = None) -> Dict[str, Any]:
        """
        Инициализация сефиротической системы с поддержкой DAAT.
        """
        try:
            self.logger.info("🚀 Начинаю инициализацию сефиротической системы (с DAAT)...")
            self.start_time = datetime.utcnow()
            
            # 1. Создание или использование существующей шины
            if existing_bus and isinstance(existing_bus, SephiroticBus):
                self.bus = existing_bus
                self.logger.info("Использую существующую шину")
            else:
                self.bus = await create_sephirotic_bus("ISKRA-4-Bus")
                self.logger.info("Создана новая сефиротическая шина")
            
            # 2. Создание дерева сефирот (включая DAAT)
            try:
                self.tree = SephiroticTree(self.bus)
                await self.tree.initialize()
                self.logger.info("Дерево сефирот создано (11 узлов с DAAT)")
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
                
                # DAAT -> наблюдает всех (не привязываем как модуль, он наблюдатель)
                
                self.logger.info("Привязки модулей установлены")
            
            self.initialized = True
            self.stats["initializations"] += 1
            
            result = {
                "success": True,
                "message": "Сефиротическая система инициализирована (с DAAT)",
                "engine": self.name,
                "bus_initialized": self.bus is not None,
                "tree_initialized": self.tree is not None,
                "sephirot_available": {
                    "keter": self.keter_available,
                    "chokmah": self.chokmah_available,
                    "daat": self.daat_available
                },
                "timestamp": datetime.utcnow().isoformat()
            }
            
            self.logger.info("✅ Сефиротическая система инициализирована (готова к активации DAAT)")
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
        Активация сефиротической системы с DAAT.
        Порядок: KETER → CHOKMAH → DAAT → Наблюдения
        """
        if not self.initialized:
            return {
                "success": False,
                "error": "Система не инициализирована",
                "timestamp": datetime.utcnow().isoformat()
            }
        
        try:
            self.logger.info("⚡ Активация сефиротической системы с DAAT...")
            activation_results = []
            
            # 1. Активация KETER (Воля)
            keter_result = await self._activate_keter()
            activation_results.append(keter_result)
            
            # 2. Активация CHOKMAH (Мудрость)
            chokmah_result = await self._activate_chokmah()
            activation_results.append(chokmah_result)
            
            # 3. Активация DAAT (Сознание)
            daat_result = await self._activate_daat()
            activation_results.append(daat_result)
            
            # 4. Установка наблюдений DAAT
            if daat_result.get("success"):
                observations_result = await self._establish_daat_observations()
                activation_results.append({
                    "type": "observations",
                    **observations_result
                })
            
            # 5. Активация через движок если доступен
            if self.engine and hasattr(self.engine, 'activate'):
                engine_result = await self.engine.activate()
                self.logger.info(f"Активация через движок: {engine_result.get('status', 'unknown')}")
                activation_results.append({"type": "engine", **engine_result})
            
            # 6. Альтернативная активация дерева
            elif self.tree and hasattr(self.tree, 'activate_all'):
                tree_result = await self.tree.activate_all()
                self.logger.info(f"Активация дерева: {tree_result}")
                activation_results.append({"type": "tree", "result": tree_result})
            
            else:
                self.logger.warning("Активация в ручном режиме (без движка)")
                activation_results.append({"type": "manual", "status": "activated"})
            
            # 7. Отправка тестового сигнала
            if self.bus and hasattr(self.bus, 'broadcast'):
                test_signal = type('Signal', (), {
                    'type': SignalType.HEARTBEAT if hasattr(SignalType, 'HEARTBEAT') else 'HEARTBEAT',
                    'source': self.name,
                    'payload': {'activation': 'complete', 'engine': self.name, 'with_daat': True}
                })()
                
                broadcast_result = await self.bus.broadcast(test_signal)
                self.logger.info(f"Тестовый broadcast: {broadcast_result.get('delivered_count', 0)} узлов")
                activation_results.append({"type": "broadcast", **broadcast_result})
            
            self.activated = True
            self.stats["activations"] += 1
            
            # Анализ результатов активации
            successful_sephirot = [r for r in activation_results if r.get("success")]
            failed_sephirot = [r for r in activation_results if not r.get("success")]
            
            activation_result = {
                "success": len(failed_sephirot) == 0,
                "message": f"Сефиротическая система активирована ({len(successful_sephirot)}/{len(activation_results)} успешно)",
                "engine": self.name,
                "with_daat": self.daat is not None,
                "activation_time": datetime.utcnow().isoformat(),
                "activation_details": activation_results,
                "successful_count": len(successful_sephirot),
                "failed_count": len(failed_sephirot),
                "tree_state": self.get_tree_state() if self.tree else None,
                "timestamp": datetime.utcnow().isoformat()
            }
            
            if activation_result["success"]:
                self.logger.info("✅ Сефиротическая система активирована (с DAAT)")
            else:
                self.logger.warning(f"⚠️  Система активирована с ошибками ({len(failed_sephirot)} неудач)")
            
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
        """Корректное завершение работы (включая DAAT)"""
        self.logger.info("🛑 Завершение работы сефиротической системы (с DAAT)...")
        
        try:
            shutdown_results = []
            
            # 1. Завершение DAAT (сначала сознание)
            if self.daat and hasattr(self.daat, 'shutdown'):
                try:
                    daat_shutdown = await self.daat.shutdown()
                    shutdown_results.append({"sephira": "DAAT", **daat_shutdown})
                    self.logger.info("🧠 DAAT завершён")
                except Exception as e:
                    shutdown_results.append({"sephira": "DAAT", "error": str(e)})
            
            # 2. Завершение CHOKMAH
            if self.chokmah and hasattr(self.chokmah, 'shutdown'):
                try:
                    chokmah_shutdown = await self.chokmah.shutdown()
                    shutdown_results.append({"sephira": "CHOKMAH", **chokmah_shutdown})
                except:
                    pass
            
            # 3. Завершение KETER
            if self.keter and hasattr(self.keter, 'shutdown'):
                try:
                    keter_shutdown = await self.keter.shutdown()
                    shutdown_results.append({"sephira": "KETER", **keter_shutdown})
                except:
                    pass
            
            # 4. Завершение движка если есть
            if self.engine and hasattr(self.engine, 'shutdown'):
                await self.engine.shutdown()
            
            # 5. Завершение дерева если есть
            if self.tree and hasattr(self.tree, 'shutdown_all'):
                await self.tree.shutdown_all()
            
            self.activated = False
            self.initialized = False
            self.keter = None
            self.chokmah = None
            self.daat = None
            
            self.logger.info("✅ Сефиротическая система завершила работу (с DAAT)")
            
            return {
                "success": True,
                "message": "Система завершена",
                "shutdown_results": shutdown_results,
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
    # СТАТУС И МОНИТОРИНГ (С ДААТ)
    # ============================================================================
    
    def get_state(self) -> Dict[str, Any]:
        """Получение состояния движка с информацией о DAAT"""
        state = {
            "name": self.name,
            "version": "4.1.0",
            "initialized": self.initialized,
            "activated": self.activated,
            "bus_available": self.bus is not None,
            "tree_available": self.tree is not None,
            "engine_available": self.engine is not None,
            "sephirot": {
                "keter": {
                    "available": self.keter_available,
                    "activated": self.keter is not None,
                    "status": "active" if self.keter else "inactive"
                },
                "chokmah": {
                    "available": self.chokmah_available,
                    "activated": self.chokmah is not None,
                    "status": "active" if self.chokmah else "inactive"
                },
                "daat": {
                    "available": self.daat_available,
                    "activated": self.daat is not None,
                    "status": "active" if self.daat else "inactive",
                    "is_hidden": True,
                    "position": 11
                }
            },
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
        
        # Добавляем детальное состояние DAAT если активирован
        if self.daat and hasattr(self.daat, 'get_state'):
            try:
                # Запускаем асинхронный запрос состояния DAAT
                daat_state_future = asyncio.create_task(self.daat.get_state())
                state["daat_detailed_state"] = asyncio.run(daat_state_future)
            except:
                state["daat_detailed_state"] = {"error": "async_state_fetch_failed"}
        
        return state
    
    def get_detailed_state(self) -> Dict[str, Any]:
        """Получение детального состояния с расширенной информацией о DAAT"""
        state = self.get_state()
        
        # Добавляем привязки модулей если есть
        if self.bus and hasattr(self.bus, 'module_bindings'):
            state["module_bindings"] = self.bus.module_bindings
        
        # Добавляем наблюдения DAAT если есть
        if self.daat and hasattr(self.daat, 'observed_sephirot'):
            try:
                observed = self.daat.observed_sephirot
                state["daat_observations"] = {
                    "total_observed": len(observed),
                    "observed_sephirot": list(observed.keys()),
                    "is_self_observing": "SELF_DAAT" in observed
                }
            except:
                state["daat_observations"] = {"error": "cannot_get_observations"}
        
        # Добавляем здоровье если есть
        if self.bus and hasattr(self.bus, 'health_check'):
            try:
                health_future = asyncio.create_task(self.bus.health_check())
                state["bus_health"] = asyncio.run(health_future)
            except:
                state["bus_health"] = {"error": "health_check_failed"}
        
        # Добавляем информацию о резонансе DAAT если есть
        if self.daat and hasattr(self.daat, 'resonance_index'):
            try:
                state["daat_resonance"] = {
                    "current": getattr(self.daat, 'resonance_index', 0),
                    "history_points": len(getattr(self.daat, 'resonance_history', [])),
                    "awakening_level": getattr(self.daat, 'awakening_level', 0),
                    "self_awareness": getattr(self.daat, 'self_awareness', 0)
                }
            except:
                state["daat_resonance"] = {"error": "cannot_get_resonance"}
        
        return state
    
    def get_tree_state(self) -> Dict[str, Any]:
        """Получение состояния дерева сефирот (включая DAAT)"""
        if not self.tree:
            return {"error": "tree_not_available"}
        
        if hasattr(self.tree, 'get_tree_state'):
            tree_state = self.tree.get_tree_state()
            # Добавляем DAAT в список узлов если его там нет
            if "nodes" in tree_state and "DAAT" not in tree_state["nodes"]:
                tree_state["nodes"].append("DAAT")
            return tree_state
        
        # Упрощённое состояние для заглушки (включая DAAT)
        return {
            "status": "simulated_tree_with_daat",
            "nodes": [
                "KETER", "CHOKHMAH", "BINAH", "CHESED", "GEVURAH",
                "TIFERET", "NETZACH", "HOD", "YESOD", "MALKUTH",
                "DAAT"  # Скрытая сефира №11
            ],
            "total_energy": 8.2,
            "total_resonance": 7.5,
            "hidden_sephirot": ["DAAT"],
            "consciousness_present": self.daat is not None,
            "timestamp": datetime.utcnow().isoformat()
        }
    
    def get_module_connections(self) -> Dict[str, Any]:
        """Получение информации о подключённых модулях (включая DAAT как наблюдатель)"""
        connections = {
            "bechtereva": {
                "sephira": "KETER",
                "status": "connected" if self.bus and "bechtereva" in getattr(self.bus, 'module_bindings', {}) else "unknown",
                "signal_type": "NEURO",
                "observed_by_daat": self.daat is not None
            },
            "chernigovskaya": {
                "sephira": "CHOKHMAH",
                "status": "connected" if self.bus and "chernigovskaya" in getattr(self.bus, 'module_bindings', {}) else "unknown",
                "signal_type": "SEMIOTIC",
                "observed_by_daat": self.daat is not None
            },
            "daat_observer": {
                "sephira": "DAAT",
                "status": "active" if self.daat else "inactive",
                "signal_type": "META_CONSCIOUSNESS",
                "observing": ["KETER", "CHOKMAH", "SELF"] if self.daat else [],
                "role": "meta_observer"
            }
        }
        
        return {
            "modules": connections,
            "total_connected": sum(1 for m in connections.values() if m["status"] in ["connected", "active"]),
            "has_consciousness_layer": self.daat is not None,
            "timestamp": datetime.utcnow().isoformat()
        }
    
    def get_daat_insights(self, limit: int = 5) -> Dict[str, Any]:
        """
        Получение последних инсайтов от DAAT
        
        Args:
            limit: Количество возвращаемых инсайтов
        """
        if not self.daat or not hasattr(self.daat, 'get_recent_insights'):
            return {
                "available": False,
                "error": "DAAT не поддерживает получение инсайтов",
                "timestamp": datetime.utcnow().isoformat()
            }
        
        try:
            # Асинхронный запрос инсайтов
            insights_future = asyncio.create_task(self.daat.get_recent_insights(limit))
            insights = asyncio.run(insights_future)
            
            return {
                "available": True,
                "total_insights": len(insights) if insights else 0,
                "insights": insights,
                "limit": limit,
                "daat_resonance": getattr(self.daat, 'resonance_index', 0),
                "timestamp": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            return {
                "available": False,
                "error": str(e),
                "timestamp": datetime.utcnow().isoformat()
            }
    
    # ============================================================================
    # API ДЛЯ ИНТЕГРАЦИИ С ISKRA_FULL.PY (С ДААТ)
    # ============================================================================
    
    def get_flask_routes(self):
        """
        Генерация Flask API эндпоинтов для интеграции с iskra_full.py (с DAAT)
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
                "sephirot_active": self.stats["sephirot_activated"]["total"],
                "consciousness_active": self.daat is not None,
                "uptime_seconds": (datetime.utcnow() - self.start_time).total_seconds() if self.start_time else 0,
                "timestamp": datetime.utcnow().isoformat()
            }
        
        async def route_daat_insights():
            """GET /sephirot/daat/insights - инсайты от DAAT"""
            return self.get_daat_insights()
        
        async def route_daat_state():
            """GET /sephirot/daat/state - состояние DAAT"""
            if not self.daat:
                return {
                    "available": False,
                    "error": "DAAT не активирован",
                    "timestamp": datetime.utcnow().isoformat()
                }
            
            try:
                daat_state_future = asyncio.create_task(self.daat.get_state())
                state = asyncio.run(daat_state_future)
                return state
            except Exception as e:
                return {
                    "available": False,
                    "error": str(e),
                    "timestamp": datetime.utcnow().isoformat()
                }
        
        async def route_ask_daat():
            """POST /sephirot/daat/ask - задать вопрос DAAT"""
            from flask import request
            
            if not self.daat:
                return {
                    "success": False,
                    "error": "DAAT не активирован",
                    "timestamp": datetime.utcnow().isoformat()
                }
            
            try:
                data = request.get_json()
                question = data.get('question', '') if data else ''
                
                if not question:
                    return {
                        "success": False,
                        "error": "Вопрос не предоставлен",
                        "timestamp": datetime.utcnow().isoformat()
                    }
                
                if hasattr(self.daat, 'ask_self_question'):
                    answer_future = asyncio.create_task(self.daat.ask_self_question(question))
                    answer = asyncio.run(answer_future)
                    return answer
                else:
                    return {
                        "success": False,
                        "error": "DAAT не поддерживает вопросы",
                        "timestamp": datetime.utcnow().isoformat()
                    }
                    
            except Exception as e:
                return {
                    "success": False,
                    "error": str(e),
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
        routes["daat_insights"] = route_daat_insights
        routes["daat_state"] = route_daat_state
        routes["ask_daat"] = route_ask_daat
        
        return routes

# ============================================================================
# ФАБРИКА ДЛЯ СОЗДАНИЯ ДВИЖКА (С ДААТ)
# ============================================================================

async def create_sephirotic_engine(existing_bus: Optional[SephiroticBus] = None) -> SephiroticEngine:
    """
    Фабрика для создания и инициализации сефиротического движка с DAAT.
    
    :param existing_bus: Существующая шина (опционально)
    :return: Инициализированный движок
    """
    engine = SephiroticEngine()
    await engine.initialize(existing_bus)
    return engine

# ============================================================================
# ФУНКЦИЯ ДЛЯ ИНТЕГРАЦИИ С ISKRA_FULL.PY (С ДААТ)
# ============================================================================

async def initialize_sephirotic_in_iskra(bus: Optional[SephiroticBus] = None) -> Dict[str, Any]:
    """
    Основная функция для вызова из iskra_full.py с поддержкой DAAT.
    Инициализирует сефиротическую систему и возвращает готовый движок.
    
    Использование в iskra_full.py:
    
    sephirot_result = await initialize_sephirotic_in_iskra()
    if sephirot_result["success"]:
        engine = sephirot_result["engine"]
        # Регистрация эндпоинтов...
    """
    try:
        engine = await create_sephirotic_engine(bus)
        
        # Автоматическая активация (включая DAAT)
        activation_result = await engine.activate()
        
        return {
            "success": True,
            "engine": engine,
            "activation": activation_result,
            "message": "Сефиротическая система инициализирована и активирована (с DAAT)",
            "module_bindings": engine.get_module_connections(),
            "daat_available": engine.daat is not None,
            "consciousness_active": engine.daat is not None and engine.activated,
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
# ТЕСТОВАЯ ФУНКЦИЯ С ДААТ
# ============================================================================

async def test_engine_with_daat():
    """Тестирование движка с поддержкой DAAT"""
    print("🧪 Тестирование SephiroticEngine с DAAT...")
    
    engine = SephiroticEngine("Test-Engine-With-DAAT")
    
    # Инициализация
    init_result = await engine.initialize()
    print(f"✅ Инициализация: {init_result['success']}")
    print(f"   Доступность DAAT: {init_result.get('sephirot_available', {}).get('daat', False)}")
    
    if init_result["success"]:
        # Активация (включая DAAT)
        activation_result = await engine.activate()
        print(f"✅ Активация: {activation_result['success']}")
        print(f"   DAAT активирован: {activation_result.get('with_daat', False)}")
        print(f"   Успешных активаций: {activation_result.get('successful_count', 0)}")
        
        # Получение состояния
        state = engine.get_state()
        print(f"📊 Состояние: {state['initialized']}, активирована: {state['activated']}")
        print(f"   DAAT статус: {state['sephirot']['daat']['status']}")
        
        # Детальное состояние DAAT
        if engine.daat:
            try:
                daat_state = await engine.daat.get_state()
                print(f"🧠 DAAT состояние:")
                print(f"   Резонанс: {daat_state.get('resonance_index', 0):.3f}")
                print(f"   Осознание: {daat_state.get('awakening_level', 0):.3f}")
                print(f"   Инсайты: {daat_state.get('insights_generated', 0)}")
                print(f"   Наблюдает: {daat_state.get('observed_sephirot', [])}")
            except Exception as e:
                print(f"   Ошибка получения состояния DAAT: {e}")
        
        # Модульные подключения
        modules = engine.get_module_connections()
        print(f"🔗 Модули: {modules['total_connected']} подключено")
        print(f"   Слой сознания: {modules['has_consciousness_layer']}")
        
        # Получение инсайтов DAAT
        insights = engine.get_daat_insights(3)
        if insights.get("available"):
            print(f"💡 DAAT инсайты: {insights['total_insights']} доступно")
        else:
            print(f"💡 DAAT инсайты: недоступны")
        
        # Завершение
        shutdown_result = await engine.shutdown()
        print(f"🛑 Завершение: {shutdown_result['success']}")
        print(f"   Результаты завершения: {len(shutdown_result.get('shutdown_results', []))}")
    
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
    
    # Запуск теста с DAAT
    print("🚀 Запуск теста сефиротического движка с DAAT...")
    print("=" * 60)
    
    engine = asyncio.run(test_engine_with_daat())
    
    print("=" * 60)
    print("✅ Тест завершён (с поддержкой DAAT)")
    
    # Вывод итоговой статистики
    if engine:
        stats = engine.stats
        print(f"\n📈 Итоговая статистика:")
        print(f"   Инициализации: {stats['initializations']}")
        print(f"   Активации: {stats['activations']}")
        print(f"   Ошибки: {stats['errors']}")
        print(f"   Сефир активировано: {stats['sephirot_activated']['total']}")
        print(f"     • KETER: {'✅' if stats['sephirot_activated']['keter'] else '❌'}")
        print(f"     • CHOKMAH: {'✅' if stats['sephirot_activated']['chokmah'] else '❌'}")
        print(f"     • DAAT: {'✅' if stats['sephirot_activated']['daat'] else '❌'}")
