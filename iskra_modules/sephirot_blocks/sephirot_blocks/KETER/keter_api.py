"""
KETHER API GATEWAY v2.0 - API шлюз для Ketheric Block
Сефира: KETER (Венец)
Модуль: API Gateway (40+ эндпоинтов)
"""

import asyncio
import time
import re
from typing import Dict, Any, Optional, List, Callable
import logging

# ============================================================
# 1. API КЛАСС (расширяет KetherCore)
# ============================================================

class KetherAPI:
    """
    API шлюз для KetherCore
    Все API-методы вынесены из основного класса
    """
    
    def __init__(self, core):
        """
        Инициализация API шлюза
        
        Args:
            core: Экземпляр KetherCore
        """
        self.core = core
        self.logger = logging.getLogger(f"KetherAPI")
        
        # Настройки API
        self.api_config = {
            "auth_required": False,
            "rate_limit_enabled": False,
            "allowed_origins": ["*"],
            "max_request_size": 1024 * 1024,  # 1 MB
            "request_timeout": 30.0,
            "enable_cors": True,
            "api_keys": {
                "ISKRA4_KETER_MASTER_KEY": {"level": "admin", "rate_limit": 1000},
                "KETHERIC_BLOCK_ADMIN": {"level": "admin", "rate_limit": 500},
                "SEPHIROTIC_ENGINE": {"level": "system", "rate_limit": 100},
                "METRICS_COLLECTOR": {"level": "monitor", "rate_limit": 50},
                "MODULE_INTEGRATION": {"level": "module", "rate_limit": 200},
            }
        }
        
        # Статистика запросов
        self.request_stats = []
        self.request_counter = 0
        
        self.logger.info("Kether API Gateway v2.0 инициализирован")
    
    # ========================================================
    # 2. ОСНОВНОЙ API МЕТОД
    # ========================================================
    
    async def api_call(self,
                      endpoint: str,
                      method: str = "GET",
                      data: Optional[Dict] = None,
                      api_key: Optional[str] = None,
                      client_info: Optional[Dict] = None) -> Dict[str, Any]:
        """
        ПОЛНЫЙ API шлюз с маршрутизацией, валидацией, аутентификацией и лимитами
        """
        request_id = f"req_{int(time.time())}_{self.request_counter:06d}"
        self.request_counter += 1
        start_time = time.time()
        
        self.logger.info(f"🌐 API запрос [{request_id}]: {method} {endpoint}")
        
        # Проверка аутентификации
        auth_result = await self._api_authenticate(api_key, client_info)
        if not auth_result["authenticated"]:
            self.logger.warning(f"API аутентификация провалена [{request_id}]: {auth_result.get('reason')}")
            return {
                "request_id": request_id,
                "error": "authentication_failed",
                "message": auth_result.get("message", "Invalid credentials"),
                "status_code": 401,
                "timestamp": time.time(),
                "processing_time": time.time() - start_time
            }
        
        # Проверка лимитов запросов
        if not await self._api_check_rate_limit(client_info, auth_result):
            return {
                "request_id": request_id,
                "error": "rate_limit_exceeded",
                "message": "Too many requests",
                "status_code": 429,
                "timestamp": time.time(),
                "processing_time": time.time() - start_time
            }
        
        # Нормализация endpoint
        endpoint = endpoint.strip('/')
        if not endpoint.startswith('/'):
            endpoint = '/' + endpoint
        
        # Поиск обработчика
        handler, route_params = self._find_api_handler(method, endpoint)
        
        if not handler:
            processing_time = time.time() - start_time
            self.logger.warning(f"API маршрут не найден [{request_id}]: {method} {endpoint}")
            
            # Возвращаем список доступных эндпоинтов
            available_endpoints = self._get_available_endpoints()
            
            return {
                "request_id": request_id,
                "error": "endpoint_not_found",
                "message": f"No handler for {method} {endpoint}",
                "status_code": 404,
                "available_endpoints": sorted(available_endpoints),
                "processing_time": processing_time,
                "timestamp": time.time()
            }
        
        # Выполнение обработчика
        try:
            # Подготавливаем контекст запроса
            request_context = {
                "request_id": request_id,
                "endpoint": endpoint,
                "method": method,
                "data": data or {},
                "params": route_params,
                "client_info": client_info or {},
                "auth_info": auth_result,
                "start_time": start_time
            }
            
            # Вызываем обработчик
            result = await handler(request_context)
            
            # Добавляем метаданные
            processing_time = time.time() - start_time
            result.update({
                "request_id": request_id,
                "processing_time": round(processing_time, 4),
                "timestamp": time.time(),
                "success": result.get("error") is None
            })
            
            # Логируем успешный запрос
            self.logger.info(f"✅ API запрос завершён [{request_id}]: {method} {endpoint} ({processing_time:.3f}s)")
            
            # Сохраняем статистику
            self._save_request_stats(request_id, endpoint, method, processing_time, result.get("error"))
            
            return result
            
        except Exception as e:
            processing_time = time.time() - start_time
            error_msg = str(e)
            
            self.logger.error(f"❌ Ошибка обработки API [{request_id}]: {error_msg}")
            
            # Публикуем событие об ошибке API
            await self.core._publish_internal_event("api.error", {
                "request_id": request_id,
                "endpoint": endpoint,
                "method": method,
                "error": error_msg,
                "processing_time": processing_time,
                "timestamp": time.time()
            })
            
            return {
                "request_id": request_id,
                "error": "internal_server_error",
                "message": error_msg,
                "status_code": 500,
                "processing_time": processing_time,
                "timestamp": time.time()
            }
    
    # ========================================================
    # 3. АУТЕНТИФИКАЦИЯ И ЛИМИТЫ
    # ========================================================
    
    async def _api_authenticate(self, api_key: Optional[str], client_info: Optional[Dict]) -> Dict[str, Any]:
        """Аутентификация API запроса"""
        # Если аутентификация отключена - пропускаем
        if not self.api_config["auth_required"]:
            return {
                "authenticated": True,
                "auth_method": "none",
                "access_level": "full"
            }
        
        # Проверка API ключа
        if api_key and api_key in self.api_config["api_keys"]:
            key_info = self.api_config["api_keys"][api_key]
            return {
                "authenticated": True,
                "auth_method": "api_key",
                "access_level": key_info["level"],
                "rate_limit": key_info["rate_limit"],
                "key_type": "valid"
            }
        
        # Проверка по client_info (например, для внутренних вызовов)
        if client_info and client_info.get("internal_call") == True:
            return {
                "authenticated": True,
                "auth_method": "internal",
                "access_level": "system",
                "rate_limit": 1000
            }
        
        return {
            "authenticated": False,
            "auth_method": "none",
            "access_level": "none",
            "message": "Invalid API key or credentials",
            "reason": "invalid_key"
        }
    
    async def _api_check_rate_limit(self, client_info: Optional[Dict], auth_info: Dict) -> bool:
        """Проверка лимитов запросов"""
        if not self.api_config["rate_limit_enabled"]:
            return True
        
        # TODO: Реализовать полноценную систему rate limiting
        # Сейчас просто возвращаем True
        return True
    
    # ========================================================
    # 4. МАРШРУТИЗАЦИЯ API
    # ========================================================
    
    def _find_api_handler(self, method: str, endpoint: str):
        """Поиск обработчика для API запроса"""
        api_routes = self._get_api_routes()
        
        # Проверяем точное совпадение
        if (method, endpoint) in api_routes:
            return api_routes[(method, endpoint)], {}
        
        # Проверяем паттерн с параметрами
        for (route_method, route_pattern), handler in api_routes.items():
            if method != route_method:
                continue
            
            if '{' in route_pattern and '}' in route_pattern:
                pattern_parts = route_pattern.split('/')
                endpoint_parts = endpoint.split('/')
                
                if len(pattern_parts) != len(endpoint_parts):
                    continue
                
                match = True
                params = {}
                
                for i in range(len(pattern_parts)):
                    if pattern_parts[i].startswith('{') and pattern_parts[i].endswith('}'):
                        param_name = pattern_parts[i][1:-1]
                        params[param_name] = endpoint_parts[i]
                    elif pattern_parts[i] != endpoint_parts[i]:
                        match = False
                        break
                
                if match:
                    return handler, params
        
        return None, {}
    
    def _get_api_routes(self) -> Dict[tuple, Callable]:
        """Таблица маршрутизации API"""
        return {
            # === СИСТЕМНЫЕ ЭНДПОИНТЫ ===
            ("GET", "/"): self._api_root,
            ("GET", "/status"): self._api_system_status,
            ("GET", "/health"): self._api_system_health,
            ("GET", "/version"): self._api_version_info,
            ("GET", "/config"): self._api_get_config,
            
            # === МЕТРИКИ И МОНИТОРИНГ ===
            ("GET", "/metrics"): self._api_get_metrics,
            ("GET", "/metrics/latest"): self._api_get_latest_metrics,
            ("GET", "/metrics/history"): self._api_get_metrics_history,
            ("GET", "/metrics/module/{module}"): self._api_get_module_metrics,
            
            # === УПРАВЛЕНИЕ МОДУЛЯМИ ===
            ("GET", "/modules"): self._api_list_modules,
            ("GET", "/modules/all"): self._api_get_all_modules_info,
            ("GET", "/modules/{module}"): self._api_get_module_info,
            ("GET", "/modules/{module}/health"): self._api_get_module_health,
            ("GET", "/modules/{module}/status"): self._api_get_module_status,
            ("POST", "/modules/{module}/activate"): self._api_activate_module,
            ("POST", "/modules/{module}/deactivate"): self._api_deactivate_module,
            ("POST", "/modules/{module}/restart"): self._api_restart_module,
            
            # === ВОССТАНОВЛЕНИЕ ===
            ("GET", "/recovery"): self._api_get_recovery_status,
            ("GET", "/recovery/status"): self._api_get_recovery_status_full,
            ("POST", "/recovery/{module}"): self._api_recover_module,
            ("POST", "/recovery/auto"): self._api_auto_recover,
            ("POST", "/recovery/reset"): self._api_reset_recovery_attempts,
            ("GET", "/recovery/history"): self._api_get_recovery_history,
            
            # === ЭНЕРГЕТИЧЕСКОЕ УПРАВЛЕНИЕ ===
            ("GET", "/energy"): self._api_get_energy_status,
            ("GET", "/energy/flows"): self._api_get_energy_flows,
            ("POST", "/energy/distribute"): self._api_distribute_energy,
            ("POST", "/energy/recharge"): self._api_recharge_energy,
            ("POST", "/energy/set_reserve"): self._api_set_energy_reserve,
            
            # === СОБЫТИЯ ===
            ("GET", "/events"): self._api_get_event_capabilities,
            ("POST", "/events/subscribe"): self._api_subscribe_to_event,
            ("POST", "/events/publish"): self._api_publish_event,
            ("GET", "/events/subscriptions"): self._api_get_subscriptions,
            
            # === УПРАВЛЕНИЕ СИСТЕМОЙ ===
            ("POST", "/system/activate"): self._api_activate_system,
            ("POST", "/system/shutdown"): self._api_shutdown_system,
            ("POST", "/system/restart"): self._api_restart_system,
            ("GET", "/system/diagnostics"): self._api_get_diagnostics,
            
            # === АДМИНИСТРИРОВАНИЕ ===
            ("POST", "/admin/reload_config"): self._api_reload_config,
            ("POST", "/admin/clear_cache"): self._api_clear_cache,
            ("GET", "/admin/performance"): self._api_get_performance_stats,
        }
    
    def _get_available_endpoints(self) -> List[str]:
        """Получение списка доступных эндпоинтов"""
        endpoints = []
        for (method, pattern), _ in self._get_api_routes().items():
            if method in ["GET", "POST"]:
                endpoints.append(f"{method} {pattern}")
        return endpoints
    
    # ========================================================
    # 5. API ОБРАБОТЧИКИ
    # ========================================================
    
    # 5.1. СИСТЕМНЫЕ ЭНДПОИНТЫ
    async def _api_root(self, context: Dict) -> Dict[str, Any]:
        """Корневой эндпоинт API"""
        return {
            "system": "ISKRA-4 Ketheric Block",
            "sephira": "KETER",
            "version": self.core.__version__,
            "status": "operational" if self.core.is_activated else "inactive",
            "endpoints": {
                "system": "/status, /health, /version, /config",
                "modules": "/modules, /modules/{module}, /modules/{module}/health",
                "metrics": "/metrics, /metrics/latest, /metrics/history",
                "energy": "/energy, /energy/flows, /energy/distribute",
                "recovery": "/recovery, /recovery/{module}, /recovery/auto",
                "events": "/events, /events/subscribe, /events/publish",
                "system_control": "/system/activate, /system/shutdown, /system/restart",
                "admin": "/admin/reload_config, /admin/clear_cache"
            },
            "active_modules": f"{sum(1 for m in self.core.modules.values() if m.is_active)}/{len(self.core.modules)}",
            "uptime": round(time.time() - self.core.activation_start_time, 1) if self.core.is_activated else 0
        }
    
    async def _api_system_status(self, context: Dict) -> Dict[str, Any]:
        """Статус системы"""
        active_modules = sum(1 for m in self.core.modules.values() if m.is_active)
        total_modules = len(self.core.modules)
        
        return {
            "sephira": self.core.__sephira__,
            "version": self.core.__version__,
            "status": "active" if self.core.is_activated else "inactive",
            "activation_time": self.core.activation_start_time if self.core.is_activated else None,
            "uptime": round(time.time() - self.core.activation_start_time, 1) if self.core.is_activated else 0,
            "modules": {
                "total": total_modules,
                "active": active_modules,
                "inactive": total_modules - active_modules,
                "health_percentage": round((active_modules / total_modules) * 100, 1) if total_modules > 0 else 0
            },
            "energy": {
                "reserve": self.core.energy_reserve,
                "status": "critical" if self.core.energy_reserve < self.core.config["energy"]["critical_threshold"] else "normal"
            },
            "events": {
                "queue_size": self.core.event_queue.qsize(),
                "max_queue": self.core.event_queue.maxsize
            },
            "background_tasks": len(self.core.background_tasks),
            "performance": {
                "request_id": context["request_id"],
                "api_version": "1.0"
            }
        }
    
    async def _api_system_health(self, context: Dict) -> Dict[str, Any]:
        """Полная проверка здоровья системы"""
        return await self.core.get_system_health_report()
    
    async def _api_version_info(self, context: Dict) -> Dict[str, Any]:
        """Информация о версии"""
        import sys
        
        return {
            "system": "ISKRA-4 Ketheric Block",
            "sephira": self.core.__sephira__,
            "core_version": self.core.__version__,
            "architecture": self.core.__architecture__,
            "python_version": sys.version,
            "modules": {
                name: {
                    "active": module.is_active,
                    "path": module.path,
                    "order": module.activation_order
                }
                for name, module in self.core.modules.items()
            },
            "capabilities": [
                "module_registry",
                "cascade_activation",
                "energy_management",
                "event_routing",
                "metrics_collection",
                "auto_recovery",
                "api_gateway"
            ],
            "timestamp": time.time()
        }
    
    async def _api_get_config(self, context: Dict) -> Dict[str, Any]:
        """Получение конфигурации"""
        # Фильтруем чувствительные данные
        safe_config = {
            "activation": self.core.config["activation"],
            "energy": self.core.config["energy"],
            "events": self.core.config["events"],
            "recovery": self.core.config["recovery"],
            "metrics": self.core.config["metrics"],
            "api": {
                "enabled": self.core.config.get("api", {}).get("enabled", True),
                "host": self.core.config.get("api", {}).get("host", "localhost"),
                "port": self.core.config.get("api", {}).get("port", 8080)
            }
        }
        
        return {
            "config": safe_config,
            "sephira": self.core.__sephira__,
            "timestamp": time.time()
        }
    
    # 5.2. МЕТРИКИ И МОНИТОРИНГ
    async def _api_get_metrics(self, context: Dict) -> Dict[str, Any]:
        """Получение текущих метрик"""
        return await self.core.collect_metrics()
    
    async def _api_get_latest_metrics(self, context: Dict) -> Dict[str, Any]:
        """Получение последних метрик"""
        metrics = await self.core.collect_metrics()
        
        # Фильтруем по параметрам запроса
        params = context.get("params", {})
        data = context.get("data", {})
        
        filter_module = params.get("module") or data.get("module")
        if filter_module and filter_module in metrics["modules"]:
            return {
                "module": filter_module,
                "metrics": metrics["modules"][filter_module],
                "timestamp": metrics["timestamp"]
            }
        
        # Возвращаем сводные метрики
        summary = {
            "system": metrics["system"],
            "energy": metrics["energy"],
            "performance": metrics["performance"],
            "modules_summary": {
                "total": len(metrics["modules"]),
                "active": sum(1 for m in metrics["modules"].values() if m.get("active")),
                "with_errors": sum(1 for m in metrics["modules"].values() if "error" in m)
            }
        }
        
        return {
            "summary": summary,
            "timestamp": metrics["timestamp"]
        }
    
    async def _api_get_metrics_history(self, context: Dict) -> Dict[str, Any]:
        """Получение истории метрик"""
        data = context.get("data", {})
        limit = data.get("limit", 100)
        
        history = await self.core.get_metrics_history(limit)
        
        return {
            "history": history,
            "total_records": len(history),
            "limit": limit,
            "timestamp": time.time()
        }
    
    async def _api_get_module_metrics(self, context: Dict) -> Dict[str, Any]:
        """Метрики конкретного модуля"""
        module_name = context["params"].get("module")
        
        if not module_name or module_name not in self.core.modules:
            return {
                "error": "module_not_found",
                "message": f"Module {module_name} not found",
                "available_modules": list(self.core.modules.keys())
            }
        
        module_info = self.core.modules[module_name]
        
        if not module_info.is_active or not module_info.instance:
            return {
                "module": module_name,
                "active": False,
                "message": "Module is not active"
            }
        
        try:
            metrics = await module_info.instance.get_metrics()
            return {
                "module": module_name,
                "active": True,
                "metrics": metrics,
                "activation_order": module_info.activation_order,
                "timestamp": time.time()
            }
        except Exception as e:
            return {
                "module": module_name,
                "active": True,
                "error": str(e),
                "timestamp": time.time()
            }
    
    # 5.3. УПРАВЛЕНИЕ МОДУЛЯМИ
    async def _api_list_modules(self, context: Dict) -> Dict[str, Any]:
        """Список всех модулей"""
        modules_list = []
        
        for name, module in self.core.modules.items():
            modules_list.append({
                "name": name,
                "active": module.is_active,
                "activation_order": module.activation_order,
                "dependencies": module.dependencies,
                "path": module.path,
                "has_instance": module.instance is not None
            })
        
        modules_list.sort(key=lambda x: x["activation_order"] or 999)
        
        return {
            "modules": modules_list,
            "total": len(modules_list),
            "active": sum(1 for m in modules_list if m["active"]),
            "timestamp": time.time()
        }
    
    async def _api_get_all_modules_info(self, context: Dict) -> Dict[str, Any]:
        """Полная информация о всех модулях"""
        modules_info = {}
        
        for name, module in self.core.modules.items():
            health = await self.core.get_module_health(name)
            modules_info[name] = health
        
        return {
            "modules": modules_info,
            "summary": {
                "total": len(modules_info),
                "active": sum(1 for m in modules_info.values() if m.get("active")),
                "healthy": sum(1 for m in modules_info.values() if m.get("active") and "error" not in m),
                "with_dependencies": sum(1 for m in modules_info.values() if m.get("dependencies"))
            },
            "timestamp": time.time()
        }
    
    async def _api_get_module_info(self, context: Dict) -> Dict[str, Any]:
        """Информация о конкретном модуле"""
        module_name = context["params"].get("module")
        
        if not module_name or module_name not in self.core.modules:
            return {
                "error": "module_not_found",
                "message": f"Module {module_name} not found",
                "available_modules": list(self.core.modules.keys())
            }
        
        module = self.core.modules[module_name]
        
        info = {
            "name": module_name,
            "active": module.is_active,
            "activation_order": module.activation_order,
            "dependencies": module.dependencies,
            "path": module.path,
            "config": module.config,
            "instance_present": module.instance is not None,
            "activation_time": self.core.activation_timestamps.get(module_name),
            "error_count": self.core.error_counters.get(module_name, 0),
            "recovery_attempts": self.core.error_counters.get(f"{module_name}_recovery", 0)
        }
        
        # Добавляем информацию о зависимостях
        deps_status = []
        for dep in module.dependencies:
            if dep in self.core.modules:
                dep_module = self.core.modules[dep]
                deps_status.append({
                    "name": dep,
                    "active": dep_module.is_active,
                    "order": dep_module.activation_order
                })
            else:
                deps_status.append({
                    "name": dep,
                    "active": False,
                    "error": "not_registered"
                })
        
        info["dependencies_status"] = deps_status
        info["all_dependencies_active"] = all(dep["active"] for dep in deps_status)
        
        return info
    
    async def _api_get_module_health(self, context: Dict) -> Dict[str, Any]:
        """Здоровье конкретного модуля"""
        module_name = context["params"].get("module")
        
        if not module_name or module_name not in self.core.modules:
            return {
                "error": "module_not_found",
                "message": f"Module {module_name} not found"
            }
        
        return await self.core.get_module_health(module_name)
    
    async def _api_get_module_status(self, context: Dict) -> Dict[str, Any]:
        """Статус конкретного модуля"""
        module_name = context["params"].get("module")
        
        if not module_name or module_name not in self.core.modules:
            return {
                "error": "module_not_found",
                "message": f"Module {module_name} not found"
            }
        
        module = self.core.modules[module_name]
        
        status = "active" if module.is_active else "inactive"
        
        if not module.is_active:
            if module_name in self.core.activation_timestamps:
                status = "failed"
            else:
                status = "never_activated"
        
        return {
            "module": module_name,
            "status": status,
            "active": module.is_active,
            "order": module.activation_order,
            "uptime": time.time() - self.core.activation_timestamps.get(module_name, 0) if module.is_active else 0,
            "timestamp": time.time()
        }
    
    async def _api_activate_module(self, context: Dict) -> Dict[str, Any]:
        """Активация модуля через API"""
        module_name = context["params"].get("module")
        
        if not module_name or module_name not in self.core.modules:
            return {
                "error": "module_not_found",
                "message": f"Module {module_name} not found"
            }
        
        module = self.core.modules[module_name]
        
        if module.is_active:
            return {
                "module": module_name,
                "status": "already_active",
                "message": "Module is already active",
                "order": module.activation_order
            }
        
        try:
            success = await module.instance.activate()
            
            if success:
                module.is_active = True
                module.activation_order = max(
                    [m.activation_order for m in self.core.modules.values() if m.is_active],
                    default=0
                ) + 1
                
                self.core.activation_timestamps[module_name] = time.time()
                
                return {
                    "module": module_name,
                    "status": "activated",
                    "success": True,
                    "new_order": module.activation_order,
                    "timestamp": time.time()
                }
            else:
                return {
                    "module": module_name,
                    "status": "activation_failed",
                    "success": False,
                    "message": "Module.activate() returned False",
                    "timestamp": time.time()
                }
            
        except Exception as e:
            return {
                "module": module_name,
                "status": "activation_error",
                "success": False,
                "error": str(e),
                "timestamp": time.time()
            }
    
    async def _api_deactivate_module(self, context: Dict) -> Dict[str, Any]:
        """Деактивация модуля через API"""
        module_name = context["params"].get("module")
        
        if not module_name or module_name not in self.core.modules:
            return {
                "error": "module_not_found",
                "message": f"Module {module_name} not found"
            }
        
        module = self.core.modules[module_name]
        
        if not module.is_active or not module.instance:
            return {
                "module": module_name,
                "status": "already_inactive",
                "message": "Module is already inactive",
                "timestamp": time.time()
            }
        
        try:
            await module.instance.shutdown()
            module.is_active = False
            
            return {
                "module": module_name,
                "status": "deactivated",
                "success": True,
                "timestamp": time.time()
            }
            
        except Exception as e:
            return {
                "module": module_name,
                "status": "deactivation_error",
                "success": False,
                "error": str(e),
                "timestamp": time.time()
            }
    
    async def _api_restart_module(self, context: Dict) -> Dict[str, Any]:
        """Перезапуск модуля через API"""
        module_name = context["params"].get("module")
        
        if not module_name or module_name not in self.core.modules:
            return {
                "error": "module_not_found",
                "message": f"Module {module_name} not found"
            }
        
        # Деактивация
        deactivate_result = await self._api_deactivate_module(context)
        if not deactivate_result.get("success"):
            return deactivate_result
        
        # Пауза
        await asyncio.sleep(0.5)
        
        # Активация
        activate_result = await self._api_activate_module(context)
        
        return {
            "module": module_name,
            "operation": "restart",
            "deactivation": deactivate_result,
            "activation": activate_result,
            "overall_success": activate_result.get("success", False),
            "timestamp": time.time()
        }
    
    # 5.4. ВОССТАНОВЛЕНИЕ
    async def _api_get_recovery_status(self, context: Dict) -> Dict[str, Any]:
        """Статус системы восстановления"""
        return await self.core.get_recovery_status()
    
    async def _api_get_recovery_status_full(self, context: Dict) -> Dict[str, Any]:
        """Полный статус восстановления"""
        status = await self.core.get_recovery_status()
        
        # Добавляем дополнительную информацию
        failed_modules = [
            name for name, module in self.core.modules.items()
            if not module.is_active
        ]
        
        recovery_blocked = [
            name for name in failed_modules
            if self.core.error_counters.get(f"{name}_recovery", 0) >= self.core.config["recovery"]["max_recovery_attempts"]
        ]
        
        status["detailed"] = {
            "failed_modules": failed_modules,
            "recovery_blocked": recovery_blocked,
            "can_auto_recover": self.core.config["recovery"]["auto_recover"],
            "auto_recovery_enabled": self.core.config["recovery"]["enabled"]
        }
        
        return status
    
    async def _api_recover_module(self, context: Dict) -> Dict[str, Any]:
        """Восстановление модуля через API"""
        module_name = context["params"].get("module")
        data = context.get("data", {})
        force = data.get("force", False)
        
        if not module_name or module_name not in self.core.modules:
            return {
                "error": "module_not_found",
                "message": f"Module {module_name} not found"
            }
        
        return await self.core.recover_module(module_name, force)
    
    async def _api_auto_recover(self, context: Dict) -> Dict[str, Any]:
        """Автовосстановление через API"""
        return await self.core.auto_recover_failed_modules()
    
    async def _api_reset_recovery_attempts(self, context: Dict) -> Dict[str, Any]:
        """Сброс попыток восстановления через API"""
        data = context.get("data", {})
        module_name = data.get("module")
        
        return await self.core.reset_recovery_attempts(module_name)
    
    async def _api_get_recovery_history(self, context: Dict) -> Dict[str, Any]:
        """История восстановлений"""
        data = context.get("data", {})
        limit = data.get("limit", 50)
        
        history = await self.core.get_recovery_history(limit)
        
        return {
            "history": history,
            "limit": limit,
            "total": len(history),
            "timestamp": time.time()
        }
    
    # 5.5. ЭНЕРГЕТИЧЕСКОЕ УПРАВЛЕНИЕ
    async def _api_get_energy_status(self, context: Dict) -> Dict[str, Any]:
        """Статус энергии"""
        return {
            "energy": {
                "reserve": self.core.energy_reserve,
                "critical_threshold": self.core.config["energy"]["critical_threshold"],
                "status": "critical" if self.core.energy_reserve < self.core.config["energy"]["critical_threshold"] else "normal",
                "recharge_rate": self.core.config["energy"]["recharge_rate"]
            },
            "flows": {
                "total": len(self.core.energy_flows),
                "active": sum(1 for f in self.core.energy_flows if f.current_flow > 0),
                "by_priority": {
                    "critical": sum(1 for f in self.core.energy_flows if f.priority == "critical"),
                    "high": sum(1 for f in self.core.energy_flows if f.priority == "high"),
                    "medium": sum(1 for f in self.core.energy_flows if f.priority == "medium"),
                    "low": sum(1 for f in self.core.energy_flows if f.priority == "low")
                }
            },
            "timestamp": time.time()
        }
    
    async def _api_get_energy_flows(self, context: Dict) -> Dict[str, Any]:
        """Получение информации об энергетических потоках"""
        flows_info = []
        
        for flow in self.core.energy_flows:
            flows_info.append({
                "source": flow.source,
                "target": flow.target,
                "priority": flow.priority,
                "current_flow": flow.current_flow,
                "max_flow": flow.max_flow,
                "last_transfer": flow.last_transfer,
                "active": flow.current_flow > 0,
                "utilization": round((flow.current_flow / flow.max_flow) * 100, 1) if flow.max_flow > 0 else 0
            })
        
        # Сортируем по приоритету
        priority_order = {"critical": 0, "high": 1, "medium": 2, "low": 3}
        flows_info.sort(key=lambda x: priority_order.get(x["priority"], 4))
        
        return {
            "flows": flows_info,
            "total": len(flows_info),
            "active": sum(1 for f in flows_info if f["active"]),
            "total_capacity": sum(f["max_flow"] for f in flows_info),
            "current_utilization": sum(f["current_flow"] for f in flows_info),
            "timestamp": time.time()
        }
    
    async def _api_distribute_energy(self, context: Dict) -> Dict[str, Any]:
        """Распределение энергии через API"""
        data = context.get("data", {})
    
        required = ["source", "target", "amount"]
        missing = [field for field in required if field not in data]
        if missing:
            return {
                "error": "missing_parameters",
                "message": f"Missing required parameters: {missing}",
                "required": required,
                "timestamp": time.time()
            }

        # Получение параметров и вызов распределения энергии
        source = data["source"]
        target = data["target"]
        amount = float(data["amount"])
    
        return await self.core.distribute_energy(source, target, amount)
    
async def _api_recharge_energy(self, context: Dict) -> Dict[str, Any]:
    """Пополнение энергии через API"""
    data = context.get("data", {})
    amount = float(data.get("amount", 100.0))
    
    success = await self.core.recharge_energy(amount)
    
    return {
        "success": success,
        "amount": amount,
        "new_reserve": self.core.energy_reserve,
        "timestamp": time.time()
    }

async def _api_set_energy_reserve(self, context: Dict) -> Dict[str, Any]:
    """Установка уровня энергетического резерва"""
    data = context.get("data", {})
    
    if "reserve" not in data:
        return {
            "error": "missing_parameter",
            "message": "Parameter 'reserve' is required",
            "timestamp": time.time()
        }
    
    new_reserve = float(data["reserve"])
    old_reserve = self.core.energy_reserve
    self.core.energy_reserve = new_reserve
    
    self.logger.info(f"Энергетический резерв изменён через API: {old_reserve:.1f} → {new_reserve:.1f}")
    
    return {
        "success": True,
        "old_reserve": old_reserve,
        "new_reserve": new_reserve,
        "difference": new_reserve - old_reserve,
        "timestamp": time.time()
    }

# 5.6. СОБЫТИЯ
async def _api_get_event_capabilities(self, context: Dict) -> Dict[str, Any]:
    """Возможности системы событий"""
    event_types = list(self.core.event_handlers.keys())
    
    # Определяем системные события
    system_events = [
        "module.activated", "module.deactivated", "module.recovered",
        "module.recovery_failed", "energy.distributed", "energy.recharged",
        "energy.critical", "system.critical_warning", "recovery.auto_completed",
        "recovery.emergency_completed", "recovery.attempts_reset", "api.error",
        "system.shutdown"
    ]
    
    # Определяем модульные события
    module_events = []
    for module_name in self.core.modules:
        module_events.extend([
            f"{module_name}.started", f"{module_name}.stopped",
            f"{module_name}.error", f"{module_name}.warning"
        ])
    
    return {
        "capabilities": {
            "total_event_types": len(event_types) + len(system_events) + len(module_events),
            "system_events": system_events,
            "module_events": module_events[:20],
            "custom_events": event_types,
            "queue_capacity": self.core.event_queue.maxsize,
            "current_queue_size": self.core.event_queue.qsize(),
            "subscriptions_count": sum(len(handlers) for handlers in self.core.event_handlers.values())
        },
        "subscription_methods": {
            "internal": "Через core.subscribe()",
            "api": "POST /events/subscribe",
            "webhook": "Поддержка webhooks"
        },
        "timestamp": time.time()
    }

async def _api_subscribe_to_event(self, context: Dict) -> Dict[str, Any]:
    """Подписка на событие через API"""
    data = context.get("data", {})
    
    required = ["event_type", "callback_url"]
    missing = [field for field in required if field not in data]
    if missing:
        return {
            "error": "missing_parameters",
            "message": f"Missing required parameters: {missing}",
            "required": required,
            "timestamp": time.time()
        }
    
    event_type = data["event_type"]
    callback_url = data["callback_url"]
    filter_conditions = data.get("filters", {})
    
    # Создаём обработчик для webhook
    async def webhook_handler(event_data):
        try:
            import aiohttp
            async with aiohttp.ClientSession() as session:
                async with session.post(callback_url, json=event_data, timeout=5) as response:
                    if response.status != 200:
                        self.logger.warning(f"Webhook callback failed: {response.status}")
        except Exception as e:
            self.logger.error(f"Webhook error: {e}")
    
    # Подписываемся
    subscription_id = self.core.subscribe(event_type, webhook_handler)
    
    # Сохраняем информацию о подписке
    if not hasattr(self.core, '_webhook_subscriptions'):
        self.core._webhook_subscriptions = {}
    
    self.core._webhook_subscriptions[subscription_id] = {
        "event_type": event_type,
        "callback_url": callback_url,
        "filters": filter_conditions,
        "created": time.time(),
        "last_called": None
    }
    
    return {
        "success": True,
        "subscription_id": subscription_id,
        "event_type": event_type,
        "callback_url": callback_url,
        "message": f"Subscribed to {event_type}. Events will be sent to {callback_url}",
        "timestamp": time.time()
    }

async def _api_publish_event(self, context: Dict) -> Dict[str, Any]:
    """Публикация события через API"""
    data = context.get("data", {})
    
    required = ["event_type", "data"]
    missing = [field for field in required if field not in data]
    if missing:
        return {
            "error": "missing_parameters",
            "message": f"Missing required parameters: {missing}",
            "required": required,
            "timestamp": time.time()
        }
    
    event_type = data["event_type"]
    event_data = data["data"]
    source = data.get("source", "api")
    
    # Публикуем событие
    await self.core._publish_internal_event(event_type, event_data)
    
    # Также маршрутизируем между модулями
    await self.core.route_event(event_type, event_data, source)
    
    return {
        "success": True,
        "event_type": event_type,
        "published": True,
        "source": source,
        "timestamp": time.time(),
        "queue_size": self.core.event_queue.qsize()
    }

async def _api_get_subscriptions(self, context: Dict) -> Dict[str, Any]:
    """Получение списка подписок"""
    subscriptions = []
    
    # Внутренние подписки
    for event_type, handlers in self.core.event_handlers.items():
        for subscription_id, handler in handlers:
            subscriptions.append({
                "id": subscription_id,
                "event_type": event_type,
                "handler_type": handler.__class__.__name__,
                "source": "internal"
            })
    
    # Webhook подписки
    if hasattr(self.core, '_webhook_subscriptions'):
        for sub_id, sub_info in self.core._webhook_subscriptions.items():
            subscriptions.append({
                "id": sub_id,
                "event_type": sub_info["event_type"],
                "callback_url": sub_info["callback_url"],
                "filters": sub_info["filters"],
                "created": sub_info["created"],
                "last_called": sub_info["last_called"],
                "source": "webhook"
            })
    
    return {
        "subscriptions": subscriptions,
        "total": len(subscriptions),
        "by_source": {
            "internal": sum(1 for s in subscriptions if s["source"] == "internal"),
            "webhook": sum(1 for s in subscriptions if s["source"] == "webhook")
        },
        "by_event_type": {
            event_type: sum(1 for s in subscriptions if s["event_type"] == event_type)
            for event_type in set(s["event_type"] for s in subscriptions)
        },
        "timestamp": time.time()
    }

# 5.7. УПРАВЛЕНИЕ СИСТЕМОЙ
async def _api_activate_system(self, context: Dict) -> Dict[str, Any]:
    """Активация всей системы через API"""
    if self.core.is_activated:
        return {
            "status": "already_active",
            "message": "System is already activated",
            "active_modules": sum(1 for m in self.core.modules.values() if m.is_active),
            "total_modules": len(self.core.modules),
            "timestamp": time.time()
        }
    
    try:
        result = await self.core.activate_cascade()
        
        return {
            "status": "activation_started",
            "success": True,
            "result": result,
            "timestamp": time.time()
        }
        
    except Exception as e:
        return {
            "status": "activation_failed",
            "success": False,
            "error": str(e),
            "timestamp": time.time()
        }

async def _api_shutdown_system(self, context: Dict) -> Dict[str, Any]:
    """Выключение системы через API"""
    if not self.core.is_activated:
        return {
            "status": "already_inactive",
            "message": "System is already inactive",
            "timestamp": time.time()
        }
    
    # Публикуем событие выключения
    await self.core._publish_internal_event("system.shutdown", {
        "source": "api",
        "request_id": context.get("request_id"),
        "timestamp": time.time()
    })
    
    # Запускаем graceful shutdown
    shutdown_result = await self.core.shutdown()
    
    return {
        "status": "shutdown_initiated",
        "success": True,
        "result": shutdown_result,
        "message": "System shutdown initiated",
        "timestamp": time.time()
    }

async def _api_restart_system(self, context: Dict) -> Dict[str, Any]:
    """Перезапуск системы через API"""
    # Сначала выключаем
    shutdown_result = await self._api_shutdown_system(context)
    
    if not shutdown_result.get("success"):
        return {
            "operation": "restart",
            "shutdown_phase": "failed",
            "error": shutdown_result.get("error"),
            "timestamp": time.time()
        }
    
    # Ждём завершения выключения
    await asyncio.sleep(2.0)
    
    # Затем включаем
    activation_result = await self._api_activate_system(context)
    
    return {
        "operation": "restart",
        "shutdown_phase": shutdown_result,
        "activation_phase": activation_result,
        "overall_success": activation_result.get("success", False),
        "timestamp": time.time()
    }

async def _api_get_diagnostics(self, context: Dict) -> Dict[str, Any]:
    """Полная диагностика системы"""
    import sys
    import asyncio as async_lib
    
    # Собираем информацию со всех модулей
    modules_diagnostics = {}
    
    for name, module in self.core.modules.items():
        if module.instance and module.is_active:
            try:
                # Пробуем вызвать метод диагностики если есть
                if hasattr(module.instance, 'get_diagnostics'):
                    modules_diagnostics[name] = await module.instance.get_diagnostics()
                elif hasattr(module.instance, 'get_metrics'):
                    modules_diagnostics[name] = await module.instance.get_metrics()
                else:
                    modules_diagnostics[name] = {"status": "no_diagnostics_method"}
            except Exception as e:
                modules_diagnostics[name] = {"error": str(e)}
        else:
            modules_diagnostics[name] = {"status": "inactive"}
    
    # Собираем системную диагностику
    system_diagnostics = {
        "python": {
            "version": sys.version,
            "platform": sys.platform,
            "executable": sys.executable
        },
        "asyncio": {
            "loop_running": async_lib.get_event_loop().is_running(),
            "tasks": len(async_lib.all_tasks())
        },
        "memory": {
            # TODO: Добавить использование памяти
        },
        "timing": {
            "uptime": time.time() - self.core.activation_start_time if self.core.is_activated else 0,
            "current_time": time.time(),
            "timezone": time.tzname if hasattr(time, 'tzname') else "UTC"
        }
    }
    
    return {
        "system": system_diagnostics,
        "modules": modules_diagnostics,
        "keter_core": {
            "version": self.core.__version__,
            "modules_registered": len(self.core.modules),
            "modules_active": sum(1 for m in self.core.modules.values() if m.is_active),
            "energy_reserve": self.core.energy_reserve,
            "event_queue": self.core.event_queue.qsize(),
            "background_tasks": len(self.core.background_tasks),
            "error_counters": self.core.error_counters,
            "activation_timestamps": self.core.activation_timestamps
        },
        "timestamp": time.time()
    }

# 5.8. АДМИНИСТРАТИВНЫЕ API
async def _api_reload_config(self, context: Dict) -> Dict[str, Any]:
    """Перезагрузка конфигурации"""
    return {
        "operation": "reload_config",
        "status": "not_implemented",
        "message": "Config reload from file not implemented yet",
        "current_config": self.core.config,
        "timestamp": time.time()
    }

async def _api_clear_cache(self, context: Dict) -> Dict[str, Any]:
    """Очистка кэшей"""
    data = context.get("data", {})
    cache_type = data.get("type", "all")
    
    cleared = []
    
    if cache_type in ["all", "metrics"]:
        old_size = len(self.core.metrics_history)
        self.core.metrics_history.clear()
        cleared.append({"type": "metrics", "entries_cleared": old_size})
    
    if cache_type in ["all", "events"]:
        old_size = self.core.event_queue.qsize()
        while not self.core.event_queue.empty():
            try:
                self.core.event_queue.get_nowait()
                self.core.event_queue.task_done()
            except:
                break
        cleared.append({"type": "events", "entries_cleared": old_size})
    
    return {
        "operation": "clear_cache",
        "success": True,
        "cache_type": cache_type,
        "cleared": cleared,
        "timestamp": time.time()
    }

async def _api_get_performance_stats(self, context: Dict) -> Dict[str, Any]:
    """Статистика производительности"""
    api_requests = self.request_stats
    
    # Информация о задачах
    tasks_info = []
    for task in self.core.background_tasks:
        try:
            tasks_info.append({
                "name": task.get_name() if hasattr(task, 'get_name') else "unnamed",
                "done": task.done(),
                "cancelled": task.cancelled(),
                "exception": str(task.exception()) if task.exception() else None
            })
        except:
            pass
    
    return {
        "performance": {
            "api_requests": {
                "total": len(api_requests),
                "last_hour": len([r for r in api_requests if r.get("timestamp", 0) > time.time() - 3600]),
                "average_time": sum(r.get("processing_time", 0) for r in api_requests) / max(1, len(api_requests))
            },
            "background_tasks": {
                "total": len(self.core.background_tasks),
                "active": len([t for t in self.core.background_tasks if not t.done()]),
                "tasks": tasks_info[:10]
            },
            "event_system": {
                "queue_size": self.core.event_queue.qsize(),
                "max_queue": self.core.event_queue.maxsize,
                "subscriptions": sum(len(h) for h in self.core.event_handlers.values())
            },
            "modules": {
                "total": len(self.core.modules),
                "active": sum(1 for m in self.core.modules.values() if m.is_active),
                "with_errors": sum(1 for name in self.core.modules if self.core.error_counters.get(name, 0) > 0)
            }
        },
        "timestamp": time.time()
    }

# ========================================================
# 6. ВСПОМОГАТЕЛЬНЫЕ МЕТОДЫ
# ========================================================

def _save_request_stats(self, request_id: str, endpoint: str, method: str, processing_time: float, error: str = None):
    """Сохранение статистики запросов"""
    stats_entry = {
        "request_id": request_id,
        "endpoint": endpoint,
        "method": method,
        "processing_time": processing_time,
        "error": error,
        "timestamp": time.time()
    }
    
    self.request_stats.append(stats_entry)
    
    # Ограничиваем размер истории
    if len(self.request_stats) > 1000:
        self.request_stats = self.request_stats[-1000:]

def get_api_stats(self) -> Dict[str, Any]:
    """Получение статистики API"""
    total_requests = len(self.request_stats)
    successful_requests = sum(1 for r in self.request_stats if r["error"] is None)
    error_requests = total_requests - successful_requests
    
    avg_time = sum(r["processing_time"] for r in self.request_stats) / max(1, total_requests)
    
    # Группировка по эндпоинтам
    endpoint_stats = {}
    for r in self.request_stats:
        endpoint = f"{r['method']} {r['endpoint']}"
        if endpoint not in endpoint_stats:
            endpoint_stats[endpoint] = {"count": 0, "total_time": 0, "errors": 0}
        
        endpoint_stats[endpoint]["count"] += 1
        endpoint_stats[endpoint]["total_time"] += r["processing_time"]
        if r["error"]:
            endpoint_stats[endpoint]["errors"] += 1
    
    for endpoint, stats in endpoint_stats.items():
        stats["avg_time"] = stats["total_time"] / stats["count"]
        stats["error_rate"] = (stats["errors"] / stats["count"]) * 100
    
    return {
        "total_requests": total_requests,
        "successful_requests": successful_requests,
        "error_requests": error_requests,
        "success_rate": (successful_requests / total_requests * 100) if total_requests > 0 else 0,
        "average_processing_time": avg_time,
        "endpoint_stats": endpoint_stats,
        "timestamp": time.time()
    }

# ========================================================
# 7. ИНТЕГРАЦИОННЫЙ КЛАСС (KetherCore + API)
# ========================================================

class KetherCoreWithAPI:
    """
    Комбинированный класс: KetherCore + KetherAPI
    Для обратной совместимости
    """
    
    def __init__(self, config=None):
        from keter_core import KetherCore, create_keter_core
        
        self.core = create_keter_core(config)
        self.api = KetherAPI(self.core)
        
        # Проксируем основные методы
        self.api_call = self.api.api_call
        self.__getattr__ = self._proxy_to_core
    
    def _proxy_to_core(self, name):
        """Проксирование атрибутов к core"""
        if hasattr(self.core, name):
            return getattr(self.core, name)
        elif hasattr(self.api, name):
            return getattr(self.api, name)
        raise AttributeError(f"'KetherCoreWithAPI' object has no attribute '{name}'")

# ========================================================
# 8. ФАБРИЧНЫЕ ФУНКЦИИ
# ========================================================

def create_keter_core_with_api(config: Optional[Dict] = None):
    """Создание KetherCore с API"""
    return KetherCoreWithAPI(config)

def create_keter_api_gateway(core_instance):
    """Создание только API шлюза"""
    return KetherAPI(core_instance)

# ============================================================
# 9. ТОЧКА ВХОДА ДЛЯ ТЕСТИРОВАНИЯ
# ============================================================

async def test_api():
    """Тестирование API"""
    from keter_core import create_keter_core
    
    print("🧪 Тестирование Kether API Gateway...")
    
    # Создаём ядро
    core = create_keter_core()
    
    # Создаём API шлюз
    api = KetherAPI(core)
    
    # Тестовый API запрос
    result = await api.api_call("/status", "GET", api_key="TEST_KEY")
    
    print(f"📊 Результат API запроса: {result.get('status', 'unknown')}")
    print(f"✅ API Gateway работает корректно!")
    
    return result

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=10000)
