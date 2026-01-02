"""
chokmah_api.py - API шлюз для сефиры CHOKMAH.
Создан по образцу keter_api.py
"""

import logging
import time
from typing import Dict, Any, Optional, Callable

logger = logging.getLogger(__name__)


class ChokmahAPI:
    """
    API шлюз для WisdomCore
    Стиль полностью соответствует KetherAPI
    """
    
    def __init__(self, core):
        """
        Инициализация API шлюза CHOKMAH
        
        Args:
            core: Экземпляр WisdomCore
        """
        self.core = core
        self.logger = logging.getLogger("ChokmahAPI")
        
        # Конфигурация API (упрощённая версия KETER)
        self.api_config = {
            "auth_required": False,  # У CHOKMAH проще безопасность
            "rate_limit_enabled": False,
            "api_keys": {
                "CHOKMAH_MASTER_KEY": {"level": "admin", "rate_limit": 100},
                "SEPHIROTIC_ENGINE": {"level": "system", "rate_limit": 50},
            }
        }
        
        # Статистика запросов
        self.request_stats = []
        self.request_counter = 0
        
        self.logger.info("Chokmah API Gateway v1.0 инициализирован")
    
    async def api_call(self,
                      endpoint: str,
                      method: str = "GET",
                      data: Optional[Dict] = None,
                      api_key: Optional[str] = None,
                      client_info: Optional[Dict] = None) -> Dict[str, Any]:
        """
        API шлюз CHOKMAH (упрощённая версия KETER)
        """
        request_id = f"chokmah_{int(time.time())}_{self.request_counter:04d}"
        self.request_counter += 1
        start_time = time.time()
        
        self.logger.info(f"🌐 CHOKMAH API запрос [{request_id}]: {method} {endpoint}")
        
        # Упрощённая аутентификация (по сравнению с KETER)
        auth_result = await self._api_authenticate(api_key, client_info)
        if not auth_result["authenticated"]:
            return {
                "request_id": request_id,
                "error": "authentication_failed",
                "message": auth_result.get("message", "Invalid credentials"),
                "status_code": 401,
                "timestamp": time.time()
            }
        
        # Нормализация endpoint
        endpoint = endpoint.strip('/')
        if not endpoint.startswith('/'):
            endpoint = '/' + endpoint
        
        # Поиск обработчика
        handler, route_params = self._find_api_handler(method, endpoint)
        
        if not handler:
            return {
                "request_id": request_id,
                "error": "endpoint_not_found",
                "message": f"No handler for {method} {endpoint}",
                "status_code": 404,
                "available_endpoints": self._get_available_endpoints(),
                "timestamp": time.time()
            }
        
        # Выполнение обработчика
        try:
            request_context = {
                "request_id": request_id,
                "endpoint": endpoint,
                "method": method,
                "data": data or {},
                "params": route_params,
                "start_time": start_time
            }
            
            result = await handler(request_context)
            
            # Добавляем метаданные
            processing_time = time.time() - start_time
            result.update({
                "request_id": request_id,
                "processing_time": round(processing_time, 4),
                "timestamp": time.time(),
                "success": result.get("error") is None
            })
            
            self.logger.info(f"✅ CHOKMAH API запрос завершён [{request_id}]: {processing_time:.3f}s")
            
            return result
            
        except Exception as e:
            error_msg = str(e)
            self.logger.error(f"❌ Ошибка CHOKMAH API [{request_id}]: {error_msg}")
            
            return {
                "request_id": request_id,
                "error": "internal_server_error",
                "message": error_msg,
                "status_code": 500,
                "timestamp": time.time()
            }
    
    async def _api_authenticate(self, api_key: Optional[str], client_info: Optional[Dict]) -> Dict[str, Any]:
        """Упрощённая аутентификация для CHOKMAH"""
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
                "access_level": key_info["level"]
            }
        
        # Для внутренних вызовов из системы
        if client_info and client_info.get("internal_call") == True:
            return {
                "authenticated": True,
                "auth_method": "internal",
                "access_level": "system"
            }
        
        return {
            "authenticated": False,
            "message": "Invalid API key for CHOKMAH"
        }
    
    def _find_api_handler(self, method: str, endpoint: str):
        """Поиск обработчика API запроса"""
        api_routes = self._get_api_routes()
        
        # Проверяем точное совпадение
        if (method, endpoint) in api_routes:
            return api_routes[(method, endpoint)], {}
        
        # Проверяем паттерн с параметрами (простая версия)
        for (route_method, route_pattern), handler in api_routes.items():
            if method == route_method and endpoint == route_pattern:
                return handler, {}
        
        return None, {}
    
    def _get_api_routes(self) -> Dict[tuple, Callable]:
        """Таблица маршрутизации API CHOKMAH (упрощённая)"""
        return {
            # === СИСТЕМНЫЕ ЭНДПОИНТЫ ===
            ("GET", "/"): self._api_root,
            ("GET", "/status"): self._api_status,
            ("GET", "/health"): self._api_health,
            
            # === ОСНОВНЫЕ ФУНКЦИИ ===
            ("POST", "/activate"): self._api_activate,
            ("POST", "/process"): self._api_process,
            ("POST", "/insight"): self._api_insight,
            
            # === ДИАГНОСТИКА ===
            ("GET", "/diagnostics"): self._api_diagnostics,
        }
    
    def _get_available_endpoints(self) -> list:
        """Список доступных эндпоинтов"""
        endpoints = []
        for (method, pattern), _ in self._get_api_routes().items():
            endpoints.append(f"{method} {pattern}")
        return endpoints
    
    # ========================================================
    # API ОБРАБОТЧИКИ CHOKMAH
    # ========================================================
    
    async def _api_root(self, context: Dict) -> Dict[str, Any]:
        """Корневой эндпоинт"""
        return {
            "sephira": "CHOKMAH",
            "name": "Поток Мудрости",
            "version": "1.0",
            "status": "active" if self.core.is_activated() else "dormant",
            "endpoints": [
                "GET /status",
                "GET /health", 
                "POST /activate",
                "POST /process",
                "POST /insight",
                "GET /diagnostics"
            ],
            "resonance": self.core.resonance,
            "energy": self.core.energy
        }
    
    async def _api_status(self, context: Dict) -> Dict[str, Any]:
        """Статус CHOKMAH"""
        status = await self.core.get_status()
        return {
            "sephira": "CHOKMAH",
            "status": status,
            "timestamp": time.time()
        }
    
    async def _api_health(self, context: Dict) -> Dict[str, Any]:
        """Проверка здоровья"""
        return {
            "sephira": "CHOKMAH",
            "healthy": True,
            "modules_loaded": self.core.intuition_matrix is not None and self.core.chernigovskaya is not None,
            "resonance": self.core.resonance,
            "is_activated": self.core.is_activated(),
            "timestamp": time.time()
        }
    
    async def _api_activate(self, context: Dict) -> Dict[str, Any]:
        """Активация CHOKMAH"""
        result = await self.core.activate()
        return {
            "sephira": "CHOKMAH",
            "operation": "activation",
            "result": result,
            "timestamp": time.time()
        }
    
    async def _api_process(self, context: Dict) -> Dict[str, Any]:
        """Обработка запроса через CHOKMAH"""
        data = context.get("data", {})
        
        if "text" not in data:
            return {
                "error": "missing_parameter",
                "message": "Parameter 'text' is required",
                "sephira": "CHOKMAH"
            }
        
        text = data["text"]
        context_data = data.get("context", {})
        
        result = await self.core.process(text, context_data)
        
        return {
            "sephira": "CHOKMAH",
            "operation": "process",
            "result": result,
            "timestamp": time.time()
        }
    
    async def _api_insight(self, context: Dict) -> Dict[str, Any]:
        """Быстрый инсайт (авто-активация)"""
        data = context.get("data", {})
        
        if "text" not in data:
            return {
                "error": "missing_parameter",
                "message": "Parameter 'text' is required",
                "sephira": "CHOKMAH"
            }
        
        # Авто-активация если нужно
        if not self.core.is_activated():
            await self.core.activate()
        
        text = data["text"]
        context_data = data.get("context", {})
        
        result = await self.core.process(text, context_data)
        
        return {
            "sephira": "CHOKMAH",
            "operation": "insight",
            "insight": result.get("insight"),
            "resonance": result.get("resonance"),
            "timestamp": time.time()
        }
    
    async def _api_diagnostics(self, context: Dict) -> Dict[str, Any]:
        """Диагностика CHOKMAH"""
        status = await self.core.get_status()
        
        return {
            "sephira": "CHOKMAH",
            "diagnostics": {
                "core": status,
                "api": {
                    "total_requests": len(self.request_stats),
                    "average_time": sum(r.get("processing_time", 0) for r in self.request_stats) / max(1, len(self.request_stats)) if self.request_stats else 0
                },
                "modules": {
                    "intuition_matrix": self.core.intuition_matrix is not None,
                    "chernigovskaya": self.core.chernigovskaya is not None
                }
            },
            "timestamp": time.time()
        }


# Фабричная функция для создания API
def create_chokmah_api(core) -> ChokmahAPI:
    """Создание API шлюза для CHOKMAH"""
    return ChokmahAPI(core)
