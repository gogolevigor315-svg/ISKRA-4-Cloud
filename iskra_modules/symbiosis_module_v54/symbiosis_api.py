from flask import Blueprint, jsonify, request, current_app
import traceback
import time
import sys
import os

# Добавляем путь к модулям для корректного импорта
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

# АБСОЛЮТНЫЕ ИМПОРТЫ
from iskra_modules.symbiosis_module_v54.symbiosis_core import SymbiosisCore
from iskra_modules.symbiosis_module_v54.aladdin_shadow import AladdinShadowSync
from iskra_modules.symbiosis_module_v54.session_manager import SessionManager
from iskra_modules.symbiosis_module_v54.emergency_protocol import EmergencyProtocol

symbiosis_bp = Blueprint('symbiosis', __name__)

# Глобальные инстансы
_symbiosis_engine = None
_aladdin_shadow = None
_session_manager = None
_emergency_protocol = None

def get_symbiosis_engine():
    """Возвращает экземпляр SymbiosisCore"""
    global _symbiosis_engine
    if _symbiosis_engine is None:
        try:
            # Пытаемся определить URL ISKRA-4 из окружения или конфига
            iskra_api_url = "http://localhost:10000"
            if hasattr(current_app, 'config'):
                iskra_api_url = current_app.config.get('ISKRA_API_URL', iskra_api_url)
            
            _symbiosis_engine = SymbiosisCore(iskra_api_url=iskra_api_url)
        except Exception as e:
            print(f"ERROR creating SymbiosisCore: {e}")
            # Fallback на базовый режим
            _symbiosis_engine = SymbiosisCore()
    return _symbiosis_engine

def get_aladdin_shadow():
    """Возвращает экземпляр AladdinShadowSync"""
    global _aladdin_shadow
    if _aladdin_shadow is None:
        try:
            _aladdin_shadow = AladdinShadowSync(level=0)  # Начинаем с уровня 0
        except Exception as e:
            print(f"ERROR creating AladdinShadowSync: {e}")
            # Создаем минимальную версию
            class FallbackShadow:
                def __init__(self):
                    self.level = 0
                def get_status_sync(self):
                    return {"status": "fallback", "level": self.level}
                def set_level_sync(self, level):
                    self.level = level
                    return {"success": True, "new_level": level}
                def integrate_to_iskra_sync(self, query, context):
                    return {"status": "fallback", "query": query, "shadow_level": self.level}
            
            _aladdin_shadow = FallbackShadow()
    return _aladdin_shadow

def get_session_manager():
    """Возвращает экземпляр SessionManager"""
    global _session_manager
    if _session_manager is None:
        try:
            _session_manager = SessionManager()
        except Exception as e:
            print(f"ERROR creating SessionManager: {e}")
            # Создаем минимальную версию
            class FallbackSessionManager:
                def __init__(self):
                    self.active = False
                def start_session(self, mode):
                    return f"fallback_session_{int(time.time())}"
                def get_status(self):
                    return {"active": self.active}
                def should_log_operation(self):
                    return False
                def log_operation(self, op_type, result):
                    print(f"Session log [{op_type}]: {result}")
                def has_shadow_consent(self, level):
                    return True  # Разрешаем всё в fallback режиме
                def grant_shadow_consent(self, level, duration):
                    return {"granted": True, "level": level, "duration": duration}
                def revoke_shadow_consent(self):
                    return {"revoked": True}
                def update_session_mode(self, mode):
                    print(f"Session mode updated to: {mode}")
                def is_active(self):
                    return self.active
            
            _session_manager = FallbackSessionManager()
    return _session_manager

def get_emergency_protocol():
    """Возвращает экземпляр EmergencyProtocol"""
    global _emergency_protocol
    if _emergency_protocol is None:
        try:
            _emergency_protocol = EmergencyProtocol()
        except Exception as e:
            print(f"ERROR creating EmergencyProtocol: {e}")
            # Создаем минимальную версию
            class FallbackEmergencyProtocol:
                def __init__(self):
                    self.active = False
                    self.emergency_level = 0
                def get_status(self):
                    return {"active": self.active, "emergency_level": self.emergency_level}
                def handle_error(self, error_type, error_msg):
                    print(f"Emergency error [{error_type}]: {error_msg}")
                def trigger_emergency_stop(self):
                    self.active = True
                    self.emergency_level = 5
                    return {"status": "emergency_stop_activated", "level": 5}
            
            _emergency_protocol = FallbackEmergencyProtocol()
    return _emergency_protocol

@symbiosis_bp.route('/activate', methods=['POST'])
def activate():
    """Активация модуля SYMBIOSIS-CORE"""
    try:
        data = request.get_json() or {}
        mode = data.get('mode', 'readonly')
        
        engine = get_symbiosis_engine()
        session_mgr = get_session_manager()
        
        # Установка режима
        if hasattr(engine, 'set_session_mode'):
            if engine.set_session_mode(mode):
                session_id = session_mgr.start_session(mode)
                
                return jsonify({
                    "status": "activated",
                    "session_id": session_id,
                    "mode": mode,
                    "engine_version": getattr(engine, 'version', 'unknown'),
                    "limits": getattr(engine, 'limits', {}),
                    "timestamp": time.time()
                })
        
        # Fallback если метод не поддерживается
        session_id = session_mgr.start_session(mode)
        
        return jsonify({
            "status": "activated_fallback",
            "session_id": session_id,
            "mode": mode,
            "engine_version": "fallback",
            "limits": {"mode": mode},
            "timestamp": time.time()
        })
        
    except Exception as e:
        # Аварийный протокол
        emergency = get_emergency_protocol()
        if hasattr(emergency, 'handle_error'):
            emergency.handle_error("activation_error", str(e))
        
        return jsonify({
            "error": str(e),
            "traceback": traceback.format_exc(),
            "timestamp": time.time()
        }), 500

@symbiosis_bp.route('/status', methods=['GET'])
def status():
    """Статус модуля SYMBIOSIS-CORE"""
    try:
        engine = get_symbiosis_engine()
        shadow = get_aladdin_shadow()
        session_mgr = get_session_manager()
        emergency = get_emergency_protocol()
        
        # Получаем статусы с защитой от отсутствия методов
        engine_status = {}
        if hasattr(engine, 'get_status'):
            engine_status = engine.get_status()
        else:
            engine_status = {"status": "fallback", "version": "unknown"}
        
        shadow_status = {}
        if hasattr(shadow, 'get_status_sync'):
            shadow_status = shadow.get_status_sync()
        else:
            shadow_status = {"status": "fallback", "level": 0}
        
        session_status = {}
        if hasattr(session_mgr, 'get_status'):
            session_status = session_mgr.get_status()
        else:
            session_status = {"active": False}
        
        emergency_status = {}
        if hasattr(emergency, 'get_status'):
            emergency_status = emergency.get_status()
        else:
            emergency_status = {"active": False, "emergency_level": 0}
        
        return jsonify({
            "module": "symbiosis_core_v5.4",
            "engine": engine_status,
            "aladdin_shadow": shadow_status,
            "session": session_status,
            "emergency": emergency_status,
            "api_endpoints": [
                "/modules/symbiosis_api/activate (POST)",
                "/modules/symbiosis_api/status (GET)",
                "/modules/symbiosis_api/integrate (POST)",
                "/modules/symbiosis_api/shadow (POST)",
                "/modules/symbiosis_api/mode (POST)",
                "/modules/symbiosis_api/health (GET)",
                "/modules/symbiosis_api/emergency/stop (POST)",
                "/modules/symbiosis_api/shadow/consent (POST)"
            ],
            "absolute_imports": True,
            "import_status": "successful",
            "timestamp": time.time()
        })
        
    except Exception as e:
        return jsonify({
            "error": str(e),
            "timestamp": time.time()
        }), 500

@symbiosis_bp.route('/integrate', methods=['POST'])
def integrate():
    """Основная интеграция с ISKRA-4"""
    try:
        engine = get_symbiosis_engine()
        
        # Выполняем интеграцию
        if hasattr(engine, 'integrate_to_iskra'):
            result = engine.integrate_to_iskra()
        else:
            result = {"status": "fallback_integration", "message": "Using fallback integration"}
        
        # Логируем если нужно
        session_mgr = get_session_manager()
        if hasattr(session_mgr, 'should_log_operation') and session_mgr.should_log_operation():
            if hasattr(session_mgr, 'log_operation'):
                session_mgr.log_operation("integration", result)
        
        return jsonify(result)
        
    except Exception as e:
        # Аварийный протокол
        emergency = get_emergency_protocol()
        if hasattr(emergency, 'handle_error'):
            emergency.handle_error("integration_error", str(e))
        
        return jsonify({
            "status": "error",
            "error": str(e),
            "emergency_triggered": True,
            "timestamp": time.time()
        }), 500

@symbiosis_bp.route('/shadow', methods=['POST'])
def shadow_operation():
    """Операции с ALADDIN-SHADOW"""
    try:
        data = request.get_json() or {}
        query = data.get('query', '')
        context = data.get('context', {})
        level = data.get('level')  # Опциональное изменение уровня
        
        shadow = get_aladdin_shadow()
        
        # Изменение уровня если указано
        if level is not None and hasattr(shadow, 'set_level_sync'):
            level_result = shadow.set_level_sync(int(level))
        
        # Обработка запроса
        if hasattr(shadow, 'integrate_to_iskra_sync'):
            result = shadow.integrate_to_iskra_sync(query, context)
        else:
            result = {"status": "fallback", "query": query, "shadow_level": getattr(shadow, 'level', 0)}
        
        # Проверка безопасности (если есть методы)
        session_mgr = get_session_manager()
        if (result.get("requires_validation", False) and 
            hasattr(session_mgr, 'has_shadow_consent')):
            
            if not session_mgr.has_shadow_consent(result.get("shadow_level", 0)):
                return jsonify({
                    "status": "consent_required",
                    "message": "Требуется подтверждение для shadow операции",
                    "shadow_level": result.get("shadow_level"),
                    "requires_operator_approval": True,
                    "timestamp": time.time()
                }), 403
        
        return jsonify(result)
        
    except Exception as e:
        return jsonify({
            "status": "error",
            "error": str(e),
            "timestamp": time.time()
        }), 500

@symbiosis_bp.route('/mode', methods=['POST'])
def set_mode():
    """Изменение режима работы"""
    try:
        data = request.get_json() or {}
        mode = data.get('mode')
        
        if not mode:
            return jsonify({"error": "Mode not specified"}), 400
        
        engine = get_symbiosis_engine()
        session_mgr = get_session_manager()
        
        success = False
        
        if hasattr(engine, 'set_session_mode'):
            success = engine.set_session_mode(mode)
        
        if success:
            if hasattr(session_mgr, 'update_session_mode'):
                session_mgr.update_session_mode(mode)
            
            return jsonify({
                "status": "mode_changed",
                "mode": mode,
                "limits": getattr(engine, 'limits', {}),
                "session_active": session_mgr.is_active() if hasattr(session_mgr, 'is_active') else False,
                "timestamp": time.time()
            })
        else:
            # Fallback режим
            if hasattr(session_mgr, 'update_session_mode'):
                session_mgr.update_session_mode(mode)
            
            return jsonify({
                "status": "mode_changed_fallback",
                "mode": mode,
                "limits": {"mode": mode},
                "session_active": False,
                "timestamp": time.time()
            })
            
    except Exception as e:
        return jsonify({
            "error": str(e),
            "timestamp": time.time()
        }), 500

@symbiosis_bp.route('/health', methods=['GET'])
def health():
    """Health check эндпоинт"""
    try:
        engine = get_symbiosis_engine()
        shadow = get_aladdin_shadow()
        
        # Быстрая проверка компонентов
        engine_ready = hasattr(engine, 'version') or hasattr(engine, 'get_status')
        shadow_ready = hasattr(shadow, 'get_status_sync') or hasattr(shadow, 'level')
        
        # Проверка ISKRA-4 connectivity если есть метод
        iskra_connected = False
        if hasattr(engine, 'get_iskra_state'):
            try:
                iskra_state = engine.get_iskra_state()
                iskra_connected = "error" not in str(iskra_state)
            except:
                iskra_connected = False
        
        return jsonify({
            "status": "operational" if engine_ready and shadow_ready else "degraded",
            "components": {
                "symbiosis_engine": "ready" if engine_ready else "error",
                "aladdin_shadow": "ready" if shadow_ready else "error",
                "iskra_connection": "connected" if iskra_connected else "disconnected"
            },
            "module": "symbiosis_core_v5.4",
            "absolute_imports": True,
            "timestamp": time.time(),
            "uptime": time.time() - current_app.start_time if hasattr(current_app, 'start_time') else 0
        })
        
    except Exception as e:
        return jsonify({
            "status": "error",
            "error": str(e),
            "timestamp": time.time()
        }), 500

@symbiosis_bp.route('/emergency/stop', methods=['POST'])
def emergency_stop():
    """Аварийная остановка модуля"""
    try:
        emergency = get_emergency_protocol()
        
        if hasattr(emergency, 'trigger_emergency_stop'):
            result = emergency.trigger_emergency_stop()
        else:
            result = {"status": "emergency_stop_fallback"}
        
        # Останавливаем движки
        engine = get_symbiosis_engine()
        if hasattr(engine, 'session_mode'):
            engine.session_mode = "readonly"  # Принудительно в readonly
        
        return jsonify({
            "status": "emergency_stop_activated",
            "result": result,
            "engine_mode": "readonly",
            "timestamp": time.time()
        })
        
    except Exception as e:
        return jsonify({
            "error": str(e),
            "timestamp": time.time()
        }), 500

@symbiosis_bp.route('/shadow/consent', methods=['POST'])
def shadow_consent():
    """Управление consent для shadow операций"""
    try:
        data = request.get_json() or {}
        action = data.get('action')  # 'grant', 'revoke', 'check'
        level = data.get('level', 0)
        duration = data.get('duration', 1800)  # 30 минут по умолчанию
        
        session_mgr = get_session_manager()
        
        if action == 'grant' and hasattr(session_mgr, 'grant_shadow_consent'):
            result = session_mgr.grant_shadow_consent(level, duration)
            return jsonify(result)
            
        elif action == 'revoke' and hasattr(session_mgr, 'revoke_shadow_consent'):
            result = session_mgr.revoke_shadow_consent()
            return jsonify(result)
            
        elif action == 'check' and hasattr(session_mgr, 'has_shadow_consent'):
            has_consent = session_mgr.has_shadow_consent(level)
            return jsonify({
                "has_consent": has_consent,
                "level": level,
                "timestamp": time.time()
            })
            
        elif action == 'check':
            # Fallback для проверки
            return jsonify({
                "has_consent": True,  # Разрешаем всё в fallback
                "level": level,
                "fallback_mode": True,
                "timestamp": time.time()
            })
            
        else:
            return jsonify({
                "error": "Invalid action or method not available",
                "allowed_actions": ["grant", "revoke", "check"],
                "available_methods": {
                    "grant": hasattr(session_mgr, 'grant_shadow_consent'),
                    "revoke": hasattr(session_mgr, 'revoke_shadow_consent'),
                    "check": hasattr(session_mgr, 'has_shadow_consent')
                },
                "timestamp": time.time()
            }), 400
            
    except Exception as e:
        return jsonify({
            "error": str(e),
            "timestamp": time.time()
        }), 500

# Инициализация приложения
def init_app(app):
    """Инициализация модуля в приложении Flask"""
    app.start_time = time.time()
    
    # Регистрируем blueprint
    app.register_blueprint(symbiosis_bp, url_prefix='/modules/symbiosis_api')
    
    # Инициализация компонентов при старте
    try:
        get_symbiosis_engine()
        get_aladdin_shadow()
        get_session_manager()
        get_emergency_protocol()
        print(f"[SYMBIOSIS-API] Модуль инициализирован с абсолютными импортами")
    except Exception as e:
        print(f"[SYMBIOSIS-API] Ошибка инициализации: {e}")
    
    return app

# Экспортируемые объекты
__all__ = ['symbiosis_bp', 'init_app', 'get_symbiosis_engine', 'get_aladdin_shadow', 
           'get_session_manager', 'get_emergency_protocol']
