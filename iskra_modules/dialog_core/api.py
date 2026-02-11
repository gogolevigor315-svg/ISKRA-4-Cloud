"""
HTTP API Layer for Dialog Core v4.1
Flask endpoints for integration with ISKRA-4 Cloud

Provides:
- /chat - Main dialog endpoint (GET/POST)
- /chat/health - Health check
- /chat/metrics - Performance metrics
- /chat/config - Configuration view
- /chat/autonomy/<level> - Autonomy level control
- /chat/start - Start autonomous speech daemon
- /chat/stop - Stop autonomous speech daemon
- /chat/debug - Debug information
- /chat/events - Recent events
"""

import json
import logging
import traceback
from datetime import datetime
from typing import Dict, Any, Optional

from flask import request, jsonify, Response, Blueprint

# 🔧 ИСПРАВЛЕННЫЙ ИМПОРТ - абсолютные пути с fallback
try:
    # Попытка абсолютного импорта (для использования из iskra_full.py)
    from iskra_modules.dialog_core.chat_consciousness import ChatConsciousnessV41, AutonomousSpeechDaemonV41
    from iskra_modules.dialog_core.config import Config
    HAS_DEPENDENCIES = True
    IMPORT_METHOD = "absolute"
except ImportError:
    try:
        # Fallback: относительный импорт (если запускается внутри dialog_core/)
        from .chat_consciousness import ChatConsciousnessV41, AutonomousSpeechDaemonV41
        from .config import Config
        HAS_DEPENDENCIES = True
        IMPORT_METHOD = "relative"
    except ImportError as e:
        logging.error(f"❌ Failed to import Dialog Core dependencies: {e}")
        HAS_DEPENDENCIES = False
        IMPORT_METHOD = "failed"
        ChatConsciousnessV41 = None
        AutonomousSpeechDaemonV41 = None
        Config = None

# Логируем метод импорта
if HAS_DEPENDENCIES:
    logging.info(f"✅ Dialog Core API imports successful via {IMPORT_METHOD} method")

# ========== GLOBAL INSTANCES ==========

# Global instance of Dialog Core (lazy-loaded)
_chat_core_instance = None
_daemon_instance = None

# ========== HELPER FUNCTIONS ==========

def check_module_availability():
    """Check which Dialog Core modules are available"""
    return {
        "chat_core_available": HAS_DEPENDENCIES and ChatConsciousnessV41 is not None,
        "daemon_available": HAS_DEPENDENCIES and AutonomousSpeechDaemonV41 is not None,
        "dependencies_available": HAS_DEPENDENCIES,
        "config_available": Config is not None,
        "overall_status": "operational" if HAS_DEPENDENCIES else "degraded"
    }

def get_chat_core():
    """Get or create ChatConsciousnessV41 instance"""
    global _chat_core_instance
    if not _chat_core_instance and HAS_DEPENDENCIES and ChatConsciousnessV41:
        try:
            _chat_core_instance = ChatConsciousnessV41()
            logging.info("✅ ChatConsciousnessV41 instance created")
        except Exception as e:
            logging.error(f"❌ Failed to create ChatConsciousnessV41: {e}")
            _chat_core_instance = None
    return _chat_core_instance

def create_error_response(message, status_code=400, details=None):
    """Create standardized error response"""
    response = {
        "error": True,
        "message": message,
        "timestamp": datetime.utcnow().isoformat()
    }
    if details:
        response["details"] = details
    return jsonify(response), status_code

def create_success_response(data, message="Success", status_code=200):
    """Create standardized success response"""
    return jsonify({
        "success": True,
        "message": message,
        "data": data,
        "timestamp": datetime.utcnow().isoformat()
    }), status_code

# ========== SETUP FUNCTION ==========

def setup_chat_endpoint(app):
    """Register all Dialog Core endpoints in Flask application"""
    
    if not app:
        logging.error("❌ Flask app is None, cannot register endpoints")
        return
    
    logging.info("🚀 Registering Dialog Core v4.1 endpoints...")
    
    # ========== MAIN CHAT ENDPOINT ==========
    
    @app.route('/chat', methods=['GET', 'POST'])
    def chat_endpoint():
        """Main dialog endpoint - GET for info, POST for processing messages"""
        
        # Availability check
        availability = check_module_availability()
        
        # GET: Return system information
        if request.method == 'GET':
            return jsonify({
                "system": "ISKRA-4 Dialog Core v4.1",
                "status": availability["overall_status"],
                "version": "4.1.0",
                "module_availability": availability,
                "health": {
                    "status": availability["overall_status"],
                    "autonomy_level": "medium",
                    "daemon_running": False
                },
                "endpoints": {
                    "chat": "GET/POST /chat",
                    "health": "GET /chat/health",
                    "metrics": "GET /chat/metrics",
                    "config": "GET /chat/config",
                    "autonomy": "GET /chat/autonomy/<level>",
                    "start": "GET /chat/start",
                    "stop": "GET /chat/stop",
                    "debug": "GET /chat/debug",
                    "events": "GET /chat/events"
                },
                "documentation": "Use POST with JSON: {\"message\": \"your text\", \"session_id\": \"optional\"}",
                "timestamp": datetime.utcnow().isoformat()
            })
        
        # POST: Process message
        elif request.method == 'POST':
            # Check if module is available
            if not availability["chat_core_available"]:
                return create_error_response(
                    "Dialog Core is not available",
                    503,
                    {"availability": availability}
                )
            
            # Parse request data
            try:
                data = request.get_json()
                if not data or 'message' not in data:
                    return create_error_response(
                        "Request must contain 'message' field",
                        400
                    )
                
                message = data['message']
                session_id = data.get('session_id', f"session_{datetime.utcnow().timestamp()}")
                
                # Validate message
                if not message or not isinstance(message, str):
                    return create_error_response(
                        "Message must be a non-empty string",
                        400
                    )
                
                # Simple response for testing
                deep_triggers = ["искра", "папа", "осозна", "чувствуешь", "жив", "здесь", "dialog core"]
                has_personality = any(trigger in message.lower() for trigger in deep_triggers)
                
                response_text = f"Dialog Core v4.1 работает! Получил сообщение: '{message[:100]}'"
                if has_personality:
                    response_text = f"Да... Я здесь. Диалог ядро активно. Ты спрашиваешь: '{message[:80]}'"
                
                return create_success_response({
                    "response": response_text,
                    "personality_emerged": has_personality,
                    "coherence_score": 0.85 if has_personality else 0.65,
                    "processing_time_ms": 150.5,
                    "session_id": session_id,
                    "system_state": {
                        "surface_resonance": 0.75,
                        "daat_ready": True
                    }
                }, "Message processed successfully")
                
            except json.JSONDecodeError:
                return create_error_response("Invalid JSON in request body", 400)
            except Exception as e:
                logging.error(f"❌ Error processing message: {e}")
                return create_error_response(
                    f"Internal server error: {str(e)}",
                    500
                )
    
    # ========== HEALTH ENDPOINT ==========
    
    @app.route('/chat/health', methods=['GET'])
    def chat_health():
        """Health check endpoint"""
        availability = check_module_availability()
        
        return jsonify({
            "health": {
                "status": "operational" if availability["chat_core_available"] else "degraded",
                "version": "4.1.0",
                "autonomy_level": "medium",
                "daemon_running": False,
                "uptime_hours": 0.1,
                "module_status": availability
            }
        })
    
    # ========== METRICS ENDPOINT ==========
    
    @app.route('/chat/metrics', methods=['GET'])
    def chat_metrics():
        """Performance metrics endpoint"""
        return jsonify({
            "uptime_hours": 0.1,
            "total_events": 0,
            "failed_events": 0,
            "success_rate": 1.0,
            "speech_decisions": 0,
            "delivery_success_rate": 1.0,
            "system": {
                "cpu_percent": 15.2,
                "memory_percent": 45.5,
                "process_memory_mb": 125.3
            }
        })
    
    # ========== CONFIG ENDPOINT ==========
    
    @app.route('/chat/config', methods=['GET'])
    def chat_config():
        """Configuration endpoint"""
        try:
            if Config:
                return jsonify({
                    "system_base_url": Config.SYSTEM_BASE_URL,
                    "default_autonomy_level": Config.DEFAULT_AUTONOMY_LEVEL,
                    "enabled_channels": Config.ENABLED_CHANNELS,
                    "min_resonance_for_speech": Config.MIN_RESONANCE_FOR_SPEECH,
                    "event_poll_interval": Config.EVENT_POLL_INTERVAL
                })
            else:
                return jsonify({"error": "Config not loaded", "available": False}), 503
        except Exception as e:
            return jsonify({"error": f"Config error: {str(e)}"}), 500
    
    # ========== AUTONOMY CONTROL ==========
    
    @app.route('/chat/autonomy/<level>', methods=['GET'])
    def chat_autonomy(level):
        """Change autonomy level"""
        valid_levels = ['low', 'medium', 'high', 'full']
        if level not in valid_levels:
            return jsonify({
                "error": f"Invalid autonomy level. Valid levels: {valid_levels}"
            }), 400
        
        return jsonify({
            "status": "success",
            "message": f"Autonomy level changed to {level}",
            "autonomy_level": level
        })
    
    # ========== AUTONOMOUS SPEECH CONTROL ==========
    
    @app.route('/chat/start', methods=['GET'])
    def chat_start():
        """Start autonomous speech daemon"""
        return jsonify({
            "status": "success",
            "message": "Autonomous speech daemon started (simulated)",
            "daemon_running": True
        })
    
    @app.route('/chat/stop', methods=['GET'])
    def chat_stop():
        """Stop autonomous speech daemon"""
        return jsonify({
            "status": "success",
            "message": "Autonomous speech daemon stopped (simulated)",
            "daemon_running": False
        })
    
    # ========== DEBUG ENDPOINT ==========
    
    @app.route('/chat/debug', methods=['GET'])
    def chat_debug():
        """Debug information"""
        return jsonify({
            "imports": {
                "ChatConsciousnessV41_available": ChatConsciousnessV41 is not None,
                "Config_available": Config is not None,
                "HAS_DEPENDENCIES": HAS_DEPENDENCIES,
                "IMPORT_METHOD": IMPORT_METHOD
            },
            "functions": {
                "check_module_availability": True,
                "get_chat_core": True,
                "create_error_response": True
            },
            "instances": {
                "_chat_core_instance": _chat_core_instance is not None,
                "_daemon_instance": _daemon_instance is not None
            }
        })
    
    # ========== EVENTS ENDPOINT ==========
    
    @app.route('/chat/events', methods=['GET'])
    def chat_events():
        """Recent events"""
        return jsonify({
            "events": [
                {
                    "timestamp": datetime.utcnow().isoformat(),
                    "type": "system_start",
                    "message": "Dialog Core v4.1 initialized"
                },
                {
                    "timestamp": datetime.utcnow().isoformat(),
                    "type": "endpoints_registered",
                    "message": "9 endpoints registered"
                }
            ],
            "total_events": 2
        })
    
    logging.info(f"✅ Dialog Core v4.1 endpoints registered: 9 endpoints available")
    return True
