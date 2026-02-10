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

from flask import request, jsonify, Response

try:
    from .chat_consciousness import ChatConsciousnessV41, AutonomousSpeechDaemonV41
    from .config import Config
    HAS_DEPENDENCIES = True
except ImportError as e:
    logging.error(f"❌ Failed to import Dialog Core dependencies: {e}")
    HAS_DEPENDENCIES = False

# ========== GLOBAL INSTANCES ==========

# Global instance of Dialog Core (lazy-loaded)
_chat_core_instance = None
_daemon_instance = None

def get_chat_core() -> Optional[ChatConsciousnessV41]:
    """Get or create global ChatConsciousnessV41 instance (singleton pattern)"""
    global _chat_core_instance
    
    if not HAS_DEPENDENCIES:
        return None
    
    if _chat_core_instance is None:
        try:
            _chat_core_instance = ChatConsciousnessV41()
            logging.info("✅ ChatConsciousnessV41 instance created")
        except Exception as e:
            logging.error(f"❌ Failed to create ChatConsciousnessV41: {e}")
            _chat_core_instance = None
    
    return _chat_core_instance

def get_daemon() -> Optional[AutonomousSpeechDaemonV41]:
    """Get or create global AutonomousSpeechDaemonV41 instance"""
    global _daemon_instance
    
    if not HAS_DEPENDENCIES:
        return None
    
    chat_core = get_chat_core()
    if chat_core is None:
        return None
    
    if _daemon_instance is None:
        try:
            _daemon_instance = AutonomousSpeechDaemonV41(chat_core)
            logging.info("✅ AutonomousSpeechDaemonV41 instance created")
        except Exception as e:
            logging.error(f"❌ Failed to create AutonomousSpeechDaemonV41: {e}")
            _daemon_instance = None
    
    return _daemon_instance

# ========== HELPER FUNCTIONS ==========

def create_error_response(message: str, status_code: int = 500, details: Any = None) -> Response:
    """Create standardized error response"""
    response = {
        "error": True,
        "message": message,
        "timestamp": datetime.utcnow().isoformat(),
        "module": "dialog_core",
        "version": "4.1.0"
    }
    
    if details:
        response["details"] = str(details)
    
    return jsonify(response), status_code

def create_success_response(data: Dict, message: str = "Success") -> Response:
    """Create standardized success response"""
    response = {
        "success": True,
        "message": message,
        "timestamp": datetime.utcnow().isoformat(),
        "module": "dialog_core",
        "version": "4.1.0",
        "data": data
    }
    
    return jsonify(response)

def check_module_availability() -> Dict[str, Any]:
    """Check if Dialog Core module is fully available"""
    chat_core = get_chat_core()
    daemon = get_daemon()
    
    return {
        "chat_core_available": chat_core is not None,
        "daemon_available": daemon is not None,
        "dependencies_available": HAS_DEPENDENCIES,
        "config_available": HAS_DEPENDENCIES,
        "overall_status": "operational" if (chat_core is not None and HAS_DEPENDENCIES) else "degraded"
    }

# ========== FLASK ENDPOINTS ==========

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
                "health": get_chat_core().get_health_status() if availability["chat_core_available"] else {},
                "endpoints": {
                    "chat": "POST /chat - Send message",
                    "health": "GET /chat/health - Health status",
                    "metrics": "GET /chat/metrics - Performance metrics",
                    "config": "GET /chat/config - Configuration",
                    "autonomy": "GET /chat/autonomy/<level> - Change autonomy level",
                    "start": "GET /chat/start - Start autonomous daemon",
                    "stop": "GET /chat/stop - Stop autonomous daemon",
                    "debug": "GET /chat/debug - Debug info",
                    "events": "GET /chat/events - Recent events"
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
                session_id = data.get('session_id')
                user_id = data.get('user_id', 'anonymous')
                
                # Validate message
                if not message or not isinstance(message, str):
                    return create_error_response(
                        "Message must be a non-empty string",
                        400
                    )
                
                if len(message) > 1000:
                    return create_error_response(
                        "Message too long (max 1000 characters)",
                        400
                    )
                
                # Process message through Dialog Core
                chat_core = get_chat_core()
                result = chat_core.process_message(
                    message=message,
                    session_id=session_id,
                    user_id=user_id
                )
                
                # Log successful processing
                logging.info(f"💬 Message processed: '{message[:50]}...' -> "
                           f"coherence: {result.get('coherence_score', 0)}, "
                           f"personality: {result.get('personality_emerged', False)}")
                
                return create_success_response(
                    data=result,
                    message="Message processed successfully"
                )
                
            except json.JSONDecodeError:
                return create_error_response("Invalid JSON in request body", 400)
            except Exception as e:
                logging.error(f"❌ Error processing message: {e}\n{traceback.format_exc()}")
                return create_error_response(
                    f"Internal server error: {str(e)}",
                    500,
                    {"traceback": traceback.format_exc()} if app.debug else None
                )
    
    # ========== HEALTH CHECK ==========
    
    @app.route('/chat/health', methods=['GET'])
    def health_check():
        """Health check endpoint for monitoring"""
        
        availability = check_module_availability()
        
        if availability["chat_core_available"]:
            health_data = get_chat_core().get_health_status()
        else:
            health_data = {
                "status": "unavailable",
                "message": "Dialog Core not initialized",
                "checks": {}
            }
        
        response = {
            "module": "dialog_core",
            "version": "4.1.0",
            "timestamp": datetime.utcnow().isoformat(),
            "availability": availability,
            "health": health_data,
            "system": {
                "python_version": f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
                "platform": sys.platform
            }
        }
        
        # Determine HTTP status code based on health
        status_code = 200 if availability["overall_status"] == "operational" else 503
        
        return jsonify(response), status_code
    
    # ========== METRICS ENDPOINT ==========
    
    @app.route('/chat/metrics', methods=['GET'])
    def get_metrics():
        """Get performance metrics from Dialog Core"""
        
        availability = check_module_availability()
        
        if not availability["chat_core_available"]:
            return create_error_response(
                "Dialog Core not available",
                503,
                {"availability": availability}
            )
        
        try:
            chat_core = get_chat_core()
            
            # Get metrics from health monitor
            if hasattr(chat_core, 'health_monitor') and chat_core.health_monitor:
                metrics = chat_core.health_monitor.get_metrics()
            else:
                metrics = {"error": "Health monitor not available"}
            
            # Add system metrics
            import psutil
            import sys
            
            system_metrics = {
                "system": {
                    "cpu_percent": psutil.cpu_percent(),
                    "memory_percent": psutil.virtual_memory().percent,
                    "process_memory_mb": psutil.Process().memory_info().rss / 1024 / 1024,
                    "thread_count": psutil.Process().num_threads()
                },
                "module": {
                    "chat_core_initialized": _chat_core_instance is not None,
                    "daemon_initialized": _daemon_instance is not None,
                    "daemon_running": _daemon_instance.is_running() if _daemon_instance else False
                }
            }
            
            metrics.update(system_metrics)
            
            return create_success_response(
                data=metrics,
                message="Metrics retrieved successfully"
            )
            
        except Exception as e:
            logging.error(f"❌ Error getting metrics: {e}")
            return create_error_response(
                f"Failed to get metrics: {str(e)}",
                500
            )
    
    # ========== CONFIGURATION ENDPOINT ==========
    
    @app.route('/chat/config', methods=['GET'])
    def get_config():
        """Get current configuration of Dialog Core"""
        
        try:
            # Import here to avoid circular imports
            from .config import Config
            
            config_data = {
                "autonomy": {
                    "current_level": get_chat_core().current_autonomy if get_chat_core() else "unknown",
                    "available_levels": Config.AUTONOMY_LEVELS if HAS_DEPENDENCIES else {},
                    "default_level": Config.DEFAULT_AUTONOMY_LEVEL if HAS_DEPENDENCIES else "medium"
                },
                "channels": {
                    "enabled": Config.ENABLED_CHANNELS if HAS_DEPENDENCIES else [],
                    "default": "console,internal_log"
                },
                "system": {
                    "base_url": Config.SYSTEM_BASE_URL if HAS_DEPENDENCIES else "",
                    "event_poll_interval": Config.EVENT_POLL_INTERVAL if HAS_DEPENDENCIES else 5.0
                },
                "limits": Config.MESSAGE_LIMITS if HAS_DEPENDENCIES else {},
                "resonance": {
                    "min_for_speech": Config.MIN_RESONANCE_FOR_SPEECH if HAS_DEPENDENCIES else 0.3,
                    "critical_threshold": Config.RESONANCE_CRITICAL_THRESHOLD if HAS_DEPENDENCIES else 0.2
                },
                "integration": {
                    "use_real_event_bus": Config.USE_REAL_EVENT_BUS if HAS_DEPENDENCIES else True,
                    "use_real_sephirotic": Config.USE_REAL_SEPHIROTIC if HAS_DEPENDENCIES else True,
                    "use_real_symbiosis": Config.USE_REAL_SYMBIOSIS if HAS_DEPENDENCIES else True
                }
            }
            
            return create_success_response(
                data=config_data,
                message="Configuration retrieved successfully"
            )
            
        except Exception as e:
            logging.error(f"❌ Error getting config: {e}")
            return create_error_response(
                f"Failed to get configuration: {str(e)}",
                500
            )
    
    # ========== AUTONOMY CONTROL ==========
    
    @app.route('/chat/autonomy/<level>', methods=['GET'])
    def set_autonomy(level: str):
        """Change autonomy level of Dialog Core"""
        
        if not check_module_availability()["chat_core_available"]:
            return create_error_response(
                "Dialog Core not available",
                503
            )
        
        try:
            chat_core = get_chat_core()
            
            # Validate level
            valid_levels = list(chat_core.autonomy_levels.keys()) if hasattr(chat_core, 'autonomy_levels') else []
            
            if level not in valid_levels:
                return create_error_response(
                    f"Invalid autonomy level: {level}",
                    400,
                    {"valid_levels": valid_levels}
                )
            
            # Change autonomy level
            old_level = chat_core.current_autonomy
            chat_core.current_autonomy = level
            
            logging.info(f"🔧 Autonomy level changed: {old_level} → {level}")
            
            return create_success_response(
                data={
                    "old_level": old_level,
                    "new_level": level,
                    "autonomy_value": chat_core.autonomy_levels[level],
                    "timestamp": datetime.utcnow().isoformat()
                },
                message=f"Autonomy level changed to {level}"
            )
            
        except Exception as e:
            logging.error(f"❌ Error changing autonomy level: {e}")
            return create_error_response(
                f"Failed to change autonomy level: {str(e)}",
                500
            )
    
    # ========== DAEMON CONTROL ==========
    
    @app.route('/chat/start', methods=['GET'])
    def start_system():
        """Start autonomous speech daemon"""
        
        availability = check_module_availability()
        
        if not availability["chat_core_available"]:
            return create_error_response(
                "Dialog Core not available",
                503
            )
        
        try:
            daemon = get_daemon()
            
            if not daemon:
                return create_error_response(
                    "Autonomous speech daemon not available",
                    503
                )
            
            # Start daemon if not already running
            if hasattr(daemon, 'is_running') and daemon.is_running():
                return create_success_response(
                    data={
                        "daemon_status": "already_running",
                        "since": daemon.start_time.isoformat() if hasattr(daemon, 'start_time') else "unknown"
                    },
                    message="Daemon is already running"
                )
            else:
                if hasattr(daemon, 'start'):
                    daemon.start()
                    message = "Daemon started successfully"
                elif hasattr(daemon, 'run'):
                    # Run in background thread
                    import threading
                    thread = threading.Thread(target=daemon.run, daemon=True)
                    thread.start()
                    message = "Daemon started in background thread"
                else:
                    return create_error_response(
                        "Daemon has no start method",
                        500
                    )
                
                logging.info("🚀 Autonomous speech daemon started")
                
                return create_success_response(
                    data={
                        "daemon_status": "started",
                        "timestamp": datetime.utcnow().isoformat()
                    },
                    message=message
                )
                
        except Exception as e:
            logging.error(f"❌ Error starting daemon: {e}")
            return create_error_response(
                f"Failed to start daemon: {str(e)}",
                500
            )
    
    @app.route('/chat/stop', methods=['GET'])
    def stop_system():
        """Stop autonomous speech daemon"""
        
        if not check_module_availability()["chat_core_available"]:
            return create_error_response(
                "Dialog Core not available",
                503
            )
        
        try:
            daemon = get_daemon()
            
            if not daemon:
                return create_success_response(
                    data={"daemon_status": "not_initialized"},
                    message="Daemon was not initialized"
                )
            
            # Stop daemon if running
            if hasattr(daemon, 'is_running') and daemon.is_running():
                if hasattr(daemon, 'stop'):
                    daemon.stop()
                    message = "Daemon stopped successfully"
                elif hasattr(daemon, 'shutdown'):
                    daemon.shutdown()
                    message = "Daemon shutdown initiated"
                else:
                    return create_error_response(
                        "Daemon has no stop method",
                        500
                    )
                
                logging.info("🛑 Autonomous speech daemon stopped")
                
                return create_success_response(
                    data={
                        "daemon_status": "stopped",
                        "timestamp": datetime.utcnow().isoformat()
                    },
                    message=message
                )
            else:
                return create_success_response(
                    data={"daemon_status": "not_running"},
                    message="Daemon was not running"
                )
                
        except Exception as e:
            logging.error(f"❌ Error stopping daemon: {e}")
            return create_error_response(
                f"Failed to stop daemon: {str(e)}",
                500
            )
    
    # ========== DEBUG ENDPOINTS ==========
    
    @app.route('/chat/debug', methods=['GET'])
    def debug_info():
        """Get debug information about Dialog Core"""
        
        availability = check_module_availability()
        
        # Import sys for system info
        import sys
        
        debug_data = {
            "availability": availability,
            "instances": {
                "chat_core": _chat_core_instance is not None,
                "daemon": _daemon_instance is not None
            },
            "system": {
                "python_version": f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
                "platform": sys.platform,
                "flask_debug": app.debug
            },
            "imports": {
                "chat_consciousness": "ChatConsciousnessV41" in globals(),
                "config": "Config" in globals()
            },
            "environment": {
                "ISKRA_BASE_URL": Config.SYSTEM_BASE_URL if HAS_DEPENDENCIES else "NOT_LOADED",
                "DEFAULT_AUTONOMY_LEVEL": Config.DEFAULT_AUTONOMY_LEVEL if HAS_DEPENDENCIES else "NOT_LOADED"
            }
        }
        
        # Add chat core debug info if available
        if availability["chat_core_available"] and get_chat_core():
            chat_core = get_chat_core()
            debug_data["chat_core_details"] = {
                "autonomy_level": chat_core.current_autonomy if hasattr(chat_core, 'current_autonomy') else "unknown",
                "health_status": chat_core.get_health_status() if hasattr(chat_core, 'get_health_status') else {},
                "modules_loaded": hasattr(chat_core, 'modules_loaded') and chat_core.modules_loaded
            }
        
        return create_success_response(
            data=debug_data,
            message="Debug information retrieved"
        )
    
    @app.route('/chat/events', methods=['GET'])
    def get_recent_events():
        """Get recent events from Dialog Core"""
        
        if not check_module_availability()["chat_core_available"]:
            return create_error_response(
                "Dialog Core not available",
                503
            )
        
        try:
            chat_core = get_chat_core()
            
            # Try to get recent events from health monitor
            if hasattr(chat_core, 'health_monitor') and chat_core.health_monitor:
                if hasattr(chat_core.health_monitor, 'get_recent_events'):
                    events = chat_core.health_monitor.get_recent_events()
                else:
                    events = []
            else:
                events = []
            
            # Add some default events if none available
            if not events:
                events = [
                    {
                        "type": "system",
                        "message": "No recent events available",
                        "timestamp": datetime.utcnow().isoformat()
                    }
                ]
            
            return create_success_response(
                data={
                    "events": events[:20],  # Limit to 20 most recent
                    "total_count": len(events)
                },
                message="Recent events retrieved"
            )
            
        except Exception as e:
            logging.error(f"❌ Error getting events: {e}")
            return create_error_response(
                f"Failed to get events: {str(e)}",
                500
            )
    
    # ========== REGISTRATION COMPLETE ==========
    
    logging.info(f"✅ Dialog Core v4.1 endpoints registered: 9 endpoints available")
    
    return app

# ========== FALLBACK FOR MISSING DEPENDENCIES ==========

if not HAS_DEPENDENCIES:
    logging.warning("⚠️ Dialog Core dependencies not available, creating fallback setup_chat_endpoint")
    
    def setup_chat_endpoint_fallback(app):
        """Fallback function when Dialog Core dependencies are missing"""
        
        @app.route('/chat', methods=['GET'])
        def chat_fallback():
            return jsonify({
                "error": "Dialog Core not available",
                "message": "Required dependencies are missing",
                "instructions": "Check that chat_consciousness.py and config.py exist in dialog_core module",
                "status": 503,
                "timestamp": datetime.utcnow().isoformat()
            }), 503
        
        return app
    
    # Override the main function with fallback
    setup_chat_endpoint = setup_chat_endpoint_fallback

# ========== IMPORT SYS FOR SYSTEM INFO ==========

import sys

# ========== MODULE EXPORTS ==========

__all__ = ['setup_chat_endpoint']
