"""
Dialog Core Module v4.1 - Production Ready
Основное диалоговое ядро ISKRA-4 с автономной речью и полной интеграцией

Экспортирует:
- ChatConsciousnessV41: Основной класс диалогового ядра
- AutonomousSpeechDaemonV41: Демон автономной речи
- SpeechEvent, SpeechDecision: Модели данных
- SpeechPriority, SpeechIntent: Перечисления
- setup_chat_endpoint: Функция регистрации HTTP эндпоинтов
- Config: Конфигурационный класс
"""

# ========== МЕТАДАННЫЕ МОДУЛЯ ==========

__version__ = "4.1.0"
__author__ = "ISKRA-4 Architect & Development Team"
__description__ = "Полноценное речевое ядро ISKRA-4 с автономной речью, " \
                  "интеграцией всех модулей и production-ready архитектурой"
__build_date__ = "2026-02-11"
__compatibility__ = "ISKRA-4 Cloud v2.0+"

# ========== ЗАЩИЩЕННЫЕ ИМПОРТЫ С FALLBACK ==========

HAS_CHAT_CONSCIOUSNESS = False
HAS_API = False
HAS_CONFIG = False

# Временные заглушки до импорта
ChatConsciousnessV41 = None
AutonomousSpeechDaemonV41 = None
SpeechEvent = None
SpeechDecision = None
SpeechPriority = None
SpeechIntent = None
RealEventBusIntegration = None
HealthMonitor = None
AsyncHTTPClient = None
setup_chat_endpoint = None
Config = None

try:
    # Основные классы из главного модуля
    from .chat_consciousness import (
        ChatConsciousnessV41 as CCV41,
        AutonomousSpeechDaemonV41 as ASDV41,
        SpeechEvent as SE,
        SpeechDecision as SD,
        SpeechPriority as SP,
        SpeechIntent as SI,
        RealEventBusIntegration as REBI,
        HealthMonitor as HM,
        AsyncHTTPClient as AHC
    )
    HAS_CHAT_CONSCIOUSNESS = True
    ChatConsciousnessV41 = CCV41
    AutonomousSpeechDaemonV41 = ASDV41
    SpeechEvent = SE
    SpeechDecision = SD
    SpeechPriority = SP
    SpeechIntent = SI
    RealEventBusIntegration = REBI
    HealthMonitor = HM
    AsyncHTTPClient = AHC
    print("✅ ChatConsciousness импортирован")
except ImportError as e:
    print(f"⚠️ ChatConsciousness import failed: {e}")

try:
    # HTTP слой (Flask эндпоинты)
    from .api import setup_chat_endpoint as sce
    HAS_API = True
    setup_chat_endpoint = sce
    print("✅ API импортирован")
except ImportError as e:
    print(f"⚠️ API import failed: {e}")

try:
    # Конфигурация
    from .config import Config as Cfg
    HAS_CONFIG = True
    Config = Cfg
    print("✅ Config импортирован")
except ImportError as e:
    print(f"⚠️ Config import failed: {e}")

# ========== FALLBACK ДЛЯ setup_chat_endpoint ==========

if not HAS_API and setup_chat_endpoint is None:
    print("⚠️ Creating fallback setup_chat_endpoint")
    
    def setup_chat_endpoint_fallback(app):
        """Fallback функция если Dialog Core не загружен"""
        from flask import jsonify
        from datetime import datetime
        
        @app.route('/chat', methods=['GET'])
        def chat_fallback():
            return jsonify({
                "system": "ISKRA-4 Dialog Core (fallback mode)",
                "status": "unavailable",
                "message": "Dialog Core module not loaded",
                "reason": "Module dependencies missing or import error",
                "timestamp": datetime.utcnow().isoformat()
            }), 503
        
        return app
    
    setup_chat_endpoint = setup_chat_endpoint_fallback

# ========== ПРОВЕРКА ЦЕЛОСТНОСТИ МОДУЛЯ ==========

def check_integrity():
    """Проверка целостности модуля Dialog Core"""
    
    integrity_report = {
        "module": "dialog_core",
        "version": __version__,
        "timestamp": __import__('datetime').datetime.now().isoformat(),
        "components": {
            "chat_consciousness": HAS_CHAT_CONSCIOUSNESS,
            "api": HAS_API,
            "config": HAS_CONFIG
        },
        "status": "operational" if HAS_CHAT_CONSCIOUSNESS and HAS_CONFIG else "degraded",
        "message": None
    }
    
    # Определяем сообщение о состоянии
    if integrity_report["status"] == "operational":
        integrity_report["message"] = "✅ Dialog Core v4.1 полностью функционален"
    else:
        missing = [k for k, v in integrity_report["components"].items() if not v]
        integrity_report["message"] = f"⚠️ Dialog Core работает в усеченном режиме. Отсутствуют: {missing}"
    
    return integrity_report

# ========== ПУБЛИЧНЫЙ ИНТЕРФЕЙС ==========

# Основные классы для импорта
__all__ = [
    # Основные классы
    "ChatConsciousnessV41",
    "AutonomousSpeechDaemonV41",
    
    # Модели данных
    "SpeechEvent",
    "SpeechDecision",
    "SpeechPriority", 
    "SpeechIntent",
    
    # Вспомогательные классы
    "RealEventBusIntegration",
    "HealthMonitor",
    "AsyncHTTPClient",
    
    # Функции
    "setup_chat_endpoint",
    
    # Конфигурация
    "Config",
    
    # Утилиты
    "check_integrity",
    "get_version_info"
]

# ========== ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ ==========

def get_version_info():
    """Получение информации о версии модуля"""
    return {
        "module": "dialog_core",
        "version": __version__,
        "description": __description__,
        "author": __author__,
        "build_date": __build_date__,
        "compatibility": __compatibility__,
        "integrity": check_integrity()
    }

def initialize_module(autostart=True):
    """Инициализация модуля Dialog Core"""
    
    print("=" * 60)
    print("🚀 ИНИЦИАЛИЗАЦИЯ DIALOG CORE v4.1")
    print("=" * 60)
    
    # Проверка целостности
    integrity = check_integrity()
    print(f"Статус: {integrity['message']}")
    
    # Вывод информации о компонентах
    for component, available in integrity["components"].items():
        status = "✅" if available else "❌"
        print(f"  {status} {component}")
    
    # Если модуль не полностью функционален
    if integrity["status"] != "operational":
        print(f"\n⚠️ ВНИМАНИЕ: Модуль работает в усеченном режиме")
        print("   Некоторые функции могут быть недоступны")
    
    print("=" * 60)
    
    # Создание экземпляра основного класса если требуется
    if autostart and HAS_CHAT_CONSCIOUSNESS and ChatConsciousnessV41:
        try:
            instance = ChatConsciousnessV41()
            print(f"✅ ChatConsciousnessV41 инициализирован")
            return instance
        except Exception as e:
            print(f"❌ Ошибка инициализации: {e}")
            return None
    
    return None

# ========== АВТО-ИНИЦИАЛИЗАЦИЯ (опционально) ==========

# При импорте модуля можно автоматически проверить его целостность
_AUTO_CHECK_ON_IMPORT = False  # Меняй на True для автоматической проверки

if _AUTO_CHECK_ON_IMPORT:
    print("🔍 Dialog Core: автоматическая проверка целостности...")
    check_result = check_integrity()
    if check_result["status"] == "operational":
        print("✅ Dialog Core v4.1 готов к работе")
    else:
        print(f"⚠️ {check_result['message']}")

# ========== ДОПОЛНИТЕЛЬНАЯ ИНФОРМАЦИЯ ==========

if __name__ == "__main__":
    # Запуск при прямом выполнении файла
    print("=" * 60)
    print("DIALOG CORE v4.1 - МОДУЛЬНЫЙ ТЕСТ")
    print("=" * 60)
    
    info = get_version_info()
    print(f"Версия: {info['version']}")
    print(f"Описание: {info['description']}")
    print(f"Совместимость: {info['compatibility']}")
    
    print("\n" + "=" * 60)
    print("Экспортируемые компоненты:")
    print("-" * 30)
    for item in __all__:
        print(f"  • {item}")
    
    print("\n" + "=" * 60)
    print("Тест целостности:")
    integrity = check_integrity()
    for component, available in integrity["components"].items():
        status = "ДОСТУПЕН" if available else "ОТСУТСТВУЕТ"
        print(f"  {component}: {status}")
    
    print(f"\nИтоговый статус: {integrity['status'].upper()}")
    print("=" * 60)
