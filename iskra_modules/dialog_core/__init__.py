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

# ========== ИМПОРТ ОСНОВНЫХ КОМПОНЕНТОВ ==========

try:
    # Основные классы из главного модуля
    from .chat_consciousness import (
        ChatConsciousnessV41,
        AutonomousSpeechDaemonV41,
        SpeechEvent,
        SpeechDecision,
        SpeechPriority,
        SpeechIntent,
        RealEventBusIntegration,
        HealthMonitor,
        AsyncHTTPClient
    )
    HAS_CHAT_CONSCIOUSNESS = True
except ImportError as e:
    print(f"⚠️ ChatConsciousness import failed: {e}")
    HAS_CHAT_CONSCIOUSNESS = False
    # Создаем заглушки для предотвращения падений
    ChatConsciousnessV41 = None
    AutonomousSpeechDaemonV41 = None
    SpeechEvent = None
    SpeechDecision = None
    SpeechPriority = None
    SpeechIntent = None

try:
    # HTTP слой (Flask эндпоинты)
    from .api import setup_chat_endpoint
    HAS_API = True
except ImportError as e:
    print(f"⚠️ API import failed: {e}")
    HAS_API = False
    setup_chat_endpoint = None

try:
    # Конфигурация
    from .config import Config
    HAS_CONFIG = True
except ImportError as e:
    print(f"⚠️ Config import failed: {e}")
    HAS_CONFIG = False
    Config = None

# ========== МЕТАДАННЫЕ МОДУЛЯ ==========

__version__ = "4.1.0"
__author__ = "ISKRA-4 Architect & Development Team"
__description__ = "Полноценное речевое ядро ISKRA-4 с автономной речью, " \
                  "интеграцией всех модулей и production-ready архитектурой"
__build_date__ = "2026-02-11"
__compatibility__ = "ISKRA-4 Cloud v2.0+"

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
        "status": "operational" if all([HAS_CHAT_CONSCIOUSNESS, HAS_API, HAS_CONFIG]) else "degraded",
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
