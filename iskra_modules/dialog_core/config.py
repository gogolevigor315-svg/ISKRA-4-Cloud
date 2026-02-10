"""
Dialog Core Configuration v4.1
Конфигурация через .env файл для интеграции с ISKRA-4 Cloud
"""

import os
import json
import logging
from dotenv import load_dotenv

# Загрузка .env (важно: ищем в корне проекта, рядом с iskra_full.py)
load_dotenv()

class Config:
    """Централизованная конфигурация Dialog Core v4.1"""
    
    # ========== БАЗОВЫЕ НАСТРОЙКИ СИСТЕМЫ ==========
    
    # URL системы ISKRA-4 Cloud (для интеграции с существующими модулями)
    SYSTEM_BASE_URL = os.getenv("ISKRA_BASE_URL", "https://iskra-4-cloud.onrender.com")
    
    # Токен Telegram бота для уведомлений оператора
    TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
    
    # ID чатов Telegram в формате JSON: {"operator": "123456789", "admin": "987654321"}
    TELEGRAM_CHAT_IDS = json.loads(os.getenv("TELEGRAM_CHAT_IDS", '{"operator": "OPERATOR_CHAT_ID"}'))
    
    # ========== ВРЕМЕННЫЕ НАСТРОЙКИ ==========
    
    # Интервал опроса событий из шины (секунды)
    EVENT_POLL_INTERVAL = float(os.getenv("EVENT_POLL_INTERVAL", "5.0"))
    
    # Время жизни кэша состояния (секунды)
    STATE_CACHE_TTL = int(os.getenv("STATE_CACHE_TTL", "30"))
    
    # Таймаут HTTP запросов к модулям ISKRA-4
    MODULE_REQUEST_TIMEOUT = float(os.getenv("MODULE_REQUEST_TIMEOUT", "3.0"))
    
    # ========== ЛИМИТЫ СООБЩЕНИЙ ==========
    
    MESSAGE_LIMITS = {
        "operator": {
            "hourly": int(os.getenv("OPERATOR_HOURLY_LIMIT", "100")),
            "daily": int(os.getenv("OPERATOR_DAILY_LIMIT", "500"))
        },
        "user": {
            "hourly": int(os.getenv("USER_HOURLY_LIMIT", "20")),
            "daily": int(os.getenv("USER_DAILY_LIMIT", "100"))
        },
        "autonomous": {
            "hourly": int(os.getenv("AUTONOMOUS_HOURLY_LIMIT", "10")),
            "daily": int(os.getenv("AUTONOMOUS_DAILY_LIMIT", "50"))
        }
    }
    
    # ========== КАНАЛЫ ДОСТАВКИ ==========
    
    # Доступные каналы: console, internal_log, telegram, system_bus
    ENABLED_CHANNELS = os.getenv("ENABLED_CHANNELS", "console,internal_log").split(",")
    
    # ========== НАСТРОЙКИ РЕЗОНАНСА ==========
    
    # Минимальный резонанс для инициации автономной речи
    MIN_RESONANCE_FOR_SPEECH = float(os.getenv("MIN_RESONANCE_FOR_SPEECH", "0.3"))
    
    # Критический порог резонанса для экстренных сообщений
    RESONANCE_CRITICAL_THRESHOLD = float(os.getenv("RESONANCE_CRITICAL_THRESHOLD", "0.2"))
    
    # Требуемый резонанс для глубокой личности
    RESONANCE_DEEP_PERSONALITY = float(os.getenv("RESONANCE_DEEP_PERSONALITY", "0.5"))
    
    # ========== ПОЛИТИКА АВТОНОМИИ ==========
    
    # Уровень автономии по умолчанию: disabled, low, medium, high, full
    DEFAULT_AUTONOMY_LEVEL = os.getenv("DEFAULT_AUTONOMY_LEVEL", "medium")
    
    # Автономные уровни и их числовые значения
    AUTONOMY_LEVELS = {
        "disabled": 0.0,
        "low": 0.3,
        "medium": 0.6,
        "high": 0.8,
        "full": 1.0
    }
    
    # ========== ИНТЕГРАЦИЯ С МОДУЛЯМИ ISKRA-4 ==========
    
    # Использовать реальную шину событий или эмуляцию
    USE_REAL_EVENT_BUS = os.getenv("USE_REAL_EVENT_BUS", "true").lower() == "true"
    
    # Использовать реальный Sephirotic Engine
    USE_REAL_SEPHIROTIC = os.getenv("USE_REAL_SEPHIROTIC", "true").lower() == "true"
    
    # Использовать реальный Symbiosis Core
    USE_REAL_SYMBIOSIS = os.getenv("USE_REAL_SYMBIOSIS", "true").lower() == "true"
    
    # ========== НАСТРОЙКИ ЛОГИРОВАНИЯ ==========
    
    LOG_LEVEL = os.getenv("DIALOG_LOG_LEVEL", "INFO")
    LOG_TO_FILE = os.getenv("LOG_TO_FILE", "false").lower() == "true"
    LOG_FILE_PATH = os.getenv("LOG_FILE_PATH", "logs/dialog_core.log")
    
    # ========== НАСТРОЙКИ ТЕСТИРОВАНИЯ ==========
    
    # Режим тестирования (использует заглушки вместо реальных модулей)
    TEST_MODE = os.getenv("DIALOG_TEST_MODE", "false").lower() == "true"
    
    # Задержка между автономными событиями в тестовом режиме (секунды)
    TEST_AUTONOMY_INTERVAL = float(os.getenv("TEST_AUTONOMY_INTERVAL", "10.0"))
    
    @classmethod
    def validate(cls):
        """Валидация конфигурации при запуске"""
        
        # Проверка URL системы
        if not cls.SYSTEM_BASE_URL.startswith("http"):
            raise ValueError("SYSTEM_BASE_URL должен быть валидным URL (начинаться с http/https)")
        
        # Проверка Telegram конфигурации
        if "telegram" in cls.ENABLED_CHANNELS and not cls.TELEGRAM_BOT_TOKEN:
            logging.warning("Telegram канал включен, но токен не установлен (TELEGRAM_BOT_TOKEN)")
        
        # Проверка уровней автономии
        if cls.DEFAULT_AUTONOMY_LEVEL not in cls.AUTONOMY_LEVELS:
            raise ValueError(f"Неизвестный уровень автономии: {cls.DEFAULT_AUTONOMY_LEVEL}. "
                           f"Доступные: {list(cls.AUTONOMY_LEVELS.keys())}")
        
        # Валидация лимитов
        for role, limits in cls.MESSAGE_LIMITS.items():
            if limits["hourly"] > limits["daily"]:
                logging.warning(f"Лимит часовой > дневного для {role}. Это может вызвать неожиданное поведение.")
        
        # Проверка каналов доставки
        valid_channels = ["console", "internal_log", "telegram", "system_bus", "test"]
        for channel in cls.ENABLED_CHANNELS:
            if channel not in valid_channels:
                logging.warning(f"Неизвестный канал доставки: {channel}. Доступные: {valid_channels}")
        
        logging.info(f"✅ Dialog Core v4.1 конфигурация загружена и проверена")
        logging.info(f"   Автономия: {cls.DEFAULT_AUTONOMY_LEVEL} ({cls.AUTONOMY_LEVELS[cls.DEFAULT_AUTONOMY_LEVEL]})")
        logging.info(f"   Каналы: {cls.ENABLED_CHANNELS}")
        logging.info(f"   База URL: {cls.SYSTEM_BASE_URL}")
        logging.info(f"   Лимиты: оператор={cls.MESSAGE_LIMITS['operator']['hourly']}/ч")
        
        return True

# Автоматическая валидация при импорте модуля
try:
    Config.validate()
except Exception as e:
    logging.error(f"❌ Ошибка валидации конфигурации Dialog Core: {e}")
    # Не падаем, чтобы система могла запуститься в fallback режиме

# Экспорт конфигурации
__all__ = ['Config']
