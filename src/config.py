import configparser
import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent

load_dotenv(ROOT / ".env", override=False)


class MissingSettingError(RuntimeError):
    """Обязательная переменная окружения не задана."""


def required_env(name: str) -> str:
    """Значение обязательной переменной окружения."""
    value = os.getenv(name, "").strip()

    if not value:
        raise MissingSettingError(
            f"Переменная окружения {name} не задана. "
            f"Скопируйте .env.example в .env и заполните значения "
            f"(в CI/CD — GitHub Secrets)."
        )

    return value


@dataclass(frozen=True)
class CassandraSettings:
    """Параметры подключения к Cassandra."""

    hosts: list[str]
    port: int
    keyspace: str
    username: str = field(repr=False)
    password: str = field(repr=False)
    connect_retries: int = 6
    retry_delay_seconds: float = 20.0
    request_timeout_seconds: float = 20.0


def cassandra_settings() -> CassandraSettings:
    """Настройки подключения к БД из переменных окружения."""
    hosts = [
        host.strip()
        for host in required_env("CASSANDRA_HOSTS").split(",")
        if host.strip()
    ]

    return CassandraSettings(
        hosts=hosts,
        port=int(required_env("CASSANDRA_PORT")),
        keyspace=required_env("CASSANDRA_KEYSPACE"),
        username=required_env("CASSANDRA_USER"),
        password=required_env("CASSANDRA_PASSWORD"),
        connect_retries=int(os.getenv("CASSANDRA_CONNECT_RETRIES", "6")),
        retry_delay_seconds=float(os.getenv("CASSANDRA_RETRY_DELAY", "20.0")),
        request_timeout_seconds=float(os.getenv("CASSANDRA_REQUEST_TIMEOUT", "20.0")),
    )


def config_path() -> Path:
    """Путь к config.ini."""
    return Path(os.getenv("CONFIG_PATH", ROOT / "config.ini"))


def load_config(path: Path | None = None) -> configparser.ConfigParser:
    """Загрузка конфига."""
    path = Path(path) if path else config_path()

    if not path.exists():
        raise FileNotFoundError(f"config.ini не найден: {path}")

    cfg = configparser.ConfigParser()
    cfg.read(path, encoding="utf-8")

    return cfg


def resolve(rel_path: Path) -> Path:
    """Относительный путь из конфига -> абсолютный от корня репозитория."""
    p = Path(rel_path)

    return p if p.is_absolute() else ROOT / p


def checkpoint_path(cfg: configparser.ConfigParser | None = None) -> Path:
    """
    Путь к чекпоинту обученного классификатора (.pth).

    CHECKPOINT_PATH из окружения имеет приоритет над config.ini.
    """
    env = os.getenv("CHECKPOINT_PATH")

    if env:
        path = Path(env)
    else:
        cfg = cfg or load_config()

        raw_path = cfg["MODEL"].get("checkpoint_path", "").strip()

        if not raw_path:
            raise ValueError("MODEL.checkpoint_path не указан в config.ini")

        path = resolve(raw_path)

    if not path.exists():
        raise FileNotFoundError(f"Чекпоинт модели не найден по пути: {path}")

    if not path.is_file():
        raise ValueError(f"checkpoint_path должен указывать на файл чекпоинта: {path}")

    return path


def images_path(cfg: configparser.ConfigParser | None = None) -> Path:
    """Путь к изображениям"""
    env = os.getenv("IMAGES_PATH")

    if env:
        return Path(env)

    cfg = cfg or load_config()

    return resolve(cfg["DATA"]["images_path"])


def target_path(cfg: configparser.ConfigParser | None = None) -> Path:
    """Путь к CSV с таргетами."""
    env = os.getenv("TARGET_PATH")

    if env:
        return Path(env)

    cfg = cfg or load_config()

    return resolve(cfg["DATA"]["csv_path"])


def split_path(cfg: configparser.ConfigParser | None = None) -> Path:
    """Путь к CSV с разбиением на train/val/test."""
    env = os.getenv("SPLIT_PATH")

    if env:
        return Path(env)

    cfg = cfg or load_config()

    return resolve(cfg["DATA"]["split_path"])
