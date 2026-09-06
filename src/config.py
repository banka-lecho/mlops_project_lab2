import configparser
import os
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


@dataclass(frozen=True)
class CassandraSettings:
    # TODO:: Здесь потом надо заменить username, password на переменные окружения, а не на захардкоженные хуйни
    hosts: list[str] = field(
        default_factory=lambda: os.getenv("CASSANDRA_HOSTS", "127.0.0.1").split(",")
    )
    port: int = int(os.getenv("CASSANDRA_PORT", 9042))
    keyspace: str = os.getenv("CASSANDRA_KEYSPACE", "dog_emotion_keyspace")
    username: str = os.getenv("CASSANDRA_USER", "cassandra")
    password: str = os.getenv("CASSANDRA_PASSWORD", "cassandra")
    connect_retries: int = int(os.getenv("CASSANDRA_CONNECT_RETRIES", 6))
    retry_delay_seconds: float = float(os.getenv("CASSANDRA_RETRY_DELAY", 20.0))
    request_timeout_seconds: float = float(os.getenv("CASSANDRA_REQUEST_TIMEOUT", 20.0))


def cassandra_settings() -> CassandraSettings:
    return CassandraSettings()


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
    """Путь к изображениям."""
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
