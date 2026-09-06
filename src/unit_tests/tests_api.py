import configparser
import io
import uuid
from datetime import datetime, timezone

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from src.api.main import app
from src.model import classifier_service

TEST_CLASSES = ["angry", "happy", "relaxed", "sad"]


class FakeRepository:
    """Подмена CassandraRepository для юнит-тестов."""

    def __init__(self):
        self.rows = {}
        self.is_ready = True
        self.fail_on_save = False

    def connect(self) -> None:
        pass

    def shutdown(self) -> None:
        pass

    def save_prediction(
        self,
        request_id,
        image_name,
        predicted_class,
        probabilities,
        process_time_ms,
        model_checkpoint,
        device,
        created_at=None,
    ):
        if self.fail_on_save:
            raise RuntimeError("Cassandra недоступна")

        created_at = created_at or datetime.now(timezone.utc)

        self.rows[request_id] = {
            "request_id": request_id,
            "prediction_day": created_at.date(),
            "created_at": created_at,
            "image_name": image_name,
            "predicted_class": predicted_class,
            "confidence": float(probabilities[predicted_class]),
            "probabilities": probabilities,
            "process_time_ms": process_time_ms,
            "model_checkpoint": model_checkpoint,
            "device": device,
        }

        return created_at

    def get_prediction(self, request_id):
        return self.rows.get(request_id)


@pytest.fixture
def fake_repo(monkeypatch):
    """Подменяет глобальный репозиторий в модуле API."""
    repo = FakeRepository()
    monkeypatch.setattr("src.api.main.cassandra_repository", repo)
    return repo


@pytest.fixture(autouse=True)
def setup_mocks(monkeypatch, fake_repo):
    """
    Эта фикстура автоматически применяется ко всем тестам.
    Она подменяет чтение реального config.ini и тяжеловесную ML-модель.
    """

    dummy_cfg = configparser.ConfigParser()
    dummy_cfg.read_dict(
        {"MODEL": {"checkpoint_path": "test-checkpoint-path", "device": "cpu"}}
    )

    monkeypatch.setattr("src.api.main.load_config", lambda *args, **kwargs: dummy_cfg)

    monkeypatch.setattr(
        "src.api.main.checkpoint_path", lambda cfg=None: "test-checkpoint-path"
    )

    def mock_load(checkpoint_path, device="cpu"):
        classifier_service.is_ready = True
        classifier_service.checkpoint_path = checkpoint_path
        classifier_service.device = device
        classifier_service.id2label = {i: c for i, c in enumerate(TEST_CLASSES)}
        classifier_service.label2id = {c: i for i, c in enumerate(TEST_CLASSES)}

    monkeypatch.setattr(classifier_service, "load", mock_load)

    def mock_predict(image):
        probs = {c: 0.1 for c in TEST_CLASSES}
        probs[TEST_CLASSES[0]] = 0.7
        return TEST_CLASSES[0], probs

    monkeypatch.setattr(classifier_service, "predict", mock_predict)


@pytest.fixture
def client():
    """
    Создаем клиент поверх приложения.
    Обязательно используем контекстный менеджер,
    чтобы принудительно запустить события lifespan.
    """
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture
def test_image():
    """Генерирует тестовую картинку в оперативной памяти."""
    file = io.BytesIO()
    image = Image.new("RGB", (224, 224), color="blue")
    image.save(file, "jpeg")
    file.seek(0)
    return file


def _predict(client, test_image, **params):
    """Отправляет картинку в /predict и возвращает разобранный ответ."""
    response = client.post(
        "/predict",
        files={"image": ("test.jpg", test_image, "image/jpeg")},
        params=params,
    )
    assert response.status_code == 200
    return response.json()


def test_health_endpoint(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {
        "status": "ok",
        "model_loaded": True,
        "db_connected": True,
    }


def test_model_info_endpoint(client):
    """Проверяем, что API отдал данные из нашего замоканного конфига."""
    response = client.get("/model/info")
    assert response.status_code == 200
    data = response.json()
    assert data["checkpoint_path"] == "test-checkpoint-path"
    assert data["classes"] == TEST_CLASSES
    assert data["is_ready"] is True


def test_predict_success(client, test_image):
    response = client.post(
        "/predict",
        files={"image": ("test.jpg", test_image, "image/jpeg")},
    )

    assert response.status_code == 200
    data = response.json()

    assert data["predicted_class"] == "angry"
    assert set(data["probabilities"]) == set(TEST_CLASSES)
    assert data["probabilities"]["angry"] == 0.7
    assert "process_time_ms" in data
    assert "X-Process-Time-Ms" in response.headers


def test_predict_invalid_image(client):
    """Отправляем текстовую фигню под видом картинки"""
    response = client.post(
        "/predict",
        files={"image": ("test.txt", b"not an image", "text/plain")},
    )
    assert response.status_code == 400
    assert "Невозможно прочитать файл" in response.json()["detail"]


def test_predict_saves_to_db(client, test_image, fake_repo):
    """По умолчанию предсказание уходит в базу."""
    data = _predict(client, test_image)

    assert data["saved"] is True

    request_id = uuid.UUID(data["request_id"])
    assert request_id in fake_repo.rows

    saved = fake_repo.rows[request_id]
    assert saved["predicted_class"] == "angry"
    assert saved["image_name"] == "test.jpg"
    assert saved["confidence"] == 0.7


def test_predict_without_saving(client, test_image, fake_repo):
    """save=false отключает запись, но не влияет на инференс."""
    data = _predict(client, test_image, save="false")

    assert data["saved"] is False
    assert data["predicted_class"] == "angry"
    assert fake_repo.rows == {}


def test_predict_survives_db_failure(client, test_image, fake_repo):
    """
    Ключевой сценарий: инференс уже отработал, и сбой записи не должен
    его обесценивать. Ответ остаётся 200, потеря фиксируется в saved.
    """
    fake_repo.fail_on_save = True

    data = _predict(client, test_image)

    assert data["saved"] is False
    assert data["predicted_class"] == "angry"
    assert data["probabilities"]["angry"] == 0.7


def test_predict_when_db_unavailable(client, test_image, fake_repo):
    """Cassandra не поднялась при старте — предсказания всё равно работают."""
    fake_repo.is_ready = False

    data = _predict(client, test_image)

    assert data["saved"] is False
    assert fake_repo.rows == {}


def test_read_prediction_round_trip(client, test_image, fake_repo):
    """Сохранённое предсказание читается обратно по request_id."""
    predicted = _predict(client, test_image)

    response = client.get(f"/predictions/{predicted['request_id']}")

    assert response.status_code == 200
    record = response.json()

    assert record["request_id"] == predicted["request_id"]
    assert record["predicted_class"] == predicted["predicted_class"]
    assert record["probabilities"] == predicted["probabilities"]
    assert record["confidence"] == 0.7
    assert record["image_name"] == "test.jpg"
    assert record["device"] == "cpu"


def test_read_prediction_not_found(client, fake_repo):
    response = client.get(f"/predictions/{uuid.uuid4()}")

    assert response.status_code == 404
    assert "не найдено" in response.json()["detail"]


def test_read_prediction_invalid_uuid(client):
    """Путь принимает только UUID, остальное отсекает валидация FastAPI."""
    response = client.get("/predictions/not-a-uuid")

    assert response.status_code == 422
