import configparser
import io

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from src.api.main import app
from src.model import classifier_service

TEST_CLASSES = ["angry", "happy", "relaxed", "sad"]


@pytest.fixture(autouse=True)
def setup_mocks(monkeypatch):
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


def test_health_endpoint(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok", "model_loaded": True}


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
