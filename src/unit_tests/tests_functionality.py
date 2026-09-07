import csv
import os
from pathlib import Path

import pytest
import requests

SMOKE_DIR = Path("tests/data/smoke")
CONFIDENCE_THRESHOLD = 0.7


def _load_smoke_cases():
    manifest = SMOKE_DIR / "expected.csv"
    if not manifest.exists():
        return []

    with open(manifest, newline="", encoding="utf-8") as f:
        return [(row["image_name"], row["label"]) for row in csv.DictReader(f)]


SMOKE_CASES = _load_smoke_cases()


@pytest.fixture
def base_url():
    return os.getenv("BASE_URL", "http://localhost:8000")


def test_health(base_url):
    response = requests.get(f"{base_url}/health")

    assert response.status_code == 200

    data = response.json()

    assert data["status"] == "ok"
    assert data["model_loaded"] is True


def test_model_info(base_url):
    response = requests.get(f"{base_url}/model/info")

    assert response.status_code == 200

    data = response.json()

    assert data["is_ready"] is True
    assert data["device"] in ["cpu", "cuda"]
    assert data["checkpoint_path"]
    assert data["classes"]


def test_prediction(base_url):
    with open("tests/data/happy_dog.jpg", "rb") as image:
        response = requests.post(
            f"{base_url}/predict",
            files={"image": ("happy_dog.jpg", image, "image/jpeg")},
        )

    assert response.status_code == 200

    data = response.json()

    assert data["predicted_class"]
    assert data["probabilities"]

    assert 0 <= data["process_time_ms"]


def test_invalid_image(base_url):
    response = requests.post(
        f"{base_url}/predict",
        files={"image": ("not_an_image.txt", b"not an image", "text/plain")},
    )

    assert response.status_code == 400


def test_prediction_round_trip(base_url):
    with open("tests/data/happy_dog.jpg", "rb") as image:
        predicted = requests.post(
            f"{base_url}/predict",
            files={"image": ("happy_dog.jpg", image, "image/jpeg")},
        ).json()
    assert predicted["saved"] is True

    record = requests.get(f"{base_url}/predictions/{predicted['request_id']}")
    assert record.status_code == 200
    assert record.json()["predicted_class"] == predicted["predicted_class"]


@pytest.mark.skipif(not SMOKE_CASES, reason=f"Нет фикстуры {SMOKE_DIR}/expected.csv")
@pytest.mark.parametrize("image_name,expected_label", SMOKE_CASES)
def test_confident_prediction_per_class(base_url, image_name, expected_label):
    with open(SMOKE_DIR / image_name, "rb") as image:
        response = requests.post(
            f"{base_url}/predict", files={"image": (image_name, image, "image/jpeg")}
        )

    assert response.status_code == 200

    data = response.json()
    probabilities = data["probabilities"]

    assert data["predicted_class"] == expected_label, (
        f"{image_name}: ожидался класс '{expected_label}', "
        f"получен '{data['predicted_class']}'. Вероятности: {probabilities}"
    )

    confidence = probabilities[expected_label]
    assert confidence >= CONFIDENCE_THRESHOLD, (
        f"{image_name}: уверенность в классе '{expected_label}' = {confidence:.4f} "
        f"< {CONFIDENCE_THRESHOLD}. Вероятности: {probabilities}"
    )
