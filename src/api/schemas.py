import uuid
from datetime import date, datetime

from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    status: str = Field(..., description="ok или degraded")
    model_loaded: bool


class ModelInfoResponse(BaseModel):
    checkpoint_path: str
    device: str
    classes: list[str]
    is_ready: bool


class DatasetLoadRequest(BaseModel):
    split: str | None = Field(
        None, description="Какую выборку загрузить: train, val, test. По умолчанию все"
    )


class DatasetLoadResponse(BaseModel):
    loaded_rows: int = Field(..., description="Сколько строк записано в Cassandra")
    split: str = Field(..., description="Загруженная выборка")


class PredictResponse(BaseModel):
    request_id: uuid.UUID
    predicted_class: str
    probabilities: dict[str, float]
    process_time_ms: float
    saved: bool = Field(False, description="Записано ли предсказание в Cassandra")


class PredictionRecord(BaseModel):
    request_id: uuid.UUID
    prediction_day: date
    created_at: datetime
    image_name: str
    predicted_class: str
    confidence: float
    probabilities: dict[str, float]
    process_time_ms: float
    model_checkpoint: str
    device: str
