from typing import Optional
from pydantic import BaseModel, Field

class HealthResponse(BaseModel):
    status: str = Field(..., description="ok или degraded")
    model_loaded: bool

class ModelInfoResponse(BaseModel):
    checkpoint_path: str
    device: str
    classes: list[str]
    is_ready: bool

class PredictResponse(BaseModel):
    predicted_class: str = Field(..., description="Класс с максимальной вероятностью")
    probabilities: dict[str, float] = Field(..., description="Распределение вероятностей по всем классам")
    process_time_ms: float = Field(..., description="Время инференса")

class DatasetLoadRequest(BaseModel):
    split: Optional[str] = Field(
        None, description="Какую выборку загрузить: train, val, test. По умолчанию все"
    )

class DatasetLoadResponse(BaseModel):
    loaded_rows: int = Field(..., description="Сколько строк записано в Cassandra")
    split: str = Field(..., description="Загруженная выборка")
