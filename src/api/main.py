import time
import io
from PIL import Image
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request, status, UploadFile, File
from fastapi.responses import JSONResponse

from src.logger import get_logger
from .schemas import (
    HealthResponse,
    ModelInfoResponse,
    PredictResponse,
    DatasetLoadRequest,
    DatasetLoadResponse,
)
from src.model import classifier_service, ModelNotLoadedError

from src.config import load_config, checkpoint_path
from src.db.load_dataset import load_split

logger = get_logger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Артефакт грузится один раз при старте."""
    cfg = load_config()

    ckpt_path = str(checkpoint_path(cfg))
    device = cfg["MODEL"].get("device", "cuda")

    try:
        classifier_service.load(checkpoint_path=ckpt_path, device=device)
        logger.info(
            "Модель успешно загружена из: %s",
            ckpt_path
        )
    except Exception as exc:
        logger.exception(
                "Ошибка загрузки модели: %s",
                ckpt_path
        )
    yield
    logger.info("Остановка сервиса, очистка ресурсов.")

app = FastAPI(
    title="Dog Emotion Classifier API",
    version="1.0.0",
    lifespan=lifespan
)

@app.middleware("http")
async def add_timing_header(request: Request, call_next):
    started = time.perf_counter()
    response = await call_next(request)
    response.headers["X-Process-Time-Ms"] = f"{(time.perf_counter() - started) * 1000:.2f}"
    return response

@app.exception_handler(ModelNotLoadedError)
async def model_not_loaded_handler(request: Request, exc: ModelNotLoadedError):
    return JSONResponse(
        status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
        content={"detail": "Модель не загружена"},
    )


@app.get("/health", response_model=HealthResponse, tags=["ops"])
async def health():
    return HealthResponse(
        status="ok" if classifier_service.is_ready else "degraded",
        model_loaded=classifier_service.is_ready,
    )

@app.get("/model/info", response_model=ModelInfoResponse, tags=["ops"])
async def model_info():
    if not classifier_service.is_ready:
        logger.exception("Модель еще не загружена или не получилось ее загрузить.")
        raise ModelNotLoadedError

    return ModelInfoResponse(
        checkpoint_path=classifier_service.checkpoint_path,
        device=classifier_service.device,
        classes=[classifier_service.id2label[i] for i in range(len(classifier_service.id2label))],
        is_ready=classifier_service.is_ready
    )

@app.post("/predict", response_model=PredictResponse, tags=["inference"])
async def predict(
    image: UploadFile = File(..., description="Изображение для анализа")
):
    """Основной метод инференса."""
    if not classifier_service.is_ready:
        logger.exception("Модель еще не загружена или не получилось ее загрузить.")
        raise ModelNotLoadedError

    try:
        image_bytes = await image.read()
        pil_image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    except Exception:
        logger.exception("Неправильный формат входного изобаражения.")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Невозможно прочитать файл как изображение"
        )

    started = time.perf_counter()
    try:
        predicted_class, probabilities = classifier_service.predict(image=pil_image)
    except Exception as exc:
        logger.exception("Ошибка инференса")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Ошибка инференса: {str(exc)}"
        )
    process_time = (time.perf_counter() - started) * 1000

    return PredictResponse(
        predicted_class=predicted_class,
        probabilities=probabilities,
        process_time_ms=process_time
    )


@app.post("/admin/dataset", response_model=DatasetLoadResponse, tags=["admin"])
async def load_dataset(request: DatasetLoadRequest):
    """Заливает разбиение датасета в Cassandra."""
    try:
        loaded = load_split(split=request.split)
    except (FileNotFoundError, ValueError) as exc:
        logger.exception("Не удалось загрузить датасет")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        )
    except Exception as exc:
        logger.exception("Ошибка загрузки датасета в Cassandra")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Ошибка загрузки датасета: {exc}",
        )

    return DatasetLoadResponse(
        loaded_rows=loaded,
        split=request.split or "все",
    )
