-- Схема хранилища результатов модели.
CREATE KEYSPACE IF NOT EXISTS dog_emotion_keyspace
WITH replication = {'class': 'SimpleStrategy', 'replication_factor': 1};

-- Предсказания за день: партиция на дату.
CREATE TABLE IF NOT EXISTS dog_emotion_keyspace.predictions (
    prediction_day   date,
    created_at       timestamp,
    request_id       uuid,
    image_name       text,
    predicted_class  text,
    confidence       double,
    probabilities    map<text, double>,
    process_time_ms  double,
    model_checkpoint text,
    device           text,
    PRIMARY KEY ((prediction_day), created_at, request_id)
) WITH CLUSTERING ORDER BY (created_at DESC);

-- Та же запись под точечный поиск по идентификатору запроса.
CREATE TABLE IF NOT EXISTS dog_emotion_keyspace.predictions_by_id (
    request_id       uuid PRIMARY KEY,
    prediction_day   date,
    created_at       timestamp,
    image_name       text,
    predicted_class  text,
    confidence       double,
    probabilities    map<text, double>,
    process_time_ms  double,
    model_checkpoint text,
    device           text
);

-- Обучающая и валидационная выборки: партиция на split.
CREATE TABLE IF NOT EXISTS dog_emotion_keyspace.dataset (
    split       text,
    label       text,
    image_name  text,
    image_path  text,
    loaded_at   timestamp,
    PRIMARY KEY ((split), label, image_name)
);
