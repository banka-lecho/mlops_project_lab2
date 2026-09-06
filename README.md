# Dog Emotion Classifier

An MLOps project that classifies dog emotions (`angry`, `happy`, `relaxed`, `sad`) from a photo. A fine-tuned EfficientNet-B0 model is served behind a FastAPI inference service, versioned with DVC, containerized with Docker, and shipped through CI/CD on GitHub Actions.

## 1. Project Overview

The service takes a dog photo and returns a predicted emotion with a probability distribution over all four classes.

- **Model**: EfficientNet-B0 (`torchvision`), fine-tuned on a labeled dataset of dog photos.
- **Serving**: FastAPI app that loads the trained checkpoint once at startup and exposes `/health`, `/model/info`, `/predict`.
- **CLI**: a single entry point (`src/model.py`) to train the model or run a one-off prediction, without starting the API.
- **Data & model versioning**: DVC, with a Google Drive remote.
- **Packaging**: Docker / docker-compose.
- **Automation**: CI builds and publishes the Docker image on every PR to `main`; CD pulls the latest image and the latest trained checkpoint and runs functional tests against a live container on a nightly schedule.

Project layout:

```
mlops_project/
├── config.ini                  # paths & runtime settings (data, model, device)
├── data/                       # dataset (DVC-tracked)
├── expirements/                # trained checkpoints (DVC-tracked)
├── models/                     # legacy / local model artifacts (not DVC-tracked)
├── notebooks/                  # EDA
├── src/
│   ├── api/                    # FastAPI app (main.py, schemas.py)
│   ├── config.py                # config.ini + env var resolution helpers
│   ├── model.py                 # dataset, model, training, inference, CLI
│   ├── logger.py
│   └── unit_tests/               # pytest suites (API unit tests + functional tests)
├── tests/data/                  # sample images used by functional tests
├── Dockerfile
├── docker-compose.yml
└── .github/workflows/           # ci.yml, cd.yml
```

## 2. Installation

Requirements: Python 3.10, `pip`.

```bash
git clone https://github.com/banka-lecho/mlops_project.git
cd mlops_project

python3 -m venv mlops_venv
source mlops_venv/bin/activate        # Windows: mlops_venv\Scripts\activate

pip install -r requirements.txt
```

Pull the dataset and the trained checkpoint (see [DVC](#10-dvc) for remote setup):

```bash
dvc pull
```

`config.ini` controls where data and the model checkpoint are read from; every path there can be overridden with an environment variable (`CONFIG_PATH`, `MODEL_PATH`... see [`src/config.py`](src/config.py)):

```ini
[DATA]
csv_path = data/dataset.csv
images_path = data/images

[MODEL]
checkpoint_path = expirements/dog_emotion_efficientnet_best.pth
device = cpu
```

### Secrets and connection settings

No credentials, database host/port or access tokens are stored in the source code. Everything sensitive comes from **environment variables**; `config.ini` holds only non-sensitive paths.

```bash
cp .env.example .env      # then fill in the values
```

[`.env.example`](.env.example) is the committed template; `.env` itself is gitignored (and excluded from the Docker image via `.dockerignore`), so real values never reach the repository.

| Variable | Required | Meaning |
| --- | --- | --- |
| `CASSANDRA_HOSTS` | yes | Comma-separated hosts. `cassandra` (compose service name) for `docker compose`, `127.0.0.1` when the API runs on the host |
| `CASSANDRA_PORT` | yes | CQL port |
| `CASSANDRA_KEYSPACE` | yes | Keyspace name (`dog_emotion_keyspace` in [`schema.cql`](src/db/schema.cql)) |
| `CASSANDRA_USER` | yes | Database login |
| `CASSANDRA_PASSWORD` | yes | Database password |
| `CASSANDRA_CONNECT_RETRIES` / `CASSANDRA_RETRY_DELAY` / `CASSANDRA_REQUEST_TIMEOUT` | no | Connection retry tuning |
| `API_PORT` | yes (compose) | Host port the API is published on |

[`src/config.py`](src/config.py) loads `.env` via `python-dotenv` (real environment variables win over the file) and raises `MissingSettingError` if a required variable is absent — there is no fallback default for a login, a password or a host. The password is excluded from `CassandraSettings.__repr__`, so it cannot leak into logs or tracebacks.

Running something outside compose, against a local Cassandra:

```bash
CASSANDRA_HOSTS=127.0.0.1 python -m src.db.load_dataset
```

## 3. Dataset

- **Source files** (DVC-tracked): [`data/dataset.csv`](data/dataset.csv), [`data/clean_dataset.csv`](data/clean_dataset.csv), [`data/images/`](data/images).
- **Format**: one row per image — `label, image_name`. Images live in `data/images/<image_name>`.
- **Classes**: `angry`, `happy`, `relaxed`, `sad` — 4 emotion labels, roughly balanced (~930–990 images per class, 3,876 images total).
- **EDA**: [`notebooks/eda.ipynb`](notebooks/eda.ipynb).

`DogEmotionDataset` ([`src/model.py`](src/model.py)) reads the CSV, builds a `label <-> id` mapping, and applies the standard ImageNet resize/normalize transform (plus augmentation for training: horizontal flip, rotation, color jitter).

## 4. CLI Usage

`src/model.py` is both a library and a CLI entry point, guarded by `if __name__ == "__main__":`. It exposes two subcommands via `argparse`:

```bash
python -m src.model --help
python -m src.model train --help
python -m src.model predict --help
```

Both subcommands default their paths to whatever is configured in `config.ini` (via `src/config.py` helpers `target_path()`, `images_path()`, `checkpoint_path()`), and every default can be overridden with a flag. See [Training](#5-training) and [Predict single image](#6-predict-single-image-api-and-cli) below for concrete examples.

## 5. Training

```bash
python -m src.model train --epochs 20 [--csv-path data/dataset.csv] [--img-path data/images]
```

What `DogEmotionClassifierService.train()` does:

1. Loads the CSV, stratified 80/20 train/val split (`random_state=42`).
2. Builds an EfficientNet-B0 (`ImageNet` pretrained weights) with the classifier head replaced for 4 classes.
3. Trains with `AdamW` + `CosineAnnealingLR`, tracking weighted/macro F1 on the validation split each epoch.
4. Saves the best checkpoint (by weighted F1) to `dog_emotion_efficientnet_best.pth` in the current working directory — `model_state_dict`, `label2id`, `id2label`.
5. After training, reloads the best checkpoint and prints a full classification report + confusion matrix on the validation set (see [Model Metrics](#11-model-metrics)).

To make the checkpoint pick up automatically for serving/inference, move it into `expirements/` (matching `config.ini`'s `checkpoint_path`) and re-track it with DVC:

```bash
mv dog_emotion_efficientnet_best.pth expirements/
dvc add expirements/dog_emotion_efficientnet_best.pth
dvc push
```

## 6. Predict single image (API and CLI)

**Via the CLI** — loads the checkpoint directly, no server needed:

```bash
python -m src.model predict --image path/to/dog.jpg [--checkpoint expirements/dog_emotion_efficientnet_best.pth] [--device cpu]
```

Prints the predicted class and per-class probabilities, sorted by confidence.

**Via the API** — start the server (see [API](#7-api)) then:

```bash
curl -X POST http://localhost:8000/predict \
  -F "image=@tests/data/happy_dog.jpg"
#{"predicted_class":"happy","probabilities":{"angry":0.00022165727568790317,"happy":0.9997721314430237,"relaxed":5.782482730865013e-06,"sad":4.358941509963188e-07},"process_time_ms":53.597124999953394}
```

```json
{
  "predicted_class": "happy",
  "probabilities": {
    "angry": 0.0002,
    "happy": 0.9997,
    "relaxed": 0.0001,
    "sad": 0.0002
  },
  "process_time_ms": 53.2
}
```

## 7. API

FastAPI app: [`src/api/main.py`](src/api/main.py). Run locally:

```bash
uvicorn src.api.main:app --host 0.0.0.0 --port 8000
```

The model checkpoint is loaded once at startup (`lifespan`), from the path resolved by `checkpoint_path()` in `config.ini` / `CHECKPOINT_PATH` env var.

| Method | Path          | Description                                                                   |
|--------|---------------|--------------------------------------------------------------------------------|
| GET    | `/health`     | Liveness/readiness check — `status` (`ok`/`degraded`) and `model_loaded`      |
| GET    | `/model/info` | Checkpoint path, device, class list, readiness                                |
| POST   | `/predict`    | `multipart/form-data` with an `image` file → predicted class + probabilities  |

If the model failed to load, `/model/info` and `/predict` return **503** (`ModelNotLoadedError`). Every response carries an `X-Process-Time-Ms` header. Interactive docs are auto-generated by FastAPI at `/docs`.

## 8. Docker

```bash
docker compose up --build
```

[`docker-compose.yml`](docker-compose.yml) mounts `./expirements` (checkpoint, read-only), `./data`, and `./logs`, and sets `CHECKPOINT_PATH=/app/expirements/dog_emotion_efficientnet_best.pth`. Database credentials, host and port are injected from `.env` (`env_file:`), not written in the compose file; the API is exposed on `http://localhost:${API_PORT}` (`8000` by default). Create `.env` from `.env.example` before `docker compose up`.

[`Dockerfile`](Dockerfile): `python:3.10-slim`, installs `requirements.txt`, copies `src/` and `config.ini`, runs `uvicorn src.api.main:app`.

## 9. CI/CD

Two GitHub Actions workflows:

- **[CI](.github/workflows/ci.yml)** — on every pull request to `main`: builds the Docker image and pushes it to Docker Hub (`latest` + commit SHA).
- **[CD](.github/workflows/cd.yml)** — nightly (`cron: 0 3 * * *`) and on manual dispatch:
  1. Pulls the latest published Docker image.
  2. Installs `dvc[gdrive]` and pulls the trained checkpoint (`expirements/dog_emotion_efficientnet_best.pth.dvc`) from the DVC remote, authenticating with a custom OAuth client (`GDRIVE_CLIENT_SECRET` secret, paired with the `gdrive_client_id` already committed in `.dvc/config`) and a cached OAuth token (`GDRIVE_CREDENTIALS_DATA` secret).
  3. Starts the container with the checkpoint mounted, waits for `/health`.
  4. Runs functional tests ([`src/unit_tests/tests_functionality.py`](src/unit_tests/tests_functionality.py)) against the live container.
  5. Always dumps container logs and tears the container down.

Required GitHub Secrets: `DOCKERHUB_USERNAME`, `DOCKERHUB_TOKEN`, `GDRIVE_CLIENT_SECRET`, `GDRIVE_CREDENTIALS_DATA`, `CASSANDRA_USER`, `CASSANDRA_PASSWORD`.

CI/CD never writes a secret into a file in the repository: Docker Hub credentials go to `docker/login-action`, the DVC client secret is applied with `dvc remote modify --local` (which writes to the gitignored `.dvc/config.local`), and the database login/password are passed to the container as `-e` environment variables. GitHub masks all of them in the workflow log.

## 10. DVC

Tracked artifacts: `data/dataset.csv`, `data/clean_dataset.csv`, `data/images/`, `expirements/dog_emotion_efficientnet_best.pth`.

Remote: Google Drive (`.dvc/config`), authenticated per-user via a custom OAuth client (`gdrive_client_id` in `.dvc/config`, `gdrive_client_secret` kept locally in the gitignored `.dvc/config.local` — never commit it).

```bash
dvc pull          # fetch data + checkpoint
dvc add <path>    # track a new/updated artifact
dvc push          # upload to the Drive remote
dvc status        # what's out of sync with the remote
```

First `dvc push`/`dvc pull` on a new machine opens a browser for Google OAuth consent (Testing app — the account must be added as a *Test user* on the OAuth consent screen; click **Advanced → Go to <app> (unsafe)** to proceed) and caches the token outside the repo, at `~/Library/Caches/pydrive2fs/<gdrive_client_id>/default.json` on macOS (XDG cache dir on Linux) — nothing is written into `.dvc/` for a custom OAuth client.

## 11. Model Metrics

Validation-set classification report for the current best checkpoint (`expirements/dog_emotion_efficientnet_best.pth`), EfficientNet-B0, 4 classes:

```
              precision    recall  f1-score   support

       angry       0.92      0.84      0.88       186
       happy       0.92      0.93      0.93       198
     relaxed       0.91      0.89      0.90       196
         sad       0.87      0.95      0.91       196

    accuracy                           0.90       776
   macro avg       0.91      0.90      0.90       776
weighted avg       0.91      0.90      0.90       776
```

Overall validation accuracy: **90%**. `sad` has the highest recall (0.95) but the lowest precision (0.87), i.e. the model over-predicts `sad` relative to the other classes; `angry` is the opposite — high precision (0.92), lower recall (0.84), meaning some angry photos get misclassified as another emotion. Re-run `python -m src.model train` to regenerate this report on a new split/checkpoint.
