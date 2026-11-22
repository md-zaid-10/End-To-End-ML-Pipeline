## MLFlow Experiments

Create a `.env` file in the project root (it is git-ignored) with:

```
MLFLOW_TRACKING_URI=https://dagshub.com/<your-username>/<your-repo>.mlflow
MLFLOW_TRACKING_USERNAME=<your-username>
MLFLOW_TRACKING_PASSWORD=<your-dagshub-token>
```

Load it before running:

```bash
set -a; source .env; set +a
python app.py
```