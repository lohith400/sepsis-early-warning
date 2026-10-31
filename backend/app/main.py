"""FastAPI application: JSON API + the static web frontend (single process, single port)."""
import io
import json
import os
from contextlib import asynccontextmanager
from pathlib import Path

import pandas as pd
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from sepsis import __version__
from sepsis.config import BASE
from sepsis.model import SepsisModel
from sepsis.samples import make_patient, sample_labels, sample_list, SAMPLES

from .schemas import PredictRequest

ROOT = Path(__file__).resolve().parents[2]
FRONTEND = ROOT / "frontend"
REFERENCE = Path(__file__).resolve().parent / "data" / "reference_results.json"
NO_MODEL_HELP = ("No trained model found. Run the export cell of the Colab notebook "
                 "(scripts/colab_export_model_cell.py), put the downloaded model.json into the artifacts/ folder "
                 "and restart the app. Alternatively train one with `make train`.")
MAX_UPLOAD_BYTES = 2_000_000


def find_model_path():
    """$MODEL_PATH, else artifacts/model.json. There is deliberately NO fallback model."""
    for p in (os.environ.get("MODEL_PATH"), ROOT / "artifacts" / "model.json"):
        if p and Path(p).is_file():
            return Path(p)
    return None


@asynccontextmanager
async def lifespan(app: FastAPI):
    path = find_model_path()
    app.state.model = SepsisModel.load(str(path)) if path else None
    app.state.model_path = str(path) if path else None
    if path is None:
        print("WARNING: " + NO_MODEL_HELP)
    yield


app = FastAPI(
    title="Sepsis Early-Warning API",
    version=__version__,
    description="GA + ANFIS + PSO fuzzy early-warning model for ICU sepsis. Research prototype, not a medical device.",
    lifespan=lifespan,
)


def _model() -> SepsisModel:
    m = getattr(app.state, "model", None)
    if m is None:
        raise HTTPException(503, NO_MODEL_HELP)
    return m


@app.get("/api/health", tags=["system"])
def health():
    if getattr(app.state, "model", None) is None:
        return {"status": "no_model", "version": __version__, "detail": NO_MODEL_HELP}
    m = _model()
    return {"status": "ok", "version": __version__, "model": m.art["name"], "model_source": m.art["source"],
            "is_demo": m.is_demo, "model_file": Path(app.state.model_path).name}


@app.get("/api/model", tags=["model"])
def model_info():
    """Everything the UI shows about the loaded model: features, rules, metrics, curves, histories."""
    return _model().info()


@app.get("/api/features", tags=["model"])
def features():
    """The 37 candidate features with a flag for the ones the GA selected."""
    return _model().feature_catalog()


@app.get("/api/schema", tags=["model"])
def schema():
    """Raw measurement fields accepted by /api/predict (with units and plausible ranges)."""
    return _model().raw_schema()


@app.get("/api/reference", tags=["model"])
def reference():
    """Static results of the Colab run reported in the AAT report."""
    return json.loads(REFERENCE.read_text(encoding="utf-8"))


@app.get("/api/samples", tags=["samples"])
def samples():
    return sample_list()


@app.get("/api/samples/{sample_id}", tags=["samples"])
def sample(sample_id: str):
    """An illustrative synthetic patient (hourly raw measurements)."""
    if sample_id not in SAMPLES:
        raise HTTPException(404, "Unknown sample id")
    hours, age = make_patient(sample_id)
    return {"id": sample_id, "title": SAMPLES[sample_id]["title"], "age": age, "hours": hours,
            "labels": sample_labels(sample_id)}


@app.post("/api/predict", tags=["predict"])
def predict(req: PredictRequest):
    """Score every hour of one patient and return risk, alerts and fuzzy rule activations."""
    hours = [h.model_dump(exclude={"Age"}, exclude_none=False) for h in req.hours]
    for h, src in zip(hours, req.hours):
        if src.Age is not None:
            h["Age"] = src.Age
    try:
        return _model().predict(hours, age=req.age, first_icu_hour=req.first_icu_hour)
    except ValueError as e:
        raise HTTPException(400, str(e))


@app.post("/api/predict/psv", tags=["predict"])
async def predict_psv(file: UploadFile = File(...)):
    """Upload a PhysioNet-format .psv file (pipe separated, one patient) and score it."""
    raw = await file.read()
    if len(raw) > MAX_UPLOAD_BYTES:
        raise HTTPException(413, "File too large (limit 2 MB).")
    try:
        df = pd.read_csv(io.BytesIO(raw), sep="|")
    except Exception:
        raise HTTPException(400, "Could not read the file. Expected a pipe-separated PhysioNet .psv file.")
    if df.empty or not set(BASE) & set(df.columns):
        raise HTTPException(400, "No known measurement columns (HR, Resp, Temp, ...) found in the file.")
    if len(df) > 500:
        raise HTTPException(400, "A single patient file may contain at most 500 hourly rows.")
    cols = [c for c in BASE + ["Age", "ICULOS"] if c in df.columns]
    hours = json.loads(df[cols].to_json(orient="records"))      # NaN -> null
    age = float(df["Age"].dropna().iloc[0]) if "Age" in df and df["Age"].notna().any() else None
    out = _model().predict(hours, age=age)
    out["filename"] = file.filename
    out["true_labels"] = df["SepsisLabel"].astype(int).tolist() if "SepsisLabel" in df else None
    out["hours_raw"] = hours
    out["age"] = age
    return out


@app.get("/api/{rest:path}", include_in_schema=False)
def api_not_found(rest: str):
    return JSONResponse({"detail": "Not found"}, status_code=404)


@app.get("/", include_in_schema=False)
def index():
    return FileResponse(FRONTEND / "index.html")


app.mount("/", StaticFiles(directory=FRONTEND), name="frontend")
