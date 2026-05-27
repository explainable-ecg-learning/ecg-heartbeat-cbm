"""
explorer/app.py — FastAPI backend for the ECG heartbeat CBM explorer.

Endpoints:
  GET  /records            — list PTB-XL record IDs (limited, fast)
  GET  /record/{record_id} — load full 12-lead signal + metadata
  GET  /concepts           — concept names and classifier weights
  POST /infer              — concept activations for a single window
  POST /sweep              — sliding-window concept activations across full record

Run with:
  cd explorer
  uvicorn app:app --reload --port 8000
"""

from __future__ import annotations

import sys
import json
import logging
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd
import torch
import wfdb
import yaml
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
import tomllib

# ── Add src/ to path ─────────────────────────────────────────────────────────
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from models import build_model, CONCEPT_NAMES  # noqa: E402
from data import bandpass_filter, z_normalize  # noqa: E402

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# ── Config ────────────────────────────────────────────────────────────────────
with open(REPO_ROOT / "config.toml", "rb") as f:
    _toml = tomllib.load(f)

_paths     = _toml["paths"]
RUN_DIR    = Path(_paths.get("run_dir", "runs/20260423_064742"))
if not RUN_DIR.is_absolute():
    RUN_DIR = REPO_ROOT / RUN_DIR
DATA_DIR   = Path(_paths.get("data_dir", "/nfs/data8/schlegel/git/ecg-cbm/data"))
DATA_PATH  = Path(_paths.get("data_path", "ptb-xl"))
if not DATA_PATH.is_absolute():
    DATA_PATH = DATA_DIR / DATA_PATH

with open(RUN_DIR / "config.yaml") as f:
    CFG = yaml.safe_load(f)

DATA_CFG    = CFG.get("data", {})
SAMPLING_RATE = DATA_CFG.get("sampling_rate", 500)
T_FIXED       = DATA_CFG.get("t_fixed", 500)
LEAD_IDX      = DATA_CFG.get("lead", 1)

# ── Model ─────────────────────────────────────────────────────────────────────
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
logger.info(f"Using device: {DEVICE}")

def _load_model() -> torch.nn.Module:
    ckpt_path = RUN_DIR / "checkpoints" / "best_model.pt"
    model = build_model(CFG).to(DEVICE)
    state = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
    model.load_state_dict(state["model"])
    model.eval()
    logger.info(f"Loaded model from {ckpt_path}")
    return model

MODEL = _load_model()

# ── PTB-XL meta ───────────────────────────────────────────────────────────────
_META: Optional[pd.DataFrame] = None

def _get_meta() -> pd.DataFrame:
    global _META
    if _META is None:
        meta_path = DATA_PATH / "ptbxl_database.csv"
        _META = pd.read_csv(meta_path, index_col="ecg_id")
        logger.info(f"Loaded PTB-XL metadata: {len(_META)} records")
    return _META


def _load_signal(ecg_id: int) -> np.ndarray:
    """Load and preprocess a PTB-XL record. Returns (12, T) float32."""
    meta = _get_meta()
    if ecg_id not in meta.index:
        raise HTTPException(status_code=404, detail=f"Record {ecg_id} not found")
    row = meta.loc[ecg_id]
    filename_col = "filename_hr" if SAMPLING_RATE == 500 else "filename_lr"
    record_path = str(DATA_PATH / row[filename_col])
    signal, _ = wfdb.rdsamp(record_path)
    signal = signal.T.astype(np.float32)  # (12, T)
    signal = bandpass_filter(signal, sampling_rate=SAMPLING_RATE)
    # Per-channel normalise for display (not masked-beat norm)
    signal = z_normalize(signal)
    return signal


def _infer_window(signal: np.ndarray, start: int) -> dict:
    """Run model on a single window [start, start+T_FIXED]. Returns concept dict."""
    end = start + T_FIXED
    C, T = signal.shape
    if end > T:
        end = T
        start = max(0, end - T_FIXED)

    window = signal[:, start:end]
    if window.shape[1] < T_FIXED:
        pad = np.zeros((C, T_FIXED - window.shape[1]), dtype=np.float32)
        window = np.concatenate([window, pad], axis=1)

    # Respect model_lead / model_leads from config
    model_leads = DATA_CFG.get("model_leads", None)
    model_lead  = DATA_CFG.get("model_lead", None)
    if model_leads is not None:
        window = window[np.array(model_leads), :]
    elif model_lead is not None:
        window = window[model_lead : model_lead + 1, :]

    x = torch.from_numpy(window[np.newaxis]).to(DEVICE)  # (1, C, T)

    with torch.no_grad():
        task_prob, c_probs, c_log_vars = MODEL(x)
        task_mean, task_std, _ = MODEL.classifier.mc_task_uncertainty(
            c_probs, c_log_vars, n_mc=50
        )

    c_probs_np   = c_probs[0].cpu().numpy().tolist()
    c_vars_np    = torch.exp(c_log_vars[0]).cpu().numpy().tolist()
    c_std_np     = [float(v ** 0.5) for v in c_vars_np]
    task_prob_np = float(task_mean[0].cpu())
    task_std_np  = float(task_std[0].cpu())

    return {
        "task_prob":      task_prob_np,
        "task_std":       task_std_np,
        "concept_probs":  c_probs_np,
        "concept_stds":   c_std_np,
        "concept_names":  CONCEPT_NAMES,
    }


# ── FastAPI app ───────────────────────────────────────────────────────────────
app = FastAPI(title="ECG Heartbeat CBM Explorer", version="1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve static files (index.html, main.js, style.css)
_static_dir = Path(__file__).parent / "static"
app.mount("/static", StaticFiles(directory=str(_static_dir)), name="static")


@app.get("/")
def root():
    return FileResponse(str(Path(__file__).parent / "index.html"))


# ── Pydantic schemas ──────────────────────────────────────────────────────────
class InferRequest(BaseModel):
    ecg_id: int
    start: int  # sample index


class SweepRequest(BaseModel):
    ecg_id: int
    stride: int = 25


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/records")
def list_records(limit: int = 200, fold: Optional[int] = None):
    """Return a list of PTB-XL record IDs (optionally filtered by strat_fold)."""
    meta = _get_meta()
    if fold is not None:
        subset = meta[meta["strat_fold"] == fold]
    else:
        subset = meta
    ids = subset.index.tolist()[:limit]
    return {"record_ids": ids, "total": len(subset)}


@app.get("/record/{ecg_id}")
def get_record(ecg_id: int, lead: int = LEAD_IDX):
    """
    Return signal data for one record.

    Returns a down-sampled version for the overview plot (every 4th sample)
    plus the full-resolution sample count for accurate window positioning.
    """
    signal = _load_signal(ecg_id)
    meta   = _get_meta().loc[ecg_id]
    C, T   = signal.shape

    # Downsample for transfer — send every Nth sample for display (keep detail)
    ds = 2
    signal_ds = signal[:, ::ds]

    # Lead names from wfdb standard 12-lead order
    lead_names = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]

    return {
        "ecg_id":        ecg_id,
        "sampling_rate": SAMPLING_RATE,
        "t_fixed":       T_FIXED,
        "n_samples":     T,
        "n_samples_ds":  signal_ds.shape[1],
        "downsample":    ds,
        "leads":         lead_names,
        "signal":        signal_ds.tolist(),   # (12, T//ds) — all leads
        "age":           float(meta.get("age", 0)),
        "sex":           int(meta.get("sex", 0)),
    }


@app.post("/infer")
def infer(req: InferRequest):
    """Return concept activations for a single window at `start`."""
    signal = _load_signal(req.ecg_id)
    return _infer_window(signal, req.start)


@app.post("/sweep")
def sweep(req: SweepRequest):
    """
    Pre-compute concept activations for all sliding-window positions.

    Returns parallel arrays of length (n_windows,) for each concept,
    plus the corresponding start indices.
    """
    signal = _load_signal(req.ecg_id)
    C, T   = signal.shape
    stride = max(1, req.stride)

    starts: List[int] = list(range(0, T - T_FIXED + 1, stride))
    if not starts:
        raise HTTPException(status_code=422, detail="Signal too short for the window size.")

    task_probs:     List[float]       = []
    task_stds:      List[float]       = []
    concept_probs:  List[List[float]] = []
    concept_stds:   List[List[float]] = []

    for start in starts:
        result = _infer_window(signal, start)
        task_probs.append(result["task_prob"])
        task_stds.append(result["task_std"])
        concept_probs.append(result["concept_probs"])
        concept_stds.append(result["concept_stds"])

    return {
        "ecg_id":         req.ecg_id,
        "stride":         stride,
        "t_fixed":        T_FIXED,
        "starts":         starts,
        "task_probs":     task_probs,
        "task_stds":      task_stds,
        "concept_probs":  concept_probs,   # (n_windows, K)
        "concept_stds":   concept_stds,    # (n_windows, K)
        "concept_names":  CONCEPT_NAMES,
    }


@app.get("/concepts")
def get_concepts():
    """Return concept names and classifier weight information."""
    weights = MODEL.classifier.get_weights_dict(CONCEPT_NAMES)
    return {
        "concept_names": CONCEPT_NAMES,
        "classifier_weights": weights,
    }


@app.get("/health")
def health():
    return {"status": "ok", "device": DEVICE, "run_dir": str(RUN_DIR)}
