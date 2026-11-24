"""
explorer/app.py — FastAPI backend for the ECG Heartbeat CBM & CEM Explorer.

Supports:
  - Thread-safe multi-model loading (auto-discovers all CBM, CEM, and Black-Box runs)
  - Side-by-side Primary vs. Compare model inference & batched sliding-window sweeps
  - Live click-to-intervene concept overrides with unified probability-space uncertainty
  - Optional MC-Dropout epistemic + aleatoric uncertainty decomposition on /infer
  - Pathology / rhythm filtering (NORM, MI, STTC, CD, HYP, AFIB, BBB, PVC) & record search
  - Ground-truth beat delineation markers (P, R, T peaks) and sliding-window GT labels
  - Environment-variable overrides for clean offline Docker deployment
"""

from __future__ import annotations

import ast
import json
import logging
import os
import pickle
import sys
import threading
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import wfdb
import yaml
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from data import (  # noqa: E402
    bandpass_filter,
    delineate_record,
    get_concept_labels,
    is_complete_heartbeat_window,
    z_normalize,
)
from models import CONCEPT_NAMES, build_model  # noqa: E402
from dense_seg import SUPERCLASS_NAMES, build_dense_model, collapse_soft_seg_targets, get_dense_concept_names  # noqa: E402
from soft_seg import SoftSegArchive, soft_seg_to_wave_curves, soft_window_concept_probs  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Configuration (env vars override config.yaml for Docker support)
# ─────────────────────────────────────────────────────────────────────────────

def _load_paths() -> Tuple[Path, Path, Path, str, str]:
    cfg: dict = {}
    cfg_file = REPO_ROOT / "config.yaml"
    if cfg_file.exists():
        with open(cfg_file) as f:
            cfg = yaml.safe_load(f) or {}

    data_cfg = cfg.get("data", {})
    train_cfg = cfg.get("training", {})
    inf_cfg = cfg.get("inference", {})

    default_models_dir = REPO_ROOT / train_cfg.get("log_dir", "models")
    if not default_models_dir.exists() and (REPO_ROOT / "runs").exists():
        default_models_dir = REPO_ROOT / "runs"
    runs_dir = Path(os.environ.get("ECG_RUNS_DIR", str(default_models_dir)))

    data_dir = Path(os.environ.get("ECG_DATA_DIR", data_cfg.get("data_dir", str(REPO_ROOT / "data"))))
    raw_data_path = Path(os.environ.get("ECG_DATA_PATH", data_cfg.get("data_path", "ptb-xl")))
    data_path = raw_data_path if raw_data_path.is_absolute() else (data_dir / raw_data_path)

    soft_seg_rel = data_cfg.get("soft_seg_path", "ptbxl_segmentations_8bit.zip")

    delin_cache_env = os.environ.get("ECG_DELIN_CACHE_DIR", "")
    if delin_cache_env:
        delin_cache = Path(delin_cache_env)
    else:
        ckpt_dir = Path(train_cfg.get("checkpoint_dir", "checkpoints"))
        if not ckpt_dir.is_absolute():
            ckpt_dir = REPO_ROOT / ckpt_dir
        delin_cache = ckpt_dir / "dataset_cache" / "delineation"

    default_run_raw = os.environ.get("ECG_DEFAULT_RUN", inf_cfg.get("run_dir", "models/cem_softseg"))
    default_run_key = default_run_raw.removeprefix("models/").removeprefix("runs/").strip("/")
    return runs_dir, data_path, delin_cache, default_run_key, soft_seg_rel


RUNS_DIR, DATA_PATH, DELIN_CACHE_DIR, DEFAULT_RUN_KEY, SOFT_SEG_REL = _load_paths()
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
logger.info(f"Explorer starting | device={DEVICE} | runs={RUNS_DIR} | data={DATA_PATH}")

# Optional Zenodo 7610236 (ECGdeli U-Net) soft segmentation archive
_SOFT_SEG_ARCHIVE: Optional[SoftSegArchive] = None
_SOFT_SEG_LOCK = threading.Lock()


def _get_soft_seg_archive() -> Optional[SoftSegArchive]:
    global _SOFT_SEG_ARCHIVE
    with _SOFT_SEG_LOCK:
        if _SOFT_SEG_ARCHIVE is None:
            env_ss = os.environ.get("ECG_SOFT_SEG_PATH", "").strip()
            candidates: List[Path] = []
            if env_ss:
                candidates.append(Path(env_ss))
            ss_cfg_path = Path(SOFT_SEG_REL)
            candidates.append(ss_cfg_path if ss_cfg_path.is_absolute() else (DATA_PATH.parent / ss_cfg_path))
            candidates.extend([
                DATA_PATH.parent / "ptbxl_segmentations_8bit.zip",
                DATA_PATH.parent / "ptbxl_segmentations_8bit",
            ])
            for c in candidates:
                if c.exists():
                    try:
                        _SOFT_SEG_ARCHIVE = SoftSegArchive(c)
                        logger.info(f"Loaded Zenodo soft-segmentation archive ({len(_SOFT_SEG_ARCHIVE)} records) from {c}")
                        break
                    except Exception as exc:
                        logger.warning(f"Could not open soft segmentation archive {c}: {exc}")
        return _SOFT_SEG_ARCHIVE


# ─────────────────────────────────────────────────────────────────────────────
# Thread-safe Multi-Model Discovery & Caching
# ─────────────────────────────────────────────────────────────────────────────

DENSE_MODEL_TYPES = {"dense_cbm", "dense_cem", "dense_blackbox", "dense_multitask"}


def _read_model_type(run_dir: Path) -> str:
    try:
        with open(run_dir / "config.yaml") as f:
            cfg = yaml.safe_load(f) or {}
        if not isinstance(cfg.get("model"), dict):
            return ""
        return str(cfg["model"].get("type", "cbm")).lower()
    except Exception:
        return ""


def discover_runs() -> Dict[str, Path]:
    found: Dict[str, Path] = {}
    if not RUNS_DIR.exists():
        return found
    for ckpt in sorted(RUNS_DIR.glob("**/checkpoints/best_model.pt")):
        run_dir = ckpt.parent.parent
        if not (run_dir / "config.yaml").exists():
            continue
        mtype = _read_model_type(run_dir)
        if not mtype or mtype in DENSE_MODEL_TYPES:
            continue
        found[str(run_dir.relative_to(RUNS_DIR))] = run_dir
    return found


def discover_dense_runs() -> Dict[str, Path]:
    found: Dict[str, Path] = {}
    if not RUNS_DIR.exists():
        return found
    for ckpt in sorted(RUNS_DIR.glob("**/checkpoints/best_model.pt")):
        run_dir = ckpt.parent.parent
        if not (run_dir / "config.yaml").exists():
            continue
        if _read_model_type(run_dir) not in DENSE_MODEL_TYPES:
            continue
        found[str(run_dir.relative_to(RUNS_DIR))] = run_dir
    return found


_DENSE_MODEL_CACHE: Dict[str, Tuple[torch.nn.Module, dict]] = {}


def get_dense_model_and_cfg(model_id: Optional[str] = None) -> Tuple[str, torch.nn.Module, dict]:
    runs = discover_dense_runs()
    if not runs:
        raise HTTPException(status_code=503, detail=f"No dense per-timestamp checkpoints found under {RUNS_DIR}")
    if model_id and model_id in runs:
        resolved = model_id
    elif model_id:
        raise HTTPException(status_code=404, detail=f"Dense model '{model_id}' not found. Available: {list(runs.keys())}")
    else:
        resolved = next(
            (p for p in ("dense_cem_softseg", "dense_cem_s42", "dense_cbm_detached_s42", "dense_cbm_softseg") if p in runs),
            next(iter(runs.keys())),
        )
    with _MODEL_LOCK:
        if resolved not in _DENSE_MODEL_CACHE:
            run_dir = runs[resolved]
            with open(run_dir / "config.yaml") as f:
                cfg = yaml.safe_load(f)
            model = build_dense_model(cfg).to(DEVICE)
            state = torch.load(run_dir / "checkpoints" / "best_model.pt", map_location=DEVICE, weights_only=False)
            model.load_state_dict(state["model"])
            model.eval()
            _DENSE_MODEL_CACHE[resolved] = (model, cfg)
            logger.info(f"Loaded dense model '{resolved}' ({getattr(model, 'model_type', 'dense').upper()}) on {DEVICE}")
        model, cfg = _DENSE_MODEL_CACHE[resolved]
    return resolved, model, cfg


_MODEL_CACHE: Dict[str, Tuple[torch.nn.Module, dict]] = {}
_MODEL_LOCK = threading.Lock()


def _resolve_model_id(model_id: Optional[str]) -> str:
    runs = discover_runs()
    if not runs:
        raise HTTPException(status_code=503, detail=f"No trained checkpoints found under {RUNS_DIR}")
    if model_id and model_id in runs:
        return model_id
    if model_id:
        raise HTTPException(status_code=404, detail=f"Model run '{model_id}' not found. Available: {list(runs.keys())}")
    if DEFAULT_RUN_KEY in runs:
        return DEFAULT_RUN_KEY
    for preferred in ("cem_v2", "cbm_randint_v2", "cbm_v2"):
        if preferred in runs:
            return preferred
    return next(iter(runs.keys()))


def get_model_and_cfg(model_id: Optional[str] = None) -> Tuple[str, torch.nn.Module, dict]:
    resolved = _resolve_model_id(model_id)
    with _MODEL_LOCK:
        if resolved not in _MODEL_CACHE:
            run_dir = discover_runs()[resolved]
            with open(run_dir / "config.yaml") as f:
                cfg = yaml.safe_load(f)
            model = build_model(cfg).to(DEVICE)
            state = torch.load(run_dir / "checkpoints" / "best_model.pt", map_location=DEVICE, weights_only=False)
            model.load_state_dict(state["model"])
            model.eval()
            _MODEL_CACHE[resolved] = (model, cfg)
            logger.info(f"Loaded model '{resolved}' ({getattr(model, 'model_type', 'cbm').upper()}) on {DEVICE}")
        model, cfg = _MODEL_CACHE[resolved]
    return resolved, model, cfg


def _validate_compare_compatibility(primary_cfg: dict, cmp_cfg: dict, cmp_id: str) -> None:
    p_data, c_data = primary_cfg.get("data", {}), cmp_cfg.get("data", {})
    if int(p_data.get("sampling_rate", 500)) != int(c_data.get("sampling_rate", 500)):
        raise HTTPException(status_code=400, detail=f"Compare model '{cmp_id}' has a different sampling_rate.")
    if int(p_data.get("t_fixed", 500)) != int(c_data.get("t_fixed", 500)):
        raise HTTPException(status_code=400, detail=f"Compare model '{cmp_id}' has a different window length t_fixed.")


def _concept_names_for_cfg(cfg: dict) -> List[str]:
    names = list(cfg.get("concept_names", CONCEPT_NAMES))
    active = cfg.get("active_concepts", None)
    if active is not None:
        return [names[i] for i in active]
    return names


def _read_json_if_exists(path: Path) -> Optional[dict]:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except Exception:
        return None


def _run_metadata_entry(run_id: str, run_dir: Path) -> dict:
    with open(run_dir / "config.yaml") as f:
        cfg = yaml.safe_load(f)
    mtype = str(cfg.get("model", {}).get("type", "cbm")).lower()
    metrics = (
        _read_json_if_exists(run_dir / "eval_test_metrics.json")
        or _read_json_if_exists(run_dir / "test_metrics.json")
    )
    interventions = _read_json_if_exists(run_dir / "intervention_test.json")
    return {
        "id":                run_id,
        "model_type":        mtype,
        "t_fixed":           int(cfg.get("data", {}).get("t_fixed", 500)),
        "sampling_rate":     int(cfg.get("data", {}).get("sampling_rate", 500)),
        "lead":              int(cfg.get("data", {}).get("lead", 0)),
        "concept_names":     _concept_names_for_cfg(cfg),
        "has_interventions": interventions is not None,
        "summary_metrics": {
            "task_acc":     metrics.get("task/accuracy"),
            "task_auroc":   metrics.get("task/auroc"),
            "task_ece":     metrics.get("task/ece"),
            "concept_f1":   metrics.get("concept/macro_f1"),
            "err_det_auc":  metrics.get("uncertainty/error_detection_auroc"),
            "flip_rate":    metrics.get("faithfulness/counterfactual_flip_rate"),
            "ois":          metrics.get("leakage/ois"),
        } if metrics else None,
    }


# ─────────────────────────────────────────────────────────────────────────────
# PTB-XL Metadata, Pathology Filters, Signal & Delineation Cache
# ─────────────────────────────────────────────────────────────────────────────

RHYTHM_GROUPS: Dict[str, set] = {
    "AFIB": {"AFIB", "AFLT"},
    "BBB":  {"CLBBB", "CRBBB", "ILBBB", "IRBBB"},
    "PVC":  {"PVC", "PAC", "SVPB", "BIGU", "TRIGU"},
}

_META: Optional[pd.DataFrame] = None
_META_LOCK = threading.Lock()
_SCP_MAP: Dict[str, str] = {}


def _get_meta() -> pd.DataFrame:
    global _META, _SCP_MAP
    with _META_LOCK:
        if _META is None:
            meta_path = DATA_PATH / "ptbxl_database.csv"
            if not meta_path.exists():
                raise HTTPException(status_code=503, detail=f"PTB-XL metadata not found at {meta_path}")
            df = pd.read_csv(meta_path, index_col="ecg_id")
            scp_path = DATA_PATH / "scp_statements.csv"
            if scp_path.exists():
                scp = pd.read_csv(scp_path, index_col=0)
                scp = scp[scp["diagnostic"] == 1.0]
                _SCP_MAP = scp["diagnostic_class"].dropna().to_dict()

            def _parse_tags(scp_str: str) -> Tuple[List[str], set]:
                try:
                    codes = {k for k, v in ast.literal_eval(str(scp_str)).items() if v > 0}
                except Exception:
                    return [], set()
                dx = sorted({_SCP_MAP[c] for c in codes if c in _SCP_MAP})
                tags = set(dx)
                for grp, kws in RHYTHM_GROUPS.items():
                    if not codes.isdisjoint(kws):
                        tags.add(grp)
                return dx, tags

            parsed = df["scp_codes"].apply(_parse_tags)
            df["_dx_list"] = [p[0] for p in parsed]
            df["_tag_set"] = [p[1] for p in parsed]
            _META = df
            logger.info(f"Loaded PTB-XL metadata: {len(_META)} records")
        return _META


@lru_cache(maxsize=16)
def _load_filtered_signal(ecg_id: int, sampling_rate: int = 500) -> np.ndarray:
    meta = _get_meta()
    if ecg_id not in meta.index:
        raise HTTPException(status_code=404, detail=f"Record {ecg_id} not found")
    row = meta.loc[ecg_id]
    filename_col = "filename_hr" if sampling_rate == 500 else "filename_lr"
    record_path = str(DATA_PATH / row[filename_col])
    signal, _ = wfdb.rdsamp(record_path)
    filtered = bandpass_filter(signal.T.astype(np.float32), sampling_rate=sampling_rate)
    filtered.setflags(write=False)   # immutable cache entry
    return filtered


@lru_cache(maxsize=32)
def _load_delineation(ecg_id: int, sampling_rate: int = 500, lead_idx: int = 0) -> dict:
    # Check lead-specific cache first, then default lead-0 cache
    candidate_dirs = [
        DELIN_CACHE_DIR.parent / f"delineation_sr{sampling_rate}_lead{lead_idx}",
    ]
    if sampling_rate == 500 and lead_idx == 0:
        candidate_dirs.append(DELIN_CACHE_DIR)
    for cdir in candidate_dirs:
        if cdir.exists():
            pkl_path = cdir / f"{ecg_id}.pkl"
            if pkl_path.exists():
                try:
                    with open(pkl_path, "rb") as f:
                        return pickle.load(f)
                except Exception:
                    pass
    sig = _load_filtered_signal(ecg_id, sampling_rate)
    return delineate_record(sig, sampling_rate=sampling_rate, lead_idx=lead_idx)


def _valid_int_list(arr) -> List[int]:
    if arr is None:
        return []
    return [int(v) for v in arr if not (isinstance(v, (float, np.floating)) and np.isnan(v))]


# ─────────────────────────────────────────────────────────────────────────────
# Batched Inference Helper (Unified Aleatoric + Optional Epistemic)
# ─────────────────────────────────────────────────────────────────────────────

def _select_model_leads(windows: np.ndarray, data_cfg: dict) -> np.ndarray:
    model_leads = data_cfg.get("model_leads", None)
    model_lead  = data_cfg.get("model_lead", None)
    if model_leads is not None:
        return windows[:, np.array(model_leads), :]
    if model_lead is not None:
        return windows[:, model_lead : model_lead + 1, :]
    return windows


@torch.no_grad()
def _batch_infer_windows(
    model: torch.nn.Module,
    cfg: dict,
    windows_np: np.ndarray,
    interventions: Optional[Dict[int, float]] = None,
    include_epistemic: bool = False,
    batch_size: int = 128,
) -> dict:
    data_cfg = cfg.get("data", {})
    normed = _select_model_leads(z_normalize(windows_np), data_cfg)
    N = normed.shape[0]
    mtype = getattr(model, "model_type", "cbm")
    K = model.num_concepts

    task_means, task_stds, task_ep_l, task_al_l = [], [], [], []
    c_probs_list, c_stds_list = [], []

    for s in range(0, N, batch_size):
        xb = torch.from_numpy(normed[s : s + batch_size]).to(DEVICE)
        B = xb.size(0)

        int_labels, int_mask = None, None
        if interventions:
            int_labels = torch.zeros(B, K, device=DEVICE)
            int_mask   = torch.zeros(B, K, device=DEVICE)
            for k_idx, val in interventions.items():
                ki = int(k_idx)
                if 0 <= ki < K:
                    int_labels[:, ki] = float(val)
                    int_mask[:, ki]   = 1.0

        if include_epistemic and mtype != "blackbox":
            unc = model.predict_with_uncertainty(
                xb, n_mc_epistemic=15, n_mc_aleatoric=30,
                intervene_labels=int_labels, intervene_mask=int_mask,
            )
            t_mean = unc["task_prob_mean"]
            t_std  = unc["task_prob_std"]
            t_ep   = unc["task_epistemic"]
            t_al   = unc["task_aleatoric"]
            c_probs = unc["concept_probs_mean"]
            c_std   = unc["concept_probs_std"]
        elif mtype in ("cbm", "cem"):
            z = model.encoder(xb)
            t_mean, t_std, _, c_probs, c_var = model.mc_task_from_z(
                z, n_mc=40, intervene_labels=int_labels, intervene_mask=int_mask
            )
            t_ep = torch.zeros_like(t_mean)
            t_al = t_std
            c_std = c_var.clamp(min=0.0).sqrt()
        else:
            if include_epistemic:
                unc = model.predict_with_uncertainty(xb, n_mc_epistemic=15, n_mc_aleatoric=1)
                t_mean, t_std = unc["task_prob_mean"], unc["task_prob_std"]
                t_ep, t_al = unc["task_epistemic"], unc["task_aleatoric"]
                c_probs = unc["concept_probs_mean"]
            else:
                tp, c_probs, _ = model(xb)
                t_mean = tp.squeeze(-1)
                t_std = t_ep = t_al = torch.zeros_like(t_mean)
            c_std = torch.zeros_like(c_probs)

        task_means.append(t_mean.cpu().numpy())
        task_stds.append(t_std.cpu().numpy())
        task_ep_l.append(t_ep.cpu().numpy())
        task_al_l.append(t_al.cpu().numpy())
        c_probs_list.append(c_probs.cpu().numpy())
        c_stds_list.append(c_std.cpu().numpy())

    return {
        "model_type":     mtype,
        "task_probs":     np.concatenate(task_means).tolist(),
        "task_stds":      np.concatenate(task_stds).tolist(),
        "task_epistemic": np.concatenate(task_ep_l).tolist(),
        "task_aleatoric": np.concatenate(task_al_l).tolist(),
        "concept_probs":  np.concatenate(c_probs_list, axis=0).tolist(),
        "concept_stds":   np.concatenate(c_stds_list, axis=0).tolist(),
    }


# ─────────────────────────────────────────────────────────────────────────────
# FastAPI App & Schemas
# ─────────────────────────────────────────────────────────────────────────────

app = FastAPI(title="ECG Heartbeat CBM & CEM Explorer", version="2.2")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

_static_dir = Path(__file__).parent / "static"
app.mount("/static", StaticFiles(directory=str(_static_dir)), name="static")


@app.get("/")
def root():
    return FileResponse(str(Path(__file__).parent / "index.html"))


class InferRequest(BaseModel):
    ecg_id: int
    start: int
    model_id: Optional[str] = None
    compare_model_id: Optional[str] = None
    interventions: Optional[Dict[int, float]] = None
    include_epistemic: bool = False


class SweepRequest(BaseModel):
    ecg_id: int
    stride: int = 20
    model_id: Optional[str] = None
    compare_model_id: Optional[str] = None


# ─────────────────────────────────────────────────────────────────────────────
# Endpoints
# ─────────────────────────────────────────────────────────────────────────────

@app.get("/models")
def list_models():
    runs = discover_runs()
    entries = [_run_metadata_entry(rid, rdir) for rid, rdir in runs.items()]
    priority = {
        "cem_softseg": 0,
        "cbm_softseg_masked": 1,
        "cbm_softseg": 2,
        "blackbox_softseg": 3,
        "cem_v2": 4,
        "cbm_randint_v2": 5,
        "cbm_v2": 6,
    }
    entries.sort(key=lambda e: (priority.get(e["id"], 7), e["id"]))
    default_id = _resolve_model_id(None) if entries else None
    compare_default = None
    for cand in ("cbm_softseg_masked", "cbm_randint_v2", "blackbox_softseg", "cbm_v2", "cem_v2"):
        if cand in runs and cand != default_id:
            compare_default = cand
            break
    return {
        "default_model":   default_id,
        "compare_default": compare_default,
        "models":          entries,
    }


@app.get("/model/{model_id:path}/summary")
def model_summary(model_id: str):
    resolved, model, cfg = get_model_and_cfg(model_id)
    run_dir = discover_runs()[resolved]
    metrics = (
        _read_json_if_exists(run_dir / "eval_test_metrics.json")
        or _read_json_if_exists(run_dir / "test_metrics.json")
    )
    interventions = _read_json_if_exists(run_dir / "intervention_test.json")
    c_names = _concept_names_for_cfg(cfg)
    return {
        "id":                 resolved,
        "model_type":         getattr(model, "model_type", "cbm"),
        "concept_names":      c_names,
        "classifier_weights": model.get_concept_importance(c_names),
        "metrics":            metrics,
        "interventions":      interventions,
    }


@app.get("/records")
def list_records(
    limit: int = 300,
    fold: Optional[int] = 10,
    dx: Optional[str] = None,
    search: Optional[str] = None,
):
    meta = _get_meta()
    subset = meta[meta["strat_fold"] == fold] if fold is not None else meta

    if dx and dx.upper() != "ALL":
        tag = dx.upper()
        subset = subset[subset["_tag_set"].apply(lambda s: tag in s)]

    if search and search.strip():
        q = search.strip()
        subset = subset[subset.index.astype(str).str.contains(q)]

    sub_head = subset.iloc[:limit]
    records = [
        {
            "ecg_id": int(eid),
            "fold":   int(row.get("strat_fold", 0)),
            "age":    float(row["age"]) if not pd.isna(row.get("age")) else 0.0,
            "sex":    int(row["sex"]) if not pd.isna(row.get("sex")) else 0,
            "dx":     row["_dx_list"],
            "tags":   sorted(row["_tag_set"]),
        }
        for eid, row in sub_head.iterrows()
    ]
    return {
        "record_ids": [r["ecg_id"] for r in records],
        "records":    records,
        "total":      len(subset),
    }


@app.get("/record/{ecg_id}")
def get_record(ecg_id: int, model_id: Optional[str] = None):
    _, _, cfg = get_model_and_cfg(model_id)
    sr       = int(cfg.get("data", {}).get("sampling_rate", 500))
    t_fixed  = int(cfg.get("data", {}).get("t_fixed", 500))
    lead_idx = int(cfg.get("data", {}).get("lead", 0))

    signal_filt = _load_filtered_signal(ecg_id, sr)
    signal_disp = z_normalize(signal_filt)
    meta        = _get_meta().loc[ecg_id]
    delin       = _load_delineation(ecg_id, sr, lead_idx)
    d_map       = delin.get("delineation", {})

    _, T = signal_disp.shape
    ds = 2
    signal_ds = signal_disp[:, ::ds]
    lead_names = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]

    # Optional Zenodo 7610236 (ECGdeli U-Net) per-timestep soft segmentation envelopes (100 Hz)
    soft_seg_payload = None
    archive = _get_soft_seg_archive()
    if archive is not None and ecg_id in archive:
        try:
            ss = archive.load(ecg_id)
            env = soft_seg_to_wave_curves(ss, lead_idx=lead_idx)
            soft_seg_payload = {
                "sampling_rate": 100,
                "P":   np.round(env["P"],   3).tolist(),
                "QRS": np.round(env["QRS"], 3).tolist(),
                "ST":  np.round(env["ST"],  3).tolist(),
                "T":   np.round(env["T"],   3).tolist(),
            }
        except Exception as exc:
            logger.warning(f"Failed loading soft_seg for ecg_id={ecg_id}: {exc}")

    return {
        "ecg_id":        ecg_id,
        "sampling_rate": sr,
        "t_fixed":       t_fixed,
        "n_samples":     T,
        "n_samples_ds":  signal_ds.shape[1],
        "downsample":    ds,
        "leads":         lead_names,
        "signal":        signal_ds.tolist(),
        "age":           float(meta["age"]) if not pd.isna(meta.get("age")) else 0.0,
        "sex":           int(meta["sex"]) if not pd.isna(meta.get("sex")) else 0,
        "dx":            meta["_dx_list"],
        "tags":          sorted(meta["_tag_set"]),
        "delineation": {
            "quality_ok": bool(delin.get("quality_ok", False)),
            "r_peaks":    _valid_int_list(delin.get("r_peaks", [])),
            "p_peaks":    _valid_int_list(d_map.get("ECG_P_Peaks", [])),
            "t_peaks":    _valid_int_list(d_map.get("ECG_T_Peaks", [])),
        },
        "soft_seg":      soft_seg_payload,
    }


@app.post("/infer")
def infer(req: InferRequest):
    resolved_id, model, cfg = get_model_and_cfg(req.model_id)
    sr      = int(cfg.get("data", {}).get("sampling_rate", 500))
    t_fixed = int(cfg.get("data", {}).get("t_fixed", 500))
    lead_i  = int(cfg.get("data", {}).get("lead", 0))
    c_names = _concept_names_for_cfg(cfg)

    cmp_bundle = None
    if req.compare_model_id:
        cmp_id, cmp_model, cmp_cfg = get_model_and_cfg(req.compare_model_id)
        _validate_compare_compatibility(cfg, cmp_cfg, cmp_id)
        cmp_bundle = (cmp_id, cmp_model, cmp_cfg)

    signal = _load_filtered_signal(req.ecg_id, sr)
    C, T = signal.shape
    end = min(T, req.start + t_fixed)
    start = max(0, end - t_fixed)
    window = signal[:, start:end]
    if window.shape[1] < t_fixed:
        pad = np.zeros((C, t_fixed - window.shape[1]), dtype=np.float32)
        window = np.concatenate([window, pad], axis=1)

    win_batch = window[np.newaxis]
    res = _batch_infer_windows(
        model, cfg, win_batch,
        interventions=req.interventions,
        include_epistemic=req.include_epistemic,
    )

    delin = _load_delineation(req.ecg_id, sr, lead_i)
    gt_concepts, gt_task = None, None
    if delin.get("quality_ok", False):
        c, w = get_concept_labels(start, start + t_fixed, delin["delineation"], delin["r_peaks"], delin.get("u_peaks"))
        active = cfg.get("active_concepts", None)
        gt_concepts = [c[i] for i in active] if active is not None else c
        gt_task = int(is_complete_heartbeat_window(c, w))

    payload = {
        "model_id":       resolved_id,
        "model_type":     res["model_type"],
        "task_prob":      res["task_probs"][0],
        "task_std":       res["task_stds"][0],
        "task_epistemic": res["task_epistemic"][0],
        "task_aleatoric": res["task_aleatoric"][0],
        "concept_probs":  res["concept_probs"][0],
        "concept_stds":   res["concept_stds"][0],
        "concept_names":  c_names,
        "gt_concepts":    gt_concepts,
        "gt_task":        gt_task,
    }

    if cmp_bundle is not None:
        cmp_id, cmp_model, cmp_cfg = cmp_bundle
        cmp_res = _batch_infer_windows(
            cmp_model, cmp_cfg, win_batch,
            interventions=req.interventions,
            include_epistemic=req.include_epistemic,
        )
        payload["compare"] = {
            "model_id":       cmp_id,
            "model_type":     cmp_res["model_type"],
            "task_prob":      cmp_res["task_probs"][0],
            "task_std":       cmp_res["task_stds"][0],
            "task_epistemic": cmp_res["task_epistemic"][0],
            "task_aleatoric": cmp_res["task_aleatoric"][0],
            "concept_probs":  cmp_res["concept_probs"][0],
            "concept_stds":   cmp_res["concept_stds"][0],
        }

    return payload


@app.post("/sweep")
def sweep(req: SweepRequest):
    resolved_id, model, cfg = get_model_and_cfg(req.model_id)
    sr      = int(cfg.get("data", {}).get("sampling_rate", 500))
    t_fixed = int(cfg.get("data", {}).get("t_fixed", 500))
    lead_i  = int(cfg.get("data", {}).get("lead", 0))
    c_names = _concept_names_for_cfg(cfg)

    cmp_bundle = None
    if req.compare_model_id:
        cmp_id, cmp_model, cmp_cfg = get_model_and_cfg(req.compare_model_id)
        _validate_compare_compatibility(cfg, cmp_cfg, cmp_id)
        cmp_bundle = (cmp_id, cmp_model, cmp_cfg)

    signal = _load_filtered_signal(req.ecg_id, sr)
    _, T   = signal.shape
    stride = max(5, req.stride)
    starts = list(range(0, T - t_fixed + 1, stride))
    if not starts:
        raise HTTPException(status_code=422, detail="Signal too short for window size.")

    windows = np.stack([signal[:, s : s + t_fixed] for s in starts], axis=0)
    res = _batch_infer_windows(model, cfg, windows)

    delin = _load_delineation(req.ecg_id, sr, lead_i)
    gt_task, gt_concepts = None, None
    active = cfg.get("active_concepts", None)
    if delin.get("quality_ok", False):
        gt_c_list, gt_t_list = [], []
        for s in starts:
            c, w = get_concept_labels(s, s + t_fixed, delin["delineation"], delin["r_peaks"], delin.get("u_peaks"))
            gt_c_list.append([c[i] for i in active] if active is not None else c)
            gt_t_list.append(int(is_complete_heartbeat_window(c, w)))
        gt_concepts = gt_c_list
        gt_task = gt_t_list

    payload = {
        "ecg_id":        req.ecg_id,
        "model_id":      resolved_id,
        "model_type":    res["model_type"],
        "stride":        stride,
        "t_fixed":       t_fixed,
        "sampling_rate": sr,
        "starts":        starts,
        "task_probs":    res["task_probs"],
        "task_stds":     res["task_stds"],
        "concept_probs": res["concept_probs"],
        "concept_stds":  res["concept_stds"],
        "concept_names": c_names,
        "gt_task":       gt_task,
        "gt_concepts":   gt_concepts,
    }

    if cmp_bundle is not None:
        cmp_id, cmp_model, cmp_cfg = cmp_bundle
        cmp_res = _batch_infer_windows(cmp_model, cmp_cfg, windows)
        payload["compare"] = {
            "model_id":      cmp_id,
            "model_type":    cmp_res["model_type"],
            "task_probs":    cmp_res["task_probs"],
            "task_stds":     cmp_res["task_stds"],
            "concept_probs": cmp_res["concept_probs"],
            "concept_stds":  cmp_res["concept_stds"],
        }

    return payload


@app.get("/concepts")
def get_concepts(model_id: Optional[str] = None):
    resolved_id, model, cfg = get_model_and_cfg(model_id)
    c_names = _concept_names_for_cfg(cfg)
    return {
        "model_id":           resolved_id,
        "model_type":         getattr(model, "model_type", "cbm"),
        "concept_names":      c_names,
        "classifier_weights": model.get_concept_importance(c_names),
    }


class DenseInferRequest(BaseModel):
    ecg_id: int
    model_id: Optional[str] = None
    compare_model_id: Optional[str] = None
    blackbox_model_id: Optional[str] = None
    include_epistemic: bool = False
    intervene_mode: Optional[str] = None            # "gt" | "ablate" | "force"
    intervene_concepts: Optional[List[str]] = None  # e.g. ["ST"] or ["QRS"]
    intervene_span: Optional[List[int]] = None      # [start_100, end_100], null = full record


@app.get("/dense-models")
def list_dense_models():
    runs = discover_dense_runs()
    entries = []
    priority = {
        "dense_cem_softseg": 0,
        "dense_cem_s42": 1,
        "dense_cbm_detached_s42": 2,
        "dense_cbm_softseg": 3,
        "dense_cbm_joint_s42": 4,
        "dense_blackbox_s42": 5,
        "dense_multitask_s42": 6,
        "dense_cbm_fiducial24": 7,
        "dense_cem_fiducial24": 8,
    }
    for rid, rdir in runs.items():
        with open(rdir / "config.yaml") as f:
            cfg = yaml.safe_load(f)
        mtype = str(cfg.get("model", {}).get("type", "dense_cem")).lower()
        c_mode = str(cfg.get("data", {}).get("concept_mode", "wave6"))
        metrics = _read_json_if_exists(rdir / "eval_test_metrics.json")
        entries.append({
            "id":                     rid,
            "model_type":             mtype,
            "concept_mode":           c_mode,
            "concept_names":          cfg.get("concept_names", get_dense_concept_names(c_mode)),
            "sampling_rate":          int(cfg.get("data", {}).get("sampling_rate", 100)),
            "has_seg":                mtype != "dense_blackbox",
            "supports_interventions": mtype in ("dense_cbm", "dense_cem"),
            "metrics":                metrics,
        })
    entries.sort(key=lambda e: (priority.get(e["id"], 20), e["id"]))
    by_id = {e["id"]: e for e in entries}
    default_id = next((c for c in ("dense_cem_softseg", "dense_cem_s42") if c in by_id), None)
    if default_id is None:
        default_id = next((e["id"] for e in entries if e["has_seg"]), entries[0]["id"] if entries else None)
    compare_id = next(
        (c for c in ("dense_cbm_detached_s42", "dense_cbm_softseg", "dense_cbm_joint_s42") if c in by_id and c != default_id),
        next((e["id"] for e in entries if e["id"] != default_id and e["has_seg"]), None),
    )
    blackbox_id = next(
        (c for c in ("dense_blackbox_s42",) if c in by_id),
        next((e["id"] for e in entries if e["model_type"] == "dense_blackbox"), None),
    )
    return {
        "default_model":    default_id,
        "compare_default":  compare_id,
        "blackbox_default": blackbox_id,
        "models":           entries,
    }


@app.post("/dense-infer")
def dense_infer(req: DenseInferRequest):
    resolved_id, model, cfg = get_dense_model_and_cfg(req.model_id)
    sr = int(cfg.get("data", {}).get("sampling_rate", 100))
    c_mode = str(cfg.get("data", {}).get("concept_mode", "wave6"))
    c_names = list(cfg.get("concept_names", get_dense_concept_names(c_mode)))

    sig = z_normalize(_load_filtered_signal(req.ecg_id, sr))
    T = sig.shape[-1]
    x = torch.from_numpy(sig).unsqueeze(0).to(DEVICE)

    # Cache Zenodo GT soft-segmentation per requested dense concept mode
    archive = _get_soft_seg_archive()
    raw_ss = None
    if archive is not None and req.ecg_id in archive:
        try:
            raw_ss = archive.load(req.ecg_id)
        except Exception as exc:
            logger.warning(f"Could not load soft_seg GT for ecg_id={req.ecg_id}: {exc}")

    gt_by_mode: Dict[str, Optional[np.ndarray]] = {}

    def _get_gt(mode_name: str) -> Optional[np.ndarray]:
        if mode_name not in gt_by_mode:
            if raw_ss is None:
                gt_by_mode[mode_name] = None
            else:
                try:
                    gt_by_mode[mode_name] = collapse_soft_seg_targets(raw_ss, concept_mode=mode_name, lead_idx=None)
                except Exception as exc:
                    logger.warning(f"Could not collapse soft_seg GT ({mode_name}) for ecg_id={req.ecg_id}: {exc}")
                    gt_by_mode[mode_name] = None
        return gt_by_mode[mode_name]

    gt_arr = _get_gt(c_mode)

    # Build optional per-timestamp intervention mask & target labels
    span_lo, span_hi = 0, T
    if req.intervene_span and len(req.intervene_span) == 2:
        span_lo = max(0, min(T, int(req.intervene_span[0])))
        span_hi = max(span_lo, min(T, int(req.intervene_span[1])))

    def _run_one(m: torch.nn.Module, m_cfg: dict) -> dict:
        mtype = getattr(m, "model_type", "dense_cem")
        m_has_seg = (mtype != "dense_blackbox")
        m_can_int = mtype in ("dense_cbm", "dense_cem")
        m_mode = str(m_cfg.get("data", {}).get("concept_mode", c_mode))
        m_c_names = list(m_cfg.get("concept_names", get_dense_concept_names(m_mode)))
        m_gt = _get_gt(m_mode)

        with torch.no_grad():
            base_out = m(x)
            base_tp = torch.sigmoid(base_out["task_logits"][0].float()).cpu().numpy()

            il, im = None, None
            mode = (req.intervene_mode or "").lower()
            if m_can_int:
                if mode == "gt" and m_gt is not None:
                    il = torch.from_numpy(m_gt).unsqueeze(0).to(DEVICE)
                    im = torch.zeros((1, 1, T), device=DEVICE)
                    im[:, :, span_lo:span_hi] = 1.0
                elif mode in ("ablate", "force") and req.intervene_concepts:
                    il = base_out["concept_probs"].clone()
                    im = torch.zeros_like(il)
                    has_match = False
                    for cname in req.intervene_concepts:
                        if cname not in m_c_names:
                            continue
                        has_match = True
                        k = m_c_names.index(cname)
                        if mode == "ablate":
                            il[:, k, span_lo:span_hi] = 0.0
                            im[:, k, span_lo:span_hi] = 1.0
                        else:
                            il[:, :, span_lo:span_hi] = 0.0
                            il[:, k, span_lo:span_hi] = 1.0
                            im[:, :, span_lo:span_hi] = 1.0
                    if not has_match:
                        il, im = None, None

            if not m_has_seg:
                if req.include_epistemic:
                    out = m.predict_with_uncertainty(x, n_mc_epistemic=8, n_mc_aleatoric=1)
                    tp = out["task_probs_mean"][0].cpu().numpy()
                    tp_std = out["task_probs_std"][0].cpu().numpy()
                else:
                    tp = base_tp
                    tp_std = np.zeros(len(SUPERCLASS_NAMES), dtype=np.float32)
                return {
                    "has_seg":                False,
                    "supports_interventions": False,
                    "concept_names":          m_c_names,
                    "concept_probs":          None,
                    "concept_stds":           None,
                    "mean_std_timeline":      None,
                    "pathology":              {name: round(float(tp[i]), 4) for i, name in enumerate(SUPERCLASS_NAMES)},
                    "base_pathology":         {name: round(float(base_tp[i]), 4) for i, name in enumerate(SUPERCLASS_NAMES)},
                    "pathology_std":          {name: round(float(tp_std[i]), 4) for i, name in enumerate(SUPERCLASS_NAMES)},
                }

            if req.include_epistemic:
                out = m.predict_with_uncertainty(
                    x, n_mc_epistemic=8, n_mc_aleatoric=12,
                    intervene_labels=il, intervene_mask=im,
                )
                cp = out["concept_probs_mean"][0].cpu().numpy()
                cs = out["concept_total_std"][0].cpu().numpy()
                tp = out["task_probs_mean"][0].cpu().numpy()
                tp_std = out["task_probs_std"][0].cpu().numpy()
            else:
                out = m(x, intervene_labels=il, intervene_mask=im) if im is not None else base_out
                cp = out["concept_probs"][0].float().cpu().numpy()
                lv = out["concept_log_vars"][0].float().cpu().numpy()
                cs = np.clip(cp * (1.0 - cp) * np.exp(0.5 * np.clip(lv, -10.0, 4.0)), 0.0, 0.5)
                if im is not None:
                    m_np = im[0].cpu().numpy()
                    if m_np.shape[0] == 1:
                        m_np = np.broadcast_to(m_np, cs.shape)
                    cs = np.where(m_np > 0, 0.0, cs)
                tp = torch.sigmoid(out["task_logits"][0].float()).cpu().numpy()
                tp_std = np.zeros(len(SUPERCLASS_NAMES), dtype=np.float32)

        return {
            "has_seg":                True,
            "supports_interventions": m_can_int,
            "concept_names":          m_c_names,
            "concept_probs":          {name: np.round(cp[i], 3).tolist() for i, name in enumerate(m_c_names)},
            "concept_stds":           {name: np.round(cs[i], 3).tolist() for i, name in enumerate(m_c_names)},
            "mean_std_timeline":      np.round(cs.mean(axis=0), 4).tolist(),
            "pathology":              {name: round(float(tp[i]), 4) for i, name in enumerate(SUPERCLASS_NAMES)},
            "base_pathology":         {name: round(float(base_tp[i]), 4) for i, name in enumerate(SUPERCLASS_NAMES)},
            "pathology_std":          {name: round(float(tp_std[i]), 4) for i, name in enumerate(SUPERCLASS_NAMES)},
        }

    prim = _run_one(model, cfg)
    payload = {
        "ecg_id":         req.ecg_id,
        "model_id":       resolved_id,
        "model_type":     getattr(model, "model_type", "dense_cem"),
        "sampling_rate":  sr,
        "gt_probs":       ({name: np.round(gt_arr[i], 3).tolist() for i, name in enumerate(c_names)} if gt_arr is not None else None),
        "intervene_mode": req.intervene_mode,
        **prim,
    }
    if req.compare_model_id:
        cmp_id, cmp_model, cmp_cfg = get_dense_model_and_cfg(req.compare_model_id)
        payload["compare"] = {
            "model_id":   cmp_id,
            "model_type": getattr(cmp_model, "model_type", "dense_cbm"),
            **_run_one(cmp_model, cmp_cfg),
        }
    if req.blackbox_model_id:
        bb_id, bb_model, bb_cfg = get_dense_model_and_cfg(req.blackbox_model_id)
        payload["blackbox"] = {
            "model_id":   bb_id,
            "model_type": getattr(bb_model, "model_type", "dense_blackbox"),
            **_run_one(bb_model, bb_cfg),
        }
    return payload


@app.get("/health")
def health():
    runs = discover_runs()
    return {
        "status":        "ok",
        "device":        DEVICE,
        "runs_dir":      str(RUNS_DIR),
        "data_path":     str(DATA_PATH),
        "n_models":      len(runs),
        "default_model": DEFAULT_RUN_KEY,
    }
