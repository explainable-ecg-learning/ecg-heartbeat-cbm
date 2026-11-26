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
import copy
import json
import logging
import math
import os
import pickle
import sys
import threading
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

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
from scipy.signal import resample_poly

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
from external_data import (  # noqa: E402
    CANONICAL_LEADS,
    canonicalize_leads,
    extract_wave_spans_and_intervals,
    load_georgia_metadata,
    map_snomed_to_superclasses,
)
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


def _load_external_paths() -> Tuple[Path, Path]:
    cfg: dict = {}
    cfg_file = REPO_ROOT / "config.yaml"
    if cfg_file.exists():
        with open(cfg_file) as f:
            cfg = yaml.safe_load(f) or {}
    ext_cfg = cfg.get("external_datasets", {}).get("georgia_ecg", {})
    raw_ga = os.environ.get("ECG_GEORGIA_DIR", ext_cfg.get("root", "data/georgia_ecg"))
    ga_path = Path(raw_ga)
    if not ga_path.is_absolute():
        ga_path = REPO_ROOT / ga_path
    bench_path = REPO_ROOT / "experiments" / "results" / "external_georgia_eval.json"
    return ga_path, bench_path


GEORGIA_ROOT, EXTERNAL_BENCH_PATH = _load_external_paths()
_raw_precalc = os.environ.get("ECG_PRECALC_DIR", str(Path(__file__).resolve().parent / "vercel" / "data"))
PRECALC_DIR = Path(_raw_precalc) if Path(_raw_precalc).is_absolute() else (REPO_ROOT / _raw_precalc)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
logger.info(
    f"Explorer starting | device={DEVICE} | runs={RUNS_DIR} | data={DATA_PATH} | "
    f"georgia={GEORGIA_ROOT} | precalc={PRECALC_DIR}"
)

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
# Georgia 12-Lead ECG (External OOD) Metadata, Signal & Delineation Cache
# ─────────────────────────────────────────────────────────────────────────────

GEORGIA_RHYTHM_GROUPS: Dict[str, set] = {
    "AFIB": {"164889003", "164890007"},
    "BBB":  {"164909002", "251120003", "59118001", "713427006", "713426002", "6374002"},
    "IAVB": {"270492004"},
    "LQT":  {"111975006"},
    "PVC":  {"17338001", "284470004", "63593006", "427172004"},
}
GEORGIA_SHOWCASE_IDS: List[str] = ["E00021", "E00005", "E10019"]

_GEORGIA_META: Optional[pd.DataFrame] = None
_GEORGIA_LOCK = threading.Lock()
_PRECALC_MANIFEST: Optional[dict] = None
_PRECALC_MANIFEST_PATH: Optional[Path] = None
_PRECALC_LOCK = threading.Lock()


def _get_precalc_manifest() -> Optional[dict]:
    global _PRECALC_MANIFEST, _PRECALC_MANIFEST_PATH
    mpath = PRECALC_DIR / "manifest.json"
    with _PRECALC_LOCK:
        if _PRECALC_MANIFEST is None or _PRECALC_MANIFEST_PATH != mpath:
            _PRECALC_MANIFEST = _read_json_if_exists(mpath)
            _PRECALC_MANIFEST_PATH = mpath
        return _PRECALC_MANIFEST


def _unpack_precalc_record(raw: dict) -> dict:
    """Unpack compact quantized static bundle arrays in Python."""
    if not raw or raw.get("_unpacked"):
        return raw
    raw["_unpacked"] = True

    for sw in (raw.get("sweeps") or {}).values():
        n_win = len(sw.get("task_probs") or [])
        k_cnt = len(sw.get("concept_names") or [])
        if sw.get("concept_probs") is None:
            sw["concept_probs"] = [[0.0] * k_cnt for _ in range(n_win)]
            sw["concept_stds"] = [[0.0] * k_cnt for _ in range(n_win)]
            sw["concept_stds_mc"] = [[0.0] * k_cnt for _ in range(n_win)]
        elif sw.get("_cstd_scale"):
            inv = 1.0 / float(sw["_cstd_scale"])
            if sw.get("concept_stds") is not None:
                sw["concept_stds"] = [[float(v) * inv for v in row] for row in sw["concept_stds"]]
            if sw.get("concept_stds_mc") is not None:
                sw["concept_stds_mc"] = [[float(v) * inv for v in row] for row in sw["concept_stds_mc"]]

    gt_scale = raw.get("dense_prob_scale") or 0
    gt_probs = raw.get("dense_gt_probs")
    if gt_scale and isinstance(gt_probs, dict):
        inv = 1.0 / float(gt_scale)
        non_tp = [k for k in ("P", "PQ", "QRS", "ST", "T") if isinstance(gt_probs.get(k), list)]
        for k in non_tp:
            gt_probs[k] = [float(v) * inv for v in gt_probs[k]]
        if non_tp and "TP" not in gt_probs:
            t_seg = len(gt_probs[non_tp[0]])
            gt_probs["TP"] = [
                max(0.0, 1.0 - sum(gt_probs[k][t] for k in non_tp)) for t in range(t_seg)
            ]
    rec = raw.get("record")
    if isinstance(rec, dict) and rec.get("soft_seg") is None and isinstance(gt_probs, dict):
        rec["soft_seg"] = {**gt_probs, "sampling_rate": 100}

    for de in (raw.get("dense") or {}).values():
        if not de or not de.get("has_seg"):
            continue
        c_names = de.get("concept_names") or ["P", "PQ", "QRS", "ST", "T", "TP"]
        non_tp = [n for n in c_names if n != "TP"]
        p_scale = de.get("_prob_scale") or 0
        s_scale = de.get("_std_scale") or 0
        cp_map = de.get("concept_probs")
        cs_map = de.get("concept_stds")
        if p_scale and isinstance(cp_map, dict):
            inv_p = 1.0 / float(p_scale)
            for n in non_tp:
                if isinstance(cp_map.get(n), list):
                    cp_map[n] = [float(v) * inv_p for v in cp_map[n]]
            t_seg = len(cp_map[non_tp[0]]) if (non_tp and isinstance(cp_map.get(non_tp[0]), list)) else 0
            if t_seg and "TP" in c_names and "TP" not in cp_map:
                cp_map["TP"] = [
                    max(0.0, 1.0 - sum(cp_map[n][t] for n in non_tp)) for t in range(t_seg)
                ]
            if isinstance(cs_map, dict):
                for n in non_tp:
                    if isinstance(cs_map.get(n), list):
                        cs_map[n] = [float(v) * inv_p for v in cs_map[n]]
                if t_seg and "TP" in c_names and "TP" not in cs_map:
                    cs_map["TP"] = [0.0] * t_seg
        if s_scale:
            inv_s = 1.0 / float(s_scale)
            if isinstance(de.get("mean_std_timeline"), list):
                de["mean_std_timeline"] = [float(v) * inv_s for v in de["mean_std_timeline"]]
            if isinstance(de.get("mean_std_timeline_mc"), list):
                de["mean_std_timeline_mc"] = [float(v) * inv_s for v in de["mean_std_timeline_mc"]]

    return raw


@lru_cache(maxsize=64)
def _load_precalc_record_cached(precalc_dir: str, ds: str, ecg_id: str) -> Optional[dict]:
    p = Path(precalc_dir) / "records" / f"{ds}_{ecg_id}.json"
    if not p.exists():
        return None
    try:
        with open(p) as f:
            raw = json.load(f)
        return _unpack_precalc_record(raw)
    except Exception as exc:
        logger.warning(f"Failed loading precalc record {p}: {exc}")
        return None


def _load_precalc_record(ds: str, ecg_id: Union[int, str]) -> Optional[dict]:
    return _load_precalc_record_cached(str(PRECALC_DIR), str(ds), str(ecg_id))


def _eval_precalc_window_infer(
    sw: dict,
    head: Optional[dict],
    ni: int,
    interventions: Optional[Dict[Union[int, str], float]],
    include_epistemic: bool = False,
) -> dict:
    c_probs = list(sw["concept_probs"][ni])
    c_std_src = sw["concept_stds_mc"][ni] if (include_epistemic and sw.get("concept_stds_mc")) else sw["concept_stds"][ni]
    c_stds = list(c_std_src)
    t_std = sw["task_stds_mc"][ni] if (include_epistemic and sw.get("task_stds_mc")) else sw["task_stds"][ni]
    t_epi = sw["task_epistemic_mc"][ni] if (include_epistemic and sw.get("task_epistemic_mc")) else 0.0
    has_int = bool(interventions)
    if not has_int or not head or head.get("model_type") == "blackbox":
        return {
            "task_prob":      float(sw["task_probs"][ni]),
            "task_std":       float(t_std),
            "task_epistemic": float(t_epi),
            "task_aleatoric": float(sw["task_stds"][ni]),
            "concept_probs":  c_probs,
            "concept_stds":   c_stds,
        }

    inactive = set(head.get("inactive_concepts") or [])
    k_cnt = len(c_probs)
    logit = float(head.get("bias", 0.0))
    int_map = {int(k): float(v) for k, v in interventions.items()}

    if head["model_type"] == "cem":
        a_pos = sw["a_pos"][ni]
        a_neg = sw["a_neg"][ni]
        for k in range(k_cnt):
            if k in inactive:
                p_eff = 0.0
            elif k in int_map:
                p_eff = int_map[k]
                c_probs[k] = p_eff
                c_stds[k] = 0.0
            else:
                p_eff = c_probs[k]
            logit += p_eff * a_pos[k] + (1.0 - p_eff) * a_neg[k]
    else:
        w_prob = head.get("w_prob") or []
        w_var = head.get("w_var") or []
        use_unc = bool(head.get("use_uncertainty", True))
        c_vars = sw["c_vars"][ni] if sw.get("c_vars") else [0.0] * k_cnt
        v_zero = math.exp(-20.0)
        for k in range(k_cnt):
            if k in inactive:
                p_eff, v_eff = 0.0, 0.0
            elif k in int_map:
                p_eff, v_eff = int_map[k], v_zero
                c_probs[k] = p_eff
                c_stds[k] = 0.0
            else:
                p_eff, v_eff = c_probs[k], c_vars[k]
            logit += (w_prob[k] if k < len(w_prob) else 0.0) * p_eff + (
                (w_var[k] if k < len(w_var) else 0.0) * v_eff if use_unc else 0.0
            )

    t_prob = 1.0 / (1.0 + math.exp(-max(-30.0, min(30.0, logit))))
    return {
        "task_prob":      round(float(t_prob), 4),
        "task_std":       0.0,
        "task_epistemic": 0.0,
        "task_aleatoric": 0.0,
        "concept_probs":  c_probs,
        "concept_stds":   c_stds,
    }


def _build_precalc_dense_entry(
    precalc: dict,
    model_id: str,
    include_epistemic: bool,
    int_mode: Optional[str],
    int_concepts: Optional[List[str]],
    int_span: Optional[List[int]] = None,
) -> Optional[dict]:
    de = (precalc.get("dense") or {}).get(model_id)
    if not de:
        return None
    if not de.get("has_seg"):
        return {
            "model_id":               model_id,
            "model_type":             de["model_type"],
            "has_seg":                False,
            "supports_interventions": False,
            "concept_names":          list(de.get("concept_names") or []),
            "concept_probs":          None,
            "concept_stds":           None,
            "seg_spans":              None,
            "mean_std_timeline":      None,
            "intervals":              None,
            "pathology":              dict(de["base_pathology"]),
            "base_pathology":         dict(de["base_pathology"]),
            "pathology_std":          dict(
                de["pathology_std_mc"] if (include_epistemic and de.get("pathology_std_mc")) else de["pathology_std"]
            ),
        }

    c_names = list(de.get("concept_names") or ["P", "PQ", "QRS", "ST", "T", "TP"])
    cp_out = {n: list(de["concept_probs"][n]) for n in c_names}
    cs_out = {n: list(de["concept_stds"][n]) for n in c_names} if de.get("concept_stds") else None
    std_src = de["mean_std_timeline_mc"] if (include_epistemic and de.get("mean_std_timeline_mc")) else de["mean_std_timeline"]
    std_line = list(std_src)
    patho = dict(de["base_pathology"])
    patho_keys = list(de["base_pathology"].keys())

    mode = (int_mode or "").lower()
    t_len = len(cp_out[c_names[0]]) if c_names and cp_out.get(c_names[0]) else 0
    is_win = isinstance(int_span, list) and len(int_span) == 2
    lo = max(0, min(t_len, int(round(int_span[0])))) if is_win else 0
    hi = max(lo, min(t_len, int(round(int_span[1])))) if is_win else t_len
    bin_idx = max(0, min(9, (lo + hi) // 200)) if is_win else -1

    def _map_win_patho(arr: List[float]) -> dict:
        return {name: float(arr[idx]) for idx, name in enumerate(patho_keys)}

    if de.get("supports_interventions") and mode:
        gt_probs = precalc.get("dense_gt_probs")
        if mode == "gt" and gt_probs:
            if bin_idx >= 0 and (de.get("interventions_win") or {}).get("gt"):
                patho = _map_win_patho(de["interventions_win"]["gt"][bin_idx])
            elif (de.get("interventions") or {}).get("gt"):
                patho = dict(de["interventions"]["gt"])
            for t in range(lo, hi):
                for n in c_names:
                    cp_out[n][t] = gt_probs[n][t]
                    if cs_out is not None:
                        cs_out[n][t] = 0.0
                std_line[t] = 0.0
        elif mode in ("ablate", "force") and int_concepts:
            target = int_concepts[0]
            ikey = f"{mode}:{target}"
            if bin_idx >= 0 and (de.get("interventions_win") or {}).get(ikey):
                patho = _map_win_patho(de["interventions_win"][ikey][bin_idx])
            elif (de.get("interventions") or {}).get(ikey):
                patho = dict(de["interventions"][ikey])
            if mode == "ablate" and target in cp_out:
                for t in range(lo, hi):
                    cp_out[target][t] = 0.0
                    s = sum(cp_out[n][t] for n in c_names)
                    denom = max(1e-6, s)
                    for n in c_names:
                        cp_out[n][t] = round(cp_out[n][t] / denom, 3)
                    if cs_out is not None and target in cs_out:
                        cs_out[target][t] = 0.0
            elif mode == "force" and target in cp_out:
                for t in range(lo, hi):
                    for n in c_names:
                        cp_out[n][t] = 1.0 if n == target else 0.0
                        if cs_out is not None:
                            cs_out[n][t] = 0.0
                    std_line[t] = 0.0

    return {
        "model_id":               model_id,
        "model_type":             de["model_type"],
        "has_seg":                True,
        "supports_interventions": bool(de.get("supports_interventions")),
        "concept_names":          c_names,
        "concept_probs":          cp_out,
        "concept_stds":           cs_out,
        "seg_spans":              (None if mode else de.get("seg_spans")),
        "mean_std_timeline":      std_line,
        "intervals":              de.get("intervals"),
        "pathology":              patho,
        "base_pathology":         dict(de["base_pathology"]),
        "pathology_std":          dict(
            de["pathology_std_mc"] if (include_epistemic and de.get("pathology_std_mc")) else de["pathology_std"]
        ),
    }


def _has_georgia_raw() -> bool:
    return (GEORGIA_ROOT / "quality" / "error_labels.csv").exists() and (GEORGIA_ROOT / "data").exists()


def _has_georgia_precalc() -> bool:
    manifest = _get_precalc_manifest()
    ga_recs = (manifest or {}).get("records_by_dataset", {}).get("georgia", [])
    if not ga_recs:
        return False
    rec_dir = PRECALC_DIR / "records"
    return any((rec_dir / f"georgia_{r['ecg_id']}.json").exists() for r in ga_recs)


def _has_georgia() -> bool:
    return _has_georgia_raw() or _has_georgia_precalc()


def _get_georgia_meta() -> pd.DataFrame:
    global _GEORGIA_META
    with _GEORGIA_LOCK:
        if _GEORGIA_META is None:
            if not _has_georgia():
                raise HTTPException(status_code=503, detail=f"Georgia 12-Lead ECG not found under {GEORGIA_ROOT}")
            df = load_georgia_metadata(GEORGIA_ROOT, max_error_severity=None, require_10s_500hz=True)

            dx_lists, label_lists, tag_sets = [], [], []
            sex_ints, sex_labels = [], []
            for _, row in df.iterrows():
                scodes_str = str(row.get("snomed_codes", ""))
                cset = {c.strip() for c in scodes_str.split("|") if c.strip()}
                sc_vec = map_snomed_to_superclasses(scodes_str)
                dx = [SUPERCLASS_NAMES[i] for i, v in enumerate(sc_vec) if v > 0.5]
                lbls = [s.strip() for s in str(row.get("label_names", "")).split("|") if s.strip()]
                tags = set(dx)
                for grp, kws in GEORGIA_RHYTHM_GROUPS.items():
                    if not cset.isdisjoint(kws):
                        tags.add(grp)

                raw_sex = str(row.get("sex", "")).strip().lower()
                if raw_sex.startswith("m"):
                    sx_i, sx_l = 0, "M"
                elif raw_sex.startswith("f"):
                    sx_i, sx_l = 1, "F"
                else:
                    sx_i, sx_l = -1, ""

                dx_lists.append(dx)
                label_lists.append(lbls)
                tag_sets.append(tags)
                sex_ints.append(sx_i)
                sex_labels.append(sx_l)

            df["_dx_list"] = dx_lists
            df["_snomed_labels"] = label_lists
            df["_tag_set"] = tag_sets
            df["_sex_int"] = sex_ints
            df["_sex_label"] = sex_labels
            df = df.set_index("record_id", drop=False)
            _GEORGIA_META = df
            logger.info(f"Loaded Georgia 12-Lead metadata: {len(_GEORGIA_META)} valid 10s records")
        return _GEORGIA_META


@lru_cache(maxsize=16)
def _load_georgia_filtered_signal(record_id: str, sampling_rate: int = 500) -> np.ndarray:
    meta = _get_georgia_meta()
    rid = str(record_id)
    if rid not in meta.index:
        raise HTTPException(status_code=404, detail=f"Georgia record '{rid}' not found")
    row = meta.loc[rid]
    rec_path = str(GEORGIA_ROOT / "data" / str(row["record_path"]))
    raw, fields = wfdb.rdsamp(rec_path)
    src_fs = int(fields.get("fs", row.get("fs", 500)))
    canon = canonicalize_leads(raw.T, fields.get("sig_name", CANONICAL_LEADS))
    canon = np.nan_to_num(canon.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    if src_fs != sampling_rate:
        if src_fs == 500 and sampling_rate == 100:
            canon = resample_poly(canon, 1, 5, axis=-1).astype(np.float32)
        else:
            from fractions import Fraction
            frac = Fraction(int(sampling_rate), int(src_fs))
            canon = resample_poly(canon, frac.numerator, frac.denominator, axis=-1).astype(np.float32)
    filtered = bandpass_filter(canon, sampling_rate=sampling_rate, lowcut=0.5, highcut=40.0)
    filtered.setflags(write=False)
    return filtered


@lru_cache(maxsize=32)
def _load_georgia_delineation(record_id: str, sampling_rate: int = 500, lead_idx: int = 1) -> dict:
    cache_file = DELIN_CACHE_DIR / f"georgia_{record_id}_sr{sampling_rate}_l{lead_idx}.pkl"
    if cache_file.exists():
        try:
            with open(cache_file, "rb") as f:
                return pickle.load(f)
        except Exception:
            pass
    sig = _load_georgia_filtered_signal(record_id, sampling_rate)
    delin = delineate_record(sig, sampling_rate=sampling_rate, lead_idx=lead_idx)
    if DELIN_CACHE_DIR.exists() and os.access(DELIN_CACHE_DIR, os.W_OK):
        try:
            with open(cache_file, "wb") as f:
                pickle.dump(delin, f)
        except OSError:
            pass
    return delin


def _normalize_dataset(dataset: Optional[str]) -> str:
    ds = (dataset or "ptbxl").strip().lower()
    if ds in ("ptbxl", "ptb-xl"):
        return "ptbxl"
    if ds in ("georgia", "georgia_ecg"):
        if not _has_georgia():
            raise HTTPException(status_code=503, detail=f"Georgia dataset not available under {GEORGIA_ROOT}")
        return "georgia"
    raise HTTPException(status_code=400, detail=f"Unknown dataset '{dataset}'. Expected 'ptbxl' or 'georgia'.")


def _parse_ptbxl_id(ecg_id: Union[int, str]) -> int:
    try:
        return int(ecg_id)
    except (TypeError, ValueError):
        raise HTTPException(status_code=422, detail=f"Invalid PTB-XL ecg_id '{ecg_id}'; expected integer.")


def _load_filtered_signal_any(ecg_id: Union[int, str], sampling_rate: int = 500, dataset: str = "ptbxl") -> np.ndarray:
    ds = _normalize_dataset(dataset)
    if ds == "georgia":
        return _load_georgia_filtered_signal(str(ecg_id), sampling_rate)
    return _load_filtered_signal(_parse_ptbxl_id(ecg_id), sampling_rate)


def _load_delineation_any(
    ecg_id: Union[int, str], sampling_rate: int = 500, lead_idx: int = 0, dataset: str = "ptbxl"
) -> dict:
    ds = _normalize_dataset(dataset)
    if ds == "georgia":
        eff_lead = 1 if lead_idx == 0 else lead_idx
        return _load_georgia_delineation(str(ecg_id), sampling_rate, eff_lead)
    return _load_delineation(_parse_ptbxl_id(ecg_id), sampling_rate, lead_idx)


def _safe_round(val: Optional[float], digits: int = 1) -> Optional[float]:
    if val is None or not np.isfinite(val):
        return None
    return round(float(val), digits)


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
    ecg_id: Union[int, str]
    dataset: str = "ptbxl"
    start: int
    model_id: Optional[str] = None
    compare_model_id: Optional[str] = None
    interventions: Optional[Dict[int, float]] = None
    include_epistemic: bool = False


class SweepRequest(BaseModel):
    ecg_id: Union[int, str]
    dataset: str = "ptbxl"
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


@app.get("/datasets")
def list_datasets():
    datasets = [
        {
            "id":              "ptbxl",
            "label":           "PTB-XL (German)",
            "split_label":     "Split",
            "has_soft_seg_gt": _get_soft_seg_archive() is not None,
            "default_record":  "9",
            "default_split":   "10",
            "splits": [
                {"value": "10",  "label": "Test (Fold 10)"},
                {"value": "9",   "label": "Val (Fold 9)"},
                {"value": "all", "label": "All Folds"},
            ],
            "conditions": ["ALL", "NORM", "MI", "STTC", "CD", "HYP", "AFIB", "BBB", "PVC"],
        }
    ]
    if _has_georgia():
        datasets.append({
            "id":              "georgia",
            "label":           "Georgia 12-Lead (US · OOD)",
            "split_label":     "Quality",
            "has_soft_seg_gt": False,
            "default_record":  GEORGIA_SHOWCASE_IDS[0],
            "default_split":   "0_clean",
            "splits": [
                {"value": "0_clean",            "label": "Clean (0_clean)"},
                {"value": "1_minor_warning",    "label": "Minor Warning (1)"},
                {"value": "2_moderate_anomaly", "label": "Moderate Anomaly (2)"},
                {"value": "3_severe_error",     "label": "Severe Error (3)"},
                {"value": "all",                "label": "All Valid 10s"},
            ],
            "conditions": ["ALL", "NORM", "STTC", "CD", "HYP", "MI", "AFIB", "BBB", "IAVB", "LQT", "PVC"],
        })
    return {"default_dataset": "ptbxl", "datasets": datasets}


@app.get("/external-benchmark")
def get_external_benchmark():
    data = _read_json_if_exists(EXTERNAL_BENCH_PATH)
    if data is None:
        data = _read_json_if_exists(PRECALC_DIR / "external_benchmark.json")
    if data is None:
        return {"available": False}
    return {"available": True, **data}


@app.get("/records")
def list_records(
    limit: int = 300,
    fold: Optional[str] = "10",
    dx: Optional[str] = None,
    search: Optional[str] = None,
    dataset: str = "ptbxl",
):
    ds = _normalize_dataset(dataset)
    if ds == "georgia":
        if not _has_georgia_raw():
            manifest = _get_precalc_manifest()
            all_recs = list((manifest or {}).get("records_by_dataset", {}).get("georgia", []))
            subset_recs = all_recs
            fval = (fold or "0_clean").strip()
            if fval.lower() not in ("all", ""):
                if fval == "10":
                    fval = "0_clean"
                subset_recs = [r for r in subset_recs if str(r.get("severity_tier", "")) == fval]
            if dx and dx.upper() != "ALL":
                tag = dx.upper()
                subset_recs = [r for r in subset_recs if tag in (r.get("tags") or [])]
            q = (search or "").strip().lower()
            if q:
                subset_recs = [
                    r for r in subset_recs
                    if q in str(r.get("ecg_id", "")).lower()
                    or q in " ".join(r.get("label_names") or []).lower()
                ]
            total = len(subset_recs)
            head_recs = subset_recs[:limit]
            if not q and (not dx or dx.upper() == "ALL") and fval in ("0_clean", "all", ""):
                by_id = {str(r["ecg_id"]): r for r in subset_recs}
                pinned = [by_id[rid] for rid in GEORGIA_SHOWCASE_IDS if rid in by_id]
                if pinned:
                    pin_ids = {str(r["ecg_id"]) for r in pinned}
                    rest = [r for r in head_recs if str(r["ecg_id"]) not in pin_ids][: max(0, limit - len(pinned))]
                    head_recs = pinned + rest
            return {
                "dataset":    "georgia",
                "record_ids": [str(r["ecg_id"]) for r in head_recs],
                "records":    head_recs,
                "total":      total,
            }
        meta = _get_georgia_meta()
        subset = meta
        fval = (fold or "0_clean").strip()
        if fval.lower() not in ("all", ""):
            if fval == "10":
                fval = "0_clean"
            subset = subset[subset["severity_tier"].astype(str) == fval]

        if dx and dx.upper() != "ALL":
            tag = dx.upper()
            subset = subset[subset["_tag_set"].apply(lambda s: tag in s)]

        q = (search or "").strip().lower()
        if q:
            m_id = subset["record_id"].astype(str).str.lower().str.contains(q, regex=False)
            m_dx = subset["label_names"].astype(str).str.lower().str.contains(q, regex=False)
            subset = subset[m_id | m_dx]

        total = len(subset)
        sub_head = subset.iloc[:limit]

        # Pin curated NSR / AF / LBBB showcase records at the top of default 0_clean view
        if not q and (not dx or dx.upper() == "ALL") and fval in ("0_clean", "all", ""):
            pinned = [rid for rid in GEORGIA_SHOWCASE_IDS if rid in subset.index]
            if pinned:
                rest = sub_head.loc[~sub_head.index.isin(pinned)].iloc[: max(0, limit - len(pinned))]
                sub_head = pd.concat([subset.loc[pinned], rest])

        records = [
            {
                "ecg_id":        str(row["record_id"]),
                "dataset":       "georgia",
                "severity_tier": str(row["severity_tier"]),
                "age":           float(row["age"]) if pd.notna(row.get("age")) else 0.0,
                "sex":           int(row["_sex_int"]),
                "sex_label":     str(row["_sex_label"]),
                "dx":            row["_dx_list"],
                "label_names":   row["_snomed_labels"],
                "tags":          sorted(row["_tag_set"]),
                "est_hr_bpm":    _safe_round(row.get("est_hr_bpm"), 1),
            }
            for _, row in sub_head.iterrows()
        ]
        return {
            "dataset":    "georgia",
            "record_ids": [r["ecg_id"] for r in records],
            "records":    records,
            "total":      total,
        }

    meta = _get_meta()
    if fold is not None and str(fold).strip().lower() not in ("all", ""):
        try:
            subset = meta[meta["strat_fold"] == int(fold)]
        except ValueError:
            raise HTTPException(status_code=400, detail=f"Invalid PTB-XL fold '{fold}'.")
    else:
        subset = meta

    if dx and dx.upper() != "ALL":
        tag = dx.upper()
        subset = subset[subset["_tag_set"].apply(lambda s: tag in s)]

    if search and search.strip():
        q = search.strip()
        subset = subset[subset.index.astype(str).str.contains(q)]

    sub_head = subset.iloc[:limit]
    records = [
        {
            "ecg_id":  int(eid),
            "dataset": "ptbxl",
            "fold":    int(row.get("strat_fold", 0)),
            "age":     float(row["age"]) if not pd.isna(row.get("age")) else 0.0,
            "sex":     int(row["sex"]) if not pd.isna(row.get("sex")) else 0,
            "dx":      row["_dx_list"],
            "tags":    sorted(row["_tag_set"]),
        }
        for eid, row in sub_head.iterrows()
    ]
    return {
        "dataset":    "ptbxl",
        "record_ids": [r["ecg_id"] for r in records],
        "records":    records,
        "total":      len(subset),
    }


@app.get("/record/{ecg_id}")
def get_record(ecg_id: str, model_id: Optional[str] = None, dataset: str = "ptbxl"):
    ds_name = _normalize_dataset(dataset)
    if ds_name == "georgia" and not _has_georgia_raw():
        precalc = _load_precalc_record("georgia", str(ecg_id))
        if precalc is None or not precalc.get("record"):
            raise HTTPException(
                status_code=404,
                detail=f"Georgia record '{ecg_id}' not found in pre-calculated cache and raw Georgia is not mounted",
            )
        return copy.deepcopy(precalc["record"])

    _, _, cfg = get_model_and_cfg(model_id)
    sr       = int(cfg.get("data", {}).get("sampling_rate", 500))
    t_fixed  = int(cfg.get("data", {}).get("t_fixed", 500))
    lead_idx = int(cfg.get("data", {}).get("lead", 0))

    signal_filt = _load_filtered_signal_any(ecg_id, sr, dataset=ds_name)
    signal_disp = z_normalize(signal_filt)

    precalc_delin = None
    if ds_name == "georgia" and sr == 500 and lead_idx in (0, 1):
        pre = _load_precalc_record("georgia", str(ecg_id))
        if pre and isinstance(pre.get("record"), dict):
            precalc_delin = pre["record"].get("delineation")

    if precalc_delin is None:
        delin = _load_delineation_any(ecg_id, sr, lead_idx, dataset=ds_name)
        d_map = delin.get("delineation", {})
        delin_payload = {
            "quality_ok": bool(delin.get("quality_ok", False)),
            "r_peaks":    _valid_int_list(delin.get("r_peaks", [])),
            "p_peaks":    _valid_int_list(d_map.get("ECG_P_Peaks", [])),
            "t_peaks":    _valid_int_list(d_map.get("ECG_T_Peaks", [])),
        }
    else:
        delin_payload = copy.deepcopy(precalc_delin)

    _, T = signal_disp.shape
    ds = 2
    signal_ds = signal_disp[:, ::ds]
    lead_names = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]

    if ds_name == "georgia":
        meta = _get_georgia_meta()
        rid = str(ecg_id)
        row = meta.loc[rid]
        return {
            "ecg_id":          rid,
            "dataset":         "georgia",
            "sampling_rate":   sr,
            "t_fixed":         t_fixed,
            "n_samples":       T,
            "n_samples_ds":    signal_ds.shape[1],
            "downsample":      ds,
            "leads":           lead_names,
            "signal":          signal_ds.tolist(),
            "age":             float(row["age"]) if pd.notna(row.get("age")) else 0.0,
            "sex":             int(row["_sex_int"]),
            "sex_label":       str(row["_sex_label"]),
            "severity_tier":   str(row["severity_tier"]),
            "est_hr_bpm":      _safe_round(row.get("est_hr_bpm"), 1),
            "dx":              row["_dx_list"],
            "label_names":     row["_snomed_labels"],
            "tags":            sorted(row["_tag_set"]),
            "has_soft_seg_gt": False,
            "delineation":     delin_payload,
            "soft_seg":        None,
        }

    eid_int = _parse_ptbxl_id(ecg_id)
    meta = _get_meta().loc[eid_int]
    # Optional Zenodo 7610236 (ECGdeli U-Net) per-timestep soft segmentation envelopes (100 Hz)
    soft_seg_payload = None
    archive = _get_soft_seg_archive()
    if archive is not None and eid_int in archive:
        try:
            ss = archive.load(eid_int)
            env = soft_seg_to_wave_curves(ss, lead_idx=lead_idx)
            soft_seg_payload = {
                "sampling_rate": 100,
                "P":   np.round(env["P"],   3).tolist(),
                "QRS": np.round(env["QRS"], 3).tolist(),
                "ST":  np.round(env["ST"],  3).tolist(),
                "T":   np.round(env["T"],   3).tolist(),
            }
        except Exception as exc:
            logger.warning(f"Failed loading soft_seg for ecg_id={eid_int}: {exc}")

    return {
        "ecg_id":          eid_int,
        "dataset":         "ptbxl",
        "sampling_rate":   sr,
        "t_fixed":         t_fixed,
        "n_samples":       T,
        "n_samples_ds":    signal_ds.shape[1],
        "downsample":      ds,
        "leads":           lead_names,
        "signal":          signal_ds.tolist(),
        "age":             float(meta["age"]) if not pd.isna(meta.get("age")) else 0.0,
        "sex":             int(meta["sex"]) if not pd.isna(meta.get("sex")) else 0,
        "sex_label":       "M" if (not pd.isna(meta.get("sex")) and int(meta["sex"]) == 0) else "F",
        "severity_tier":   None,
        "dx":              meta["_dx_list"],
        "label_names":     [],
        "tags":            sorted(meta["_tag_set"]),
        "has_soft_seg_gt": soft_seg_payload is not None,
        "delineation":     delin_payload,
        "soft_seg":        soft_seg_payload,
    }


@app.post("/infer")
def infer(req: InferRequest):
    ds_name = _normalize_dataset(req.dataset)
    if ds_name == "georgia" and not _has_georgia_raw():
        resolved_id = _resolve_model_id(req.model_id)
        cmp_id = _resolve_model_id(req.compare_model_id) if req.compare_model_id else None
        precalc = _load_precalc_record("georgia", str(req.ecg_id))
        manifest = _get_precalc_manifest() or {}
        sweeps = (precalc or {}).get("sweeps") or {}
        heads = manifest.get("window_heads") or {}
        if resolved_id not in sweeps:
            raise HTTPException(
                status_code=503,
                detail=f"Georgia raw data not mounted and '{req.ecg_id}/{resolved_id}' not in pre-calculated cache",
            )
        sw = sweeps[resolved_id]
        starts = sw["starts"]
        ni = min(range(len(starts)), key=lambda i: abs(starts[i] - req.start))
        prim = _eval_precalc_window_infer(sw, heads.get(resolved_id), ni, req.interventions, req.include_epistemic)
        gt_concepts = precalc["sweep_gt_concepts"][ni] if precalc.get("sweep_gt_concepts") else None
        gt_task = precalc["sweep_gt_task"][ni] if precalc.get("sweep_gt_task") else None
        payload = {
            "ecg_id":        req.ecg_id,
            "dataset":       "georgia",
            "model_id":      resolved_id,
            "model_type":    sw["model_type"],
            **prim,
            "concept_names": list(sw["concept_names"]),
            "gt_concepts":   gt_concepts,
            "gt_task":       gt_task,
        }
        if cmp_id and cmp_id in sweeps:
            csw = sweeps[cmp_id]
            payload["compare"] = {
                "model_id":   cmp_id,
                "model_type": csw["model_type"],
                **_eval_precalc_window_infer(csw, heads.get(cmp_id), ni, req.interventions, req.include_epistemic),
            }
        return payload

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

    signal = _load_filtered_signal_any(req.ecg_id, sr, dataset=ds_name)
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

    delin = _load_delineation_any(req.ecg_id, sr, lead_i, dataset=ds_name)
    gt_concepts, gt_task = None, None
    if delin.get("quality_ok", False):
        c, w = get_concept_labels(start, start + t_fixed, delin["delineation"], delin["r_peaks"], delin.get("u_peaks"))
        active = cfg.get("active_concepts", None)
        gt_concepts = [c[i] for i in active] if active is not None else c
        gt_task = int(is_complete_heartbeat_window(c, w))

    payload = {
        "ecg_id":         req.ecg_id,
        "dataset":        ds_name,
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
    ds_name = _normalize_dataset(req.dataset)
    resolved_id = _resolve_model_id(req.model_id)
    cmp_id = _resolve_model_id(req.compare_model_id) if req.compare_model_id else None
    stride = max(5, req.stride)

    # Fast-path: serve pre-calculated curated 500 Hz sweeps (ptbxl & georgia) directly
    precalc = _load_precalc_record(ds_name, str(req.ecg_id))
    if precalc is not None:
        sweeps = precalc.get("sweeps") or {}
        sw = sweeps.get(resolved_id)
        csw = sweeps.get(cmp_id) if cmp_id else None
        if sw is not None and sw.get("stride") == stride and (cmp_id is None or csw is not None):
            return {
                "ecg_id":        req.ecg_id,
                "dataset":       ds_name,
                "model_id":      resolved_id,
                "model_type":    sw["model_type"],
                "stride":        sw["stride"],
                "t_fixed":       sw["t_fixed"],
                "sampling_rate": sw["sampling_rate"],
                "starts":        list(sw["starts"]),
                "task_probs":    list(sw["task_probs"]),
                "task_stds":     list(sw["task_stds"]),
                "concept_probs": [list(r) for r in sw["concept_probs"]],
                "concept_stds":  [list(r) for r in sw["concept_stds"]],
                "concept_names": list(sw["concept_names"]),
                "gt_task":       list(precalc["sweep_gt_task"]) if precalc.get("sweep_gt_task") is not None else None,
                "gt_concepts":   [list(r) for r in precalc["sweep_gt_concepts"]] if precalc.get("sweep_gt_concepts") is not None else None,
                "compare": (
                    {
                        "model_id":      cmp_id,
                        "model_type":    csw["model_type"],
                        "task_probs":    list(csw["task_probs"]),
                        "task_stds":     list(csw["task_stds"]),
                        "concept_probs": [list(r) for r in csw["concept_probs"]],
                        "concept_stds":  [list(r) for r in csw["concept_stds"]],
                    }
                    if csw is not None
                    else None
                ),
            }

    if ds_name == "georgia" and not _has_georgia_raw():
        raise HTTPException(
            status_code=503,
            detail=f"Georgia raw data not mounted and '{req.ecg_id}' (stride={stride}) not in pre-calculated cache",
        )

    _, model, cfg = get_model_and_cfg(resolved_id)
    sr      = int(cfg.get("data", {}).get("sampling_rate", 500))
    t_fixed = int(cfg.get("data", {}).get("t_fixed", 500))
    lead_i  = int(cfg.get("data", {}).get("lead", 0))
    c_names = _concept_names_for_cfg(cfg)

    cmp_bundle = None
    if cmp_id:
        _, cmp_model, cmp_cfg = get_model_and_cfg(cmp_id)
        _validate_compare_compatibility(cfg, cmp_cfg, cmp_id)
        cmp_bundle = (cmp_id, cmp_model, cmp_cfg)

    signal = _load_filtered_signal_any(req.ecg_id, sr, dataset=ds_name)
    _, T   = signal.shape
    stride = max(5, req.stride)
    starts = list(range(0, T - t_fixed + 1, stride))
    if not starts:
        raise HTTPException(status_code=422, detail="Signal too short for window size.")

    windows = np.stack([signal[:, s : s + t_fixed] for s in starts], axis=0)
    res = _batch_infer_windows(model, cfg, windows)

    delin = _load_delineation_any(req.ecg_id, sr, lead_i, dataset=ds_name)
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
        "dataset":       ds_name,
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
    ecg_id: Union[int, str]
    dataset: str = "ptbxl"
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
    ds_name = _normalize_dataset(req.dataset)
    precalc = _load_precalc_record(ds_name, str(req.ecg_id))
    if precalc is not None:
        dense_map = precalc.get("dense") or {}
        req_mid = req.model_id or next(
            (c for c in ("dense_cem_softseg", "dense_cem_s42", "dense_cbm_detached_s42", "dense_cbm_softseg") if c in dense_map),
            next(iter(dense_map.keys()), None),
        )
        req_cmp = req.compare_model_id
        req_bb = req.blackbox_model_id
        have_models = (
            req_mid in dense_map
            and (not req_cmp or req_cmp in dense_map)
            and (not req_bb or req_bb in dense_map)
        )
        mode = (req.intervene_mode or "").lower()
        single_concept_full = (
            not mode
            or (
                not req.intervene_span
                and mode in ("gt", "ablate", "force")
                and (mode == "gt" or (isinstance(req.intervene_concepts, list) and len(req.intervene_concepts) == 1))
            )
        )
        precalc_only_fallback = (ds_name == "georgia" and not _has_georgia_raw())
        if have_models and (single_concept_full or precalc_only_fallback):
            prim = _build_precalc_dense_entry(
                precalc, req_mid, req.include_epistemic, req.intervene_mode, req.intervene_concepts, req.intervene_span
            )
            payload = {
                "ecg_id":         req.ecg_id,
                "dataset":        ds_name,
                "sampling_rate":  100,
                "gt_probs":       ({k: list(v) for k, v in precalc["dense_gt_probs"].items()} if precalc.get("dense_gt_probs") else None),
                "gt_spans":       (list(precalc["dense_gt_spans"]) if precalc.get("dense_gt_spans") else None),
                "intervene_mode": req.intervene_mode,
                **prim,
            }
            if req_cmp:
                cmp_e = _build_precalc_dense_entry(
                    precalc, req_cmp, req.include_epistemic, req.intervene_mode, req.intervene_concepts, req.intervene_span
                )
                if cmp_e is not None:
                    payload["compare"] = cmp_e
            if req_bb:
                bb_e = _build_precalc_dense_entry(
                    precalc, req_bb, req.include_epistemic, None, None, None
                )
                if bb_e is not None:
                    payload["blackbox"] = bb_e
            return payload

    if ds_name == "georgia" and not _has_georgia_raw():
        raise HTTPException(
            status_code=503,
            detail=f"Georgia raw data not mounted and '{req.ecg_id}' dense inference not fully covered by pre-calculated cache",
        )

    resolved_id, model, cfg = get_dense_model_and_cfg(req.model_id)
    sr = int(cfg.get("data", {}).get("sampling_rate", 100))
    c_mode = str(cfg.get("data", {}).get("concept_mode", "wave6"))
    c_names = list(cfg.get("concept_names", get_dense_concept_names(c_mode)))

    sig = z_normalize(_load_filtered_signal_any(req.ecg_id, sr, dataset=ds_name))
    T = sig.shape[-1]
    x = torch.from_numpy(sig).unsqueeze(0).to(DEVICE)

    # Cache Zenodo GT soft-segmentation per requested dense concept mode (PTB-XL only)
    archive = _get_soft_seg_archive() if ds_name == "ptbxl" else None
    raw_ss = None
    if archive is not None:
        eid_int = _parse_ptbxl_id(req.ecg_id)
        if eid_int in archive:
            try:
                raw_ss = archive.load(eid_int)
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

        intervals = None
        if m_mode == "wave6" and cp.shape[0] == 6:
            iv = extract_wave_spans_and_intervals(cp.argmax(axis=0), fs=sr)
            intervals = {
                "qrs_ms":       _safe_round(iv.get("qrs_ms"), 1),
                "pr_ms":        _safe_round(iv.get("pr_ms"), 1),
                "qt_ms":        _safe_round(iv.get("qt_ms"), 1),
                "pred_hr_bpm":  _safe_round(iv.get("pred_hr_bpm"), 1),
                "canon_cycle":  _safe_round(iv.get("canonical_transition_rate"), 4),
            }

        return {
            "has_seg":                True,
            "supports_interventions": m_can_int,
            "concept_names":          m_c_names,
            "concept_probs":          {name: np.round(cp[i], 3).tolist() for i, name in enumerate(m_c_names)},
            "concept_stds":           {name: np.round(cs[i], 3).tolist() for i, name in enumerate(m_c_names)},
            "mean_std_timeline":      np.round(cs.mean(axis=0), 4).tolist(),
            "intervals":              intervals,
            "pathology":              {name: round(float(tp[i]), 4) for i, name in enumerate(SUPERCLASS_NAMES)},
            "base_pathology":         {name: round(float(base_tp[i]), 4) for i, name in enumerate(SUPERCLASS_NAMES)},
            "pathology_std":          {name: round(float(tp_std[i]), 4) for i, name in enumerate(SUPERCLASS_NAMES)},
        }

    prim = _run_one(model, cfg)
    payload = {
        "ecg_id":         req.ecg_id,
        "dataset":        ds_name,
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
    ga_raw = _has_georgia_raw()
    ga_pre = _has_georgia_precalc()
    ga_mode = (
        "mounted+precalc" if (ga_raw and ga_pre)
        else ("mounted" if ga_raw else ("precalc" if ga_pre else "unavailable"))
    )
    return {
        "status":        "ok",
        "device":        DEVICE,
        "runs_dir":      str(RUNS_DIR),
        "data_path":     str(DATA_PATH),
        "datasets":      (["ptbxl", "georgia"] if (ga_raw or ga_pre) else ["ptbxl"]),
        "georgia_mode":  ga_mode,
        "n_models":      len(runs),
        "default_model": DEFAULT_RUN_KEY,
    }
