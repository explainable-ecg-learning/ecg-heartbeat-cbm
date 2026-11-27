"""
explorer/convert_numpy.py — Convert arbitrary NumPy ECG arrays (.npy, .npz, or
a directory of files) into Explorer-ready pre-calculated bundles under
`explorer/vercel/data/` so custom cohorts appear immediately in both the
live/Docker Explorer and the static Vercel deployment.

Examples:
    # Single .npz archive with multiple signals + metadata
    .venv_new/bin/python explorer/convert_numpy.py cohort.npz \
        --dataset-id my_cohort --dataset-label "My Cohort" --fs 500

    # Directory of .npy files at 250 Hz with lead-first axis and a metadata CSV
    .venv_new/bin/python explorer/convert_numpy.py ./npy_folder \
        --dataset-id ward_ecgs --fs 250 --meta labels.csv --channel-axis first

    # Remove a previously imported custom NumPy dataset
    .venv_new/bin/python explorer/convert_numpy.py --remove-dataset my_cohort
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
import time
from fractions import Fraction
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
import torch
from scipy.signal import resample_poly

REPO_ROOT = Path(__file__).resolve().parent.parent
EXPLORER_DIR = REPO_ROOT / "explorer"
SRC_DIR = REPO_ROOT / "src"
for p in (str(EXPLORER_DIR), str(SRC_DIR)):
    if p not in sys.path:
        sys.path.insert(0, p)

import app as exp_app  # noqa: E402
import build_vercel as bv  # noqa: E402
from data import (  # noqa: E402
    bandpass_filter,
    delineate_record,
    get_concept_labels,
    is_complete_heartbeat_window,
    z_normalize,
)
from dense_seg import (  # noqa: E402
    SUPERCLASS_NAMES,
    get_dense_concept_names,
)
from external_data import (  # noqa: E402
    CANONICAL_LEADS,
    canonicalize_leads,
    extract_wave_spans_and_intervals,
)
from models import LOG_VAR_MAX  # noqa: E402

LEAD_DISPLAY_NAMES = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]
STANDARD_NPZ_SIGNAL_KEYS = ("signals", "signal", "X", "x", "ecg", "ecgs", "data", "arr_0")


def _sanitize_id(raw_id: Any, fallback: str = "rec_0") -> str:
    s = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(raw_id).strip()).strip("._")
    return s or fallback


def _parse_str_list(val: Any) -> List[str]:
    if val is None or (isinstance(val, float) and math.isnan(val)):
        return []
    if isinstance(val, np.ndarray):
        return [str(x).strip() for x in val.tolist() if str(x).strip()]
    if isinstance(val, (list, tuple, set)):
        return [str(x).strip() for x in val if str(x).strip()]
    s = str(val).strip()
    if not s or s.lower() in ("nan", "none"):
        return []
    if s.startswith("[") and s.endswith("]"):
        try:
            parsed = json.loads(s.replace("'", '"'))
            if isinstance(parsed, list):
                return [str(x).strip() for x in parsed if str(x).strip()]
        except Exception:
            pass
    sep = "|" if "|" in s else ","
    return [part.strip() for part in s.split(sep) if part.strip()]


def _parse_sex(val: Any) -> Tuple[int, str]:
    if val is None or (isinstance(val, float) and math.isnan(val)):
        return -1, ""
    if isinstance(val, (int, np.integer)) and int(val) in (0, 1):
        return (0, "M") if int(val) == 0 else (1, "F")
    s = str(val).strip().upper()
    if s in ("0", "M", "MALE"):
        return 0, "M"
    if s in ("1", "F", "FEMALE"):
        return 1, "F"
    return -1, ""


def _parse_age(val: Any) -> float:
    try:
        v = float(val)
        return v if math.isfinite(v) and v >= 0 else 0.0
    except (TypeError, ValueError):
        return 0.0


def _load_meta_lookup(meta_path: Optional[Union[str, Path]]) -> Tuple[Dict[str, dict], List[dict]]:
    if meta_path is None:
        return {}, []
    p = Path(meta_path)
    if not p.exists():
        raise FileNotFoundError(f"Metadata file not found: {p}")
    rows_by_id: Dict[str, dict] = {}
    rows_list: List[dict] = []
    if p.suffix.lower() == ".json":
        raw = json.loads(p.read_text())
        if isinstance(raw, dict):
            for k, v in raw.items():
                entry = dict(v) if isinstance(v, dict) else {"dx": v}
                entry.setdefault("ecg_id", str(k))
                rows_by_id[str(entry["ecg_id"])] = entry
                rows_by_id[_sanitize_id(entry["ecg_id"])] = entry
                rows_list.append(entry)
        elif isinstance(raw, list):
            for item in raw:
                if not isinstance(item, dict):
                    continue
                rows_list.append(item)
                rid = item.get("ecg_id", item.get("record_id", item.get("id")))
                if rid is not None:
                    rows_by_id[str(rid)] = item
                    rows_by_id[_sanitize_id(rid)] = item
        return rows_by_id, rows_list

    df = pd.read_csv(p)
    for _, row in df.iterrows():
        item = {k: row[k] for k in df.columns}
        rows_list.append(item)
        rid = item.get("ecg_id", item.get("record_id", item.get("id")))
        if rid is not None and not (isinstance(rid, float) and math.isnan(rid)):
            rows_by_id[str(rid)] = item
            rows_by_id[_sanitize_id(rid)] = item
    return rows_by_id, rows_list


def _orient_2d_signal(arr: np.ndarray, channel_axis: str) -> np.ndarray:
    """Return a 2D ECG array as (C, T)."""
    if arr.ndim == 1:
        return arr[np.newaxis, :]
    if arr.ndim != 2:
        raise ValueError(f"Expected 1D or 2D ECG array, got shape {arr.shape}")
    r, c = arr.shape
    if channel_axis == "first":
        return arr
    if channel_axis == "last":
        return arr.T
    # auto
    if r <= 12 and c > 12:
        return arr
    if c <= 12 and r > 12:
        return arr.T
    if r <= 12 and c <= 12:
        return arr
    raise ValueError(
        f"Ambiguous 2D ECG shape {arr.shape} under --channel-axis=auto; pass --channel-axis first|last."
    )


def _orient_3d_batch(arr: np.ndarray, channel_axis: str) -> np.ndarray:
    """Return a 3D batch of ECG arrays as (N, C, T)."""
    if arr.ndim != 3:
        raise ValueError(f"Expected 3D batch shape, got {arr.shape}")
    _, a, b = arr.shape
    if channel_axis == "first":
        return arr
    if channel_axis == "last":
        return np.transpose(arr, (0, 2, 1))
    if a <= 12 and b > 12:
        return arr
    if b <= 12 and a > 12:
        return np.transpose(arr, (0, 2, 1))
    if a <= 12 and b <= 12:
        return arr
    raise ValueError(
        f"Ambiguous 3D ECG batch shape {arr.shape} under --channel-axis=auto; pass --channel-axis first|last."
    )


def _extract_npz_records(
    npz_path: Path,
    *,
    key: Optional[str],
    channel_axis: str,
    default_fs: int,
    default_leads: Optional[List[str]],
    prefix_with_stem: bool,
) -> List[dict]:
    records: List[dict] = []
    with np.load(npz_path, allow_pickle=False) as zf:
        keys = list(zf.files)
        sig_key = key
        if sig_key is None:
            sig_key = next((k for k in STANDARD_NPZ_SIGNAL_KEYS if k in keys), None)

        if sig_key is not None:
            if sig_key not in zf:
                raise KeyError(f"Key '{sig_key}' not found in {npz_path} (keys={keys})")
            arr = np.asarray(zf[sig_key])
            ids_arr = next((zf[k] for k in ("ids", "record_ids", "ecg_ids") if k in keys), None)
            dx_arr = next((zf[k] for k in ("dx", "labels", "classes") if k in keys), None)
            lbl_arr = zf["label_names"] if "label_names" in keys else None
            age_arr = zf["age"] if "age" in keys else None
            sex_arr = zf["sex"] if "sex" in keys else None
            split_arr = next((zf[k] for k in ("splits", "split", "folds", "fold") if k in keys), None)
            fs_val = int(np.asarray(zf["fs"]).item()) if ("fs" in keys and zf["fs"].ndim == 0) else default_fs
            leads_val = (
                [str(x).strip() for x in zf["leads"].tolist()]
                if "leads" in keys
                else default_leads
            )

            if arr.ndim in (1, 2):
                arr_batch = _orient_2d_signal(arr, channel_axis)[np.newaxis, ...]
            elif arr.ndim == 3:
                arr_batch = _orient_3d_batch(arr, channel_axis)
            else:
                raise ValueError(f"Unsupported array ndim={arr.ndim} for key '{sig_key}' in {npz_path}")

            n = arr_batch.shape[0]
            for i in range(n):
                if ids_arr is not None and i < len(ids_arr):
                    rid = str(ids_arr[i])
                elif n == 1:
                    rid = npz_path.stem
                elif prefix_with_stem:
                    rid = f"{npz_path.stem}_{i:04d}"
                else:
                    rid = f"rec_{i:04d}"

                dx_list = _parse_str_list(dx_arr[i]) if (dx_arr is not None and i < len(dx_arr)) else []
                lbl_list = _parse_str_list(lbl_arr[i]) if (lbl_arr is not None and i < len(lbl_arr)) else dx_list
                sex_i, sex_lbl = _parse_sex(sex_arr[i] if (sex_arr is not None and i < len(sex_arr)) else None)
                split_v = str(split_arr[i]) if (split_arr is not None and i < len(split_arr)) else "custom"
                rec_fs = int(zf["fs"][i]) if ("fs" in keys and zf["fs"].ndim > 0 and i < len(zf["fs"])) else fs_val
                records.append({
                    "ecg_id":      rid,
                    "signal":      arr_batch[i],
                    "fs":          rec_fs,
                    "leads":       leads_val,
                    "dx":          dx_list,
                    "label_names": lbl_list,
                    "tags":        sorted({x.upper() for x in (dx_list + lbl_list)}),
                    "age":         _parse_age(age_arr[i] if (age_arr is not None and i < len(age_arr)) else None),
                    "sex":         sex_i,
                    "sex_label":   sex_lbl,
                    "split":       split_v,
                })
            return records

        # Treat every 1D/2D numeric array key in .npz as an individual record
        for k in keys:
            arr = np.asarray(zf[k])
            if arr.ndim not in (1, 2) or not np.issubdtype(arr.dtype, np.number):
                continue
            rid = f"{npz_path.stem}_{k}" if prefix_with_stem else str(k)
            records.append({
                "ecg_id":      rid,
                "signal":      _orient_2d_signal(arr, channel_axis),
                "fs":          default_fs,
                "leads":       default_leads,
                "dx":          [],
                "label_names": [],
                "tags":        [],
                "age":         0.0,
                "sex":         -1,
                "sex_label":   "",
                "split":       "custom",
            })
        if not records:
            raise ValueError(f"No 1D/2D/3D numeric ECG arrays found in {npz_path} (keys={keys})")
    return records


def load_numpy_records(
    input_path: Union[str, Path],
    *,
    fs: int = 500,
    key: Optional[str] = None,
    channel_axis: str = "auto",
    leads: Optional[Sequence[str]] = None,
    meta_path: Optional[Union[str, Path]] = None,
    max_records: Optional[int] = None,
) -> List[dict]:
    """Load 1D/2D/3D ECG recordings from a .npy file, .npz archive, or folder of .npy/.npz files."""
    p = Path(input_path)
    if not p.exists():
        raise FileNotFoundError(f"Input path does not exist: {p}")

    default_leads = [str(l).strip() for l in leads] if leads else None
    raw_records: List[dict] = []

    def _from_npy(npy_path: Path, prefix_idx: bool) -> List[dict]:
        arr = np.load(npy_path, allow_pickle=False)
        out: List[dict] = []
        if arr.ndim in (1, 2):
            sig2 = _orient_2d_signal(arr, channel_axis)
            out.append({
                "ecg_id":      npy_path.stem,
                "signal":      sig2,
                "fs":          fs,
                "leads":       default_leads,
                "dx":          [],
                "label_names": [],
                "tags":        [],
                "age":         0.0,
                "sex":         -1,
                "sex_label":   "",
                "split":       "custom",
            })
        elif arr.ndim == 3:
            batch = _orient_3d_batch(arr, channel_axis)
            for i in range(batch.shape[0]):
                rid = f"{npy_path.stem}_{i:04d}" if prefix_idx else f"rec_{i:04d}"
                out.append({
                    "ecg_id":      rid,
                    "signal":      batch[i],
                    "fs":          fs,
                    "leads":       default_leads,
                    "dx":          [],
                    "label_names": [],
                    "tags":        [],
                    "age":         0.0,
                    "sex":         -1,
                    "sex_label":   "",
                    "split":       "custom",
                })
        else:
            raise ValueError(f"Unsupported .npy shape {arr.shape} in {npy_path}")
        return out

    if p.is_dir():
        files = sorted([*p.glob("*.npy"), *p.glob("*.npz")])
        if not files:
            raise ValueError(f"No .npy or .npz files found under {p}")
        for fp in files:
            if fp.suffix.lower() == ".npy":
                raw_records.extend(_from_npy(fp, prefix_idx=True))
            else:
                raw_records.extend(
                    _extract_npz_records(
                        fp, key=key, channel_axis=channel_axis,
                        default_fs=fs, default_leads=default_leads, prefix_with_stem=True,
                    )
                )
    elif p.suffix.lower() == ".npy":
        raw_records.extend(_from_npy(p, prefix_idx=False))
    elif p.suffix.lower() == ".npz":
        raw_records.extend(
            _extract_npz_records(
                p, key=key, channel_axis=channel_axis,
                default_fs=fs, default_leads=default_leads, prefix_with_stem=False,
            )
        )
    else:
        raise ValueError(f"Unsupported input '{p}'; expected .npy, .npz, or directory of .npy/.npz files.")

    # Apply optional CSV / JSON metadata table
    meta_by_id, meta_rows = _load_meta_lookup(meta_path)
    used_ids: Dict[str, int] = {}
    for idx, rec in enumerate(raw_records):
        meta_row = (
            meta_by_id.get(str(rec["ecg_id"]))
            or meta_by_id.get(_sanitize_id(rec["ecg_id"]))
            or (meta_rows[idx] if idx < len(meta_rows) else None)
        )
        if meta_row:
            if meta_row.get("ecg_id", meta_row.get("record_id", meta_row.get("id"))) is not None:
                cand_id = str(meta_row.get("ecg_id", meta_row.get("record_id", meta_row.get("id")))).strip()
                if cand_id and cand_id.lower() != "nan":
                    rec["ecg_id"] = cand_id
            dx_m = _parse_str_list(meta_row.get("dx"))
            lbl_m = _parse_str_list(meta_row.get("label_names", meta_row.get("labels")))
            tag_m = _parse_str_list(meta_row.get("tags"))
            if dx_m:
                rec["dx"] = dx_m
            if lbl_m:
                rec["label_names"] = lbl_m
            elif dx_m and not rec["label_names"]:
                rec["label_names"] = list(dx_m)
            all_tags = set(rec["tags"]) | {t.upper() for t in (dx_m + lbl_m + tag_m)}
            rec["tags"] = sorted(all_tags)
            if "age" in meta_row:
                rec["age"] = _parse_age(meta_row.get("age"))
            if "sex" in meta_row:
                sx_i, sx_l = _parse_sex(meta_row.get("sex"))
                if sx_i >= 0:
                    rec["sex"] = sx_i
                    rec["sex_label"] = sx_l
            sp = meta_row.get("split", meta_row.get("fold"))
            if sp is not None and not (isinstance(sp, float) and math.isnan(sp)):
                rec["split"] = str(sp).strip() or rec["split"]
            fs_m = meta_row.get("fs")
            if fs_m is not None and not (isinstance(fs_m, float) and math.isnan(fs_m)):
                rec["fs"] = int(float(fs_m))
            leads_m = _parse_str_list(meta_row.get("leads"))
            if leads_m:
                rec["leads"] = leads_m

        # Enforce unique sanitized IDs
        safe = _sanitize_id(rec["ecg_id"], fallback=f"rec_{idx:04d}")
        if safe in used_ids:
            used_ids[safe] += 1
            safe = f"{safe}_{used_ids[safe]}"
        else:
            used_ids[safe] = 0
        rec["ecg_id"] = safe

    if max_records is not None and max_records > 0:
        raw_records = raw_records[: int(max_records)]
    return raw_records


def standardize_ecg_array(
    signal_c_t: np.ndarray,
    *,
    src_fs: int = 500,
    leads: Optional[Sequence[str]] = None,
    single_lead: str = "II",
    target_sec: float = 10.0,
    apply_bandpass: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Standardize a raw (C, T) NumPy ECG recording into filtered `(12, target_500)`
    at 500 Hz and `(12, target_100)` at 100 Hz in PTB-XL canonical lead order.
    """
    arr = np.nan_to_num(np.asarray(signal_c_t, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    if arr.ndim == 1:
        arr = arr[np.newaxis, :]
    if arr.ndim != 2:
        raise ValueError(f"Expected (C, T) array, got shape {arr.shape}")

    c_cnt, t_len = arr.shape
    can_upper = {name: i for i, name in enumerate(CANONICAL_LEADS)}

    if leads and len(leads) == c_cnt:
        lead_keys = [str(l).strip().upper() for l in leads]
        if all(name in lead_keys for name in CANONICAL_LEADS):
            canon = canonicalize_leads(arr, leads)
        else:
            canon = np.zeros((12, t_len), dtype=np.float32)
            for row_i, lkey in enumerate(lead_keys):
                if lkey in can_upper:
                    canon[can_upper[lkey]] = arr[row_i]
    elif c_cnt == 1:
        canon = np.zeros((12, t_len), dtype=np.float32)
        target_idx = can_upper.get(str(single_lead).strip().upper(), 1)
        canon[target_idx] = arr[0]
    elif c_cnt >= 12:
        canon = arr[:12].astype(np.float32, copy=False)
    else:
        canon = np.zeros((12, t_len), dtype=np.float32)
        canon[:c_cnt] = arr.astype(np.float32, copy=False)

    def _to_target_rate(target_sr: int) -> np.ndarray:
        target_samples = int(round(target_sec * target_sr))
        if int(src_fs) != target_sr:
            frac = Fraction(int(target_sr), int(src_fs))
            rs = resample_poly(canon, frac.numerator, frac.denominator, axis=-1).astype(np.float32)
        else:
            rs = canon.copy()
        cur = rs.shape[-1]
        if cur > target_samples:
            rs = rs[:, :target_samples]
        elif cur < target_samples:
            rs = np.pad(rs, ((0, 0), (0, target_samples - cur)), mode="constant", constant_values=0.0)
        if apply_bandpass:
            rs = bandpass_filter(rs, sampling_rate=target_sr, lowcut=0.5, highcut=40.0)
        return np.nan_to_num(rs.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)

    return _to_target_rate(500), _to_target_rate(100)


@torch.no_grad()
def _compute_window_sweep(
    model: torch.nn.Module,
    cfg: dict,
    windows_np: np.ndarray,
    device: str,
    run_mc: bool = True,
) -> dict:
    if run_mc:
        entry, _ = bv._compute_window_sweep_with_export(model, cfg, windows_np, next(model.parameters()).device)
        return entry

    data_cfg = cfg.get("data", {})
    normed = exp_app._select_model_leads(z_normalize(windows_np), data_cfg)
    model_dev = next(model.parameters()).device
    xb = torch.from_numpy(normed).to(model_dev)
    mtype = getattr(model, "model_type", "cbm")
    k_cnt = model.num_concepts
    if mtype in ("cbm", "cem"):
        z = model.encoder(xb)
        t_mean, t_std, _, c_probs, c_var = model.mc_task_from_z(z, n_mc=40)
        c_std_np = c_var.clamp(min=0.0).sqrt().cpu().numpy()
        c_std_q = np.clip(np.round(c_std_np * 255.0), 0, 255).astype(np.int32).tolist()
        t_mean_l = bv._round_list(t_mean.cpu().numpy(), 4)
        t_std_l = bv._round_list(t_std.cpu().numpy(), 4)
        entry: Dict[str, Any] = {
            "model_type":        mtype,
            "_cstd_scale":       255,
            "task_probs":        t_mean_l,
            "task_stds":         t_std_l,
            "task_stds_mc":      t_std_l,
            "task_epistemic_mc": [0.0] * len(t_mean_l),
            "concept_probs":     bv._round_list(c_probs.cpu().numpy(), 3),
            "concept_stds":      c_std_q,
            "concept_stds_mc":   c_std_q,
        }
        if mtype == "cem":
            c_pos, c_neg, _, _ = model.concept_layer.compute_embeddings_and_scores(z)
            emb_size = model.concept_layer.emb_size
            w = model.classifier.linear.weight.data[0].view(1, k_cnt, emb_size)
            entry["a_pos"] = bv._round_list((c_pos * w).sum(dim=-1).cpu().numpy(), 4)
            entry["a_neg"] = bv._round_list((c_neg * w).sum(dim=-1).cpu().numpy(), 4)
        else:
            _, c_lv = model.concept_layer(z)
            entry["c_vars"] = bv._round_list(torch.exp(c_lv.clamp(-25.0, LOG_VAR_MAX)).cpu().numpy(), 5)
        return entry

    tp, _, _ = model(xb)
    t_mean_l = bv._round_list(tp.squeeze(-1).cpu().numpy(), 4)
    zeros = [0.0] * len(t_mean_l)
    return {
        "model_type":        mtype,
        "task_probs":        t_mean_l,
        "task_stds":         zeros,
        "task_stds_mc":      zeros,
        "task_epistemic_mc": zeros,
        "concept_probs":     None,
        "concept_stds":      None,
        "concept_stds_mc":   None,
    }


@torch.no_grad()
def _compute_dense_export_from_signal(
    run_id: str,
    sig_100: np.ndarray,
    device: str,
    run_mc: bool = True,
) -> dict:
    _, model, cfg = exp_app.get_dense_model_and_cfg(run_id)
    sr = int(cfg.get("data", {}).get("sampling_rate", 100))
    c_mode = str(cfg.get("data", {}).get("concept_mode", "wave6"))
    c_names = list(cfg.get("concept_names", get_dense_concept_names(c_mode)))
    mtype = getattr(model, "model_type", "dense_cem")
    m_has_seg = mtype != "dense_blackbox"
    m_can_int = mtype in ("dense_cbm", "dense_cem")

    sig = z_normalize(sig_100.astype(np.float32, copy=False))
    t_len = sig.shape[-1]
    model_dev = next(model.parameters()).device
    x = torch.from_numpy(sig).unsqueeze(0).to(model_dev)

    base_out = model(x)
    base_tp = torch.sigmoid(base_out["task_logits"][0].float()).cpu().numpy()
    base_patho = {name: round(float(base_tp[i]), 4) for i, name in enumerate(SUPERCLASS_NAMES)}

    if not m_has_seg:
        if run_mc:
            unc = model.predict_with_uncertainty(x, n_mc_epistemic=8, n_mc_aleatoric=1)
            tp_std_mc = unc["task_probs_std"][0].cpu().numpy()
        else:
            tp_std_mc = np.zeros(len(SUPERCLASS_NAMES), dtype=np.float32)
        return {
            "model_type":             mtype,
            "has_seg":                False,
            "supports_interventions": False,
            "concept_names":          c_names,
            "base_pathology":         base_patho,
            "pathology_std":          {name: 0.0 for name in SUPERCLASS_NAMES},
            "pathology_std_mc":       {name: round(float(tp_std_mc[i]), 4) for i, name in enumerate(SUPERCLASS_NAMES)},
        }

    cp = base_out["concept_probs"][0].float().cpu().numpy()
    lv = base_out["concept_log_vars"][0].float().cpu().numpy()
    cs = np.clip(cp * (1.0 - cp) * np.exp(0.5 * np.clip(lv, -10.0, 4.0)), 0.0, 0.5)

    if run_mc:
        unc = model.predict_with_uncertainty(x, n_mc_epistemic=8, n_mc_aleatoric=12)
        cs_mc = unc["concept_total_std"][0].cpu().numpy()
        tp_std_mc = unc["task_probs_std"][0].cpu().numpy()
    else:
        cs_mc = cs
        tp_std_mc = np.zeros(len(SUPERCLASS_NAMES), dtype=np.float32)

    intervals = None
    if c_mode == "wave6" and cp.shape[0] == 6:
        iv = extract_wave_spans_and_intervals(cp.argmax(axis=0), fs=sr)
        intervals = {
            "qrs_ms":      exp_app._safe_round(iv.get("qrs_ms"), 1),
            "pr_ms":       exp_app._safe_round(iv.get("pr_ms"), 1),
            "qt_ms":       exp_app._safe_round(iv.get("qt_ms"), 1),
            "pred_hr_bpm": exp_app._safe_round(iv.get("pred_hr_bpm"), 1),
            "canon_cycle": exp_app._safe_round(iv.get("canonical_transition_rate"), 4),
        }

    interventions: Dict[str, Dict[str, float]] = {}
    interventions_win: Dict[str, List[List[float]]] = {}
    if m_can_int:
        n_bins = max(1, math.ceil(t_len / 100))
        x_b = x.expand(n_bins, -1, -1)
        base_cp_b = base_out["concept_probs"].expand(n_bins, -1, -1)
        bin_slices = [(b * 100, min(t_len, (b + 1) * 100)) for b in range(n_bins)]

        for k, cname in enumerate(c_names):
            il_a = base_out["concept_probs"].clone()
            il_a[:, k, :] = 0.0
            im_a = torch.zeros_like(il_a)
            im_a[:, k, :] = 1.0
            tp_a = torch.sigmoid(model(x, intervene_labels=il_a, intervene_mask=im_a)["task_logits"][0].float()).cpu().numpy()
            interventions[f"ablate:{cname}"] = {name: round(float(tp_a[i]), 4) for i, name in enumerate(SUPERCLASS_NAMES)}

            il_f = torch.zeros_like(base_out["concept_probs"])
            il_f[:, k, :] = 1.0
            im_f = torch.ones_like(il_f)
            tp_f = torch.sigmoid(model(x, intervene_labels=il_f, intervene_mask=im_f)["task_logits"][0].float()).cpu().numpy()
            interventions[f"force:{cname}"] = {name: round(float(tp_f[i]), 4) for i, name in enumerate(SUPERCLASS_NAMES)}

            il_aw = base_cp_b.clone()
            im_aw = torch.zeros_like(il_aw)
            il_fw = base_cp_b.clone()
            im_fw = torch.zeros_like(il_fw)
            for b, (s0, s1) in enumerate(bin_slices):
                il_aw[b, k, s0:s1] = 0.0
                im_aw[b, k, s0:s1] = 1.0
                il_fw[b, :, s0:s1] = 0.0
                il_fw[b, k, s0:s1] = 1.0
                im_fw[b, :, s0:s1] = 1.0
            tp_aw = torch.sigmoid(model(x_b, intervene_labels=il_aw, intervene_mask=im_aw)["task_logits"].float()).cpu().numpy()
            tp_fw = torch.sigmoid(model(x_b, intervene_labels=il_fw, intervene_mask=im_fw)["task_logits"].float()).cpu().numpy()
            interventions_win[f"ablate:{cname}"] = bv._round_list(tp_aw, 3)
            interventions_win[f"force:{cname}"] = bv._round_list(tp_fw, 3)

    seg_spans = bv._compute_seg_spans(cp) if (c_mode == "wave6" and cp.shape[0] == 6) else None
    std_line_q = np.clip(np.round(cs.mean(axis=0) * 10000.0), 0, 10000).astype(np.int32).tolist()
    std_line_mc_q = np.clip(np.round(cs_mc.mean(axis=0) * 10000.0), 0, 10000).astype(np.int32).tolist()

    return {
        "model_type":             mtype,
        "has_seg":                True,
        "supports_interventions": m_can_int,
        "concept_names":          c_names,
        "_prob_scale":            255,
        "_std_scale":             10000,
        "concept_probs":          bv._quantize_u8_dict(cp, c_names, omit_tp=True),
        "concept_stds":           bv._quantize_u8_dict(cs, c_names, omit_tp=True),
        "seg_spans":              seg_spans,
        "mean_std_timeline":      std_line_q,
        "mean_std_timeline_mc":   std_line_mc_q,
        "intervals":              intervals,
        "base_pathology":         base_patho,
        "pathology_std":          {name: 0.0 for name in SUPERCLASS_NAMES},
        "pathology_std_mc":       {name: round(float(tp_std_mc[i]), 4) for i, name in enumerate(SUPERCLASS_NAMES)},
        "interventions":          interventions,
        "interventions_win":      interventions_win,
    }


def _atomic_write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp")
    tmp.write_text(json.dumps(payload, separators=(",", ":"), allow_nan=False))
    tmp.replace(path)


def _ensure_manifest_models(manifest: dict, win_ids: List[str], dense_ids: List[str]) -> None:
    if not manifest.get("window_heads"):
        heads, summaries = {}, {}
        for mid in win_ids:
            _, wm, _ = exp_app.get_model_and_cfg(mid)
            heads[mid] = bv._export_window_head(wm)
            summaries[mid] = exp_app.model_summary(mid)
        all_win = exp_app.list_models()
        manifest["models"] = {
            "default_model":   "cem_softseg" if "cem_softseg" in win_ids else (win_ids[0] if win_ids else None),
            "compare_default": "cbm_softseg_masked" if "cbm_softseg_masked" in win_ids else None,
            "models":          [e for e in all_win["models"] if e["id"] in win_ids],
        }
        manifest["window_heads"] = heads
        manifest["summaries"] = summaries

    if dense_ids and not (manifest.get("dense_models") or {}).get("models"):
        all_dense = exp_app.list_dense_models()
        manifest["dense_models"] = {
            "default_model":    "dense_cem_s42" if "dense_cem_s42" in dense_ids else dense_ids[0],
            "compare_default":  "dense_cbm_detached_s42" if "dense_cbm_detached_s42" in dense_ids else None,
            "blackbox_default": "dense_blackbox_s42" if "dense_blackbox_s42" in dense_ids else None,
            "models":           [e for e in all_dense["models"] if e["id"] in dense_ids],
        }


def remove_dataset_from_explorer(
    dataset_id: str,
    out_dir: Union[str, Path] = bv.DATA_OUT,
    *,
    force: bool = False,
) -> int:
    """Remove a dataset's pre-calculated records and manifest entry."""
    ds_id = _sanitize_id(dataset_id).lower()
    if ds_id in ("ptbxl", "georgia") and not force:
        raise ValueError(f"Refusing to remove built-in dataset '{ds_id}' without force=True.")
    out_path = Path(out_dir)
    rec_dir = out_path / "records"
    removed = 0
    if rec_dir.exists():
        for rf in rec_dir.glob(f"{ds_id}_*.json"):
            rf.unlink()
            removed += 1
    manifest_path = out_path / "manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        manifest["datasets"] = [d for d in (manifest.get("datasets") or []) if d.get("id") != ds_id]
        (manifest.get("records_by_dataset") or {}).pop(ds_id, None)
        _atomic_write_json(manifest_path, manifest)
    print(f"Removed dataset '{ds_id}' ({removed} record JSONs) from {out_path}")
    return removed


def convert_numpy_to_explorer(
    input_path: Union[str, Path],
    *,
    dataset_id: str,
    dataset_label: Optional[str] = None,
    out_dir: Union[str, Path] = bv.DATA_OUT,
    fs: int = 500,
    key: Optional[str] = None,
    channel_axis: str = "auto",
    leads: Optional[Sequence[str]] = None,
    single_lead: str = "II",
    meta_path: Optional[Union[str, Path]] = None,
    target_sec: float = 10.0,
    apply_bandpass: bool = True,
    append: bool = False,
    max_records: Optional[int] = None,
    skip_dense: bool = False,
    run_mc: bool = True,
    device: Optional[str] = None,
) -> dict:
    """Convert NumPy ECG files into Explorer pre-calculated JSONs and register in manifest.json."""
    t0 = time.perf_counter()
    ds_id = _sanitize_id(dataset_id).lower()
    if ds_id in ("ptbxl", "georgia"):
        raise ValueError(f"Dataset ID '{ds_id}' is reserved; choose a custom dataset ID.")

    eff_device = device or exp_app.DEVICE
    out_path = Path(out_dir)
    rec_dir = out_path / "records"
    rec_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = out_path / "manifest.json"

    raw_records = load_numpy_records(
        input_path,
        fs=fs,
        key=key,
        channel_axis=channel_axis,
        leads=leads,
        meta_path=meta_path,
        max_records=max_records,
    )
    if not raw_records:
        raise ValueError(f"No ECG records loaded from {input_path}")

    win_runs = exp_app.discover_runs()
    win_ids = [mid for mid in bv.WINDOW_EXPORT_IDS if mid in win_runs]
    if not win_ids:
        win_ids = list(win_runs.keys())[:3]
    if not win_ids:
        raise RuntimeError(f"No trained window models found under {exp_app.RUNS_DIR}")

    dense_runs = exp_app.discover_dense_runs()
    dense_ids = [] if skip_dense else [mid for mid in bv.DENSE_EXPORT_IDS if mid in dense_runs]

    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {"default_dataset": "ptbxl", "datasets": [], "records_by_dataset": {}}
    _ensure_manifest_models(manifest, win_ids, dense_ids)

    if not append:
        for old_f in rec_dir.glob(f"{ds_id}_*.json"):
            old_f.unlink()
        existing_meta: List[dict] = []
    else:
        existing_meta = list((manifest.get("records_by_dataset") or {}).get(ds_id, []))

    single_lead_idx = CANONICAL_LEADS.index(str(single_lead).strip().upper()) if str(single_lead).strip().upper() in CANONICAL_LEADS else 1
    meta_by_id = {str(r["ecg_id"]): r for r in existing_meta}

    for rec in raw_records:
        eid = rec["ecg_id"]
        sig_500, sig_100 = standardize_ecg_array(
            rec["signal"],
            src_fs=int(rec["fs"]),
            leads=rec.get("leads"),
            single_lead=single_lead,
            target_sec=target_sec,
            apply_bandpass=apply_bandpass,
        )
        sig_disp = z_normalize(sig_500)
        sig_ds4 = sig_disp[:, ::4]
        _, t500 = sig_500.shape

        delin_lead = 0 if len(rec["signal"]) >= 12 else single_lead_idx
        delin = delineate_record(sig_500, sampling_rate=500, lead_idx=delin_lead)
        d_map = delin.get("delineation", {})
        est_hr = None
        r_peaks = delin.get("r_peaks", [])
        if len(r_peaks) >= 2:
            rr_sec = np.diff(r_peaks) / 500.0
            med_rr = float(np.median(rr_sec))
            if med_rr > 0:
                est_hr = exp_app._safe_round(60.0 / med_rr, 1)

        rec_payload = {
            "ecg_id":          eid,
            "dataset":         ds_id,
            "sampling_rate":   500,
            "t_fixed":         500,
            "n_samples":       int(t500),
            "n_samples_ds":    int(sig_ds4.shape[1]),
            "downsample":      4,
            "leads":           LEAD_DISPLAY_NAMES,
            "signal":          bv._round_list(sig_ds4, 2),
            "age":             float(rec["age"]),
            "sex":             int(rec["sex"]),
            "sex_label":       str(rec["sex_label"]),
            "split":           str(rec["split"]),
            "severity_tier":   None,
            "est_hr_bpm":      est_hr,
            "dx":              rec["dx"],
            "label_names":     rec["label_names"],
            "tags":            rec["tags"],
            "has_soft_seg_gt": False,
            "delineation": {
                "quality_ok": bool(delin.get("quality_ok", False)),
                "r_peaks":    exp_app._valid_int_list(r_peaks),
                "p_peaks":    exp_app._valid_int_list(d_map.get("ECG_P_Peaks", [])),
                "t_peaks":    exp_app._valid_int_list(d_map.get("ECG_T_Peaks", [])),
            },
            "soft_seg":        None,
        }

        t_fixed = 500
        starts = list(range(0, t500 - t_fixed + 1, bv.STATIC_STRIDE))
        windows_np = np.stack([sig_500[:, s : s + t_fixed] for s in starts], axis=0)
        gt_concepts, gt_task = None, None
        if delin.get("quality_ok", False):
            gc, gt = [], []
            for s in starts:
                c, w = get_concept_labels(s, s + t_fixed, d_map, r_peaks, delin.get("u_peaks"))
                gc.append(c)
                gt.append(int(is_complete_heartbeat_window(c, w)))
            gt_concepts, gt_task = gc, gt

        sweeps: Dict[str, dict] = {}
        for mid in win_ids:
            _, wm, wcfg = exp_app.get_model_and_cfg(mid)
            sw_entry = _compute_window_sweep(wm, wcfg, windows_np, eff_device, run_mc=run_mc)
            sw_entry.update({
                "stride":        bv.STATIC_STRIDE,
                "t_fixed":       t_fixed,
                "sampling_rate": 500,
                "starts":        starts,
                "concept_names": exp_app._concept_names_for_cfg(wcfg),
            })
            sweeps[mid] = sw_entry

        dense_map: Dict[str, dict] = {}
        for dmid in dense_ids:
            dense_map[dmid] = _compute_dense_export_from_signal(dmid, sig_100, eff_device, run_mc=run_mc)

        out_obj = {
            "record":            rec_payload,
            "sweep_gt_task":     gt_task,
            "sweep_gt_concepts": gt_concepts,
            "sweeps":            sweeps,
            "dense_prob_scale":  255,
            "dense_gt_probs":    None,
            "dense_gt_spans":    None,
            "dense":             dense_map,
        }
        _atomic_write_json(rec_dir / f"{ds_id}_{eid}.json", out_obj)

        meta_by_id[str(eid)] = {
            "ecg_id":      eid,
            "dataset":     ds_id,
            "split":       str(rec["split"]),
            "age":         float(rec["age"]),
            "sex":         int(rec["sex"]),
            "sex_label":   str(rec["sex_label"]),
            "dx":          rec["dx"],
            "label_names": rec["label_names"],
            "tags":        rec["tags"],
            "est_hr_bpm":  est_hr,
        }

    records_list = list(meta_by_id.values())
    manifest.setdefault("records_by_dataset", {})[ds_id] = records_list

    splits_vals = sorted({str(r.get("split", "custom")) for r in records_list})
    all_tags = sorted({t for r in records_list for t in (r.get("tags") or [])})
    split_entries = [{"value": "all", "label": "All Imported"}] + [{"value": sv, "label": sv} for sv in splits_vals]

    ds_entry = {
        "id":              ds_id,
        "label":           dataset_label or f"{ds_id} (NumPy)",
        "split_label":     "Split",
        "has_soft_seg_gt": False,
        "default_record":  str(records_list[0]["ecg_id"]),
        "default_split":   "all",
        "splits":          split_entries,
        "conditions":      ["ALL"] + all_tags,
    }
    manifest["datasets"] = [d for d in (manifest.get("datasets") or []) if d.get("id") != ds_id] + [ds_entry]
    _atomic_write_json(manifest_path, manifest)

    print(
        f"Converted {len(raw_records)} records into dataset '{ds_id}' under {out_path} "
        f"in {time.perf_counter() - t0:.1f}s (total in dataset: {len(records_list)})"
    )
    return {"dataset": ds_entry, "n_converted": len(raw_records), "record_ids": [r["ecg_id"] for r in raw_records]}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert NumPy ECG arrays (.npy/.npz/folder) into Explorer pre-calculated bundles."
    )
    parser.add_argument("input", nargs="?", help="Path to .npy file, .npz archive, or folder of .npy/.npz files.")
    parser.add_argument("--dataset-id", help="Short lowercase dataset identifier (e.g. my_cohort).")
    parser.add_argument("--dataset-label", default=None, help="Display label in the Explorer dataset dropdown.")
    parser.add_argument("--out-dir", default=str(bv.DATA_OUT), help=f"Output data directory (default: {bv.DATA_OUT}).")
    parser.add_argument("--fs", type=int, default=500, help="Input sampling frequency in Hz (default: 500).")
    parser.add_argument("--key", default=None, help="Array key inside .npz archive (auto-detected if omitted).")
    parser.add_argument("--channel-axis", choices=["auto", "first", "last"], default="auto", help="Lead channel axis.")
    parser.add_argument("--leads", default=None, help="Comma-separated lead names matching channels (e.g. I,II,...).")
    parser.add_argument("--single-lead", default="II", help="Target lead name when input is 1D single-lead (default: II).")
    parser.add_argument("--meta", default=None, help="Optional CSV or JSON file with record metadata.")
    parser.add_argument("--target-sec", type=float, default=10.0, help="Target duration in seconds (default: 10.0).")
    parser.add_argument("--no-bandpass", action="store_true", help="Skip 0.5-40 Hz Butterworth bandpass filter.")
    parser.add_argument("--append", action="store_true", help="Append/update records in an existing custom dataset.")
    parser.add_argument("--max-records", type=int, default=None, help="Optional cap on number of imported records.")
    parser.add_argument("--skip-dense", action="store_true", help="Skip dense 100 Hz models (window models only).")
    parser.add_argument("--no-mc", action="store_true", help="Skip MC-Dropout passes for faster import.")
    parser.add_argument("--device", default=None, help="Torch device override (default: cuda if available else cpu).")
    parser.add_argument("--remove-dataset", default=None, help="Delete an existing custom dataset by ID and exit.")
    parser.add_argument("--force", action="store_true", help="Allow --remove-dataset on built-in dataset IDs.")

    args = parser.parse_args()
    if args.remove_dataset:
        remove_dataset_from_explorer(args.remove_dataset, args.out_dir, force=args.force)
        return

    if not args.input or not args.dataset_id:
        parser.error("Positional `input` path and `--dataset-id` are required unless `--remove-dataset` is used.")

    leads_list = _parse_str_list(args.leads) if args.leads else None
    convert_numpy_to_explorer(
        args.input,
        dataset_id=args.dataset_id,
        dataset_label=args.dataset_label,
        out_dir=args.out_dir,
        fs=args.fs,
        key=args.key,
        channel_axis=args.channel_axis,
        leads=leads_list,
        single_lead=args.single_lead,
        meta_path=args.meta,
        target_sec=args.target_sec,
        apply_bandpass=not args.no_bandpass,
        append=args.append,
        max_records=args.max_records,
        skip_dense=args.skip_dense,
        run_mc=not args.no_mc,
        device=args.device,
    )


if __name__ == "__main__":
    main()
