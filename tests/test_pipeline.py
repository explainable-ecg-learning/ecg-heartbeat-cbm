"""Unit & smoke tests for Heartbeat CBM/CEM/BlackBox, Pathology CBM/CEM, Faithfulness/OIS, and Explorer."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from data import CONCEPT_COLUMNS, HeartbeatDataset, get_concept_labels  # noqa: E402
from evaluate import (  # noqa: E402
    evaluate,
    evaluate_faithfulness_and_leakage,
    evaluate_interventions,
    evaluate_uncertainty_quality,
)
from losses import CombinedCBMLoss  # noqa: E402
from models import build_model  # noqa: E402
from pathology import ConceptNormalizer, build_pathology_model  # noqa: E402
from dense_seg import (  # noqa: E402
    WAVE6_NAMES,
    DenseSoftSegLoss,
    build_dense_model,
    collapse_soft_seg_targets,
    evaluate_dense,
    evaluate_dense_leakage_probe,
    evaluate_dense_task_uncertainty,
)
from external_data import (  # noqa: E402
    CANONICAL_LEADS,
    canonicalize_leads,
    extract_wave_spans_and_intervals,
    map_snomed_to_superclasses,
    preprocess_to_dense_100hz,
)
from utils import bootstrap_multilabel_auroc, set_seed  # noqa: E402


class TestECGModelsAndPipeline(unittest.TestCase):
    def _base_cfg(self, model_type: str = "cbm") -> dict:
        return {
            "data": {"t_fixed": 120, "batch_size": 4, "num_workers": 0},
            "model": {
                "type": model_type,
                "encoder": {"in_channels": 12, "base_channels": 16, "dropout": 0.2},
                "concept_layer": {"num_concepts": 10, "emb_size": 8, "rand_int_prob": 0.5},
                "classifier": {"use_uncertainty": True, "detach_uncertainty": True},
                "mc_dropout_rate": 0.2,
            },
        }

    def test_cbm_cem_blackbox_forward_backward_randint_and_interventions(self) -> None:
        criterion = CombinedCBMLoss()
        x = torch.randn(4, 12, 120)
        y_task = torch.tensor([1, 0, 1, 0])
        y_conc = torch.randint(0, 2, (4, 10)).float()
        w_conc = torch.ones(4, 10)
        w_conc[:, 5] = 0.0

        for mtype in ("cbm", "cem", "blackbox"):
            model = build_model(self._base_cfg(mtype))
            model.train()
            tp, cp, clv, clog = model.forward_with_logits(
                x, concept_labels=y_conc, concept_weights=w_conc
            )
            self.assertEqual(tp.shape, (4, 1))
            self.assertEqual(cp.shape, (4, 10))
            losses = criterion(
                tp, cp, clv, clog, y_task, y_conc, w_conc,
                epoch=5, compute_concepts=(mtype != "blackbox"),
            )
            losses["total"].backward()

            # Partial intervention uncertainty propagation via mc_task_from_z
            if mtype in ("cbm", "cem"):
                model.eval()
                z = model.encoder(x)
                mask_p = torch.zeros(4, 10)
                mask_p[:, 0] = 1.0
                tm, ts, _, cp_eff, cv = model.mc_task_from_z(
                    z, n_mc=10, intervene_labels=torch.ones(4, 10), intervene_mask=mask_p
                )
                self.assertEqual(tm.shape, (4,))
                self.assertTrue(torch.allclose(cp_eff[:, 0], torch.ones(4)))
                self.assertTrue(torch.allclose(cv[:, 0], torch.zeros(4), atol=1e-6))

                # Full intervention flips prediction
                mask_all = torch.ones(4, 10)
                tp_zeros = model.intervene(x, torch.zeros(4, 10), mask_all)
                tp_ones  = model.intervene(x, torch.ones(4, 10), mask_all)
                self.assertFalse(torch.allclose(tp_zeros, tp_ones))

    def test_mc_dropout_preserves_batchnorm_and_supports_batch_size_1(self) -> None:
        for mtype in ("cbm", "cem", "blackbox"):
            model = build_model(self._base_cfg(mtype))
            model.eval()
            bn = model.encoder.stem[1]
            rm_before = bn.running_mean.clone()
            rv_before = bn.running_var.clone()

            x1 = torch.randn(1, 12, 120)
            unc = model.predict_with_uncertainty(x1, n_mc_epistemic=4, n_mc_aleatoric=6)
            self.assertEqual(unc["task_prob_mean"].shape, (1,))
            self.assertTrue(torch.equal(bn.running_mean, rm_before))
            self.assertTrue(torch.equal(bn.running_var, rv_before))
            self.assertFalse(model.training)

    def test_beat_level_concept_weights(self) -> None:
        r_peaks = np.array([100, 600])
        delin = {
            "ECG_P_Peaks":   np.array([60.0, np.nan]),
            "ECG_Q_Peaks":   np.array([90.0, 590.0]),
            "ECG_S_Peaks":   np.array([110.0, 610.0]),
            "ECG_T_Peaks":   np.array([200.0, 700.0]),
            "ECG_P_Onsets":  np.array([40.0, np.nan]),
            "ECG_R_Onsets":  np.array([85.0, 585.0]),
            "ECG_R_Offsets": np.array([115.0, 615.0]),
            "ECG_T_Offsets": np.array([230.0, np.nan]),
        }
        c1, w1 = get_concept_labels(450, 850, delin, r_peaks, np.array([np.nan, np.nan]))
        self.assertEqual(c1[0], 0)
        self.assertEqual(w1[0], 0)
        self.assertEqual(w1[5], 0)
        self.assertEqual(c1[2], 1)
        self.assertEqual(w1[2], 1)

    def test_evaluate_interventions_faithfulness_and_ois(self) -> None:
        rng = np.random.default_rng(0)
        samples = [
            {
                "signal": rng.standard_normal((12, 120)).astype(np.float32),
                "task_label": i % 2,
                "concept_labels": [int(v) for v in rng.integers(0, 2, size=10)],
                "concept_weights": [1, 1, 1, 1, 1, 0, 1, 1, 1, 1],
                "record_id": i,
            }
            for i in range(60)
        ]
        ds = HeartbeatDataset(samples)
        loader = DataLoader(ds, batch_size=16)
        model = build_model(self._base_cfg("cem"))

        metrics = evaluate(model, loader, device="cpu", criterion=CombinedCBMLoss())
        self.assertIn("val/loss_total", metrics)
        self.assertIn("task/ece", metrics)
        self.assertNotIn("concept/U/f1", metrics)

        int_res = evaluate_interventions(model, loader, device="cpu", n_random_trials=2)
        self.assertEqual(len(int_res["uncertainty_acc"]), 11)

        faith_res = evaluate_faithfulness_and_leakage(model, loader, device="cpu", max_ois_samples=60)
        self.assertIn("faithfulness/counterfactual_flip_rate", faith_res)
        self.assertIn("leakage/ois", faith_res)

        unc_res = evaluate_uncertainty_quality(model, loader, device="cpu", n_mc_epistemic=3, n_mc_aleatoric=4)
        self.assertIn("uncertainty/error_detection_auroc", unc_res)

    def test_pathology_models_and_outlier_normalizer(self) -> None:
        # Simulate extreme delineation outliers like -9338 / +8636 in concepts.parquet
        rng = np.random.default_rng(7)
        mat = rng.standard_normal((50, len(CONCEPT_COLUMNS))).astype(np.float32)
        mat[0, 7] = -9338.0
        mat[1, 15] = 8636.0
        df = pd.DataFrame(mat, index=list(range(50)), columns=CONCEPT_COLUMNS)
        norm = ConceptNormalizer(q_lo=0.02, q_hi=0.98, clip_z=4.0).fit(df, list(range(50)))

        raw_t = torch.from_numpy(mat[:4])
        mask_t = torch.ones_like(raw_t)
        normed = norm.transform_tensor(raw_t, mask_t)
        self.assertTrue(torch.all(normed.abs() <= 4.0 + 1e-5))

        x = torch.randn(4, 12, 200)
        for mtype in ("cbm", "cem", "blackbox"):
            m = build_pathology_model(model_type=mtype, base_channels=16, emb_size=8)
            logits, c_pred = m(x, concept_targets=normed, concept_mask=mask_t)
            self.assertEqual(logits.shape, (4, 5))
            self.assertEqual(c_pred.shape, (4, len(CONCEPT_COLUMNS)))

    def test_dense_per_timestamp_cbm_and_cem(self) -> None:
        raw_ss = np.random.default_rng(1).random((200, 12, 24)).astype(np.float32)
        seg_t = collapse_soft_seg_targets(raw_ss, concept_mode="wave6")
        self.assertEqual(seg_t.shape, (len(WAVE6_NAMES), 200))
        self.assertTrue(np.allclose(seg_t.sum(axis=0), 1.0, atol=1e-5))

        x = torch.randn(3, 12, 200)
        y_seg = torch.from_numpy(seg_t).unsqueeze(0).expand(3, -1, -1).clone()
        y_task = torch.randint(0, 2, (3, 5)).float()
        crit = DenseSoftSegLoss(n_mc_samples=2)

        for mtype in ("dense_cbm", "dense_cem"):
            cfg = {"model": {"type": mtype, "base_channels": 16, "num_concepts": 6, "emb_size": 8, "rand_int_prob": 0.5}}
            model = build_dense_model(cfg)
            model.train()
            out = model(x, concept_targets=y_seg)
            self.assertEqual(out["concept_probs"].shape, (3, 6, 200))
            self.assertEqual(out["task_logits"].shape, (3, 5))
            self.assertTrue(torch.allclose(out["concept_probs"].sum(dim=1), torch.ones(3, 200), atol=1e-5))
            losses = crit(out, y_seg, y_task)
            losses["total"].backward()

            # Span-level per-timestamp intervention & BatchNorm-safe MC-Dropout
            model.eval()
            bn = model.unet.stem[1]
            rm_before = bn.running_mean.clone()
            span_mask = torch.zeros(3, 1, 200)
            span_mask[:, :, 40:90] = 1.0
            span_lbl = torch.zeros(3, 6, 200)
            span_lbl[:, 2, 40:90] = 1.0  # force QRS=1 on [40:90]
            int_out = model.intervene(x, span_lbl, span_mask)
            self.assertTrue(torch.allclose(int_out["concept_probs"][:, 2, 40:90], torch.ones(3, 50), atol=1e-5))

            unc = model.predict_with_uncertainty(x[:1], n_mc_epistemic=3, n_mc_aleatoric=4, use_bf16=False)
            self.assertEqual(unc["concept_probs_mean"].shape, (1, 6, 200))
            self.assertEqual(unc["concept_total_std"].shape, (1, 6, 200))
            self.assertTrue(torch.equal(bn.running_mean, rm_before))

    def test_legacy_checkpoint_compatibility(self) -> None:
        legacy_run = REPO_ROOT / "models" / "20260423_064742"
        ckpt_path = legacy_run / "checkpoints" / "best_model.pt"
        if not ckpt_path.exists():
            self.skipTest("Legacy checkpoint not present on disk")
        with open(legacy_run / "config.yaml") as f:
            cfg = yaml.safe_load(f)
        model = build_model(cfg)
        state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        model.load_state_dict(state["model"], strict=True)

    def test_inactive_concept_masking_blocks_unsupervised_leakage(self) -> None:
        x = torch.randn(4, 12, 120)
        for mtype in ("cbm", "cem"):
            cfg = self._base_cfg(mtype)
            cfg["data"]["delineator"] = "soft_seg"
            cfg["model"]["concept_layer"]["mask_unsupervised_concepts"] = True
            model = build_model(cfg)
            self.assertEqual(model.inactive_concepts, (5,))

            # Task-only backward pass must send zero gradient into inactive concept weights
            model.train()
            tp, _, _, _ = model.forward_with_logits(x)
            tp.sum().backward()
            if mtype == "cbm":
                # Classifier weight for inactive U concept (idx 5) receives zero gradient
                self.assertAlmostEqual(float(model.classifier.linear.weight.grad[0, 5].item()), 0.0, places=7)
            else:
                # CEM classifier columns for U (slice 5*8 : 6*8) receive zero gradient
                w_grad = model.classifier.linear.weight.grad[0, 5 * 8 : 6 * 8]
                self.assertTrue(torch.allclose(w_grad, torch.zeros_like(w_grad)))

    def test_dense_baselines_detach_and_leakage_probe(self) -> None:
        set_seed(123)
        x = torch.randn(4, 12, 200)
        y_seg = torch.softmax(torch.randn(4, 6, 200), dim=1)
        y_task = torch.randint(0, 2, (4, 5)).float()

        # 1. DenseBlackBox & DenseMultiTask forward/backward + MC uncertainty
        for mtype, comp_seg in (("dense_blackbox", False), ("dense_multitask", True)):
            cfg = {"model": {"type": mtype, "base_channels": 16, "num_concepts": 6}}
            m = build_dense_model(cfg)
            m.train()
            out = m(x, concept_targets=y_seg)
            self.assertEqual(out["task_logits"].shape, (4, 5))
            crit = DenseSoftSegLoss(n_mc_samples=2, compute_seg=comp_seg, use_heteroscedastic=False)
            losses = crit(out, y_seg, y_task)
            losses["total"].backward()

            m.eval()
            bn_m = m.unet.stem[1]
            rm_before = bn_m.running_mean.clone()
            unc_m = m.predict_with_uncertainty(x[:2], n_mc_epistemic=3, use_bf16=False)
            self.assertEqual(unc_m["task_probs_mean"].shape, (2, 5))
            self.assertEqual(unc_m["task_probs_std"].shape, (2, 5))
            self.assertTrue(torch.equal(bn_m.running_mean, rm_before))

        # 2. DenseSoftSegCBM with detach_concepts=True blocks task gradient into concept_proj
        cfg_det = {"model": {"type": "dense_cbm", "base_channels": 16, "num_concepts": 6, "detach_concepts": True}}
        m_det = build_dense_model(cfg_det)
        m_det.train()
        out_det = m_det(x, concept_targets=y_seg)
        out_det["task_logits"].sum().backward()
        self.assertIsNone(m_det.concept_proj.weight.grad)

        # 3. Leakage probe & bootstrap CI smoke test
        class _DummyDenseDS(torch.utils.data.Dataset):
            def __init__(self, n: int = 24):
                g = np.random.default_rng(7)
                self.x = g.standard_normal((n, 12, 100)).astype(np.float32)
                raw = g.random((n, 6, 100)).astype(np.float32)
                self.seg = raw / raw.sum(axis=1, keepdims=True)
                self.y = np.zeros((n, 5), dtype=np.float32)
                for c in range(5):
                    self.y[c::2, c] = 1.0
            def __len__(self): return len(self.x)
            def __getitem__(self, i):
                return {"signal": torch.from_numpy(self.x[i]), "seg": torch.from_numpy(self.seg[i]),
                        "pathology": torch.from_numpy(self.y[i]), "record_id": i}

        dl = DataLoader(_DummyDenseDS(24), batch_size=8)
        m_cem = build_dense_model({"model": {"type": "dense_cem", "base_channels": 16, "num_concepts": 6, "emb_size": 4}})
        ev = evaluate_dense(m_cem, dl, "cpu", WAVE6_NAMES, use_bf16=False, compute_bootstrap=True)
        self.assertIn("pathology/macro_auroc_ci95_lo", ev)
        self.assertIn("pathology/macro_ece", ev)
        self.assertIn("pathology/macro_brier", ev)
        tunc = evaluate_dense_task_uncertainty(m_cem, dl, "cpu", n_mc_epistemic=3, use_bf16=False)
        self.assertIn("uncertainty/task_error_detection_auroc", tunc)
        self.assertIn("pathology/mc_ece", tunc)
        lk = evaluate_dense_leakage_probe(m_cem, dl, dl, "cpu", use_bf16=False)
        self.assertIn("leakage/bottleneck_excess_auroc", lk)

        # 4. PathologyCBM with detach_concepts=True blocks task gradient into concept_head
        pm = build_pathology_model("cbm", base_channels=16, detach_concepts=True)
        pm.train()
        plog, _ = pm(torch.randn(4, 12, 200))
        plog.sum().backward()
        self.assertIsNone(pm.concept_head.weight.grad)

        # 5. Bootstrap helper sanity
        yt = np.tile(np.array([[1, 0, 1, 0, 1], [0, 1, 0, 1, 0]], dtype=np.float32), (10, 1))
        yp = np.clip(yt * 0.8 + 0.1, 0.0, 1.0)
        bci = bootstrap_multilabel_auroc(yt, yp, ["NORM", "MI", "STTC", "CD", "HYP"], n_boot=50, seed=0)
        self.assertAlmostEqual(bci["pathology/macro_auroc_ci95_lo"], 1.0, places=5)

    def test_external_georgia_preprocessing_and_intervals(self) -> None:
        # 1. Lead canonicalization
        perm_names = ["v6", "II", "aVR", "I", "III", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5"]
        raw = np.arange(12 * 100, dtype=np.float32).reshape(12, 100)
        canon = canonicalize_leads(raw, perm_names)
        self.assertEqual(canon.shape, (12, 100))
        self.assertTrue(np.allclose(canon[0], raw[3]))  # I
        self.assertTrue(np.allclose(canon[1], raw[1]))  # II
        self.assertTrue(np.allclose(canon[3], raw[2]))  # aVR
        with self.assertRaises(ValueError):
            canonicalize_leads(raw[:11], CANONICAL_LEADS[:11])

        # 2. 500 Hz -> 100 Hz preprocessing + 60 Hz suppression + edge-pad for short signals
        fs = 500
        t = np.arange(fs * 10, dtype=np.float32) / fs
        ecg_like = np.sin(2.0 * np.pi * 1.2 * t) + 0.5 * np.sin(2.0 * np.pi * 60.0 * t)
        sig_500 = np.tile(ecg_like, (12, 1)).astype(np.float32)
        sig_500[0, 100] = np.nan
        out_100 = preprocess_to_dense_100hz(sig_500, fs=500, target_sr=100, target_sec=10.0)
        self.assertEqual(out_100.shape, (12, 1000))
        self.assertTrue(np.isfinite(out_100).all())
        self.assertAlmostEqual(float(out_100[1].mean()), 0.0, places=4)
        self.assertAlmostEqual(float(out_100[1].std()), 1.0, delta=0.05)

        short_100 = preprocess_to_dense_100hz(sig_500[:, :2500], fs=500, target_sr=100, target_sec=10.0)
        self.assertEqual(short_100.shape, (12, 1000))

        # 3. SNOMED-CT -> PTB-XL 5-superclass mapping
        self.assertEqual(map_snomed_to_superclasses(["426783006"]).tolist(), [1, 0, 0, 0, 0])         # NSR only -> NORM
        self.assertEqual(map_snomed_to_superclasses(["426783006", "164889003"]).tolist(), [0, 0, 0, 0, 0])  # AF excludes NORM
        self.assertEqual(map_snomed_to_superclasses(["164909002"]).tolist(), [0, 0, 0, 1, 0])         # LBBB -> CD
        self.assertEqual(map_snomed_to_superclasses(["164873001", "59931005"]).tolist(), [0, 0, 1, 0, 1])  # LVH + TInv -> STTC+HYP

        # 4. Wave-span clinical interval extraction on synthetic periodic wave6 cycle (60 bpm)
        cycle = np.array(
            [0] * 10 + [1] * 6 + [2] * 10 + [3] * 10 + [4] * 20 + [5] * 44,
            dtype=np.int64,
        )  # 100 samples = 1.0 s -> PR=160ms, QRS=100ms, QT=400ms
        seq = np.tile(cycle, 10)[:1000]
        info = extract_wave_spans_and_intervals(seq, fs=100)
        self.assertEqual(info["n_beats"], 10)
        self.assertAlmostEqual(info["pred_hr_bpm"], 60.0, delta=1.0)
        self.assertAlmostEqual(info["qrs_ms"], 100.0, delta=1.0)
        self.assertAlmostEqual(info["pr_ms"], 160.0, delta=1.0)
        self.assertAlmostEqual(info["qt_ms"], 400.0, delta=1.0)
        self.assertAlmostEqual(info["p_present_beat_frac"], 1.0, places=5)
        self.assertAlmostEqual(info["canonical_transition_rate"], 1.0, places=5)

    def test_explorer_georgia_and_ptbxl_endpoints(self) -> None:
        exp_dir = REPO_ROOT / "explorer"
        if str(exp_dir) not in sys.path:
            sys.path.insert(0, str(exp_dir))
        import app as exp_app  # noqa: E402

        if not exp_app._has_georgia():
            self.skipTest("Georgia dataset symlink not present")

        ds_payload = exp_app.list_datasets()
        ds_ids = [d["id"] for d in ds_payload["datasets"]]
        self.assertIn("ptbxl", ds_ids)
        self.assertIn("georgia", ds_ids)

        rec_list = exp_app.list_records(limit=10, fold="0_clean", dataset="georgia")
        self.assertEqual(rec_list["dataset"], "georgia")
        self.assertGreater(rec_list["total"], 8000)
        self.assertEqual(rec_list["record_ids"][:3], ["E00021", "E00005", "E10019"])

        rec_ga = exp_app.get_record("E00021", dataset="georgia")
        self.assertEqual(rec_ga["dataset"], "georgia")
        self.assertFalse(rec_ga["has_soft_seg_gt"])
        self.assertEqual(len(rec_ga["signal"]), 12)

        dense_runs = exp_app.discover_dense_runs()
        if "dense_cem_s42" in dense_runs:
            dreq = exp_app.DenseInferRequest(
                ecg_id="E10019",
                dataset="georgia",
                model_id="dense_cem_s42",
                compare_model_id="dense_cbm_detached_s42" if "dense_cbm_detached_s42" in dense_runs else None,
                blackbox_model_id="dense_blackbox_s42" if "dense_blackbox_s42" in dense_runs else None,
            )
            dres = exp_app.dense_infer(dreq)
            self.assertEqual(dres["dataset"], "georgia")
            self.assertIsNone(dres["gt_probs"])
            self.assertIn("qrs_ms", dres["intervals"])
            self.assertGreater(dres["intervals"]["qrs_ms"], 120.0)

    def test_vercel_exact_intervention_math(self) -> None:
        exp_dir = REPO_ROOT / "explorer"
        if str(exp_dir) not in sys.path:
            sys.path.insert(0, str(exp_dir))
        import build_vercel  # noqa: E402

        torch.manual_seed(0)
        x = torch.randn(2, 12, 120)
        int_dict = {0: 1.0, 2: 0.0, 4: 1.0, 7: 0.0}
        int_labels = torch.zeros(2, 10)
        int_mask = torch.zeros(2, 10)
        for k, v in int_dict.items():
            int_labels[:, k] = v
            int_mask[:, k] = 1.0

        for mtype in ("cem", "cbm"):
            cfg = self._base_cfg(mtype)
            cfg["data"]["delineator"] = "soft_seg"
            cfg["model"]["concept_layer"]["mask_unsupervised_concepts"] = True
            model = build_model(cfg).eval()
            head = build_vercel._export_window_head(model)
            sw_entry, xb_np = build_vercel._compute_window_sweep_with_export(model, cfg, x.numpy(), "cpu")

            with torch.no_grad():
                pt_out = model.intervene(torch.from_numpy(xb_np), int_labels, int_mask)

            inactive = set(head.get("inactive_concepts", []))
            for bi in range(2):
                logit = float(head["bias"])
                c_probs = sw_entry["concept_probs"][bi]
                if mtype == "cem":
                    for k in range(10):
                        p_eff = 0.0 if k in inactive else int_dict.get(k, c_probs[k])
                        logit += p_eff * sw_entry["a_pos"][bi][k] + (1.0 - p_eff) * sw_entry["a_neg"][bi][k]
                else:
                    v_zero = float(np.exp(-20.0))
                    for k in range(10):
                        if k in inactive:
                            p_eff, v_eff = 0.0, 0.0
                        elif k in int_dict:
                            p_eff, v_eff = int_dict[k], v_zero
                        else:
                            p_eff, v_eff = c_probs[k], sw_entry["c_vars"][bi][k]
                        logit += head["w_prob"][k] * p_eff + (head["w_var"][k] * v_eff if head["use_uncertainty"] else 0.0)
                exp_prob = 1.0 / (1.0 + float(np.exp(-np.clip(logit, -30.0, 30.0))))
                self.assertAlmostEqual(float(pt_out[bi, 0].item()), exp_prob, delta=5e-3)


if __name__ == "__main__":
    unittest.main()
