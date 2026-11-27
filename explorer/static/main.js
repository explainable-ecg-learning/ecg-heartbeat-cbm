/* ── ECG Heartbeat CBM & CEM Explorer — main.js ───────────────────────────
 * Multi-model D3 v7 frontend:
 *  - Dynamic CBM / CEM model selection & side-by-side comparison
 *  - Synchronized ECG waveform + sliding-window concept activation lineplots & heatmap
 *  - Live click-to-intervene concept pills updating Task prediction in real time
 *  - Test set benchmark table + Concept Intervention Curves (Uncertainty vs Random)
 * ────────────────────────────────────────────────────────────────────────── */

const URL_PARAMS = new URLSearchParams(window.location.search);
const API = (URL_PARAMS.get("api") || window.EXPLORER_API_BASE || "").replace(/\/$/, "");
const STATIC_MODE = Boolean(!URL_PARAMS.get("api") && window.EXPLORER_STATIC_MODE);

let _staticManifestPromise = null;
let _staticExtBenchPromise = null;
const _staticRecordCache = new Map();

function _sigmoid(x) {
  const c = Math.max(-30, Math.min(30, x));
  return 1.0 / (1.0 + Math.exp(-c));
}

async function _loadStaticManifest() {
  if (!_staticManifestPromise) {
    _staticManifestPromise = fetch("./data/manifest.json").then(r => {
      if (!r.ok) throw new Error(`Static manifest HTTP ${r.status}`);
      return r.json();
    });
  }
  return _staticManifestPromise;
}

async function _loadStaticExternalBenchmark() {
  if (!_staticExtBenchPromise) {
    _staticExtBenchPromise = fetch("./data/external_benchmark.json")
      .then(r => (r.ok ? r.json() : { available: false }))
      .catch(() => ({ available: false }));
  }
  return _staticExtBenchPromise;
}

function _unpackStaticRecord(raw) {
  if (!raw || raw._unpacked) return raw;
  raw._unpacked = true;

  // 1. Dequantize sweep arrays & materialize Black-Box zero arrays in RAM
  Object.values(raw.sweeps || {}).forEach(sw => {
    const N = (sw.task_probs || []).length;
    const K = (sw.concept_names || []).length;
    if (sw.concept_probs === null) {
      sw.concept_probs   = Array.from({ length: N }, () => new Array(K).fill(0));
      sw.concept_stds    = Array.from({ length: N }, () => new Array(K).fill(0));
      sw.concept_stds_mc = Array.from({ length: N }, () => new Array(K).fill(0));
    } else if (sw._cstd_scale) {
      const inv = 1.0 / sw._cstd_scale;
      if (sw.concept_stds)    sw.concept_stds    = sw.concept_stds.map(row => row.map(v => v * inv));
      if (sw.concept_stds_mc) sw.concept_stds_mc = sw.concept_stds_mc.map(row => row.map(v => v * inv));
    }
  });

  // 2. Dequantize dense Zenodo GT probabilities & reconstruct record.soft_seg without duplicating JSON bytes
  const gtScale = raw.dense_prob_scale || 0;
  if (gtScale > 0 && raw.dense_gt_probs) {
    const inv = 1.0 / gtScale;
    const nonTp = ["P", "PQ", "QRS", "ST", "T"].filter(k => Array.isArray(raw.dense_gt_probs[k]));
    nonTp.forEach(k => { raw.dense_gt_probs[k] = raw.dense_gt_probs[k].map(v => v * inv); });
    const Tseg = nonTp.length ? raw.dense_gt_probs[nonTp[0]].length : 0;
    if (Tseg > 0 && !raw.dense_gt_probs.TP) {
      const tp = new Array(Tseg);
      for (let t = 0; t < Tseg; t++) {
        let s = 0.0;
        for (let i = 0; i < nonTp.length; i++) s += raw.dense_gt_probs[nonTp[i]][t];
        tp[t] = Math.max(0.0, 1.0 - s);
      }
      raw.dense_gt_probs.TP = tp;
    }
  }
  if (raw.record && !raw.record.soft_seg && raw.dense_gt_probs) {
    raw.record.soft_seg = { ...raw.dense_gt_probs, sampling_rate: 100 };
  }

  // 3. Dequantize dense segmentation models
  Object.values(raw.dense || {}).forEach(de => {
    if (!de || !de.has_seg) return;
    const cNames = de.concept_names || ["P", "PQ", "QRS", "ST", "T", "TP"];
    const nonTp  = cNames.filter(n => n !== "TP");
    const pScale = de._prob_scale || 0;
    const sScale = de._std_scale || 0;
    if (pScale > 0 && de.concept_probs) {
      const invP = 1.0 / pScale;
      nonTp.forEach(n => {
        if (Array.isArray(de.concept_probs[n])) de.concept_probs[n] = de.concept_probs[n].map(v => v * invP);
      });
      const Tseg = nonTp.length && de.concept_probs[nonTp[0]] ? de.concept_probs[nonTp[0]].length : 0;
      if (Tseg > 0 && cNames.includes("TP") && !de.concept_probs.TP) {
        const tp = new Array(Tseg);
        for (let t = 0; t < Tseg; t++) {
          let s = 0.0;
          for (let i = 0; i < nonTp.length; i++) s += de.concept_probs[nonTp[i]][t];
          tp[t] = Math.max(0.0, 1.0 - s);
        }
        de.concept_probs.TP = tp;
      }
      if (de.concept_stds) {
        nonTp.forEach(n => {
          if (Array.isArray(de.concept_stds[n])) de.concept_stds[n] = de.concept_stds[n].map(v => v * invP);
        });
        if (Tseg > 0 && cNames.includes("TP") && !de.concept_stds.TP) {
          de.concept_stds.TP = new Array(Tseg).fill(0.0);
        }
      }
    }
    if (sScale > 0) {
      const invS = 1.0 / sScale;
      if (Array.isArray(de.mean_std_timeline))    de.mean_std_timeline    = de.mean_std_timeline.map(v => v * invS);
      if (Array.isArray(de.mean_std_timeline_mc)) de.mean_std_timeline_mc = de.mean_std_timeline_mc.map(v => v * invS);
    }
  });

  return raw;
}

async function _loadStaticRecord(dataset, ecgId) {
  const key = `${dataset}_${ecgId}`;
  if (!_staticRecordCache.has(key)) {
    const p = fetch(`./data/records/${encodeURIComponent(key)}.json`).then(r => {
      if (!r.ok) throw new Error(`Static record ${key} HTTP ${r.status}`);
      return r.json();
    }).then(_unpackStaticRecord);
    _staticRecordCache.set(key, p);
  }
  return _staticRecordCache.get(key);
}

function _nearestIdx(arr, target) {
  let best = 0, bestDist = Infinity;
  for (let i = 0; i < arr.length; i++) {
    const d = Math.abs(arr[i] - target);
    if (d < bestDist) { bestDist = d; best = i; }
  }
  return best;
}

function _evalStaticWindowInfer(sw, head, ni, interventions, includeEpistemic = false) {
  const cProbs = [...sw.concept_probs[ni]];
  const cStdSrc = (includeEpistemic && sw.concept_stds_mc) ? sw.concept_stds_mc[ni] : sw.concept_stds[ni];
  const cStds  = [...cStdSrc];
  const tStd   = (includeEpistemic && sw.task_stds_mc) ? sw.task_stds_mc[ni] : sw.task_stds[ni];
  const tEpi   = (includeEpistemic && sw.task_epistemic_mc) ? sw.task_epistemic_mc[ni] : 0.0;
  const hasInt = Boolean(interventions && Object.keys(interventions).length > 0);
  if (!hasInt || !head || head.model_type === "blackbox") {
    return {
      task_prob:      sw.task_probs[ni],
      task_std:       tStd,
      task_epistemic: tEpi,
      task_aleatoric: sw.task_stds[ni],
      concept_probs:  cProbs,
      concept_stds:   cStds,
    };
  }
  const inactive = new Set(head.inactive_concepts || []);
  const K = cProbs.length;
  let logit = head.bias || 0.0;

  if (head.model_type === "cem") {
    const aPos = sw.a_pos[ni];
    const aNeg = sw.a_neg[ni];
    for (let k = 0; k < K; k++) {
      let pEff = cProbs[k];
      if (inactive.has(k)) {
        pEff = 0.0;
      } else if (interventions[k] !== undefined || interventions[String(k)] !== undefined) {
        const iv = interventions[k] !== undefined ? interventions[k] : interventions[String(k)];
        pEff = Number(iv);
        cProbs[k] = pEff;
        cStds[k] = 0.0;
      }
      logit += pEff * aPos[k] + (1.0 - pEff) * aNeg[k];
    }
  } else {
    const wProb = head.w_prob || [];
    const wVar  = head.w_var || [];
    const cVars = sw.c_vars ? sw.c_vars[ni] : new Array(K).fill(0.0);
    const vZero = Math.exp(-20.0);
    for (let k = 0; k < K; k++) {
      let pEff = cProbs[k];
      let vEff = cVars[k] || 0.0;
      if (inactive.has(k)) {
        pEff = 0.0;
        vEff = 0.0;
      } else if (interventions[k] !== undefined || interventions[String(k)] !== undefined) {
        const iv = interventions[k] !== undefined ? interventions[k] : interventions[String(k)];
        pEff = Number(iv);
        vEff = vZero;
        cProbs[k] = pEff;
        cStds[k] = 0.0;
      }
      logit += (wProb[k] || 0.0) * pEff + (head.use_uncertainty ? (wVar[k] || 0.0) * vEff : 0.0);
    }
  }

  return {
    task_prob:      _sigmoid(logit),
    task_std:       0.0,
    task_epistemic: 0.0,
    task_aleatoric: 0.0,
    concept_probs:  cProbs,
    concept_stds:   cStds,
  };
}

function _buildStaticDenseEntry(rec, modelId, includeEpistemic, intMode, intConcepts, intSpan = null) {
  const de = rec.dense?.[modelId];
  if (!de) return null;
  if (!de.has_seg) {
    return {
      model_id:               modelId,
      model_type:             de.model_type,
      has_seg:                false,
      supports_interventions: false,
      concept_names:          de.concept_names,
      concept_probs:          null,
      concept_stds:           null,
      seg_spans:              null,
      mean_std_timeline:      null,
      intervals:              null,
      pathology:              de.base_pathology,
      base_pathology:         de.base_pathology,
      pathology_std:          (includeEpistemic && de.pathology_std_mc) ? de.pathology_std_mc : de.pathology_std,
    };
  }

  const cNames = de.concept_names || ["P", "PQ", "QRS", "ST", "T", "TP"];
  const cpOut  = {};
  cNames.forEach(n => { cpOut[n] = [...de.concept_probs[n]]; });
  const csOut = de.concept_stds ? {} : null;
  if (csOut) cNames.forEach(n => { csOut[n] = [...de.concept_stds[n]]; });
  const stdLine = [...((includeEpistemic && de.mean_std_timeline_mc) ? de.mean_std_timeline_mc : de.mean_std_timeline)];
  let patho = { ...de.base_pathology };

  const mode = (intMode || "").toLowerCase();
  const T = cpOut[cNames[0]]?.length || 0;
  const isWinSpan = Array.isArray(intSpan) && intSpan.length === 2;
  const lo = isWinSpan ? Math.max(0, Math.min(T, Math.round(intSpan[0]))) : 0;
  const hi = isWinSpan ? Math.max(lo, Math.min(T, Math.round(intSpan[1]))) : T;
  const binIdx = isWinSpan ? Math.max(0, Math.min(9, Math.floor((lo + hi) / 200))) : -1;
  const pathoKeys = Object.keys(de.base_pathology || {});

  const _mapWinPatho = arr => {
    const mapped = {};
    pathoKeys.forEach((name, idx) => { mapped[name] = Number(arr[idx]); });
    return mapped;
  };

  if (de.supports_interventions && mode) {
    if (mode === "gt" && rec.dense_gt_probs) {
      if (binIdx >= 0 && de.interventions_win?.gt?.[binIdx]) {
        patho = _mapWinPatho(de.interventions_win.gt[binIdx]);
      } else if (de.interventions?.gt) {
        patho = { ...de.interventions.gt };
      }
      for (let t = lo; t < hi; t++) {
        cNames.forEach(n => { cpOut[n][t] = rec.dense_gt_probs[n][t]; });
        if (csOut) cNames.forEach(n => { csOut[n][t] = 0; });
        stdLine[t] = 0;
      }
    } else if ((mode === "ablate" || mode === "force") && intConcepts?.length) {
      const target = intConcepts[0];
      const ikey = `${mode}:${target}`;
      if (binIdx >= 0 && de.interventions_win?.[ikey]?.[binIdx]) {
        patho = _mapWinPatho(de.interventions_win[ikey][binIdx]);
      } else if (de.interventions?.[ikey]) {
        patho = { ...de.interventions[ikey] };
      }
      if (mode === "ablate" && cpOut[target]) {
        for (let t = lo; t < hi; t++) {
          cpOut[target][t] = 0.0;
          let sum = 0.0;
          cNames.forEach(n => { sum += cpOut[n][t]; });
          const denom = Math.max(1e-6, sum);
          cNames.forEach(n => { cpOut[n][t] = Number((cpOut[n][t] / denom).toFixed(3)); });
          if (csOut?.[target]) csOut[target][t] = 0.0;
        }
      } else if (mode === "force" && cpOut[target]) {
        for (let t = lo; t < hi; t++) {
          cNames.forEach(n => { cpOut[n][t] = n === target ? 1.0 : 0.0; });
          if (csOut) cNames.forEach(n => { csOut[n][t] = 0.0; });
          stdLine[t] = 0.0;
        }
      }
    }
  }

  return {
    model_id:               modelId,
    model_type:             de.model_type,
    has_seg:                true,
    supports_interventions: Boolean(de.supports_interventions),
    concept_names:          cNames,
    concept_probs:          cpOut,
    concept_stds:           csOut,
    seg_spans:              mode ? null : (de.seg_spans || null),
    mean_std_timeline:      stdLine,
    intervals:              de.intervals || null,
    pathology:              patho,
    base_pathology:         de.base_pathology,
    pathology_std:          (includeEpistemic && de.pathology_std_mc) ? de.pathology_std_mc : de.pathology_std,
  };
}

let _globalLoadCount = 0;

function beginGlobalLoad() {
  _globalLoadCount += 1;
  document.getElementById("global-loading-bar")?.classList.add("active");
}

function endGlobalLoad() {
  _globalLoadCount = Math.max(0, _globalLoadCount - 1);
  if (_globalLoadCount === 0) {
    document.getElementById("global-loading-bar")?.classList.remove("active");
  }
}

function setSectionLoading(sectionId, loading) {
  const el = document.getElementById(sectionId);
  if (el) el.classList.toggle("is-loading", Boolean(loading));
}

function setLoadingPill(pillId, loading, text = null) {
  const el = document.getElementById(pillId);
  if (!el) return;
  if (text !== null && el.lastElementChild) {
    el.lastElementChild.textContent = text;
  }
  el.style.display = loading ? "" : "none";
}

async function apiGet(path) {
  beginGlobalLoad();
  try {
    return await _apiGetInner(path);
  } finally {
    endGlobalLoad();
  }
}

async function _apiGetInner(path) {
  if (!STATIC_MODE) {
    const res = await fetch(`${API}${path}`);
    if (!res.ok) throw new Error(await res.text());
    return res.json();
  }
  const u = new URL(path, "http://local");
  const p = u.pathname;
  const manifest = await _loadStaticManifest();
  if (p === "/health") {
    return {
      status:   "ok",
      device:   "STATIC CDN",
      datasets: (manifest.datasets || []).map(d => d.id),
      n_models: (manifest.models?.models || []).length,
    };
  }
  if (p === "/datasets") {
    return { default_dataset: manifest.default_dataset || "ptbxl", datasets: manifest.datasets || [] };
  }
  if (p === "/external-benchmark") {
    return _loadStaticExternalBenchmark();
  }
  if (p === "/models") return manifest.models;
  if (p === "/dense-models") return manifest.dense_models;
  if (p.startsWith("/model/") && p.endsWith("/summary")) {
    const mid = decodeURIComponent(p.slice(7, -8));
    return manifest.summaries[mid] || null;
  }
  if (p === "/records") {
    const ds = u.searchParams.get("dataset") || "ptbxl";
    const fold = u.searchParams.get("fold");
    const dx = (u.searchParams.get("dx") || "ALL").toUpperCase();
    const search = (u.searchParams.get("search") || "").trim().toLowerCase();
    let list = manifest.records_by_dataset[ds] || [];
    if (fold && fold !== "all") {
      list = list.filter(r => String(r.split ?? (ds === "georgia" ? r.severity_tier : r.fold) ?? "") === String(fold));
    }
    if (dx && dx !== "ALL") {
      list = list.filter(r => (r.tags || []).includes(dx));
    }
    if (search) {
      list = list.filter(r =>
        String(r.ecg_id).toLowerCase().includes(search) ||
        (r.label_names || []).join(" ").toLowerCase().includes(search) ||
        (r.dx || []).join(" ").toLowerCase().includes(search)
      );
    }
    return { dataset: ds, record_ids: list.map(r => r.ecg_id), records: list, total: list.length };
  }
  if (p.startsWith("/record/")) {
    const ecgId = decodeURIComponent(p.slice(8));
    const ds = u.searchParams.get("dataset") || "ptbxl";
    const rec = await _loadStaticRecord(ds, ecgId);
    return rec.record;
  }
  throw new Error(`Unsupported static GET route: ${path}`);
}

async function apiPost(path, body) {
  beginGlobalLoad();
  try {
    return await _apiPostInner(path, body);
  } finally {
    endGlobalLoad();
  }
}

async function _apiPostInner(path, body) {
  if (!STATIC_MODE) {
    const res = await fetch(`${API}${path}`, {
      method:  "POST",
      headers: { "Content-Type": "application/json" },
      body:    JSON.stringify(body),
    });
    if (!res.ok) throw new Error(await res.text());
    return res.json();
  }
  const manifest = await _loadStaticManifest();
  const ds = body.dataset || "ptbxl";
  const rec = await _loadStaticRecord(ds, body.ecg_id);

  if (path === "/sweep") {
    const mid = body.model_id || manifest.models.default_model;
    const sw = rec.sweeps[mid];
    const cid = body.compare_model_id;
    const csw = (cid && rec.sweeps[cid]) ? rec.sweeps[cid] : null;
    return {
      ecg_id:        body.ecg_id,
      dataset:       ds,
      model_id:      mid,
      model_type:    sw.model_type,
      stride:        sw.stride,
      t_fixed:       sw.t_fixed,
      sampling_rate: sw.sampling_rate,
      starts:        sw.starts,
      task_probs:    sw.task_probs,
      task_stds:     sw.task_stds,
      concept_probs: sw.concept_probs,
      concept_stds:  sw.concept_stds,
      concept_names: sw.concept_names,
      gt_task:       rec.sweep_gt_task,
      gt_concepts:   rec.sweep_gt_concepts,
      compare: csw ? {
        model_id:      cid,
        model_type:    csw.model_type,
        task_probs:    csw.task_probs,
        task_stds:     csw.task_stds,
        concept_probs: csw.concept_probs,
        concept_stds:  csw.concept_stds,
      } : null,
    };
  }
  if (path === "/infer") {
    const mid = body.model_id || manifest.models.default_model;
    const sw = rec.sweeps[mid];
    const head = manifest.window_heads[mid];
    const ni = _nearestIdx(sw.starts, body.start);
    const prim = _evalStaticWindowInfer(sw, head, ni, body.interventions, Boolean(body.include_epistemic));
    const out = {
      ecg_id:        body.ecg_id,
      dataset:       ds,
      model_id:      mid,
      model_type:    sw.model_type,
      ...prim,
      concept_names: sw.concept_names,
      gt_concepts:   rec.sweep_gt_concepts ? rec.sweep_gt_concepts[ni] : null,
      gt_task:       rec.sweep_gt_task ? rec.sweep_gt_task[ni] : null,
    };
    const cid = body.compare_model_id;
    if (cid && rec.sweeps[cid]) {
      const csw = rec.sweeps[cid];
      out.compare = {
        model_id:   cid,
        model_type: csw.model_type,
        ..._evalStaticWindowInfer(csw, manifest.window_heads[cid], ni, body.interventions, Boolean(body.include_epistemic)),
      };
    }
    return out;
  }
  if (path === "/dense-infer") {
    const mid = body.model_id || manifest.dense_models.default_model;
    const prim = _buildStaticDenseEntry(
      rec, mid, body.include_epistemic, body.intervene_mode, body.intervene_concepts, body.intervene_span || null
    );
    const out = {
      ecg_id:         body.ecg_id,
      dataset:        ds,
      sampling_rate:  100,
      gt_probs:       rec.dense_gt_probs || null,
      gt_spans:       rec.dense_gt_spans || null,
      intervene_mode: body.intervene_mode || null,
      ...prim,
    };
    if (body.compare_model_id) {
      const cmp = _buildStaticDenseEntry(
        rec, body.compare_model_id, body.include_epistemic, body.intervene_mode, body.intervene_concepts, body.intervene_span || null
      );
      if (cmp) out.compare = cmp;
    }
    if (body.blackbox_model_id) {
      const bb = _buildStaticDenseEntry(rec, body.blackbox_model_id, body.include_epistemic, null, null, null);
      if (bb) out.blackbox = bb;
    }
    return out;
  }
  throw new Error(`Unsupported static POST route: ${path}`);
}
const FALLBACK_PALETTE = [
  "#2b5b84", "#7c5295", "#b83b2a", "#c87137", "#2d7a56",
  "#a68a2c", "#2a7f8e", "#4b5599", "#9e4268", "#2d7a56",
];
const CONCEPT_COLORS = {
  Task: "#b83b2a",
  P: "#2b5b84", Q: "#7c5295", R: "#b83b2a", S: "#c87137", T: "#2d7a56", U: "#a68a2c",
  PR: "#2a7f8e", PQ: "#2a7f8e", QRS: "#b83b2a", ST: "#9e4268", QT: "#2d7a56", TP: "#767676",
};

function conceptColor(name, idx) {
  return CONCEPT_COLORS[name] || FALLBACK_PALETTE[idx % FALLBACK_PALETTE.length];
}

const state = {
  models:         [],
  datasets:       [],
  dataset:        "ptbxl",
  denseBenchDataset: "ptbxl",
  activePreset:   "ptbxl_norm",
  hasSoftSegGt:   true,
  externalBenchmark: null,
  primaryModel:   null,
  compareModel:   "",
  primarySummary: null,
  compareSummary: null,
  conceptNames:   ["P", "Q", "R", "S", "T", "U", "PR", "QRS", "ST", "QT"],
  ecgId:          null,
  samplingRate:   500,
  signal:         null,
  delineation:    null,
  nSamples:       0,
  nSamplesDs:     0,
  downsample:     2,
  tFixed:         500,
  leadIdx:        0,
  windowStart:    0,
  sweepData:      null,
  timelineView:   "intervals",
  interventions:  {},
  playTimer:      null,
  denseModels:    null,
  densePrimary:   null,
  denseCompare:   "",
  denseBlackbox:  "",
  denseData:      null,
  denseIntMode:   null,     // null | "gt" | "ablate" | "force"
  denseIntConcept: null,    // e.g. "ST"
  ecgSegOverlay:  "dense_cem",
};

let signalG;
let _xScaleSig = null;
let _winWpx = 0;

const SIG_MARGIN  = { top: 16, right: 24, bottom: 32, left: 48 };
const SIG_HEIGHT  = 175;
const LINE_HEIGHT = 185;

function chartWidth() {
  const el = document.getElementById("signal-container");
  return Math.max(480, (el ? el.clientWidth : 960) - SIG_MARGIN.left - SIG_MARGIN.right);
}

function taskColour(p) {
  if (p >= 0.7) return "#2d7a56";
  if (p >= 0.4) return "#a67c00";
  return "#b83b2a";
}

function stopPlayback() {
  if (state.playTimer) {
    clearInterval(state.playTimer);
    state.playTimer = null;
    document.getElementById("play-btn").textContent = "▶ Play Sweep";
  }
}

// ── Boot ───────────────────────────────────────────────────────────────────
async function init() {
  try {
    const [h, dsData, extData] = await Promise.all([
      apiGet("/health"),
      apiGet("/datasets"),
      apiGet("/external-benchmark").catch(() => ({ available: false })),
    ]);
    state.datasets = dsData.datasets || [];
    state.dataset = dsData.default_dataset || "ptbxl";
    populateDatasetSelect();
    if (extData?.available) state.externalBenchmark = extData;
    const nDs = (h.datasets || state.datasets || []).length;
    const devLbl = STATIC_MODE ? "STATIC CDN" : String(h.device || "CPU").toUpperCase();
    document.getElementById("device-badge").textContent =
      `${devLbl} · ${h.n_models} models${nDs > 1 ? ` · ${nDs} datasets` : ""}`;
  } catch (_) {}

  if (STATIC_MODE) {
    const strideInp = document.getElementById("stride-input");
    if (strideInp) {
      strideInp.value = "25";
      strideInp.disabled = true;
      strideInp.title = "Precomputed stride (25 samples = 50 ms) in static Vercel mode; append ?api=https://... for custom strides";
    }
    const scopeSel = document.getElementById("dense-int-scope");
    if (scopeSel) {
      scopeSel.title = "Full 10s record interventions are exact; Selected 1.0s Window uses the nearest precomputed 1.0s bin in static mode";
    }
  }

  await loadModels();
  await loadDenseModels();
  applyDatasetControls(false);
  renderExternalComparisonTable();
  setupPresetButtons();
  setupStoryScrollSpy();
  setupQuestionNavigator();
  setupQuickOptions();
  updateExplorerModeNote();
  await loadRecordList();
  await refreshBenchmarks();

  // Auto-load first test record (Record #9 is in fold 10!) for an immediate visual experience
  const recSel = document.getElementById("record-select");
  if (recSel.options.length > 0 && recSel.value) {
    await loadCurrentRecord();
  }
  prefetchPresetRecords();
}

function prefetchPresetRecords() {
  if (!STATIC_MODE) return;
  const targets = [
    ["ptbxl", 8215],
    ["georgia", "E00021"],
    ["georgia", "E00005"],
    ["georgia", "E10019"],
    ["georgia", "E00067"],
    ["georgia", "E00006"],
  ];
  const run = async () => {
    for (const [ds, id] of targets) {
      try {
        await _loadStaticRecord(ds, id);
      } catch (_) {}
    }
  };
  if (typeof window.requestIdleCallback === "function") {
    window.requestIdleCallback(() => { run(); }, { timeout: 2000 });
  } else {
    setTimeout(run, 900);
  }
}

const PRESET_CASES = {
  ptbxl_norm:       { dataset: "ptbxl",   fold: "10",             id: "9",      lead: 0, dxLabel: "Normal Sinus (PTB-XL #9)" },
  ptbxl_afib:       { dataset: "ptbxl",   fold: "10",             id: "8215",   lead: 1, dxLabel: "Atrial Fibrillation (PTB-XL #8215)" },
  georgia_nsr:      { dataset: "georgia", fold: "0_clean",        id: "E00021", lead: 1, dxLabel: "Normal Sinus (Georgia E00021)" },
  georgia_af:       { dataset: "georgia", fold: "0_clean",        id: "E00005", lead: 1, dxLabel: "RBBB + AF (Georgia E00005)" },
  georgia_lbbb:     { dataset: "georgia", fold: "0_clean",        id: "E10019", lead: 1, dxLabel: "LBBB (Georgia E10019)" },
  georgia_iavb:     { dataset: "georgia", fold: "0_clean",        id: "E00067", lead: 1, dxLabel: "1st-Deg AVB (Georgia E00067)" },
  georgia_artifact: { dataset: "georgia", fold: "3_severe_error", id: "E00006", lead: 1, dxLabel: "Signal Artifact (Georgia E00006)" },
};

function updatePresetHighlights(presetKey) {
  state.activePreset = presetKey || "";
  document.querySelectorAll("[data-preset]").forEach(btn => {
    btn.classList.toggle("active", Boolean(presetKey && btn.dataset.preset === presetKey));
  });
}

function updateActiveCaseSummary(text) {
  const el = document.getElementById("active-case-summary");
  if (el && text) el.textContent = text;
}

async function applyPresetCase(presetKey, scrollTargetId = null) {
  const p = PRESET_CASES[presetKey];
  if (!p) return;
  const hasDs = (state.datasets || []).some(d => d.id === p.dataset);
  if (!hasDs) return;

  const chipBtn = document.querySelector(`[data-preset="${presetKey}"]`);
  if (chipBtn) {
    chipBtn.classList.add("loading");
    chipBtn.disabled = true;
  }
  try {
    updatePresetHighlights(presetKey);
    state.dataset = p.dataset;
    const dsSel = document.getElementById("dataset-select");
    if (dsSel) dsSel.value = p.dataset;

    applyDatasetControls(false);

    const foldSel = document.getElementById("fold-select");
    if (foldSel && [...foldSel.options].some(o => o.value === p.fold)) {
      foldSel.value = p.fold;
    }
    const dxSel = document.getElementById("dx-select");
    if (dxSel) dxSel.value = "ALL";
    const leadSel = document.getElementById("lead-select");
    if (leadSel) {
      leadSel.value = String(p.lead);
      state.leadIdx = p.lead;
    }

    await loadRecordList();
    const recSel = document.getElementById("record-select");
    if (recSel && ![...recSel.options].some(o => o.value === p.id)) {
      const searchInp = document.getElementById("record-search");
      if (searchInp) searchInp.value = p.id;
      await loadRecordList();
    }
    if (recSel && [...recSel.options].some(o => o.value === p.id)) {
      recSel.value = p.id;
    }
    await loadCurrentRecord();

    if (scrollTargetId) {
      const target = document.getElementById(scrollTargetId.replace(/^#/, ""));
      if (target) target.scrollIntoView({ behavior: "smooth", block: "start" });
    }
  } finally {
    if (chipBtn) {
      chipBtn.classList.remove("loading");
      chipBtn.disabled = false;
    }
  }
}

function setupPresetButtons() {
  const hasGeorgia = (state.datasets || []).some(d => d.id === "georgia");
  document.querySelectorAll("[data-preset]").forEach(btn => {
    const p = PRESET_CASES[btn.dataset.preset];
    if (p && p.dataset === "georgia" && !hasGeorgia) {
      btn.disabled = true;
      btn.title = "Georgia 12-Lead dataset not available on this backend";
    }
    btn.addEventListener("click", () => {
      applyPresetCase(btn.dataset.preset, btn.dataset.scrollTarget || null);
    });
  });
}

function setupStoryScrollSpy() {
  const navLinks = Array.from(document.querySelectorAll(".primary-nav a[data-act]"));
  if (!navLinks.length || typeof IntersectionObserver === "undefined") return;
  const visible = new Set();
  const obs = new IntersectionObserver(entries => {
    entries.forEach(en => {
      if (en.isIntersecting) visible.add(en.target.id);
      else visible.delete(en.target.id);
    });
    const first = navLinks.find(a => visible.has(a.dataset.act));
    if (first) {
      navLinks.forEach(a => a.classList.toggle("active", a === first));
    }
  }, { rootMargin: "-18% 0px -65% 0px", threshold: 0.01 });

  navLinks.forEach(a => {
    const sec = document.getElementById(a.dataset.act);
    if (sec) obs.observe(sec);
  });
}

function renderExternalComparisonTable() {
  const el = document.getElementById("external-comparison-table");
  const ext = state.externalBenchmark;
  if (!el) return;
  if (!ext?.suite1_full_clean_cohort?.models || !ext?.suite2_neurokit2_delineation?.models) {
    el.innerHTML = `<div class="hint">External Georgia benchmark summary not present on server.</div>`;
    return;
  }
  const s1 = ext.suite1_full_clean_cohort.models;
  const s2 = ext.suite2_neurokit2_delineation.models;
  const s3 = ext.suite3_uncertainty_and_noise?.models || {};
  const order = [
    "dense_cem_s42",
    "dense_cbm_detached_s42",
    "dense_cbm_joint_s42",
    "dense_multitask_s42",
    "dense_blackbox_s42",
  ].filter(k => s1[k]);

  let html = `<table class="metrics-table"><thead><tr>
    <th>Model (PTB-XL Seed 42, Zero-Shot)</th>
    <th>NK2 QRS / P / T F1 (Georgia)</th>
    <th>PTB-XL Fold-10 Ceiling</th>
    <th>HR MAE</th>
    <th>Canon. Cycle</th>
    <th>Bnd / Int σ</th>
    <th>4-Class Path AUROC</th>
  </tr></thead><tbody>`;

  order.forEach(k => {
    const m1 = s1[k];
    const g2 = s2[k]?.georgia;
    const p2 = s2[k]?.ptbxl_fold10;
    const u3 = s3[k];
    const nkG = g2 ? `${g2.QRS.tol_150ms.f1.toFixed(3)} / ${g2.P.tol_150ms.f1.toFixed(3)} / ${g2.T.tol_150ms.f1.toFixed(3)}` : "—";
    const nkP = p2 ? `${p2.QRS.tol_150ms.f1.toFixed(3)} / ${p2.P.tol_150ms.f1.toFixed(3)} / ${p2.T.tol_150ms.f1.toFixed(3)}` : "—";
    const hr = m1.segmentation ? `${m1.segmentation.hr_agreement.mae_bpm.toFixed(2)} bpm` : "—";
    const cy = m1.segmentation ? `${(m1.segmentation.canonical_transition_rate * 100).toFixed(2)}%` : "—";
    const bi = u3 ? `${u3.georgia_clean_bnd_vs_int.toFixed(2)}×` : "—";
    const pa = m1.pathology.macro_auroc_4_ex_mi.toFixed(4);
    html += `<tr>
      <td><strong>${m1.display_name}</strong></td>
      <td>${nkG}</td>
      <td>${nkP}</td>
      <td>${hr}</td>
      <td>${cy}</td>
      <td>${bi}</td>
      <td class="${k === "dense_cem_s42" ? "best" : ""}">${pa}</td>
    </tr>`;
  });
  html += `</tbody></table>`;
  el.innerHTML = html;
}

const CONDITION_LABELS = {
  ALL: "All",
  BBB: "BBB (LBBB/RBBB)",
  IAVB: "IAVB (1st AV Block)",
  LQT: "LQT (Prolonged QT)",
  PVC: "PVC / PAC",
};

function currentDatasetMeta() {
  return (state.datasets || []).find(d => d.id === state.dataset) || {
    id: "ptbxl",
    split_label: "Split",
    has_soft_seg_gt: true,
    default_record: "9",
    default_split: "10",
    splits: [
      { value: "10", label: "Test (Fold 10)" },
      { value: "9", label: "Val (Fold 9)" },
      { value: "all", label: "All Folds" },
    ],
    conditions: ["ALL", "NORM", "MI", "STTC", "CD", "HYP", "AFIB", "BBB", "PVC"],
  };
}

function populateDatasetSelect() {
  const dsSel = document.getElementById("dataset-select");
  if (!dsSel || !state.datasets.length) return;
  dsSel.innerHTML = "";
  state.datasets.forEach(d => {
    dsSel.appendChild(new Option(d.label, d.id, false, d.id === state.dataset));
  });
}

function applyDatasetControls(switchLead = true) {
  const meta = currentDatasetMeta();
  state.hasSoftSegGt = Boolean(meta.has_soft_seg_gt);

  const foldLabel = document.getElementById("fold-label");
  if (foldLabel) foldLabel.textContent = meta.split_label || "Split";

  const foldSel = document.getElementById("fold-select");
  if (foldSel) {
    foldSel.innerHTML = "";
    (meta.splits || []).forEach(sp => {
      foldSel.appendChild(new Option(sp.label, sp.value, false, sp.value === meta.default_split));
    });
  }

  const dxSel = document.getElementById("dx-select");
  if (dxSel) {
    dxSel.innerHTML = "";
    (meta.conditions || ["ALL"]).forEach(c => {
      dxSel.appendChild(new Option(CONDITION_LABELS[c] || c, c, false, c === "ALL"));
    });
  }

  const searchInp = document.getElementById("record-search");
  if (searchInp) searchInp.value = "";

  // Toggle Zenodo GT options cleanly when viewing Georgia OOD
  const overlaySel = document.getElementById("ecg-seg-overlay");
  if (overlaySel) {
    const zenOpt = overlaySel.querySelector('option[value="zenodo"]');
    if (zenOpt) zenOpt.disabled = !state.hasSoftSegGt;
    if (!state.hasSoftSegGt && overlaySel.value === "zenodo") {
      overlaySel.value = "dense_cem";
      state.ecgSegOverlay = "dense_cem";
    }
  }

  const cmpModeSel = document.getElementById("seg-compare-mode");
  if (cmpModeSel) {
    ["cem_vs_gt", "cbm_vs_gt", "gt_only"].forEach(v => {
      const opt = cmpModeSel.querySelector(`option[value="${v}"]`);
      if (opt) opt.disabled = !state.hasSoftSegGt;
    });
    if (!state.hasSoftSegGt && ["cem_vs_gt", "cbm_vs_gt", "gt_only"].includes(cmpModeSel.value)) {
      cmpModeSel.value = "cem_vs_cbm";
    }
  }

  const gtBtn = document.getElementById("dense-gt-int-btn");
  if (gtBtn) {
    gtBtn.disabled = !state.hasSoftSegGt;
    gtBtn.style.opacity = state.hasSoftSegGt ? "1" : "0.45";
    gtBtn.title = state.hasSoftSegGt
      ? "Substitute Zenodo GT soft masks into Dense CBM & Dense CEM"
      : "Zenodo 7610236 GT masks are only available on PTB-XL";
  }

  if (switchLead) {
    const leadSel = document.getElementById("lead-select");
    if (leadSel) {
      const targetLead = state.dataset === "georgia" ? 1 : 0;
      leadSel.value = String(targetLead);
      state.leadIdx = targetLead;
    }
  }

  state.denseBenchDataset = (state.dataset === "georgia") ? "georgia" : "ptbxl";
  document.querySelectorAll("#dense-bench-toggle .tab-btn").forEach(btn => {
    btn.classList.toggle("active", btn.dataset.benchDs === state.denseBenchDataset);
  });
  renderDenseMetricsTable();
  renderMetricsTable();
}

const MODEL_DESCRIPTIONS = {
  cem_softseg:            "CEM (500 Hz): concept embeddings (c+, c-) match Black-Box accuracy while allowing exact interventions",
  cbm_softseg_masked:     "CBM Masked (500 Hz): scalar bottleneck with unlabeled U-wave blocked to cut concept impurity (OIS)",
  blackbox_softseg:       "Black-Box (500 Hz): unconstrained 1D ResNet predicting completeness without concept explanations",
  dense_cem_s42:          "DenseCEM (100 Hz): per-timestamp wave embeddings — matches Black-Box pathology with 2.23× boundary σ(t)",
  dense_cbm_detached_s42: "DenseCBM Detached (100 Hz): stops task gradients into masks — sharpest Dice (0.947) and zero leakage (Δ=-0.002)",
  dense_cbm_joint_s42:    "DenseCBM Joint (100 Hz): end-to-end mask + pathology co-training",
  dense_multitask_s42:    "DenseMultiTask (100 Hz): shared backbone predicting pathology & masks in parallel (non-causal)",
  dense_blackbox_s42:     "DenseBlackBox (100 Hz): standard ResNet classifier with no wave segmentation or concept bottleneck",
};

function describeModel(id, mtype) {
  if (id && MODEL_DESCRIPTIONS[id]) return MODEL_DESCRIPTIONS[id];
  const t = String(mtype || "").toLowerCase();
  if (t.includes("cem")) return "Concept Embedding Model (embeddings per concept; intervenable)";
  if (t.includes("cbm")) return "Concept Bottleneck Model (scalar concept probabilities; causally faithful)";
  if (t.includes("multitask")) return "Multi-Task ResNet (auxiliary masks only; non-causal pathology head)";
  return "Black-Box ResNet baseline (no concept bottleneck)";
}

function updateExplorerModeNote() {
  const el = document.getElementById("explorer-mode-note");
  if (!el) return;
  const modeMsg = STATIC_MODE
    ? "Running on static CDN bundles (exact linear head math + 10-bin window interventions). Append <code>?api=https://your-host</code> for a live FastAPI GPU/CPU backend."
    : "Connected to live FastAPI PyTorch backend (full cohort search & arbitrary strides/spans, using pre-calculated cache where available).";
  el.innerHTML = `${modeMsg} Import custom NumPy arrays with: <code>.venv_new/bin/python explorer/convert_numpy.py cohort.npz --dataset-id my_cohort --fs 500</code>`;
}

function updateDenseModelNote() {
  const el = document.getElementById("dense-model-note");
  if (!el) return;
  const parts = [];
  if (state.densePrimary) parts.push(`<strong>Primary</strong>: ${describeModel(state.densePrimary, "")}`);
  if (state.denseCompare) parts.push(`<strong>Compare</strong>: ${describeModel(state.denseCompare, "")}`);
  el.innerHTML = parts.join(" &nbsp;|&nbsp; ");
}

function setupQuestionNavigator() {
  document.querySelectorAll("[data-jump-act]").forEach(btn => {
    btn.addEventListener("click", async () => {
      const actSel = btn.dataset.jumpAct;
      const preset = btn.dataset.jumpPreset;
      const wantEp = btn.dataset.jumpEpistemic === "1";
      if (wantEp) {
        const ep = document.getElementById("epistemic-chk");
        const qEp = document.getElementById("quick-epistemic-chk");
        if (ep && !ep.checked) {
          ep.checked = true;
          if (qEp) qEp.checked = true;
        }
      }
      if (preset) {
        await applyPresetCase(preset, actSel);
        if (wantEp) {
          updatePipeline();
          runDenseInfer();
        }
        return;
      }
      if (wantEp && state.ecgId !== null) {
        updatePipeline();
        runDenseInfer();
      }
      if (actSel) {
        const target = document.getElementById(actSel.replace(/^#/, ""));
        if (target) target.scrollIntoView({ behavior: "smooth", block: "start" });
      }
    });
  });
}

function syncQuickOptions() {
  const pairBtns = document.querySelectorAll("[data-model-pair]");
  const p = state.primaryModel || "";
  const c = state.compareModel || "";
  const isCem = p.includes("cem");
  const isCbm = p.includes("cbm");
  const cmpCbm = c.includes("cbm");
  const cmpBb  = c.includes("blackbox");

  pairBtns.forEach(btn => {
    const k = btn.dataset.modelPair;
    const active =
      (k === "cem_cbm" && isCem && cmpCbm) ||
      (k === "cem_bb"  && isCem && cmpBb)  ||
      (k === "cbm_bb"  && isCbm && cmpBb);
    btn.classList.toggle("active", active);
  });

  const ep  = document.getElementById("epistemic-chk");
  const qEp = document.getElementById("quick-epistemic-chk");
  if (ep && qEp) qEp.checked = ep.checked;
}

function setupQuickOptions() {
  const models = state.models || [];
  const findId = (pred, fallback = "") => (models.find(pred)?.id || fallback);
  const cemId = findId(m => m.id === "cem_softseg" || m.model_type === "cem");
  const cbmId = findId(m => m.id === "cbm_softseg_masked" || m.model_type === "cbm");
  const bbId  = findId(m => m.id === "blackbox_softseg" || m.model_type === "blackbox");

  document.querySelectorAll("[data-model-pair]").forEach(btn => {
    btn.addEventListener("click", async () => {
      const pair = btn.dataset.modelPair;
      let nextP = cemId, nextC = cbmId;
      if (pair === "cem_bb") { nextP = cemId; nextC = bbId; }
      else if (pair === "cbm_bb") { nextP = cbmId; nextC = bbId; }
      if (!nextP) return;

      state.primaryModel = nextP;
      state.compareModel = nextC || "";
      const pSel = document.getElementById("model-select");
      const cSel = document.getElementById("compare-select");
      if (pSel) pSel.value = state.primaryModel;
      if (cSel) cSel.value = state.compareModel;
      syncQuickOptions();
      await refreshBenchmarks();
      if (state.ecgId !== null) await runSweep();
    });
  });

  const qEp = document.getElementById("quick-epistemic-chk");
  const ep  = document.getElementById("epistemic-chk");
  if (qEp && ep) {
    qEp.addEventListener("change", () => {
      ep.checked = qEp.checked;
      updatePipeline();
      runDenseInfer();
    });
  }

  const openBtn = document.getElementById("open-controls-btn");
  const drawer  = document.getElementById("controls-drawer");
  if (openBtn && drawer) {
    openBtn.addEventListener("click", () => {
      drawer.open = !drawer.open;
    });
  }
  syncQuickOptions();
}

async function loadModels() {
  const data = await apiGet("/models");
  state.models = data.models || [];
  state.primaryModel = data.default_model || (state.models[0] ? state.models[0].id : "");
  state.compareModel = data.compare_default || "";

  const primSel = document.getElementById("model-select");
  const cmpSel  = document.getElementById("compare-select");
  primSel.innerHTML = "";
  cmpSel.innerHTML  = `<option value="">None</option>`;

  state.models.forEach(m => {
    const label = `${m.id} [${m.model_type.toUpperCase()}]`;
    const o1 = new Option(label, m.id, false, m.id === state.primaryModel);
    const o2 = new Option(label, m.id, false, m.id === state.compareModel);
    const tip = describeModel(m.id, m.model_type);
    o1.title = tip;
    o2.title = tip;
    primSel.appendChild(o1);
    cmpSel.appendChild(o2);
  });
  syncQuickOptions();
}

async function loadRecordList() {
  const foldVal = document.getElementById("fold-select").value;
  const dxVal   = document.getElementById("dx-select").value;
  const search  = document.getElementById("record-search").value.trim();
  const params  = new URLSearchParams({ limit: "300", dataset: state.dataset });
  if (foldVal !== "all") params.set("fold", foldVal);
  if (dxVal && dxVal !== "ALL") params.set("dx", dxVal);
  if (search) params.set("search", search);

  const sel = document.getElementById("record-select");
  sel.disabled = true;
  sel.innerHTML = `<option value="">Loading records…</option>`;
  let data;
  try {
    data = await apiGet(`/records?${params.toString()}`);
  } finally {
    sel.disabled = false;
  }
  sel.innerHTML = "";
  (data.records || []).forEach(r => {
    let text;
    if (state.dataset === "ptbxl") {
      const tags = r.tags && r.tags.length ? ` [${r.tags.join(",")}]` : "";
      text = `Record #${r.ecg_id}${tags}`;
    } else {
      const lbls = (r.label_names && r.label_names.length) ? r.label_names : (r.dx && r.dx.length ? r.dx : (r.tags || []));
      const fallbackTag = r.severity_tier || r.split || "";
      const tagStr = lbls.length ? ` [${lbls.join(", ")}]` : (fallbackTag ? ` [${fallbackTag}]` : "");
      text = `${r.ecg_id}${tagStr}`;
    }
    sel.appendChild(new Option(text, String(r.ecg_id)));
  });
  const defId = String(currentDatasetMeta().default_record || "");
  if (!search && dxVal === "ALL" && defId && [...sel.options].some(o => o.value === defId)) {
    sel.value = defId;
  }
}

// ── Record loading & automatic sweep ───────────────────────────────────────
async function loadCurrentRecord() {
  stopPlayback();
  const rawVal = (document.getElementById("record-select").value || "").trim();
  if (!rawVal) return;
  const ecgId = state.dataset === "ptbxl" ? parseInt(rawVal, 10) : rawVal;
  if (state.dataset === "ptbxl" && isNaN(ecgId)) return;

  state.ecgId         = ecgId;
  state.leadIdx       = parseInt(document.getElementById("lead-select").value, 10) || 0;
  state.sweepData       = null;
  state.denseData       = null;
  state.denseIntMode    = null;
  state.denseIntConcept = null;
  state.interventions   = {};
  updateResetIntButton();
  updateDenseIntButtons();

  const loadBtn = document.getElementById("load-btn");
  if (loadBtn) {
    loadBtn.disabled = true;
    loadBtn.textContent = "Loading…";
  }
  const sigHint = document.getElementById("signal-hint");
  if (!state.signal && sigHint) {
    sigHint.innerHTML = `<span class="spinner" aria-hidden="true"></span> Loading ECG ${ecgId}…`;
    sigHint.style.display = "";
  }
  setSectionLoading("signal-section", true);
  setSectionLoading("act-segment", true);
  setSectionLoading("dense-seg-section", true);
  document.getElementById("pipeline-section").style.display = "";
  const segSec = document.getElementById("act-segment");
  if (segSec) segSec.style.display = "";

  try {
    let data;
    try {
      data = await apiGet(
        `/record/${encodeURIComponent(ecgId)}?model_id=${encodeURIComponent(state.primaryModel || "")}&dataset=${encodeURIComponent(state.dataset)}`
      );
    } catch (err) {
      alert(`Failed to load record: ${err.message}`);
      return;
    }
    if (sigHint) sigHint.style.display = "none";
    state.signal       = data.signal;
    state.delineation  = data.delineation;
    state.softSeg      = data.soft_seg || null;
    state.nSamples     = data.n_samples;
    state.nSamplesDs   = data.n_samples_ds;
    state.downsample   = data.downsample;
    state.tFixed       = data.t_fixed;
    state.samplingRate = data.sampling_rate || 500;

    // Position initial window on the 3rd R-peak if delineated, else midpoint
    const rps = data.delineation?.r_peaks || [];
    if (rps.length >= 3) {
      state.windowStart = Math.max(0, Math.min(state.nSamples - state.tFixed, rps[2] - Math.floor(state.tFixed / 2)));
    } else {
      state.windowStart = Math.max(0, Math.floor(state.nSamples / 2) - Math.floor(state.tFixed / 2));
    }

    const age = data.age > 0 ? `age ${Math.round(data.age)}` : "";
    const leadName = document.getElementById("lead-select")?.selectedOptions?.[0]?.textContent || "I";
    if (data.dataset === "ptbxl") {
      const sex = data.sex === 0 ? "M" : "F";
      const dx  = data.dx && data.dx.length ? `· ${data.dx.join(", ")}` : "";
      document.getElementById("record-meta").textContent =
        `— Record #${ecgId}  ${age} ${sex} ${dx}  (${rps.length} R-peaks)`;
      updateActiveCaseSummary(`PTB-XL #${ecgId} ${dx} · Lead ${leadName}`);
    } else {
      const dsLabel = data.dataset === "georgia" ? "Georgia" : (currentDatasetMeta().label || data.dataset);
      const tierStr = data.severity_tier ? ` (${data.severity_tier})` : (data.split ? ` (${data.split})` : "");
      const sexTag = data.sex_label || "";
      const dxArr = (data.label_names && data.label_names.length) ? data.label_names : (data.dx || []);
      const dxStr = dxArr.length ? `· ${dxArr.join(", ")}` : "";
      const hrStr = data.est_hr_bpm ? ` · ${data.est_hr_bpm} bpm` : "";
      document.getElementById("record-meta").textContent =
        `— ${dsLabel} ${ecgId}${tierStr}  ${age} ${sexTag} ${dxStr}${hrStr}  (${rps.length} R-peaks)`;
      updateActiveCaseSummary(`${dsLabel} ${ecgId}${tierStr}${dxStr} · Lead ${leadName}`);
    }

    renderSignal();
    renderSegmentedSignal();
    await Promise.all([runSweep(), runDenseInfer()]);
  } finally {
    if (loadBtn) {
      loadBtn.disabled = false;
      loadBtn.textContent = "Load";
    }
    setSectionLoading("signal-section", false);
  }
}

// ── Signal Waveform Chart ──────────────────────────────────────────────────
function renderSignal() {
  const container = document.getElementById("signal-container");
  container.innerHTML = "";
  if (!state.signal) return;

  const W = chartWidth();
  const H = SIG_HEIGHT;
  const { left: ml, right: mr, top: mt, bottom: mb } = SIG_MARGIN;

  const svg = d3.select(container).append("svg")
    .attr("width", W + ml + mr)
    .attr("height", H + mt + mb);

  signalG = svg.append("g").attr("transform", `translate(${ml},${mt})`);

  const raw    = state.signal[state.leadIdx];
  const ds     = state.downsample;
  const T      = state.nSamples;
  const sr     = state.samplingRate;
  const tFixed = state.tFixed;

  _xScaleSig = d3.scaleLinear().domain([0, T]).range([0, W]);
  _winWpx    = _xScaleSig(tFixed) - _xScaleSig(0);

  const yMin = d3.min(raw), yMax = d3.max(raw);
  const yPad = (yMax - yMin) * 0.12 || 0.2;
  const yScale = d3.scaleLinear().domain([yMin - yPad, yMax + yPad]).range([H, 0]);

  signalG.append("g").attr("class", "grid")
    .call(d3.axisLeft(yScale).tickSize(-W).tickFormat("").ticks(5));
  signalG.append("g").attr("class", "axis")
    .attr("transform", `translate(0,${H})`)
    .call(d3.axisBottom(_xScaleSig).ticks(10).tickFormat(d => `${(d / sr).toFixed(1)}s`));
  signalG.append("g").attr("class", "axis")
    .call(d3.axisLeft(yScale).ticks(5));

  // Delineated R-peaks (dashed vertical lines)
  const delin = state.delineation || {};
  (delin.r_peaks || []).forEach(r => {
    const rx = _xScaleSig(r);
    signalG.append("line")
      .attr("x1", rx).attr("x2", rx).attr("y1", 0).attr("y2", H)
      .attr("stroke", "#b83b2a").attr("stroke-width", 0.85)
      .attr("stroke-dasharray", "3,3").attr("opacity", 0.45);
  });

  // ECG trace
  const lineGen = d3.line()
    .x((_, i) => _xScaleSig(i * ds))
    .y(d => yScale(d));
  signalG.append("path").datum(raw).attr("class", "ecg-line").attr("d", lineGen);

  // Delineated P-peaks (slate blue) & T-peaks (forest green)
  const drawPeakDots = (peaks, color) => {
    (peaks || []).forEach(p => {
      const idxDs = Math.min(raw.length - 1, Math.max(0, Math.round(p / ds)));
      signalG.append("circle")
        .attr("cx", _xScaleSig(p))
        .attr("cy", yScale(raw[idxDs]))
        .attr("r", 2.8)
        .attr("fill", color)
        .attr("opacity", 0.9);
    });
  };
  drawPeakDots(delin.p_peaks, "#2b5b84");
  drawPeakDots(delin.t_peaks, "#2d7a56");

  // Draggable 1.0s Window Box, L/R Handles & Midpoint Cursor
  attachSyncedWindowOverlay("signal", signalG, _xScaleSig, H);
  updateWindowPosition(state.windowStart);

  const halfWin = tFixed / 2;
  svg.style("cursor", "pointer").on("click", async e => {
    if (e.defaultPrevented) return;
    const [mx] = d3.pointer(e, signalG.node());
    if (mx < 0 || mx > W) return;
    const start = Math.max(0, Math.min(T - tFixed, Math.round(_xScaleSig.invert(mx) - halfWin)));
    updateWindowPosition(start);
    await updatePipeline();
  });
}

// ── Act 03 Waveform Chart with Soft-Segmentation Background Shading ─────────
function renderSegmentedSignal() {
  const container = document.getElementById("seg-signal-container");
  if (!container) return;
  container.innerHTML = "";
  if (!state.signal) return;

  const W = chartWidth();
  const H = 128;
  const { left: ml, right: mr, top: mt, bottom: mb } = SIG_MARGIN;

  const svg = d3.select(container).append("svg")
    .attr("width", W + ml + mr)
    .attr("height", H + mt + mb);
  const g = svg.append("g").attr("transform", `translate(${ml},${mt})`);

  const raw    = state.signal[state.leadIdx];
  const ds     = state.downsample;
  const T      = state.nSamples;
  const sr     = state.samplingRate;
  const tFixed = state.tFixed;

  const xScale = d3.scaleLinear().domain([0, T]).range([0, W]);
  const winWpx = xScale(tFixed) - xScale(0);

  const yMin = d3.min(raw), yMax = d3.max(raw);
  const yPad = (yMax - yMin) * 0.12 || 0.2;
  const yScale = d3.scaleLinear().domain([yMin - yPad, yMax + yPad]).range([H, 0]);

  g.append("g").attr("class", "grid")
    .call(d3.axisLeft(yScale).tickSize(-W).tickFormat("").ticks(4));
  g.append("g").attr("class", "axis")
    .attr("transform", `translate(0,${H})`)
    .call(d3.axisBottom(xScale).ticks(10).tickFormat(d => `${(d / sr).toFixed(1)}s`));
  g.append("g").attr("class", "axis")
    .call(d3.axisLeft(yScale).ticks(4));

  // Background wave-segmentation shading aligned with 100 Hz soft-seg curves below
  const segPalette = { P: "#2b5b84", PQ: "#2a7f8e", QRS: "#b83b2a", ST: "#9e4268", T: "#2d7a56" };
  const spanCols   = ["#2b5b84", "#2a7f8e", "#b83b2a", "#9e4268", "#2d7a56"];
  const hasSegProbs = obj => Boolean(obj && obj.has_seg !== false && obj.concept_probs);
  let segSrc = null, segSpans = null, segSr = 100;
  if (state.ecgSegOverlay === "dense_cem" && state.denseData) {
    const src = (hasSegProbs(state.denseData) && state.denseData.model_type === "dense_cem")
      ? state.denseData
      : (hasSegProbs(state.denseData.compare) && state.denseData.compare.model_type === "dense_cem" ? state.denseData.compare : null);
    segSrc   = src?.concept_probs || null;
    segSpans = !state.denseIntMode ? (src?.seg_spans || null) : null;
    segSr    = state.denseData.sampling_rate || 100;
  } else if (state.ecgSegOverlay === "dense_cbm" && state.denseData) {
    const src = (hasSegProbs(state.denseData) && state.denseData.model_type === "dense_cbm")
      ? state.denseData
      : (hasSegProbs(state.denseData.compare) && state.denseData.compare.model_type === "dense_cbm" ? state.denseData.compare : null);
    segSrc   = src?.concept_probs || null;
    segSpans = !state.denseIntMode ? (src?.seg_spans || null) : null;
    segSr    = state.denseData.sampling_rate || 100;
  } else if (state.ecgSegOverlay === "zenodo") {
    segSrc   = state.denseData?.gt_probs || state.softSeg;
    segSpans = !state.denseIntMode ? (state.denseData?.gt_spans || null) : null;
    segSr    = state.softSeg?.sampling_rate || 100;
  }
  const scaleSeg = sr / segSr;
  if (Array.isArray(segSpans) && segSpans.length) {
    for (let idx = 0; idx < segSpans.length; idx++) {
      const [s0, s1, wIdx] = segSpans[idx];
      const x0 = xScale(s0 * scaleSeg);
      const x1 = xScale(s1 * scaleSeg);
      g.append("rect")
        .attr("x", x0).attr("y", 0)
        .attr("width", Math.max(1, x1 - x0)).attr("height", H)
        .attr("fill", spanCols[wIdx]).attr("opacity", 0.16);
    }
  } else if (segSrc) {
    const keys = ["P", "PQ", "QRS", "ST", "T"].filter(k => Array.isArray(segSrc[k]) && segSrc[k].length);
    const nSeg = keys.length ? segSrc[keys[0]].length : 0;
    let curKey = null, runStart = 0;
    const flushRun = endIdx => {
      if (!curKey || endIdx <= runStart) return;
      const x0 = xScale(runStart * scaleSeg);
      const x1 = xScale(endIdx * scaleSeg);
      g.append("rect")
        .attr("x", x0).attr("y", 0)
        .attr("width", Math.max(1, x1 - x0)).attr("height", H)
        .attr("fill", segPalette[curKey]).attr("opacity", 0.16);
    };
    for (let i = 0; i < nSeg; i++) {
      let bestK = null, bestV = 0.35;
      keys.forEach(k => {
        const v = segSrc[k][i];
        if (v > bestV) { bestV = v; bestK = k; }
      });
      if (bestK !== curKey) {
        flushRun(i);
        curKey = bestK;
        runStart = i;
      }
    }
    flushRun(nSeg);
  }

  // ECG trace + Synced Draggable 1.0s Window Overlay
  const lineGen = d3.line()
    .x((_, i) => xScale(i * ds))
    .y(d => yScale(d));
  g.append("path").datum(raw).attr("class", "ecg-line").attr("d", lineGen);

  attachSyncedWindowOverlay("seg_signal", g, xScale, H);

  const halfWin = tFixed / 2;
  svg.style("cursor", "pointer").on("click", async e => {
    if (e.defaultPrevented) return;
    const [mx] = d3.pointer(e, g.node());
    if (mx < 0 || mx > W) return;
    const start = Math.max(0, Math.min(T - tFixed, Math.round(xScale.invert(mx) - halfWin)));
    updateWindowPosition(start);
    await updatePipeline();
  });
}

// ── Sliding-Window Sweep ───────────────────────────────────────────────────
async function runSweep() {
  if (state.ecgId === null) return;
  const stride   = parseInt(document.getElementById("stride-input").value, 10) || 20;
  const status   = document.getElementById("sweep-status");
  const sweepBtn = document.getElementById("sweep-btn");
  if (sweepBtn) sweepBtn.disabled = true;
  status.textContent = "Computing batched sweep…";
  status.className   = "status-label running";

  try {
    state.sweepData = await apiPost("/sweep", {
      ecg_id:           state.ecgId,
      dataset:          state.dataset,
      stride,
      model_id:         state.primaryModel,
      compare_model_id: state.compareModel || null,
    });
  } catch (e) {
    status.textContent = `Sweep error: ${e.message}`;
    status.className   = "status-label";
    return;
  } finally {
    if (sweepBtn) sweepBtn.disabled = false;
  }

  const cmpTag = state.sweepData.compare ? ` + ${state.sweepData.compare.model_id}` : "";
  status.textContent = `Ready — ${state.sweepData.starts.length} windows (${state.sweepData.model_id}${cmpTag})`;
  status.className   = "status-label done";
  document.getElementById("play-btn").disabled = false;
  document.getElementById("timeline-wrapper").style.display = "";

  renderTimelineView();
  await updatePipeline();
}

function renderTimelineView() {
  if (!state.sweepData) return;
  const lp = document.getElementById("lineplot-container");
  const hm = document.getElementById("heatmap-container");
  if (state.timelineView === "heatmap") {
    lp.style.display = "none";
    hm.style.display = "";
    renderHeatmap(state.sweepData);
  } else {
    hm.style.display = "none";
    lp.style.display = "";
    renderActivationLineplot(state.sweepData);
  }
}

async function loadDenseModels() {
  try {
    state.denseModels = await apiGet("/dense-models");
  } catch (_) {
    state.denseModels = { models: [] };
  }
  const sec = document.getElementById("dense-seg-section");
  const models = state.denseModels?.models || [];
  if (models.length > 0) {
    sec.style.display = "";
    state.densePrimary  = state.denseModels.default_model    || models[0].id;
    state.denseCompare  = state.denseModels.compare_default  || "";
    state.denseBlackbox = state.denseModels.blackbox_default || "";

    const primSel = document.getElementById("dense-primary-select");
    const cmpSel  = document.getElementById("dense-compare-select");
    const bbSel   = document.getElementById("dense-blackbox-select");
    if (primSel && cmpSel && bbSel) {
      primSel.innerHTML = "";
      cmpSel.innerHTML  = `<option value="">None</option>`;
      bbSel.innerHTML   = `<option value="">None</option>`;
      models.forEach(m => {
        const tag = m.model_type.replace(/^dense_/, "").toUpperCase();
        const label = `${m.id} [${tag}]`;
        const tip = describeModel(m.id, m.model_type);
        const o1 = new Option(label, m.id, false, m.id === state.densePrimary);
        const o2 = new Option(label, m.id, false, m.id === state.denseCompare);
        const o3 = new Option(label, m.id, false, m.id === state.denseBlackbox);
        o1.title = tip; o2.title = tip; o3.title = tip;
        primSel.appendChild(o1);
        cmpSel.appendChild(o2);
        bbSel.appendChild(o3);
      });
    }
    updateDenseModelNote();
    renderDenseMetricsTable();
    renderDenseWavePills();
  } else {
    sec.style.display = "none";
  }
}

async function runDenseInfer() {
  const modelId = state.densePrimary || state.denseModels?.default_model;
  if (state.ecgId === null || !state.denseModels || !modelId) return;
  const wantEp = document.getElementById("epistemic-chk").checked;
  const scope  = document.getElementById("dense-int-scope")?.value || "record";
  let span = null;
  if (scope === "window" && state.denseIntMode) {
    const scale = (state.samplingRate || 500) / 100;
    span = [
      Math.round(state.windowStart / scale),
      Math.round((state.windowStart + state.tFixed) / scale),
    ];
  }
  const body = {
    ecg_id:            state.ecgId,
    dataset:           state.dataset,
    model_id:          modelId,
    compare_model_id:  state.denseCompare  || null,
    blackbox_model_id: state.denseBlackbox || null,
    include_epistemic: wantEp,
    intervene_mode:    state.denseIntMode,
    intervene_concepts: state.denseIntConcept ? [state.denseIntConcept] : null,
    intervene_span:    span,
  };
  const st = document.getElementById("dense-seg-status");
  const loadLabel = wantEp
    ? "Running MC-Dropout…"
    : (state.denseIntMode ? "Applying intervention…" : "Running dense U-Net…");
  if (st) st.textContent = "— Computing dense segmentation…";
  setLoadingPill("dense-loading-indicator", true, loadLabel);
  setSectionLoading("act-segment", true);
  setSectionLoading("dense-seg-section", true);
  try {
    state.denseData = await apiPost("/dense-infer", body);
    const intTag = state.denseIntMode
      ? ` · Intervention: ${state.denseIntMode.toUpperCase()}${state.denseIntConcept ? " " + state.denseIntConcept : ""} (${scope})`
      : "";
    const ids = [state.denseData.model_id];
    if (state.denseData.compare)  ids.push(state.denseData.compare.model_id);
    if (state.denseData.blackbox) ids.push(state.denseData.blackbox.model_id);
    const iv = state.denseData.intervals || state.denseData.compare?.intervals;
    const ivParts = [];
    if (iv?.qrs_ms !== null && iv?.qrs_ms !== undefined) ivParts.push(`QRS ${iv.qrs_ms}ms`);
    if (iv?.pr_ms  !== null && iv?.pr_ms  !== undefined) ivParts.push(`PR ${iv.pr_ms}ms`);
    if (iv?.qt_ms  !== null && iv?.qt_ms  !== undefined) ivParts.push(`QT ${iv.qt_ms}ms`);
    if (iv?.pred_hr_bpm !== null && iv?.pred_hr_bpm !== undefined) ivParts.push(`HR ${iv.pred_hr_bpm}bpm`);
    const ivTag = ivParts.length ? ` · ${ivParts.join(" · ")}` : "";
    if (st) st.textContent = `— ${ids.join(" vs. ")}${ivTag}${intTag}`;

    renderDenseIntervalBadges(iv);
    renderDensePathologyBars();
    renderDenseWavePills();
    renderUnifiedSegmentationView();
    renderSegmentedSignal();
  } catch (e) {
    console.warn("Dense infer failed:", e);
    if (st) st.textContent = "— Dense inference failed";
  } finally {
    setLoadingPill("dense-loading-indicator", false);
    setSectionLoading("act-segment", false);
    setSectionLoading("dense-seg-section", false);
  }
}

function renderDenseIntervalBadges(iv) {
  const el = document.getElementById("dense-interval-badges");
  if (!el) return;
  if (!iv) {
    el.innerHTML = "";
    return;
  }
  const badges = [];
  if (iv.qrs_ms !== null && iv.qrs_ms !== undefined) badges.push(`QRS ${iv.qrs_ms} ms`);
  if (iv.pr_ms  !== null && iv.pr_ms  !== undefined) badges.push(`PR ${iv.pr_ms} ms`);
  if (iv.qt_ms  !== null && iv.qt_ms  !== undefined) badges.push(`QT ${iv.qt_ms} ms`);
  if (iv.pred_hr_bpm !== null && iv.pred_hr_bpm !== undefined) badges.push(`HR ${iv.pred_hr_bpm} bpm`);
  if (iv.canon_cycle !== null && iv.canon_cycle !== undefined) badges.push(`Cycle ${(iv.canon_cycle * 100).toFixed(1)}%`);
  el.innerHTML = badges.map(b => `<span class="iv-badge">${b}</span>`).join("");
}

function updateDenseIntButtons() {
  const rst = document.getElementById("dense-reset-int-btn");
  if (rst) rst.style.display = state.denseIntMode ? "" : "none";
}

function renderDenseWavePills() {
  const bar = document.getElementById("dense-wave-pills");
  if (!bar) return;
  const d = state.denseData;
  const anyCanInt = !d || d.supports_interventions !== false || d.compare?.supports_interventions === true;
  const names = (d?.has_seg !== false && d?.concept_names)
    ? d.concept_names
    : (d?.compare?.concept_names || ["P", "PQ", "QRS", "ST", "T", "TP"]);
  bar.innerHTML = "";
  names.forEach(name => {
    const btn = document.createElement("button");
    const isThis = state.denseIntConcept === name;
    let cls = "int-pill";
    let lbl = `${name}: Auto`;
    if (isThis && state.denseIntMode === "ablate") { cls += " force-0"; lbl = `${name}: 0`; }
    else if (isThis && state.denseIntMode === "force") { cls += " force-1"; lbl = `${name}: 1`; }
    btn.className = cls;
    btn.textContent = lbl;
    btn.disabled = !anyCanInt;
    btn.style.opacity = anyCanInt ? "1" : "0.45";
    if (!anyCanInt) btn.title = "Selected Black-Box model has no concept bottleneck to intervene on";
    btn.onclick = async () => {
      if (!anyCanInt) return;
      if (state.denseIntConcept === name && state.denseIntMode === "ablate") {
        state.denseIntMode = "force";
      } else if (state.denseIntConcept === name && state.denseIntMode === "force") {
        state.denseIntMode = null;
        state.denseIntConcept = null;
      } else {
        state.denseIntConcept = name;
        state.denseIntMode = "ablate";
      }
      updateDenseIntButtons();
      await runDenseInfer();
    };
    bar.appendChild(btn);
  });
}

function renderDensePathologyBars() {
  const container = document.getElementById("dense-pathology-bars");
  if (!container || !state.denseData) return;
  container.innerHTML = "";

  const classes = ["NORM", "MI", "STTC", "CD", "HYP"];
  const d = state.denseData;
  const modelsToPlot = [
    { key: "prim", id: d.model_id,          map: d.pathology || {},           base: d.base_pathology || d.pathology || {},                   std: d.pathology_std || {},          col: "#b83b2a" },
    ...(d.compare  ? [{ key: "cmp", id: d.compare.model_id,  map: d.compare.pathology || {},  base: d.compare.base_pathology || d.compare.pathology || {},   std: d.compare.pathology_std || {},  col: "#2b5b84" }] : []),
    ...(d.blackbox ? [{ key: "bb",  id: d.blackbox.model_id, map: d.blackbox.pathology || {}, base: d.blackbox.base_pathology || d.blackbox.pathology || {}, std: d.blackbox.pathology_std || {}, col: "#555555" }] : []),
  ];

  const W = Math.max(360, container.clientWidth - 8);
  const H = 198;
  const pad = { top: 34, right: 14, bottom: 30, left: 40 };
  const iW = W - pad.left - pad.right, iH = H - pad.top - pad.bottom;

  const svg = d3.select(container).append("svg").attr("width", W).attr("height", H);
  const g   = svg.append("g").attr("transform", `translate(${pad.left},${pad.top})`);
  const x0  = d3.scaleBand().domain(classes).range([0, iW]).padding(0.22);
  const x1  = d3.scaleBand().domain(modelsToPlot.map(m => m.key)).range([0, x0.bandwidth()]).padding(0.10);
  const yScale = d3.scaleLinear().domain([0, 1]).range([iH, 0]);

  g.append("g").attr("class", "grid").call(d3.axisLeft(yScale).tickSize(-iW).tickFormat("").ticks(4));
  g.append("g").attr("class", "axis").attr("transform", `translate(0,${iH})`).call(d3.axisBottom(x0));
  g.append("g").attr("class", "axis").call(d3.axisLeft(yScale).ticks(4).tickFormat(d3.format(".0%")));

  classes.forEach(cls => {
    modelsToPlot.forEach(m => {
      const val  = m.map[cls]  || 0;
      const base = m.base[cls] || 0;
      const sd   = m.std[cls]  || 0;
      const bx = x0(cls) + x1(m.key);
      const bw = x1.bandwidth();
      const yT = yScale(val);

      g.append("rect").attr("x", bx).attr("y", 0).attr("width", bw).attr("height", iH).attr("fill", "#eeeeee").attr("rx", 2);
      g.append("rect").attr("x", bx).attr("y", yT).attr("width", bw).attr("height", Math.max(0, iH - yT))
        .attr("fill", m.col).attr("opacity", 0.88).attr("rx", 2);

      if (sd > 0.002) {
        const yHi = Math.max(0, yScale(Math.min(1, val + sd)));
        const yLo = Math.min(iH, yScale(Math.max(0, val - sd)));
        g.append("line").attr("x1", bx + bw / 2).attr("x2", bx + bw / 2).attr("y1", yHi).attr("y2", yLo)
          .attr("stroke", "#222222").attr("stroke-width", 1.4);
      }

      if (state.denseIntMode && Math.abs(val - base) >= 0.005) {
        const yb = yScale(base);
        g.append("line").attr("x1", bx).attr("x2", bx + bw).attr("y1", yb).attr("y2", yb)
          .attr("stroke", "#222222").attr("stroke-width", 2);
      }
      g.append("text").attr("x", bx + bw / 2).attr("y", Math.max(yT - 4, 9))
        .attr("text-anchor", "middle").attr("fill", "#222222").attr("font-size", modelsToPlot.length >= 3 ? 7.8 : 8.5)
        .text(`${(val * 100).toFixed(0)}%`);
    });
  });

  // Clean multi-item legend spaced horizontally across the top (wraps cleanly if needed)
  let lx = 0, ly = -24;
  modelsToPlot.forEach(m => {
    const estW = 22 + m.id.length * 5.3;
    if (lx > 0 && lx + estW > iW) { lx = 0; ly += 13; }
    g.append("rect").attr("x", lx).attr("y", ly).attr("width", 9).attr("height", 8).attr("fill", m.col).attr("rx", 1.5);
    g.append("text").attr("x", lx + 12).attr("y", ly + 7).attr("fill", "#444444").attr("font-size", 8.5).text(m.id);
    lx += estW + 10;
  });
  if (state.denseIntMode) {
    if (lx > 0 && lx + 95 > iW) { lx = 0; ly += 13; }
    g.append("line").attr("x1", lx).attr("x2", lx + 12).attr("y1", ly + 4).attr("y2", ly + 4).attr("stroke", "#222222").attr("stroke-width", 2);
    g.append("text").attr("x", lx + 16).attr("y", ly + 7).attr("fill", "#222222").attr("font-size", 8.5).text("Pre-Int");
  }
}

function _resolveGeorgiaDenseId(modelId, s1Models) {
  if (!s1Models) return { key: null, mapped: false };
  if (s1Models[modelId]) return { key: modelId, mapped: false };
  if (modelId.startsWith("dense_cbm_detached")) return { key: "dense_cbm_detached_s42", mapped: true };
  if (modelId.startsWith("dense_cbm_joint") || modelId === "dense_cbm_softseg") return { key: "dense_cbm_joint_s42", mapped: true };
  if (modelId.startsWith("dense_cem")) return { key: "dense_cem_s42", mapped: true };
  if (modelId.startsWith("dense_multitask")) return { key: "dense_multitask_s42", mapped: true };
  if (modelId.startsWith("dense_blackbox")) return { key: "dense_blackbox_s42", mapped: true };
  return { key: null, mapped: false };
}

function renderDenseMetricsTable() {
  const el = document.getElementById("dense-metrics-table");
  const titleEl = document.getElementById("dense-benchmark-title");
  const allModels = state.denseModels?.models || [];
  if (!el || !allModels.length) return;

  const byId = new Map(allModels.map(m => [m.id, m]));
  const chosenIds = [
    state.densePrimary || state.denseModels?.default_model || allModels[0].id,
    state.denseCompare,
    state.denseBlackbox,
  ].filter(Boolean);
  const shown = [];
  chosenIds.forEach(id => {
    const m = byId.get(id);
    if (m && !shown.some(x => x.id === m.id)) shown.push(m);
  });
  if (!shown.length) return;

  // External Zero-Shot Georgia 12-Lead Benchmark view
  const ext = state.externalBenchmark;
  const benchDs = state.denseBenchDataset || state.dataset;
  if (benchDs === "georgia" && ext?.suite1_full_clean_cohort?.models) {
    if (titleEl) titleEl.textContent = "Zero-Shot External Benchmark · Georgia 12-Lead (8,770 Clean ECGs)";
    const s1 = ext.suite1_full_clean_cohort.models;
    const s2 = ext.suite2_neurokit2_delineation?.models || {};
    const s3 = ext.suite3_uncertainty_and_noise?.models || {};
    const resolved = shown.map(m => ({ m, ..._resolveGeorgiaDenseId(m.id, s1) }));

    const extRows = [
      { label: "NK2 QRS / P / T F1 (±150 ms)", combo: true, higher: true },
      { label: "QRS Peak MAE ↓ (ms)",           get: k => s2[k]?.georgia?.QRS?.tol_150ms?.mae_ms,               higher: false, fmt: v => v.toFixed(1) + " ms" },
      { label: "HR MAE vs. Ref ↓ (bpm)",        get: k => s1[k]?.segmentation?.hr_agreement?.mae_bpm,          higher: false, fmt: v => v.toFixed(2) },
      { label: "Canonical Cycle Order ↑",       get: k => s1[k]?.segmentation?.canonical_transition_rate,     higher: true,  fmt: v => (v * 100).toFixed(2) + "%" },
      { label: "Boundary / Interior σ ↑",       get: k => s3[k]?.georgia_clean_bnd_vs_int,                    higher: true,  fmt: v => v.toFixed(2) + "×" },
      { label: "LBBB QRS Widening AUROC",       get: k => s1[k]?.segmentation?.clinical_morphology?.lbbb_vs_nsr_qrs_ms?.auroc, higher: true, fmt: v => v.toFixed(3) },
      { label: "IAVB PR Prolongation AUROC",    get: k => s1[k]?.segmentation?.clinical_morphology?.iavb_vs_nsr_pr_ms?.auroc,  higher: true, fmt: v => v.toFixed(3) },
      { label: "4-Class Pathology AUROC",       get: k => s1[k]?.pathology?.macro_auroc_4_ex_mi,              higher: true,  fmt: v => v.toFixed(4) },
    ];

    let html = `<table class="metrics-table"><thead><tr><th>Zero-Shot Metric (Georgia)</th>`;
    resolved.forEach(({ m, key, mapped }) => {
      const tag = m.model_type.replace(/^dense_/, "").toUpperCase();
      const note = mapped && key ? ` · ref ${key}` : "";
      html += `<th>${m.id}<br/><span class="meta-label">[${tag}${note}]</span></th>`;
    });
    html += `</tr></thead><tbody>`;

    extRows.forEach(r => {
      if (r.combo) {
        const fmtC = k => {
          const g = k ? s2[k]?.georgia : null;
          if (!g) return "—";
          return `${g.QRS.tol_150ms.f1.toFixed(3)} / ${g.P.tol_150ms.f1.toFixed(3)} / ${g.T.tol_150ms.f1.toFixed(3)}`;
        };
        html += `<tr><td>${r.label}</td>${resolved.map(({ key }) => `<td>${fmtC(key)}</td>`).join("")}</tr>`;
        return;
      }
      const vals = resolved.map(({ key }) => {
        const v = key ? r.get(key) : undefined;
        return (typeof v === "number" && !Number.isNaN(v)) ? v : undefined;
      });
      if (vals.every(v => v === undefined)) return;
      const valid = vals.filter(v => v !== undefined);
      const bestVal = valid.length > 1 ? valid.reduce((a, b) => (r.higher ? (b > a ? b : a) : (b < a ? b : a))) : null;
      html += `<tr><td>${r.label}</td>`;
      vals.forEach(v => {
        const isBest = v !== undefined && bestVal !== null && v === bestVal;
        html += `<td class="${isBest ? "best" : ""}">${v !== undefined ? r.fmt(v) : "—"}</td>`;
      });
      html += `</tr>`;
    });
    html += `</tbody></table>`;
    el.innerHTML = html;
    return;
  }

  if (titleEl) titleEl.textContent = "Per-Timestamp Test Benchmark · Fold 10 (2.20M Timestamps)";

  const rows = [
    { label: "Per-Timestamp Acc",         key: "seg/argmax_acc",                         higher: true,  fmt: v => (v * 100).toFixed(2) + "%" },
    { label: "Hard Macro-Dice",           key: "seg/macro_hard_dice",                    higher: true,  fmt: v => v.toFixed(4) },
    { label: "QRS / T / P / ST Dice",     key: "_combo_dice",                            higher: true,  fmt: null },
    { label: "Mean Soft MAE ↓",           key: "seg/mean_mae",                           higher: false, fmt: v => v.toFixed(4) },
    { label: "5-Class Pathology AUROC",   key: "pathology/macro_auroc",                  higher: true,  fmt: v => v.toFixed(4) },
    { label: "Pathology ECE ↓",           key: "pathology/macro_ece",                    higher: false, fmt: v => v.toFixed(4) },
    { label: "Task Err-Det AUROC ↑",      key: "uncertainty/task_error_detection_auroc", higher: true,  fmt: v => v.toFixed(4) },
    { label: "GT-Mask Intervened AUROC",  key: "pathology/intervened_macro_auroc",       higher: true,  fmt: v => v.toFixed(4) },
    { label: "Excess Leakage Δ ↓",        key: "leakage/bottleneck_excess_auroc",        higher: false, absCmp: true, fmt: v => (v >= 0 ? "+" : "") + v.toFixed(4) },
    { label: "Seg Err-Det AUROC ↑",       key: "uncertainty/seg_error_detection_auroc",  higher: true,  fmt: v => v.toFixed(4) },
    { label: "Boundary / Interior σ ↑",   key: "uncertainty/boundary_vs_interior",       higher: true,  fmt: v => v.toFixed(2) + "×" },
    { label: "Ablate ST → Task Shift",    key: "counterfactual/ablate_ST_task_shift",    higher: true,  fmt: v => v.toFixed(3) },
  ];

  let html = `<table class="metrics-table"><thead><tr><th>Metric</th>`;
  shown.forEach(m => {
    const tag = m.model_type.replace(/^dense_/, "").toUpperCase();
    html += `<th>${m.id}<br/><span class="meta-label">[${tag}]</span></th>`;
  });
  html += `</tr></thead><tbody>`;

  rows.forEach(r => {
    if (r.key === "_combo_dice") {
      const fmtCombo = m => {
        const mt = m?.metrics;
        if (!mt || mt["seg/QRS/hard_dice"] === undefined) return "—";
        return `${mt["seg/QRS/hard_dice"].toFixed(2)} / ${mt["seg/T/hard_dice"].toFixed(2)} / ${mt["seg/P/hard_dice"].toFixed(2)} / ${mt["seg/ST/hard_dice"].toFixed(2)}`;
      };
      if (shown.every(m => fmtCombo(m) === "—")) return;
      html += `<tr><td>${r.label}</td>${shown.map(m => `<td>${fmtCombo(m)}</td>`).join("")}</tr>`;
      return;
    }

    const vals = shown.map(m => {
      const v = m.metrics?.[r.key];
      return (typeof v === "number" && !Number.isNaN(v)) ? v : undefined;
    });
    if (vals.every(v => v === undefined)) return;

    const validVals = vals.filter(v => v !== undefined);
    let bestVal = null;
    if (validVals.length > 1) {
      bestVal = validVals.reduce((a, b) => {
        const va = r.absCmp ? Math.abs(a) : a;
        const vb = r.absCmp ? Math.abs(b) : b;
        return (r.higher ? vb > va : vb < va) ? b : a;
      });
    }

    html += `<tr><td>${r.label}</td>`;
    vals.forEach(v => {
      const isBest = v !== undefined && bestVal !== null && v === bestVal;
      html += `<td class="${isBest ? "best" : ""}">${v !== undefined ? r.fmt(v) : "—"}</td>`;
    });
    html += `</tr>`;
  });
  html += `</tbody></table>`;
  el.innerHTML = html;
  renderDenseLeakageBreakdown(shown);
}

function renderDenseLeakageBreakdown(shownModels) {
  const el = document.getElementById("dense-leakage-table");
  if (!el) return;
  const shown = shownModels || [];
  if (!shown.length) {
    el.innerHTML = "";
    return;
  }
  const rows = [
    { label: "P Soft-Dice",                 key: "seg/P/soft_dice",               higher: true,  fmt: v => v.toFixed(4) },
    { label: "PQ Soft-Dice",                key: "seg/PQ/soft_dice",              higher: true,  fmt: v => v.toFixed(4) },
    { label: "QRS Soft-Dice",               key: "seg/QRS/soft_dice",             higher: true,  fmt: v => v.toFixed(4) },
    { label: "ST Soft-Dice",                key: "seg/ST/soft_dice",              higher: true,  fmt: v => v.toFixed(4) },
    { label: "T Soft-Dice",                 key: "seg/T/soft_dice",               higher: true,  fmt: v => v.toFixed(4) },
    { label: "TP Soft-Dice",                key: "seg/TP/soft_dice",              higher: true,  fmt: v => v.toFixed(4) },
    { label: "GT Mask Probe AUROC",         key: "leakage/gt_mask_probe_auroc",   higher: true,  fmt: v => v.toFixed(4) },
    { label: "Pred Mask Probe AUROC",       key: "leakage/pred_mask_probe_auroc", higher: false, fmt: v => v.toFixed(4) },
    { label: "Soft Mask Excess ΔAUROC ↓",   key: "leakage/soft_mask_excess_auroc",higher: false, fmt: v => v.toFixed(4), absCmp: true },
  ];
  let html = `<table class="metrics-table"><thead><tr><th>Dense Wave Dice &amp; Leakage Probes</th>`;
  shown.forEach(m => { html += `<th>${m.id}</th>`; });
  html += `</tr></thead><tbody>`;
  rows.forEach(r => {
    const vals = shown.map(m => {
      const v = m.metrics?.[r.key];
      return (typeof v === "number" && !Number.isNaN(v)) ? v : undefined;
    });
    if (vals.every(v => v === undefined)) return;
    const valid = vals.filter(v => v !== undefined);
    const best = valid.length > 1
      ? valid.reduce((a, b) => {
          const va = r.absCmp ? Math.abs(a) : a;
          const vb = r.absCmp ? Math.abs(b) : b;
          return (r.higher ? vb > va : vb < va) ? b : a;
        })
      : null;
    html += `<tr><td>${r.label}</td>`;
    vals.forEach(v => {
      const cls = (v !== undefined && best !== null && v === best) ? "best" : "";
      html += `<td class="${cls}">${v !== undefined ? r.fmt(v) : "—"}</td>`;
    });
    html += `</tr>`;
  });
  html += `</tbody></table>`;
  el.innerHTML = html;
}

function renderUnifiedSegmentationView() {
  const container = document.getElementById("dense-seg-canvas");
  const uncBox    = document.getElementById("unc-strip-container");
  const legBox    = document.getElementById("dense-seg-legend");
  if (!container || !uncBox) return;
  container.innerHTML = "";
  uncBox.innerHTML    = "";

  const mode = document.getElementById("seg-compare-mode")?.value || "cem_vs_gt";
  const showUnc = document.getElementById("seg-show-unc")?.checked;
  const d = state.denseData;
  const gtMap = d?.gt_probs || state.softSeg;

  // Resolve segmentation-capable objects safely (guarding against dense_blackbox where has_seg === false)
  const hasSeg = obj => Boolean(obj && obj.has_seg !== false && obj.concept_probs);
  const primSeg = hasSeg(d) ? d : null;
  const cmpSeg  = hasSeg(d?.compare) ? d.compare : null;
  const cemObj  = (primSeg?.model_type === "dense_cem" ? primSeg : (cmpSeg?.model_type === "dense_cem" ? cmpSeg : null));
  const cbmObj  = (primSeg?.model_type === "dense_cbm" ? primSeg : (cmpSeg?.model_type === "dense_cbm" ? cmpSeg : null));

  let solidProbs = null, solidStds = null, solidLabel = "";
  let backProbs  = null, backLabel = "", backIsDashed = false;

  if (mode === "gt_only") {
    solidProbs = gtMap; solidLabel = "Zenodo 7610236 / ECGdeli GT";
  } else if (mode === "cem_vs_gt") {
    const obj = cemObj || primSeg || cmpSeg;
    solidProbs = obj?.concept_probs; solidStds = obj?.concept_stds; solidLabel = obj?.model_id || "DenseCEM";
    backProbs  = gtMap; backLabel = "Zenodo GT"; backIsDashed = false;
  } else if (mode === "cbm_vs_gt") {
    const obj = cbmObj || cmpSeg || primSeg;
    solidProbs = obj?.concept_probs; solidStds = obj?.concept_stds; solidLabel = obj?.model_id || "DenseCBM";
    backProbs  = gtMap; backLabel = "Zenodo GT"; backIsDashed = false;
  } else if (mode === "cem_vs_cbm") {
    const sObj = cemObj || primSeg;
    const bObj = cbmObj || (cmpSeg && cmpSeg !== sObj ? cmpSeg : null);
    solidProbs = sObj?.concept_probs; solidStds = sObj?.concept_stds; solidLabel = sObj?.model_id || "DenseCEM";
    backProbs  = bObj?.concept_probs; backLabel = bObj?.model_id || "DenseCBM"; backIsDashed = true;
  }

  if (!solidProbs && !backProbs) {
    const noSegHint = (d && !primSeg && !cmpSeg)
      ? "Selected dense model (Black-Box) has no soft-segmentation head; choose DenseCEM, DenseCBM, or DenseMultiTask, or switch View to Zenodo GT Only."
      : "Waiting for dense soft-segmentation inference…";
    container.innerHTML = `<div class="hint" style="padding:1rem">${noSegHint}</div>`;
    uncBox.style.display = "none";
    return;
  }

  const W  = chartWidth();
  const H  = LINE_HEIGHT;
  const { left: ml, right: mr, top: mt, bottom: mb } = SIG_MARGIN;
  const T  = state.nSamples;
  const sr = state.samplingRate;
  const segSr = d?.sampling_rate || state.softSeg?.sampling_rate || 100;
  const scaleSeg = sr / segSr;

  const svg = d3.select(container).append("svg").attr("width", W + ml + mr).attr("height", H + mt + mb);
  const g   = svg.append("g").attr("transform", `translate(${ml},${mt})`);
  const xScale = d3.scaleLinear().domain([0, T]).range([0, W]);
  const yScale = d3.scaleLinear().domain([0, 1]).range([H, 0]);

  g.append("g").attr("class", "grid").call(d3.axisLeft(yScale).tickSize(-W).tickFormat("").ticks(4));
  g.append("g").attr("class", "axis").attr("transform", `translate(0,${H})`)
    .call(d3.axisBottom(xScale).ticks(10).tickFormat(v => `${(v / sr).toFixed(1)}s`));
  g.append("g").attr("class", "axis").call(d3.axisLeft(yScale).ticks(4).tickFormat(d3.format(".1f")));

  const groups = [
    { key: "P",   col: "#2b5b84" },
    { key: "PQ",  col: "#2a7f8e" },
    { key: "QRS", col: "#b83b2a" },
    { key: "ST",  col: "#9e4268" },
    { key: "T",   col: "#2d7a56" },
  ];

  groups.forEach(gr => {
    const bArr = backProbs?.[gr.key] || [];
    if (bArr.length) {
      if (backIsDashed) {
        g.append("path").datum(bArr).attr("fill", "none").attr("stroke", gr.col)
          .attr("stroke-width", 1.25).attr("stroke-dasharray", "4,3").attr("opacity", 0.8)
          .attr("d", d3.line().x((_, i) => xScale(i * scaleSeg)).y(v => yScale(v)));
      } else {
        g.append("path").datum(bArr).attr("fill", gr.col).attr("opacity", 0.22)
          .attr("d", d3.area().x((_, i) => xScale(i * scaleSeg)).y0(H).y1(v => yScale(v)));
      }
    }
    const sArr = solidProbs?.[gr.key] || [];
    const uArr = solidStds?.[gr.key]  || [];
    if (sArr.length) {
      if (!backProbs || backIsDashed) {
        g.append("path").datum(sArr).attr("fill", gr.col).attr("opacity", 0.14)
          .attr("d", d3.area().x((_, i) => xScale(i * scaleSeg)).y0(H).y1(v => yScale(v)));
      }
      if (uArr.length === sArr.length) {
        g.append("path").datum(sArr).attr("fill", gr.col).attr("opacity", 0.16)
          .attr("d", d3.area()
            .x((_, i) => xScale(i * scaleSeg))
            .y0((v, i) => yScale(Math.max(0, v - uArr[i])))
            .y1((v, i) => yScale(Math.min(1, v + uArr[i]))));
      }
      g.append("path").datum(sArr).attr("fill", "none").attr("stroke", gr.col).attr("stroke-width", 1.75)
        .attr("d", d3.line().x((_, i) => xScale(i * scaleSeg)).y(v => yScale(v)));
    }
  });

  attachSyncedWindowOverlay("dense_seg", g, xScale, H);

  // Interactive hover readout + click to move ECG window
  const hoverLine = g.append("line").attr("y1", 0).attr("y2", H)
    .attr("stroke", "#767676").attr("stroke-width", 1).attr("stroke-dasharray", "2,2").style("display", "none");
  const readout = document.getElementById("seg-hover-readout");
  svg.on("mousemove", e => {
    const [mx] = d3.pointer(e, g.node());
    if (mx < 0 || mx > W) { hoverLine.style("display", "none"); return; }
    hoverLine.style("display", null).attr("x1", mx).attr("x2", mx);
    const sigSample = xScale.invert(mx);
    const ti = Math.max(0, Math.min(999, Math.round(sigSample / scaleSeg)));
    const src = solidProbs || backProbs;
    const vals = groups.filter(gr => src[gr.key]).map(gr => `${gr.key}:${(src[gr.key][ti] || 0).toFixed(2)}`).join(" ");
    const sigT = cemObj?.mean_std_timeline?.[ti];
    readout.textContent = `t=${(ti / segSr).toFixed(2)}s · ${vals}${sigT !== undefined ? ` · σ=${sigT.toFixed(3)}` : ""}`;
  }).on("mouseleave", () => { hoverLine.style("display", "none"); });

  const halfWin = state.tFixed / 2;
  svg.style("cursor", "pointer").on("click", async e => {
    if (e.defaultPrevented) return;
    const [mx] = d3.pointer(e, g.node());
    if (mx < 0 || mx > W) return;
    const start = Math.max(0, Math.min(T - state.tFixed, Math.round(xScale.invert(mx) - halfWin)));
    updateWindowPosition(start);
    await updatePipeline();
  });

  const backDesc = backProbs ? (backIsDashed ? ` · Dashed: ${backLabel}` : ` · Shaded: ${backLabel}`) : "";
  if (legBox) {
    legBox.innerHTML =
      groups.map(gr => `<span class="leg-item"><i class="swatch" style="background:${gr.col}"></i>${gr.key}</span>`).join("") +
      `<span class="meta-label">Solid: ${solidLabel}${backDesc}</span>`;
  }

  // Optional synchronized per-timestamp boundary uncertainty strip σ(t)
  if (showUnc && (cemObj?.mean_std_timeline || cbmObj?.mean_std_timeline)) {
    uncBox.style.display = "";
    const UH = 66;
    const uSvg = d3.select(uncBox).append("svg").attr("width", W + ml + mr).attr("height", UH + 24);
    const ug   = uSvg.append("g").attr("transform", `translate(${ml},8)`);
    const uCem = cemObj?.mean_std_timeline || [];
    const uCbm = cbmObj?.mean_std_timeline || [];
    const uMax = Math.max(0.03, d3.max(uCem) || 0, d3.max(uCbm) || 0) * 1.15;
    const uy   = d3.scaleLinear().domain([0, uMax]).range([UH, 0]);

    ug.append("g").attr("class", "grid").call(d3.axisLeft(uy).tickSize(-W).tickFormat("").ticks(2));
    ug.append("g").attr("class", "axis").call(d3.axisLeft(uy).ticks(2).tickFormat(d3.format(".3f")));
    if (uCem.length) {
      ug.append("path").datum(uCem).attr("fill", "#b83b2a").attr("opacity", 0.14)
        .attr("d", d3.area().x((_, i) => xScale(i * scaleSeg)).y0(UH).y1(v => uy(v)));
      ug.append("path").datum(uCem).attr("fill", "none").attr("stroke", "#b83b2a").attr("stroke-width", 1.35)
        .attr("d", d3.line().x((_, i) => xScale(i * scaleSeg)).y(v => uy(v)));
    }
    if (uCbm.length) {
      ug.append("path").datum(uCbm).attr("fill", "none").attr("stroke", "#2b5b84")
        .attr("stroke-width", 1.2).attr("stroke-dasharray", "4,3").attr("opacity", 0.85)
        .attr("d", d3.line().x((_, i) => xScale(i * scaleSeg)).y(v => uy(v)));
    }
    ug.append("text").attr("x", 4).attr("y", 10).attr("fill", "#767676").attr("font-size", 8.5)
      .text("Per-Timestamp Boundary Uncertainty σ(t) — Terracotta: DenseCEM · Slate dashed: DenseCBM");
    attachSyncedWindowOverlay("unc", ug, xScale, UH);
    uSvg.style("cursor", "pointer").on("click", async e => {
      if (e.defaultPrevented) return;
      const [mx] = d3.pointer(e, ug.node());
      if (mx < 0 || mx > W) return;
      const start = Math.max(0, Math.min(T - state.tFixed, Math.round(xScale.invert(mx) - halfWin)));
      updateWindowPosition(start);
      await updatePipeline();
    });
  } else {
    _syncedWindowOverlays.delete("unc");
    uncBox.style.display = "none";
  }
}

// ── Synchronized Concept Activation Lineplot (Primary solid + Compare dashed)
function renderActivationLineplot(data) {
  const container = document.getElementById("lineplot-container");
  container.innerHTML = "";

  const W  = chartWidth();
  const H  = LINE_HEIGHT;
  const { left: ml, right: mr, top: mt, bottom: mb } = SIG_MARGIN;
  const T  = state.nSamples;
  const sr = data.sampling_rate || state.samplingRate;
  const halfWin = data.t_fixed / 2;

  const svg = d3.select(container).append("svg")
    .attr("width", W + ml + mr)
    .attr("height", H + mt + mb);
  const g = svg.append("g").attr("transform", `translate(${ml},${mt})`);

  const xScale = d3.scaleLinear().domain([0, T]).range([0, W]);
  const yScale = d3.scaleLinear().domain([0, 1]).range([H, 0]);

  // Ground-truth complete-beat shading
  if (data.gt_task) {
    const stepW = data.starts.length > 1 ? (xScale(data.starts[1]) - xScale(data.starts[0])) : 4;
    data.starts.forEach((s, i) => {
      if (data.gt_task[i] === 1) {
        g.append("rect")
          .attr("x", xScale(s + halfWin) - stepW / 2)
          .attr("y", 0)
          .attr("width", Math.max(1, stepW + 0.5))
          .attr("height", H)
          .attr("fill", "rgba(45, 122, 86, 0.09)");
      }
    });
  }

  // Grid & axes
  g.append("g").attr("class", "grid")
    .call(d3.axisLeft(yScale).tickSize(-W).tickFormat("").ticks(4));
  g.append("g").attr("class", "axis")
    .attr("transform", `translate(0,${H})`)
    .call(d3.axisBottom(xScale).ticks(10).tickFormat(d => `${(d / sr).toFixed(1)}s`));
  g.append("g").attr("class", "axis")
    .call(d3.axisLeft(yScale).ticks(4).tickFormat(d3.format(".1f")));

  // 0.5 threshold
  g.append("line")
    .attr("x1", 0).attr("x2", W).attr("y1", yScale(0.5)).attr("y2", yScale(0.5))
    .attr("stroke", "#d0d0d0").attr("stroke-dasharray", "4,4").attr("stroke-width", 0.9);

  const cNames = data.concept_names || state.conceptNames;
  state.conceptNames = cNames;
  const intervalSet = new Set(["PR", "QRS", "ST", "QT"]);
  const series = state.timelineView === "intervals"
    ? [{ label: "Task", idx: -1, col: CONCEPT_COLORS.Task }].concat(
        cNames.map((n, i) => ({ label: n, idx: i, col: conceptColor(n, i) }))
              .filter(s => intervalSet.has(s.label) || cNames.length <= 6)
      )
    : cNames.map((n, i) => ({ label: n, idx: i, col: conceptColor(n, i) }))
            .filter(s => !intervalSet.has(s.label) || cNames.length <= 6);

  const drawSet = (taskArr, concArr, dashed, width, opacity) => {
    series.forEach(s => {
      const lineGen = d3.line()
        .x((_, i) => xScale(data.starts[i] + halfWin))
        .y((_, i) => yScale(s.idx === -1 ? taskArr[i] : concArr[i][s.idx]));
      const path = g.append("path")
        .datum(data.starts)
        .attr("fill", "none")
        .attr("stroke", s.col)
        .attr("stroke-width", width)
        .attr("opacity", opacity)
        .attr("d", lineGen);
      if (dashed) path.attr("stroke-dasharray", "5,3");
    });
  };

  drawSet(data.task_probs, data.concept_probs, false, 1.9, 0.95);
  if (data.compare) {
    drawSet(data.compare.task_probs, data.compare.concept_probs, true, 1.4, 0.78);
  }

  // Synced 1.0s Window Box, L/R Handles & Midpoint Cursor
  attachSyncedWindowOverlay("lineplot", g, xScale, H);

  svg.style("cursor", "pointer").on("click", async e => {
    if (e.defaultPrevented) return;
    const [mx] = d3.pointer(e, g.node());
    if (mx < 0 || mx > W) return;
    const start = Math.max(0, Math.min(T - data.t_fixed, Math.round(xScale.invert(mx) - halfWin)));
    updateWindowPosition(start);
    await updatePipeline();
  });

  // Update legend
  const styleNote = data.compare
    ? `<span class="meta-label">Solid: ${data.model_id} · Dashed: ${data.compare.model_id} · Green shade: GT Complete</span>`
    : `<span class="meta-label">${data.model_id} · Green shade: GT Complete</span>`;
  document.getElementById("timeline-legend").innerHTML =
    series.map(s => `<span class="leg-item"><i class="swatch" style="background:${s.col}"></i>${s.label}</span>`).join("") +
    styleNote;
}

// ── Sweep Heatmap (HiDPI Canvas + lightweight SVG axis/cursor overlay) ──────
let _HEATMAP_LUT = null;
function _getHeatmapLUT() {
  if (_HEATMAP_LUT) return _HEATMAP_LUT;
  const color = d3.scaleSequential(d3.interpolateYlOrRd).domain([0, 255]);
  _HEATMAP_LUT = new Array(256);
  for (let i = 0; i < 256; i++) _HEATMAP_LUT[i] = color(i);
  return _HEATMAP_LUT;
}

function renderHeatmap(data) {
  const container = document.getElementById("heatmap-container");
  container.innerHTML = "";
  const names  = data.concept_names;
  const starts = data.starts;
  const cProbs = data.concept_probs;
  const K      = names.length;
  const N      = starts.length;
  const W      = chartWidth();
  const { left: ml, right: mr } = SIG_MARGIN;

  const CELL_H  = 20;
  const CELL_W  = W / N;
  const MAP_TOP = 8;
  const MAP_BOT = 28;
  const MAP_H   = CELL_H * K;
  const sr      = data.sampling_rate || state.samplingRate;
  const lut     = _getHeatmapLUT();

  const wrap = document.createElement("div");
  wrap.style.position = "relative";
  wrap.style.width = `${W + ml + mr}px`;
  wrap.style.height = `${MAP_TOP + MAP_H + MAP_BOT}px`;
  container.appendChild(wrap);

  const dpr = Math.min(window.devicePixelRatio || 1, 2);
  const canvas = document.createElement("canvas");
  canvas.width = Math.max(1, Math.round(W * dpr));
  canvas.height = Math.max(1, Math.round(MAP_H * dpr));
  canvas.style.position = "absolute";
  canvas.style.left = `${ml}px`;
  canvas.style.top = `${MAP_TOP}px`;
  canvas.style.width = `${W}px`;
  canvas.style.height = `${MAP_H}px`;
  canvas.style.pointerEvents = "none";
  wrap.appendChild(canvas);

  const ctx = canvas.getContext("2d");
  ctx.scale(dpr, dpr);
  const drawW = Math.ceil(CELL_W) + 0.5;
  const drawH = CELL_H - 1;
  for (let ki = 0; ki < K; ki++) {
    const y = ki * CELL_H;
    for (let ni = 0; ni < N; ni++) {
      const idx = Math.max(0, Math.min(255, Math.round((cProbs[ni][ki] || 0) * 255)));
      ctx.fillStyle = lut[idx];
      ctx.fillRect(ni * CELL_W, y, drawW, drawH);
    }
  }

  const svg = d3.select(wrap).append("svg")
    .attr("width", W + ml + mr)
    .attr("height", MAP_TOP + MAP_H + MAP_BOT)
    .style("position", "relative")
    .style("z-index", "1");
  const g = svg.append("g").attr("transform", `translate(${ml},${MAP_TOP})`);

  names.forEach((name, ki) => {
    svg.append("text")
      .attr("x", ml - 6).attr("y", MAP_TOP + ki * CELL_H + CELL_H / 2 + 4)
      .attr("text-anchor", "end").attr("fill", "#767676").attr("font-size", 10)
      .text(name);
  });

  const xScale = d3.scaleLinear().domain([0, state.nSamples]).range([0, W]);
  g.append("g").attr("class", "axis")
    .attr("transform", `translate(0,${MAP_H})`)
    .call(d3.axisBottom(xScale).ticks(10).tickFormat(d => `${(d / sr).toFixed(1)}s`));

  g.append("rect")
    .attr("x", 0).attr("y", 0).attr("width", W).attr("height", MAP_H)
    .attr("fill", "transparent")
    .style("cursor", "pointer")
    .on("click", async e => {
      if (e.defaultPrevented) return;
      const [mx] = d3.pointer(e, g.node());
      const ni = Math.max(0, Math.min(N - 1, Math.floor(mx / CELL_W)));
      updateWindowPosition(starts[ni]);
      await updatePipeline();
    });

  attachSyncedWindowOverlay("heatmap", g, xScale, MAP_H);
  document.getElementById("timeline-legend").innerHTML =
    `<span class="meta-label">Heatmap: ${data.model_id} concept probabilities (0 → 1)</span>`;
}

// ── Pipeline Update & Live Concept Interventions ───────────────────────────
function findNearestSweepIndex(windowStart) {
  const starts = state.sweepData.starts;
  let best = 0, bestDist = Infinity;
  starts.forEach((s, i) => {
    const d = Math.abs(s - windowStart);
    if (d < bestDist) { bestDist = d; best = i; }
  });
  return best;
}

function updatePipelineFromSweep() {
  if (!state.sweepData) return;
  const ni = findNearestSweepIndex(state.windowStart);
  const d  = state.sweepData;
  const gtT = d.gt_task ? d.gt_task[ni] : null;
  const gtC = d.gt_concepts ? d.gt_concepts[ni] : null;
  const cmp = d.compare ? {
    model_id:      d.compare.model_id,
    model_type:    d.compare.model_type,
    task_prob:     d.compare.task_probs[ni],
    task_std:      d.compare.task_stds[ni],
    concept_probs: d.compare.concept_probs[ni],
  } : null;

  renderPipelineValues({
    model_id:       d.model_id,
    model_type:     d.model_type,
    task_prob:      d.task_probs[ni],
    task_std:       d.task_stds[ni],
    task_aleatoric: d.task_stds[ni],
    task_epistemic: 0.0,
    concept_probs:  d.concept_probs[ni],
    concept_stds:   d.concept_stds[ni],
    gt_task:        gtT,
    gt_concepts:    gtC,
    compare:        cmp,
  });
}

async function updatePipeline() {
  if (state.ecgId === null) return;
  renderWindowPlot();

  const scope = document.getElementById("dense-int-scope")?.value || "record";
  const wantDenseWinSync = Boolean(state.denseIntMode && scope === "window");
  const hasInt = Object.keys(state.interventions).length > 0;
  const wantEpistemic = document.getElementById("epistemic-chk").checked;
  if (state.sweepData && !hasInt && !wantEpistemic) {
    updatePipelineFromSweep();
    if (wantDenseWinSync) await runDenseInfer();
    return;
  }

  setLoadingPill("pipeline-loading-indicator", true, wantEpistemic ? "Running MC-Dropout…" : "Updating…");
  setSectionLoading("pipeline-section", true);
  try {
    const result = await apiPost("/infer", {
      ecg_id:            state.ecgId,
      dataset:           state.dataset,
      start:             state.windowStart,
      model_id:          state.primaryModel,
      compare_model_id:  state.compareModel || null,
      interventions:     hasInt ? state.interventions : null,
      include_epistemic: wantEpistemic,
    });
    if (result.concept_names) state.conceptNames = result.concept_names;
    renderPipelineValues(result);
    if (wantDenseWinSync) await runDenseInfer();
  } catch (e) {
    console.error("Infer error:", e);
  } finally {
    setLoadingPill("pipeline-loading-indicator", false);
    setSectionLoading("pipeline-section", false);
  }
}

function renderPipelineValues(res) {
  document.getElementById("predictor-label").textContent =
    res.model_type === "cem" ? "CEM Mix" : (res.model_type === "blackbox" ? "ResNet Head" : "CBM Linear");

  const primIsBB = res.model_type === "blackbox";
  const cmpHasConcepts = Boolean(res.compare && res.compare.model_type !== "blackbox");
  const titleEl = document.getElementById("concepts-panel-title");
  if (titleEl) {
    titleEl.textContent = primIsBB
      ? "Concept Bottleneck (Unavailable — Black-Box Model)"
      : "Concept Activations & Interventions";
  }

  const gtPill = document.getElementById("gt-window-pill");
  if (res.gt_task === 1) {
    gtPill.textContent = "GT: Complete Beat";
    gtPill.className   = "gt-pill complete";
  } else if (res.gt_task === 0) {
    gtPill.textContent = "GT: Incomplete";
    gtPill.className   = "gt-pill incomplete";
  } else {
    gtPill.textContent = "";
    gtPill.className   = "gt-pill";
  }

  const hasInt = Object.keys(state.interventions || {}).length > 0;
  let baseProb = null, cmpBaseProb = null;
  if (hasInt && state.sweepData?.starts?.length) {
    const ni = findNearestSweepIndex(state.windowStart);
    baseProb = state.sweepData.task_probs?.[ni] ?? null;
    if (state.sweepData.compare?.task_probs) {
      cmpBaseProb = state.sweepData.compare.task_probs[ni] ?? null;
    }
  }

  const cmpProbs = cmpHasConcepts ? res.compare.concept_probs : null;
  renderHistogram(res.concept_probs, res.concept_stds, res.gt_concepts, cmpProbs, primIsBB);
  renderInterventionPills(!primIsBB);
  renderTaskPrediction(res.model_id, res.task_prob, res.task_std, res.compare, {
    taskAle: res.task_aleatoric,
    taskEpi: res.task_epistemic,
    baseProb,
    cmpBaseProb,
  });
}

function renderInterventionPills(canIntervene = true) {
  const bar = document.getElementById("intervention-controls");
  const names = state.conceptNames;
  bar.style.gridTemplateColumns = `repeat(${names.length}, 1fr)`;
  bar.innerHTML = "";
  names.forEach((name, idx) => {
    const cur = state.interventions[idx];
    const btn = document.createElement("button");
    btn.className = "int-pill" + (cur === 1 ? " force-1" : cur === 0 ? " force-0" : "");
    btn.textContent = cur === 1 ? `${name}=1` : cur === 0 ? `${name}=0` : `${name}`;
    btn.disabled = !canIntervene;
    btn.style.opacity = canIntervene ? "1" : "0.45";
    btn.title = canIntervene
      ? `Click to cycle intervention for ${name}: Auto → Force 1 → Force 0 → Auto`
      : "Black-Box model has no concept bottleneck to intervene on";
    btn.addEventListener("click", async () => {
      if (!canIntervene) return;
      if (state.interventions[idx] === undefined)      state.interventions[idx] = 1;
      else if (state.interventions[idx] === 1)         state.interventions[idx] = 0;
      else                                             delete state.interventions[idx];
      updateResetIntButton();
      await updatePipeline();
    });
    bar.appendChild(btn);
  });
}

function updateResetIntButton() {
  const n = Object.keys(state.interventions).length;
  const btn = document.getElementById("reset-int-btn");
  btn.style.display = n > 0 ? "" : "none";
  btn.textContent   = `Reset Interventions (${n})`;
}

// ── Window Mini-Plot ───────────────────────────────────────────────────────
function renderWindowPlot() {
  const container = document.getElementById("window-plot");
  container.innerHTML = "";
  if (!state.signal) return;

  const raw     = state.signal[state.leadIdx];
  const ds      = state.downsample;
  const sr      = state.samplingRate;
  const startDs = Math.floor(state.windowStart / ds);
  const endDs   = Math.min(Math.ceil((state.windowStart + state.tFixed) / ds), raw.length);
  const slice   = raw.slice(startDs, endDs);
  if (!slice.length) return;

  const W = 206, H = 185;
  const pad = { top: 10, right: 10, bottom: 28, left: 34 };
  const iW = W - pad.left - pad.right, iH = H - pad.top - pad.bottom;

  const svg = d3.select(container).append("svg").attr("width", W).attr("height", H);
  const g   = svg.append("g").attr("transform", `translate(${pad.left},${pad.top})`);

  const xScale = d3.scaleLinear().domain([0, slice.length - 1]).range([0, iW]);
  const yMin = d3.min(slice), yMax = d3.max(slice);
  const yPad = (yMax - yMin) * 0.15 || 0.2;
  const yScale = d3.scaleLinear().domain([yMin - yPad, yMax + yPad]).range([iH, 0]);

  g.append("g").attr("class", "grid").call(d3.axisLeft(yScale).tickSize(-iW).tickFormat("").ticks(3));
  g.append("g").attr("class", "axis").attr("transform", `translate(0,${iH})`)
    .call(d3.axisBottom(xScale).ticks(4).tickFormat(d => `${((state.windowStart + d * ds) / sr).toFixed(1)}s`));
  g.append("g").attr("class", "axis").call(d3.axisLeft(yScale).ticks(3));

  g.append("path").datum(slice).attr("fill", "rgba(184, 59, 42, 0.06)")
    .attr("d", d3.area().x((_, i) => xScale(i)).y0(iH).y1(d => yScale(d)));
  g.append("path").datum(slice).attr("fill", "none").attr("stroke", "#222222").attr("stroke-width", 1.35)
    .attr("d", d3.line().x((_, i) => xScale(i)).y(d => yScale(d)));

  const midX = xScale((slice.length - 1) / 2);
  g.append("line")
    .attr("class", "sync-cursor")
    .attr("x1", midX).attr("x2", midX)
    .attr("y1", 0).attr("y2", iH);
}

// ── Concept Histogram (with ±1σ, GT dots, and optional Compare marker) ─────
function renderHistogram(probs, stds, gtConcepts, cmpProbs, primIsBlackbox = false) {
  const container = document.getElementById("histogram-container");
  container.innerHTML = "";
  if (primIsBlackbox && !cmpProbs) {
    container.innerHTML = `<div class="hint" style="padding:1.25rem 0.5rem">Black-Box ResNet predicts heartbeat completeness directly from z ∈ ℝ⁵¹² without a concept bottleneck; select a CBM or CEM in Compare to view concept activations.</div>`;
    return;
  }
  const names = state.conceptNames;

  const W = 510, H = 175;
  const pad = { top: 20, right: 12, bottom: 26, left: 36 };
  const innerW = W - pad.left - pad.right;
  const innerH = H - pad.top - pad.bottom;

  const svg = d3.select(container).append("svg").attr("width", W).attr("height", H);
  const g   = svg.append("g").attr("transform", `translate(${pad.left},${pad.top})`);

  const xScale = d3.scaleBand().domain(names).range([0, innerW]).padding(0.26);
  const yScale = d3.scaleLinear().domain([0, 1]).range([innerH, 0]);

  g.append("g").attr("class", "grid").call(d3.axisLeft(yScale).tickSize(-innerW).tickFormat("").ticks(4));
  g.append("g").attr("class", "axis").attr("transform", `translate(0,${innerH})`).call(d3.axisBottom(xScale));
  g.append("g").attr("class", "axis").call(d3.axisLeft(yScale).ticks(4).tickFormat(d3.format(".1f")));

  const barW = xScale.bandwidth();
  names.forEach((name, i) => {
    const p   = Math.max(0, Math.min(1, probs[i] || 0));
    const s   = Math.min(Math.max(stds[i] || 0, 0), 0.5);
    const x   = xScale(name);
    const col = conceptColor(name, i);
    const cx  = x + barW / 2;

    g.append("rect").attr("class", "hist-bar-bg")
      .attr("x", x).attr("y", 0).attr("width", barW).attr("height", innerH).attr("rx", 3);

    // ±σ uncertainty band
    const bTop = Math.max(0, yScale(Math.min(1, p + s)));
    const bBot = Math.min(innerH, yScale(Math.max(0, p - s)));
    g.append("rect").attr("x", x).attr("y", bTop).attr("width", barW).attr("height", Math.max(1, bBot - bTop))
      .attr("rx", 2).attr("fill", "#a67c00").attr("opacity", 0.22);

    // Primary bar
    const yT = yScale(p);
    g.append("rect").attr("class", "hist-bar-fill")
      .attr("x", x + 2).attr("y", yT).attr("width", barW - 4).attr("height", Math.max(0, innerH - yT))
      .attr("fill", col).attr("opacity", 0.88);

    // Compare model marker (dark crisp horizontal tick)
    if (cmpProbs) {
      const cy = yScale(Math.max(0, Math.min(1, cmpProbs[i] || 0)));
      g.append("line")
        .attr("x1", x).attr("x2", x + barW).attr("y1", cy).attr("y2", cy)
        .attr("stroke", "#222222").attr("stroke-width", 2).attr("opacity", 0.9);
    }

    // Ground truth dot
    if (gtConcepts && gtConcepts[i] !== undefined) {
      g.append("circle")
        .attr("cx", cx).attr("cy", yScale(gtConcepts[i]))
        .attr("r", 3.2).attr("fill", "#222222").attr("stroke", "#ffffff").attr("stroke-width", 1.2);
    }

    g.append("text").attr("x", cx).attr("y", Math.max(yT - 4, 9))
      .attr("text-anchor", "middle").attr("fill", "#222222").attr("font-size", 9)
      .text(p.toFixed(2));
  });

  // Compact top legend
  let lx = innerW - 210;
  if (cmpProbs) {
    g.append("line").attr("x1", lx).attr("x2", lx + 10).attr("y1", -11).attr("y2", -11)
      .attr("stroke", "#222222").attr("stroke-width", 2);
    g.append("text").attr("x", lx + 14).attr("y", -8).attr("fill", "#767676").attr("font-size", 8)
      .text("Compare");
    lx += 64;
  }
  g.append("circle").attr("cx", lx + 4).attr("cy", -11).attr("r", 3).attr("fill", "#222222");
  g.append("text").attr("x", lx + 11).attr("y", -8).attr("fill", "#767676").attr("font-size", 8).text("GT");
  lx += 36;
  g.append("rect").attr("x", lx).attr("y", -15).attr("width", 9).attr("height", 8).attr("fill", "#a67c00").attr("opacity", 0.4);
  g.append("text").attr("x", lx + 13).attr("y", -8).attr("fill", "#767676").attr("font-size", 8).text("±1σ");
}

// ── Task Prediction Bar Chart (Primary vs Compare + Baseline + Epistemic) ──
function renderTaskPrediction(modelId, taskProb, taskStd, cmpObj, opts = {}) {
  const container = document.getElementById("task-pred-container");
  container.innerHTML = "";
  const uncBox = document.getElementById("task-unc-breakdown");

  const W = 196, H = 188;
  const pad = { top: 22, right: 12, bottom: 38, left: 36 };
  const iW = W - pad.left - pad.right, iH = H - pad.top - pad.bottom;

  const svg = d3.select(container).append("svg").attr("width", W).attr("height", H);
  const g   = svg.append("g").attr("transform", `translate(${pad.left},${pad.top})`);

  const items = cmpObj
    ? [
        { label: modelId,         p: taskProb,        s: taskStd || 0,        b: opts.baseProb ?? null },
        { label: cmpObj.model_id, p: cmpObj.task_prob, s: cmpObj.task_std || 0, b: (cmpObj.model_type === "blackbox" ? null : (opts.cmpBaseProb ?? null)) },
      ]
    : [
        { label: "Complete",   p: taskProb,     s: taskStd || 0, b: opts.baseProb ?? null },
        { label: "Incomplete", p: 1 - taskProb, s: 0,            b: null },
      ];

  const xScale = d3.scaleBand().domain(items.map(d => d.label)).range([0, iW]).padding(0.32);
  const yScale = d3.scaleLinear().domain([0, 1]).range([iH, 0]);

  g.append("g").attr("class", "grid").call(d3.axisLeft(yScale).tickSize(-iW).tickFormat("").ticks(4));
  g.append("g").attr("class", "axis").attr("transform", `translate(0,${iH})`)
    .call(d3.axisBottom(xScale).tickSize(0));
  g.append("g").attr("class", "axis").call(d3.axisLeft(yScale).ticks(4).tickFormat(d3.format(".0%")));

  items.forEach((it, idx) => {
    const x  = xScale(it.label);
    const bW = xScale.bandwidth();
    const yT = yScale(it.p);
    const fill = (!cmpObj && idx === 1) ? "#d0d0d0" : taskColour(it.p);

    g.append("rect").attr("x", x).attr("y", 0).attr("width", bW).attr("height", iH).attr("fill", "#eeeeee").attr("rx", 2);
    g.append("rect").attr("x", x).attr("y", yT).attr("width", bW).attr("height", Math.max(0, iH - yT))
      .attr("fill", fill).attr("rx", 2).attr("opacity", 0.88);

    if (it.s > 0) {
      const eT = Math.max(0, yScale(Math.min(1, it.p + it.s)));
      const eB = Math.min(iH, yScale(Math.max(0, it.p - it.s)));
      g.append("rect").attr("x", x).attr("y", eT).attr("width", bW).attr("height", Math.max(1, eB - eT))
        .attr("fill", "#a67c00").attr("opacity", 0.24);
    }

    if (it.b !== null && it.b !== undefined) {
      const yb = yScale(it.b);
      g.append("line")
        .attr("x1", x - 2).attr("x2", x + bW + 2).attr("y1", yb).attr("y2", yb)
        .attr("stroke", "#222222").attr("stroke-width", 2.0);
    }

    g.append("text").attr("x", x + bW / 2).attr("y", Math.max(yT - 5, 9))
      .attr("text-anchor", "middle").attr("fill", "#222222").attr("font-size", 10).attr("font-weight", 600)
      .text(`${(it.p * 100).toFixed(1)}%`);

    if (it.s > 0) {
      g.append("text").attr("x", x + bW / 2).attr("y", iH + 26)
        .attr("text-anchor", "middle").attr("fill", "#767676").attr("font-size", 8)
        .text(`±${(it.s * 100).toFixed(1)}%`);
    }
  });

  if (uncBox) {
    const parts = [];
    if (opts.baseProb !== null && opts.baseProb !== undefined) {
      const diff = (taskProb - opts.baseProb) * 100;
      const sign = diff > 0.05 ? "+" : "";
      const cls  = diff > 0.5 ? "pos" : (diff < -0.5 ? "neg" : "neu");
      parts.push(`<span class="delta-badge ${cls}" title="Shift vs pre-intervention probability (${(opts.baseProb * 100).toFixed(1)}%)">Δ ${sign}${diff.toFixed(1)}% vs base</span>`);
    }
    const wantEp = document.getElementById("epistemic-chk")?.checked;
    const ale = opts.taskAle ?? taskStd ?? 0;
    const epi = opts.taskEpi ?? 0;
    if (wantEp || epi > 0) {
      parts.push(`<span title="Aleatoric = concept noise; Epistemic = MC-Dropout model/OOD disagreement">Ale ±${(ale * 100).toFixed(1)}% · Epi ±${(epi * 100).toFixed(1)}%</span>`);
    } else if (ale > 0) {
      parts.push(`<span>Ale ±${(ale * 100).toFixed(1)}% (enable Epistemic MC for split)</span>`);
    }
    uncBox.innerHTML = parts.join("");
  }
}

// ── Benchmark Summary & Concept Intervention Curves ────────────────────────
async function refreshBenchmarks() {
  if (!state.primaryModel) return;
  try {
    state.primarySummary = await apiGet(`/model/${encodeURIComponent(state.primaryModel)}/summary`);
  } catch (_) { state.primarySummary = null; }

  if (state.compareModel) {
    try {
      state.compareSummary = await apiGet(`/model/${encodeURIComponent(state.compareModel)}/summary`);
    } catch (_) { state.compareSummary = null; }
  } else {
    state.compareSummary = null;
  }

  renderConceptWeights();
  renderMetricsTable();
  renderInterventionCurves();
}

function renderConceptWeights() {
  const panel = document.getElementById("concept-weights-panel");
  const bars  = document.getElementById("concept-weights-bars");
  const note  = document.getElementById("concept-weights-note");
  if (!panel || !bars) return;

  const s = state.primarySummary;
  if (!s) {
    bars.innerHTML = "";
    return;
  }
  if (s.model_type === "blackbox") {
    if (note) note.textContent = `${s.id} has no concept bottleneck`;
    bars.innerHTML = `<div class="hint" style="grid-column:1/-1">Black-Box ResNet predicts directly from dense features z ∈ ℝ⁵¹² — no interpretable concept weights exist.</div>`;
    return;
  }
  const cw = s.classifier_weights || {};
  const names = s.concept_names || Object.keys(cw);
  const absMax = Math.max(1e-6, ...names.map(n => Math.abs(cw[n]?.prob_weight || 0)));
  if (note) {
    note.textContent = s.model_type === "cem"
      ? `${s.id} · mean |w_c| L2 norm of concept embeddings (0 = masked)`
      : `${s.id} · linear classifier weight w_c (+ pushes Complete, 0 = masked)`;
  }

  bars.innerHTML = names.map((n, i) => {
    const entry = cw[n] || {};
    const w = Number(entry.prob_weight || 0);
    const u = Number(entry.uncert_weight || 0);
    const pct = Math.min(100, Math.round((Math.abs(w) / absMax) * 100));
    const isMasked = Math.abs(w) < 1e-4;
    const col = isMasked ? "#bbbbbb" : conceptColor(n, i);
    const title = `${n}: prob_weight=${w.toFixed(4)}${u ? `, uncert_weight=${u.toFixed(4)}` : ""}${isMasked ? " (masked/unsupervised)" : ""}`;
    return `
      <div class="weight-item" title="${title}">
        <div class="weight-top${isMasked ? " masked" : ""}"><span>${n}</span><span>${w >= 0 && s.model_type !== "cem" ? "+" : ""}${w.toFixed(2)}</span></div>
        <div class="weight-track"><div class="weight-fill" style="width:${pct}%;background:${col}"></div></div>
      </div>`;
  }).join("");
}

function renderMetricsTable() {
  const el = document.getElementById("metrics-summary-container");
  const p = state.primarySummary, c = state.compareSummary;
  if (!p || !p.metrics) {
    el.innerHTML = `<div class="hint">No test evaluation JSON saved for ${state.primaryModel}.</div>`;
    return;
  }

  const rows = [
    { label: "Task Accuracy",         key: "task/accuracy",                        higher: true,  fmt: v => (v * 100).toFixed(2) + "%" },
    { label: "Task AUROC",            key: "task/auroc",                           higher: true,  fmt: v => v.toFixed(4) },
    { label: "Calibration (ECE ↓)",   key: "task/ece",                             higher: false, fmt: v => v.toFixed(4) },
    { label: "Concept Macro-F1",      key: "concept/macro_f1",                     higher: true,  fmt: v => v.toFixed(4) },
    { label: "Concept Macro-Acc",     key: "concept/macro_accuracy",               higher: true,  fmt: v => (v * 100).toFixed(2) + "%" },
    { label: "ST Interval F1",        key: "concept/ST/f1",                        higher: true,  fmt: v => v.toFixed(4) },
    { label: "QRS Interval F1",       key: "concept/QRS/f1",                       higher: true,  fmt: v => v.toFixed(4) },
    { label: "Error-Detection AUROC", key: "uncertainty/error_detection_auroc",     higher: true,  fmt: v => v.toFixed(4) },
    { label: "Noisy / Reliable σ²",   key: "uncertainty/noisy_vs_reliable_ratio",   higher: true,  fmt: v => v.toFixed(2) + "×" },
    { label: "Counterfactual Flip ↑", key: "faithfulness/counterfactual_flip_rate", higher: true,  fmt: v => (v * 100).toFixed(1) + "%" },
    { label: "Concept Impurity OIS ↓",key: "leakage/ois",                           higher: false, fmt: v => v.toFixed(4) },
  ];

  const hasCmp = Boolean(c && c.metrics);
  const s4 = (state.dataset === "georgia") ? state.externalBenchmark?.suite4_window_500hz_zeroshot : null;
  const subEl = document.getElementById("benchmark-subtitle");
  if (subEl) {
    subEl.textContent = s4
      ? `PTB-XL Fold 10 + Georgia 500 Hz Zero-Shot (${s4.n_windows} windows)`
      : "PTB-XL Fold 10";
  }

  let html = `<table class="metrics-table"><thead><tr><th>Metric (Test Fold 10)</th><th>${p.id} (${p.model_type.toUpperCase()})</th>`;
  if (hasCmp) html += `<th>${c.id} (${c.model_type.toUpperCase()})</th>`;
  html += `</tr></thead><tbody>`;

  rows.forEach(r => {
    const vp = p.metrics[r.key];
    const vc = hasCmp ? c.metrics[r.key] : undefined;
    if (vp === undefined && vc === undefined) return;
    let pCls = "", cCls = "";
    if (vp !== undefined && vc !== undefined) {
      if ((r.higher && vp > vc) || (!r.higher && vp < vc)) pCls = "best";
      else if ((r.higher && vc > vp) || (!r.higher && vc < vp)) cCls = "best";
    }
    html += `<tr><td>${r.label}</td><td class="${pCls}">${vp !== undefined ? r.fmt(vp) : "—"}</td>`;
    if (hasCmp) html += `<td class="${cCls}">${vc !== undefined ? r.fmt(vc) : "—"}</td>`;
    html += `</tr>`;
  });

  if (s4?.models && (s4.models[p.id] || (hasCmp && s4.models[c.id]))) {
    const gaRows = [
      { label: "Georgia 500 Hz Task Acc",      key: "task_accuracy",           fmt: v => (v * 100).toFixed(2) + "%" },
      { label: "Georgia 500 Hz Task AUROC",    key: "task_auroc",              fmt: v => v.toFixed(4) },
      { label: "Georgia Concept Macro-F1",     key: "shared_concept_macro_f1", fmt: v => v.toFixed(4) },
    ];
    gaRows.forEach(gr => {
      const vp = s4.models[p.id]?.[gr.key];
      const vc = hasCmp ? s4.models[c.id]?.[gr.key] : undefined;
      if (vp === undefined && vc === undefined) return;
      let pCls = "", cCls = "";
      if (vp !== undefined && vc !== undefined) {
        if (vp > vc) pCls = "best";
        else if (vc > vp) cCls = "best";
      }
      html += `<tr><td>${gr.label}</td><td class="${pCls}">${vp !== undefined ? gr.fmt(vp) : "—"}</td>`;
      if (hasCmp) html += `<td class="${cCls}">${vc !== undefined ? gr.fmt(vc) : "—"}</td>`;
      html += `</tr>`;
    });
  }

  html += `</tbody></table>`;
  el.innerHTML = html;
  renderPerConceptBreakdown();
}

function renderPerConceptBreakdown() {
  const el = document.getElementById("per-concept-table");
  if (!el) return;
  const p = state.primarySummary, c = state.compareSummary;
  if (!p?.metrics) {
    el.innerHTML = "";
    return;
  }
  const hasCmp = Boolean(c?.metrics);
  const concepts = p.concept_names || ["P", "Q", "R", "S", "T", "U", "PR", "QRS", "ST", "QT"];
  let html = `<table class="metrics-table"><thead><tr><th>Concept (500 Hz F1 / AUROC)</th><th>${p.id}</th>`;
  if (hasCmp) html += `<th>${c.id}</th>`;
  html += `</tr></thead><tbody>`;

  const fmtPair = (m, name) => {
    const f1 = m[`concept/${name}/f1`];
    const au = m[`concept/${name}/auroc`];
    if (f1 === undefined && au === undefined) return "—";
    return `${f1 !== undefined ? f1.toFixed(3) : "—"} / ${au !== undefined ? au.toFixed(3) : "—"}`;
  };

  concepts.forEach(name => {
    const fp = p.metrics[`concept/${name}/f1`];
    const fc = hasCmp ? c.metrics[`concept/${name}/f1`] : undefined;
    const sp = fmtPair(p.metrics, name);
    const sc = hasCmp ? fmtPair(c.metrics, name) : "—";
    if (sp === "—" && sc === "—") return;
    let pCls = "", cCls = "";
    if (fp !== undefined && fc !== undefined) {
      if (fp > fc) pCls = "best";
      else if (fc > fp) cCls = "best";
    }
    html += `<tr><td>${name}</td><td class="${pCls}">${sp}</td>${hasCmp ? `<td class="${cCls}">${sc}</td>` : ""}</tr>`;
  });
  html += `</tbody></table>`;
  el.innerHTML = html;
}

function renderInterventionCurves() {
  const container = document.getElementById("intervention-curve-container");
  container.innerHTML = "";
  const pInt = state.primarySummary?.interventions;
  const cInt = state.compareSummary?.interventions;
  if (!pInt && !cInt) {
    container.innerHTML = `<div class="hint">No intervention curve saved (run evaluate.py --interventions).</div>`;
    return;
  }

  const W = Math.max(380, container.clientWidth - 24);
  const H = 225;
  const pad = { top: 24, right: 18, bottom: 34, left: 48 };
  const iW = W - pad.left - pad.right, iH = H - pad.top - pad.bottom;

  const svg = d3.select(container).append("svg").attr("width", W).attr("height", H);
  const g   = svg.append("g").attr("transform", `translate(${pad.left},${pad.top})`);

  const allVals = [];
  [pInt, cInt].forEach(d => {
    if (!d) return;
    allVals.push(...d.uncertainty_acc, ...d.random_acc);
  });
  const yMin = Math.max(0, Math.min(...allVals) - 0.02);
  const ks = (pInt || cInt).num_intervened;

  const xScale = d3.scaleLinear().domain([0, ks[ks.length - 1]]).range([0, iW]);
  const yScale = d3.scaleLinear().domain([yMin, 1.005]).range([iH, 0]);

  g.append("g").attr("class", "grid").call(d3.axisLeft(yScale).tickSize(-iW).tickFormat("").ticks(5));
  g.append("g").attr("class", "axis").attr("transform", `translate(0,${iH})`)
    .call(d3.axisBottom(xScale).ticks(ks.length).tickFormat(d3.format("d")));
  g.append("g").attr("class", "axis").call(d3.axisLeft(yScale).ticks(5).tickFormat(d3.format(".1%")));

  g.append("text").attr("x", iW / 2).attr("y", iH + 30).attr("text-anchor", "middle")
    .attr("fill", "#767676").attr("font-size", 10).text("Number of Intervened Concepts (Test Set)");
  g.append("text").attr("x", 0).attr("y", -8).attr("fill", "#222222").attr("font-size", 10.5).attr("font-weight", 600)
    .text("Test-Time Concept Intervention Curve (Solid: Uncertainty-guided · Dashed: Random)");

  const drawCurve = (intData, col) => {
    const lineGen = d3.line().x((_, i) => xScale(intData.num_intervened[i])).y(v => yScale(v));
    g.append("path").datum(intData.uncertainty_acc).attr("fill", "none")
      .attr("stroke", col).attr("stroke-width", 2).attr("d", lineGen);
    g.append("path").datum(intData.random_acc).attr("fill", "none")
      .attr("stroke", col).attr("stroke-width", 1.35).attr("stroke-dasharray", "4,3").attr("opacity", 0.65).attr("d", lineGen);
    intData.uncertainty_acc.forEach((v, i) => {
      g.append("circle").attr("cx", xScale(intData.num_intervened[i])).attr("cy", yScale(v))
        .attr("r", 2.8).attr("fill", col);
    });
  };

  if (pInt) drawCurve(pInt, "#b83b2a");
  if (cInt) drawCurve(cInt, "#2b5b84");

  // Legend top-right
  let ly = 14;
  if (pInt) {
    g.append("circle").attr("cx", iW - 105).attr("cy", ly).attr("r", 3.2).attr("fill", "#b83b2a");
    g.append("text").attr("x", iW - 96).attr("y", ly + 3).attr("fill", "#444444").attr("font-size", 9.5)
      .text(state.primarySummary.id);
    ly += 15;
  }
  if (cInt) {
    g.append("circle").attr("cx", iW - 105).attr("cy", ly).attr("r", 3.2).attr("fill", "#2b5b84");
    g.append("text").attr("x", iW - 96).attr("y", ly + 3).attr("fill", "#444444").attr("font-size", 9.5)
      .text(state.compareSummary.id);
  }
}

// ── Shared Synced 1.0s Window Box, L/R Handles & Midpoint Cursor ────────────
const _syncedWindowOverlays = new Map();

function attachSyncedWindowOverlay(key, g, xScale, height) {
  const T = state.nSamples;
  const tFixed = state.tFixed;
  const winWpx = xScale(tFixed) - xScale(0);
  const handleH = Math.min(20, Math.max(12, Math.round(height * 0.28)));

  const rect = g.append("rect")
    .attr("class", "window-rect").attr("y", 0).attr("height", height).attr("width", winWpx);
  const handleL = g.append("rect")
    .attr("class", "window-handle").attr("width", 7).attr("height", handleH).attr("y", (height - handleH) / 2);
  const handleR = g.append("rect")
    .attr("class", "window-handle").attr("width", 7).attr("height", handleH).attr("y", (height - handleH) / 2);
  const cursor = g.append("line")
    .attr("class", "sync-cursor").attr("y1", 0).attr("y2", height);

  const startFromX = px => Math.max(0, Math.min(T - tFixed, Math.round(xScale.invert(px))));
  const onDragMove = newStart => {
    updateWindowPosition(newStart);
    renderWindowPlot();
    if (state.sweepData && Object.keys(state.interventions).length === 0) {
      updatePipelineFromSweep();
    }
  };

  rect.call(d3.drag()
    .on("drag", e => onDragMove(startFromX(e.x - winWpx / 2)))
    .on("end",  () => updatePipeline()));
  handleL.call(d3.drag()
    .on("drag", e => onDragMove(startFromX(e.x)))
    .on("end",  () => updatePipeline()));
  handleR.call(d3.drag()
    .on("drag", e => onDragMove(startFromX(e.x - winWpx)))
    .on("end",  () => updatePipeline()));

  _syncedWindowOverlays.set(key, { rect, handleL, handleR, cursor, xScale, winWpx });
  updateSyncCursors(state.windowStart);
}

function updateSyncCursors(windowStart) {
  const centre = windowStart + state.tFixed / 2;
  _syncedWindowOverlays.forEach(ov => {
    const px = ov.xScale(windowStart);
    const cx = ov.xScale(centre);
    ov.rect.attr("x", px);
    ov.handleL.attr("x", px - 3.5);
    ov.handleR.attr("x", px + ov.winWpx - 3.5);
    ov.cursor.attr("x1", cx).attr("x2", cx);
  });
}

function updateWindowPosition(start) {
  state.windowStart = start;
  const sr = state.samplingRate || 500;
  document.getElementById("window-pos-label").textContent =
    `window: ${(start / sr).toFixed(2)}s – ${((start + state.tFixed) / sr).toFixed(2)}s`;
  updateSyncCursors(start);
}

// ── Event Listeners ────────────────────────────────────────────────────────
document.getElementById("load-btn").addEventListener("click", loadCurrentRecord);
document.getElementById("sweep-btn").addEventListener("click", runSweep);

document.getElementById("lead-select").addEventListener("change", () => {
  state.leadIdx = parseInt(document.getElementById("lead-select").value, 10) || 0;
  if (state.signal) { renderSignal(); renderSegmentedSignal(); renderWindowPlot(); }
});

document.getElementById("dataset-select")?.addEventListener("change", async e => {
  state.dataset = e.target.value || "ptbxl";
  applyDatasetControls(true);
  renderMetricsTable();
  await loadRecordList();
  const recSel = document.getElementById("record-select");
  if (recSel && recSel.options.length > 0 && recSel.value) {
    await loadCurrentRecord();
  }
});

document.getElementById("fold-select").addEventListener("change", loadRecordList);
document.getElementById("dx-select").addEventListener("change", loadRecordList);
document.getElementById("record-search").addEventListener("input", () => {
  clearTimeout(window._searchDeb);
  window._searchDeb = setTimeout(loadRecordList, 200);
});
document.getElementById("epistemic-chk").addEventListener("change", () => {
  syncQuickOptions();
  updatePipeline();
  runDenseInfer();
});

document.getElementById("ecg-seg-overlay").addEventListener("change", e => {
  state.ecgSegOverlay = e.target.value;
  if (state.signal) renderSegmentedSignal();
});

document.getElementById("seg-compare-mode").addEventListener("change", () => {
  if (state.denseData || state.softSeg) renderUnifiedSegmentationView();
});

document.getElementById("seg-show-unc").addEventListener("change", () => {
  if (state.denseData || state.softSeg) renderUnifiedSegmentationView();
});

document.querySelectorAll("#dense-bench-toggle .tab-btn").forEach(btn => {
  btn.addEventListener("click", () => {
    document.querySelectorAll("#dense-bench-toggle .tab-btn").forEach(b => b.classList.remove("active"));
    btn.classList.add("active");
    state.denseBenchDataset = btn.dataset.benchDs || "ptbxl";
    renderDenseMetricsTable();
  });
});

document.getElementById("dense-int-scope").addEventListener("change", () => {
  if (state.denseIntMode) runDenseInfer();
});

document.getElementById("dense-primary-select")?.addEventListener("change", async e => {
  state.densePrimary = e.target.value;
  updateDenseModelNote();
  renderDenseMetricsTable();
  await runDenseInfer();
});

document.getElementById("dense-compare-select")?.addEventListener("change", async e => {
  state.denseCompare = e.target.value;
  updateDenseModelNote();
  renderDenseMetricsTable();
  await runDenseInfer();
});

document.getElementById("dense-blackbox-select")?.addEventListener("change", async e => {
  state.denseBlackbox = e.target.value;
  updateDenseModelNote();
  renderDenseMetricsTable();
  await runDenseInfer();
});

document.getElementById("dense-gt-int-btn").addEventListener("click", async () => {
  state.denseIntMode = "gt";
  state.denseIntConcept = null;
  updateDenseIntButtons();
  await runDenseInfer();
});

document.getElementById("dense-reset-int-btn").addEventListener("click", async () => {
  state.denseIntMode = null;
  state.denseIntConcept = null;
  updateDenseIntButtons();
  await runDenseInfer();
});

// Responsive chart redraw on container resize
let _resizeTimer = null;
new ResizeObserver(() => {
  clearTimeout(_resizeTimer);
  _resizeTimer = setTimeout(() => {
    if (state.signal) { renderSignal(); renderSegmentedSignal(); }
    if (state.sweepData) renderTimelineView();
    if (state.denseData || state.softSeg) renderUnifiedSegmentationView();
    if (state.primarySummary || state.compareSummary) renderInterventionCurves();
  }, 150);
}).observe(document.getElementById("signal-section"));

document.getElementById("model-select").addEventListener("change", async e => {
  state.primaryModel = e.target.value;
  syncQuickOptions();
  await refreshBenchmarks();
  if (state.ecgId !== null) await runSweep();
});

document.getElementById("compare-select").addEventListener("change", async e => {
  state.compareModel = e.target.value;
  syncQuickOptions();
  await refreshBenchmarks();
  if (state.ecgId !== null) await runSweep();
});

document.getElementById("reset-int-btn").addEventListener("click", async () => {
  state.interventions = {};
  updateResetIntButton();
  await updatePipeline();
});

document.getElementById("play-btn").addEventListener("click", () => {
  if (!state.sweepData) return;
  if (state.playTimer) {
    stopPlayback();
    return;
  }
  document.getElementById("play-btn").textContent = "⏸ Pause";
  let ni = findNearestSweepIndex(state.windowStart);
  state.playTimer = setInterval(() => {
    ni = (ni + 1) % state.sweepData.starts.length;
    updateWindowPosition(state.sweepData.starts[ni]);
    renderWindowPlot();
    if (Object.keys(state.interventions).length === 0) updatePipelineFromSweep();
  }, 85);
});

document.querySelectorAll("#timeline-tabs .tab-btn").forEach(btn => {
  btn.addEventListener("click", () => {
    document.querySelectorAll("#timeline-tabs .tab-btn").forEach(b => b.classList.remove("active"));
    btn.classList.add("active");
    state.timelineView = btn.dataset.view;
    renderTimelineView();
  });
});

init();
