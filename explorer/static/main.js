/* ── ECG Heartbeat CBM & CEM Explorer — main.js ───────────────────────────
 * Multi-model D3 v7 frontend:
 *  - Dynamic CBM / CEM model selection & side-by-side comparison
 *  - Synchronized ECG waveform + sliding-window concept activation lineplots & heatmap
 *  - Live click-to-intervene concept pills updating Task prediction in real time
 *  - Test set benchmark table + Concept Intervention Curves (Uncertainty vs Random)
 * ────────────────────────────────────────────────────────────────────────── */

const API = "";
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

let signalG, windowRect, windowHandleL, windowHandleR;
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
    const h = await (await fetch(`${API}/health`)).json();
    document.getElementById("device-badge").textContent =
      `${h.device.toUpperCase()} · ${h.n_models} models`;
  } catch (_) {}

  await loadModels();
  await loadDenseModels();
  await loadRecordList();
  await refreshBenchmarks();

  // Auto-load first test record (Record #9 is in fold 10!) for an immediate visual experience
  const recSel = document.getElementById("record-select");
  if (recSel.options.length > 0 && recSel.value) {
    await loadCurrentRecord();
  }
}

async function loadModels() {
  const res = await fetch(`${API}/models`);
  const data = await res.json();
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
    primSel.appendChild(o1);
    cmpSel.appendChild(o2);
  });
}

async function loadRecordList() {
  const foldVal = document.getElementById("fold-select").value;
  const dxVal   = document.getElementById("dx-select").value;
  const search  = document.getElementById("record-search").value.trim();
  const params  = new URLSearchParams({ limit: "300" });
  if (foldVal !== "all") params.set("fold", foldVal);
  if (dxVal && dxVal !== "ALL") params.set("dx", dxVal);
  if (search) params.set("search", search);

  const res  = await fetch(`${API}/records?${params.toString()}`);
  const data = await res.json();
  const sel  = document.getElementById("record-select");
  sel.innerHTML = "";
  (data.records || []).forEach(r => {
    const tags = r.tags && r.tags.length ? ` [${r.tags.join(",")}]` : "";
    sel.appendChild(new Option(`Record #${r.ecg_id}${tags}`, String(r.ecg_id)));
  });
  if (!search && dxVal === "ALL" && [...sel.options].some(o => o.value === "9")) {
    sel.value = "9";
  }
}

// ── Record loading & automatic sweep ───────────────────────────────────────
async function loadCurrentRecord() {
  stopPlayback();
  const ecgId = parseInt(document.getElementById("record-select").value, 10);
  if (isNaN(ecgId)) return;

  state.ecgId         = ecgId;
  state.leadIdx       = parseInt(document.getElementById("lead-select").value, 10) || 0;
  state.sweepData       = null;
  state.denseData       = null;
  state.denseIntMode    = null;
  state.denseIntConcept = null;
  state.interventions   = {};
  updateResetIntButton();
  updateDenseIntButtons();

  document.getElementById("signal-hint").style.display = "none";
  document.getElementById("pipeline-section").style.display = "";

  const res = await fetch(`${API}/record/${ecgId}?model_id=${encodeURIComponent(state.primaryModel || "")}`);
  if (!res.ok) {
    alert(`Failed to load record: ${await res.text()}`);
    return;
  }
  const data = await res.json();
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
  const sex = data.sex === 0 ? "M" : "F";
  const dx  = data.dx && data.dx.length ? `· ${data.dx.join(", ")}` : "";
  document.getElementById("record-meta").textContent =
    `— Record #${ecgId}  ${age} ${sex} ${dx}  (${rps.length} R-peaks)`;

  renderSignal();
  await Promise.all([runSweep(), runDenseInfer()]);
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

  // Optional per-timestamp segmentation shading directly behind the ECG waveform
  const segPalette = { P: "#2b5b84", PQ: "#2a7f8e", QRS: "#b83b2a", ST: "#9e4268", T: "#2d7a56" };
  const hasSegProbs = obj => Boolean(obj && obj.has_seg !== false && obj.concept_probs);
  let segSrc = null, segSr = 100;
  if (state.ecgSegOverlay === "dense_cem" && state.denseData) {
    const src = (hasSegProbs(state.denseData) && state.denseData.model_type === "dense_cem")
      ? state.denseData
      : (hasSegProbs(state.denseData.compare) && state.denseData.compare.model_type === "dense_cem" ? state.denseData.compare : null);
    segSrc = src?.concept_probs || null;
    segSr = state.denseData.sampling_rate || 100;
  } else if (state.ecgSegOverlay === "dense_cbm" && state.denseData) {
    const src = (hasSegProbs(state.denseData) && state.denseData.model_type === "dense_cbm")
      ? state.denseData
      : (hasSegProbs(state.denseData.compare) && state.denseData.compare.model_type === "dense_cbm" ? state.denseData.compare : null);
    segSrc = src?.concept_probs || null;
    segSr = state.denseData.sampling_rate || 100;
  } else if (state.ecgSegOverlay === "zenodo") {
    segSrc = state.denseData?.gt_probs || state.softSeg;
    segSr = state.softSeg?.sampling_rate || 100;
  }
  if (segSrc) {
    const keys = ["P", "PQ", "QRS", "ST", "T"].filter(k => Array.isArray(segSrc[k]) && segSrc[k].length);
    const nSeg = keys.length ? segSrc[keys[0]].length : 0;
    const scaleSeg = sr / segSr;
    // Merge contiguous spans of the winning wave class (>0.35) into crisp background bands
    let curKey = null, runStart = 0;
    const flushRun = endIdx => {
      if (!curKey || endIdx <= runStart) return;
      const x0 = _xScaleSig(runStart * scaleSeg);
      const x1 = _xScaleSig(endIdx * scaleSeg);
      signalG.append("rect")
        .attr("x", x0).attr("y", 0)
        .attr("width", Math.max(1, x1 - x0)).attr("height", H)
        .attr("fill", segPalette[curKey]).attr("opacity", 0.14);
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

  // Draggable 1.0s Window
  const startFromX = px => Math.max(0, Math.min(T - tFixed, Math.round(_xScaleSig.invert(px))));

  windowRect = signalG.append("rect")
    .attr("class", "window-rect").attr("y", 0).attr("height", H).attr("width", _winWpx);
  windowHandleL = signalG.append("rect")
    .attr("class", "window-handle").attr("width", 7).attr("height", 20).attr("y", H / 2 - 10);
  windowHandleR = signalG.append("rect")
    .attr("class", "window-handle").attr("width", 7).attr("height", 20).attr("y", H / 2 - 10);

  updateWindowPosition(state.windowStart);

  const onDragMove = newStart => {
    updateWindowPosition(newStart);
    renderWindowPlot();
    if (state.sweepData && Object.keys(state.interventions).length === 0) {
      updatePipelineFromSweep();
    }
  };

  windowRect.call(d3.drag()
    .on("drag", e => onDragMove(startFromX(e.x - _winWpx / 2)))
    .on("end",  () => updatePipeline()));
  windowHandleL.call(d3.drag()
    .on("drag", e => onDragMove(startFromX(e.x)))
    .on("end",  () => updatePipeline()));
  windowHandleR.call(d3.drag()
    .on("drag", e => onDragMove(startFromX(e.x - _winWpx)))
    .on("end",  () => updatePipeline()));

  svg.on("click", async e => {
    const [mx] = d3.pointer(e, signalG.node());
    if (mx < 0 || mx > W) return;
    updateWindowPosition(startFromX(mx - _winWpx / 2));
    await updatePipeline();
  });
}

// ── Sliding-Window Sweep ───────────────────────────────────────────────────
async function runSweep() {
  if (state.ecgId === null) return;
  const stride = parseInt(document.getElementById("stride-input").value, 10) || 20;
  const status = document.getElementById("sweep-status");
  status.textContent = "Computing batched sweep…";
  status.className   = "status-label running";

  try {
    const res = await fetch(`${API}/sweep`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        ecg_id:           state.ecgId,
        stride,
        model_id:         state.primaryModel,
        compare_model_id: state.compareModel || null,
      }),
    });
    if (!res.ok) throw new Error(await res.text());
    state.sweepData = await res.json();
  } catch (e) {
    status.textContent = `Sweep error: ${e.message}`;
    status.className   = "status-label";
    return;
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
  if (!state.sweepData && !state.denseData) return;
  const lp  = document.getElementById("lineplot-container");
  const hm  = document.getElementById("heatmap-container");
  const scb = document.getElementById("seg-controls-bar");
  const unc = document.getElementById("unc-strip-container");

  if (state.timelineView === "heatmap") {
    scb.style.display = "none";
    unc.style.display = "none";
    lp.style.display  = "none";
    hm.style.display  = "";
    if (state.sweepData) renderHeatmap(state.sweepData);
  } else if (state.timelineView === "dense" || state.timelineView === "softseg") {
    hm.style.display  = "none";
    scb.style.display = "";
    lp.style.display  = "";
    const modeSel = document.getElementById("seg-compare-mode");
    if (state.timelineView === "dense" && modeSel.value === "gt_only") {
      modeSel.value = "cem_vs_cbm";
    }
    renderUnifiedSegmentationView();
  } else {
    scb.style.display = "none";
    unc.style.display = "none";
    hm.style.display  = "none";
    lp.style.display  = "";
    if (state.sweepData) renderActivationLineplot(state.sweepData);
  }
}

async function loadDenseModels() {
  try {
    const res = await fetch(`${API}/dense-models`);
    state.denseModels = res.ok ? await res.json() : { models: [] };
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
        primSel.appendChild(new Option(label, m.id, false, m.id === state.densePrimary));
        cmpSel.appendChild(new Option(label, m.id, false, m.id === state.denseCompare));
        bbSel.appendChild(new Option(label, m.id, false, m.id === state.denseBlackbox));
      });
    }
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
    model_id:          modelId,
    compare_model_id:  state.denseCompare  || null,
    blackbox_model_id: state.denseBlackbox || null,
    include_epistemic: wantEp,
    intervene_mode:    state.denseIntMode,
    intervene_concepts: state.denseIntConcept ? [state.denseIntConcept] : null,
    intervene_span:    span,
  };
  try {
    const res = await fetch(`${API}/dense-infer`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
    if (!res.ok) throw new Error(await res.text());
    state.denseData = await res.json();
    const st = document.getElementById("dense-seg-status");
    const intTag = state.denseIntMode
      ? ` · Intervention: ${state.denseIntMode.toUpperCase()}${state.denseIntConcept ? " " + state.denseIntConcept : ""} (${scope})`
      : "";
    const ids = [state.denseData.model_id];
    if (state.denseData.compare)  ids.push(state.denseData.compare.model_id);
    if (state.denseData.blackbox) ids.push(state.denseData.blackbox.model_id);
    st.textContent = `— ${ids.join(" vs. ")}${intTag}`;

    renderDensePathologyBars();
    renderDenseWavePills();
    if (state.ecgSegOverlay !== "none") renderSignal();
    if (state.timelineView === "dense" || state.timelineView === "softseg") {
      renderUnifiedSegmentationView();
    }
  } catch (e) {
    console.warn("Dense infer failed:", e);
  }
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

function renderDenseMetricsTable() {
  const el = document.getElementById("dense-metrics-table");
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
}

function renderUnifiedSegmentationView() {
  const container = document.getElementById("lineplot-container");
  const uncBox    = document.getElementById("unc-strip-container");
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

  const cursor = g.append("line").attr("class", "sync-cursor").attr("y1", 0).attr("y2", H);
  window._lineCursorScale = xScale;
  window._lineCursor      = cursor;
  updateSyncCursors(state.windowStart);

  // Interactive hover readout + click/drag to move ECG window
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
  svg.on("click", async e => {
    const [mx] = d3.pointer(e, g.node());
    if (mx < 0 || mx > W) return;
    const start = Math.max(0, Math.min(T - state.tFixed, Math.round(xScale.invert(mx) - halfWin)));
    updateWindowPosition(start);
    await updatePipeline();
  });

  const backDesc = backProbs ? (backIsDashed ? ` · Dashed: ${backLabel}` : ` · Shaded: ${backLabel}`) : "";
  document.getElementById("timeline-legend").innerHTML =
    groups.map(gr => `<span class="leg-item"><i class="swatch" style="background:${gr.col}"></i>${gr.key}</span>`).join("") +
    `<span class="meta-label">Solid: ${solidLabel}${backDesc}</span>`;

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
    const uCur = ug.append("line").attr("class", "sync-cursor").attr("y1", 0).attr("y2", UH);
    window._uncCursor = uCur;
    window._uncCursorScale = xScale;
    updateSyncCursors(state.windowStart);
  } else {
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

  // Synced vertical cursor
  const cursor = g.append("line").attr("class", "sync-cursor").attr("y1", 0).attr("y2", H);
  window._lineCursorScale = xScale;
  window._lineCursor      = cursor;
  updateSyncCursors(state.windowStart);

  // Click / drag on lineplot to move ECG window
  const moveFromX = mx => {
    const centreSample = xScale.invert(mx);
    const start = Math.max(0, Math.min(T - data.t_fixed, Math.round(centreSample - halfWin)));
    updateWindowPosition(start);
    renderWindowPlot();
    if (Object.keys(state.interventions).length === 0) updatePipelineFromSweep();
  };
  svg.call(d3.drag()
    .on("start drag", e => moveFromX(e.x - ml))
    .on("end", () => updatePipeline()));

  // Update legend
  const styleNote = data.compare
    ? `<span class="meta-label">Solid: ${data.model_id} · Dashed: ${data.compare.model_id} · Green shade: GT Complete</span>`
    : `<span class="meta-label">${data.model_id} · Green shade: GT Complete</span>`;
  document.getElementById("timeline-legend").innerHTML =
    series.map(s => `<span class="leg-item"><i class="swatch" style="background:${s.col}"></i>${s.label}</span>`).join("") +
    styleNote;
}

// ── Sweep Heatmap ──────────────────────────────────────────────────────────
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
  const sr      = data.sampling_rate || state.samplingRate;
  const color   = d3.scaleSequential(d3.interpolateYlOrRd).domain([0, 1]);

  const svg = d3.select(container).append("svg")
    .attr("width", W + ml + mr)
    .attr("height", MAP_TOP + CELL_H * K + MAP_BOT);
  const g = svg.append("g").attr("transform", `translate(${ml},${MAP_TOP})`);

  names.forEach((name, ki) => {
    starts.forEach((start, ni) => {
      g.append("rect")
        .attr("x", ni * CELL_W).attr("y", ki * CELL_H)
        .attr("width", Math.ceil(CELL_W) + 0.5).attr("height", CELL_H - 1)
        .attr("fill", color(cProbs[ni][ki]))
        .style("cursor", "pointer")
        .on("click", async () => {
          updateWindowPosition(start);
          await updatePipeline();
        });
    });
    svg.append("text")
      .attr("x", ml - 6).attr("y", MAP_TOP + ki * CELL_H + CELL_H / 2 + 4)
      .attr("text-anchor", "end").attr("fill", "#767676").attr("font-size", 10)
      .text(name);
  });

  const xScale = d3.scaleLinear().domain([0, state.nSamples]).range([0, W]);
  g.append("g").attr("class", "axis")
    .attr("transform", `translate(0,${CELL_H * K})`)
    .call(d3.axisBottom(xScale).ticks(10).tickFormat(d => `${(d / sr).toFixed(1)}s`));

  const cursor = g.append("line").attr("class", "sync-cursor").attr("y1", 0).attr("y2", CELL_H * K);
  window._heatmapCursorScale = xScale;
  window._heatmapCursor      = cursor;
  updateSyncCursors(state.windowStart);
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
    model_id:      d.model_id,
    model_type:    d.model_type,
    task_prob:     d.task_probs[ni],
    task_std:      d.task_stds[ni],
    concept_probs: d.concept_probs[ni],
    concept_stds:  d.concept_stds[ni],
    gt_task:       gtT,
    gt_concepts:   gtC,
    compare:       cmp,
  });
}

async function updatePipeline() {
  if (state.ecgId === null) return;
  renderWindowPlot();

  const hasInt = Object.keys(state.interventions).length > 0;
  const wantEpistemic = document.getElementById("epistemic-chk").checked;
  if (state.sweepData && !hasInt && !wantEpistemic) {
    updatePipelineFromSweep();
    return;
  }

  try {
    const res = await fetch(`${API}/infer`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        ecg_id:            state.ecgId,
        start:             state.windowStart,
        model_id:          state.primaryModel,
        compare_model_id:  state.compareModel || null,
        interventions:     hasInt ? state.interventions : null,
        include_epistemic: wantEpistemic,
      }),
    });
    if (!res.ok) throw new Error(await res.text());
    const result = await res.json();
    if (result.concept_names) state.conceptNames = result.concept_names;
    renderPipelineValues(result);
  } catch (e) {
    console.error("Infer error:", e);
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

  const cmpProbs = cmpHasConcepts ? res.compare.concept_probs : null;
  renderHistogram(res.concept_probs, res.concept_stds, res.gt_concepts, cmpProbs, primIsBB);
  renderInterventionPills(!primIsBB);
  renderTaskPrediction(res.model_id, res.task_prob, res.task_std, res.compare);
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

// ── Task Prediction Bar Chart (Primary vs Compare side-by-side) ────────────
function renderTaskPrediction(modelId, taskProb, taskStd, cmpObj) {
  const container = document.getElementById("task-pred-container");
  container.innerHTML = "";

  const W = 196, H = 195;
  const pad = { top: 22, right: 12, bottom: 42, left: 36 };
  const iW = W - pad.left - pad.right, iH = H - pad.top - pad.bottom;

  const svg = d3.select(container).append("svg").attr("width", W).attr("height", H);
  const g   = svg.append("g").attr("transform", `translate(${pad.left},${pad.top})`);

  const items = cmpObj
    ? [
        { label: modelId,        p: taskProb,        s: taskStd || 0 },
        { label: cmpObj.model_id, p: cmpObj.task_prob, s: cmpObj.task_std || 0 },
      ]
    : [
        { label: "Complete",   p: taskProb,     s: taskStd || 0 },
        { label: "Incomplete", p: 1 - taskProb, s: 0 },
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

    g.append("text").attr("x", x + bW / 2).attr("y", Math.max(yT - 5, 9))
      .attr("text-anchor", "middle").attr("fill", "#222222").attr("font-size", 10).attr("font-weight", 600)
      .text(`${(it.p * 100).toFixed(1)}%`);

    if (it.s > 0) {
      g.append("text").attr("x", x + bW / 2).attr("y", iH + 28)
        .attr("text-anchor", "middle").attr("fill", "#767676").attr("font-size", 8)
        .text(`±${(it.s * 100).toFixed(1)}%`);
    }
  });
}

// ── Benchmark Summary & Concept Intervention Curves ────────────────────────
async function refreshBenchmarks() {
  if (!state.primaryModel) return;
  try {
    state.primarySummary = await (await fetch(`${API}/model/${encodeURIComponent(state.primaryModel)}/summary`)).json();
  } catch (_) { state.primarySummary = null; }

  if (state.compareModel) {
    try {
      state.compareSummary = await (await fetch(`${API}/model/${encodeURIComponent(state.compareModel)}/summary`)).json();
    } catch (_) { state.compareSummary = null; }
  } else {
    state.compareSummary = null;
  }

  renderMetricsTable();
  renderInterventionCurves();
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

// ── Shared Cursor Sync ─────────────────────────────────────────────────────
function updateSyncCursors(windowStart) {
  const centre = windowStart + state.tFixed / 2;
  if (window._lineCursor && window._lineCursorScale) {
    const px = window._lineCursorScale(centre);
    window._lineCursor.attr("x1", px).attr("x2", px);
  }
  if (window._heatmapCursor && window._heatmapCursorScale) {
    const px = window._heatmapCursorScale(centre);
    window._heatmapCursor.attr("x1", px).attr("x2", px);
  }
  if (window._uncCursor && window._uncCursorScale) {
    const px = window._uncCursorScale(centre);
    window._uncCursor.attr("x1", px).attr("x2", px);
  }
}

function updateWindowPosition(start) {
  state.windowStart = start;
  if (signalG && _xScaleSig) {
    const px = _xScaleSig(start);
    windowRect.attr("x", px);
    windowHandleL.attr("x", px - 3.5);
    windowHandleR.attr("x", px + _winWpx - 3.5);
  }
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
  if (state.signal) { renderSignal(); renderWindowPlot(); }
});

document.getElementById("fold-select").addEventListener("change", loadRecordList);
document.getElementById("dx-select").addEventListener("change", loadRecordList);
document.getElementById("record-search").addEventListener("input", () => {
  clearTimeout(window._searchDeb);
  window._searchDeb = setTimeout(loadRecordList, 200);
});
document.getElementById("epistemic-chk").addEventListener("change", () => {
  updatePipeline();
  runDenseInfer();
});

document.getElementById("ecg-seg-overlay").addEventListener("change", e => {
  state.ecgSegOverlay = e.target.value;
  if (state.signal) renderSignal();
});

document.getElementById("seg-compare-mode").addEventListener("change", () => {
  if (state.timelineView === "dense" || state.timelineView === "softseg") {
    renderUnifiedSegmentationView();
  }
});

document.getElementById("seg-show-unc").addEventListener("change", () => {
  if (state.timelineView === "dense" || state.timelineView === "softseg") {
    renderUnifiedSegmentationView();
  }
});

document.getElementById("dense-int-scope").addEventListener("change", () => {
  if (state.denseIntMode) runDenseInfer();
});

document.getElementById("dense-primary-select")?.addEventListener("change", async e => {
  state.densePrimary = e.target.value;
  renderDenseMetricsTable();
  await runDenseInfer();
});

document.getElementById("dense-compare-select")?.addEventListener("change", async e => {
  state.denseCompare = e.target.value;
  renderDenseMetricsTable();
  await runDenseInfer();
});

document.getElementById("dense-blackbox-select")?.addEventListener("change", async e => {
  state.denseBlackbox = e.target.value;
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
    if (state.signal) renderSignal();
    if (state.sweepData) renderTimelineView();
    if (state.primarySummary || state.compareSummary) renderInterventionCurves();
  }, 150);
}).observe(document.getElementById("signal-section"));

document.getElementById("model-select").addEventListener("change", async e => {
  state.primaryModel = e.target.value;
  await refreshBenchmarks();
  if (state.ecgId !== null) await runSweep();
});

document.getElementById("compare-select").addEventListener("change", async e => {
  state.compareModel = e.target.value;
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
