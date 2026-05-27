/* ── ECG Heartbeat CBM Explorer — main.js ─────────────────────────────────
 * D3 v7 frontend.
 *
 * Pipeline layout (below full-record signal chart):
 *   [Input Window plot] ──Encoder──► [Concept Histogram] ──Predictor──► [Task Bar]
 *
 * Sweep heatmap (optional) shows concept activations as a colour matrix.
 * ────────────────────────────────────────────────────────────────────────── */

const API = "";           // same origin
const CONCEPT_COLORS = [
  "#38bdf8", "#a78bfa", "#4ade80", "#fb923c", "#f472b6",
  "#facc15", "#34d399", "#60a5fa", "#f87171", "#c084fc",
];

// ── App state ─────────────────────────────────────────────────────────────
let state = {
  ecgId:       null,
  signal:      null,   // (12, T_ds) nested array
  nSamples:    0,      // full-res sample count
  nSamplesDs:  0,      // downsampled count
  downsample:  1,
  tFixed:      500,
  leadIdx:     1,
  windowStart: 0,      // full-res sample index
  sweepData:   null,   // result from /sweep
};

// ── Module-level D3 helpers (populated after first render) ────────────────
let signalSvg, signalG, windowRect, windowHandleL, windowHandleR;
// xScale shared between renderSignal and updateWindowPosition
let _xScaleSig = null;
let _winWpx    = 0;

const SIG_MARGIN = { top: 16, right: 24, bottom: 36, left: 48 };
const SIG_HEIGHT = 180;

// Window mini-plot
const WIN_W   = 210;
const WIN_H   = 168;
const WIN_PAD = { top: 10, right: 10, bottom: 30, left: 36 };

// Concept histogram (sized for pipeline)
const HIST_W   = 500;
const HIST_H   = 195;
const HIST_PAD = { top: 22, right: 14, bottom: 54, left: 38 };

// Task prediction bar chart
const TASK_W   = 170;
const TASK_H   = 178;
const TASK_PAD = { top: 24, right: 14, bottom: 54, left: 40 };

// ── Utility ───────────────────────────────────────────────────────────────
function sigWidth() {
  const el = document.getElementById("signal-container");
  return (el ? el.clientWidth : 900) - SIG_MARGIN.left - SIG_MARGIN.right;
}

function taskColour(p) {
  if (p >= 0.7) return "#4ade80";
  if (p >= 0.4) return "#facc15";
  return "#f87171";
}

// ── Boot: load record list ─────────────────────────────────────────────────
async function init() {
  try {
    const res  = await fetch(`${API}/records?limit=500`);
    const data = await res.json();
    const sel  = document.getElementById("record-select");
    sel.innerHTML = "";
    data.record_ids.forEach(id => {
      const opt   = document.createElement("option");
      opt.value   = id;
      opt.textContent = `Record ${id}`;
      sel.appendChild(opt);
    });
  } catch (e) {
    console.error("Could not fetch records:", e);
  }
}

// ── Load & render a record ─────────────────────────────────────────────────
document.getElementById("load-btn").addEventListener("click", async () => {
  const ecgId   = parseInt(document.getElementById("record-select").value, 10);
  const leadIdx = parseInt(document.getElementById("lead-select").value, 10);
  if (isNaN(ecgId)) return;

  document.getElementById("signal-hint").style.display = "none";
  document.getElementById("sweep-section").style.display = "none";
  document.getElementById("pipeline-section").style.display = "";
  state.sweepData = null;
  state.leadIdx   = leadIdx;
  state.ecgId     = ecgId;

  let data;
  try {
    const res = await fetch(`${API}/record/${ecgId}`);
    if (!res.ok) throw new Error(await res.text());
    data = await res.json();
  } catch (e) {
    alert(`Failed to load record: ${e.message}`);
    return;
  }

  state.signal     = data.signal;
  state.nSamples   = data.n_samples;
  state.nSamplesDs = data.n_samples_ds;
  state.downsample = data.downsample;
  state.tFixed     = data.t_fixed;
  state.windowStart = Math.max(0, Math.floor(state.nSamples / 2) - state.tFixed / 2);

  const age = data.age > 0 ? `age ${Math.round(data.age)}` : "";
  const sex = data.sex === 0 ? "M" : "F";
  document.getElementById("record-meta").textContent =
    `— ID ${ecgId}  ${age}  ${sex}`;

  renderSignal();
  await updatePipeline();
});

// ── Lead selection ─────────────────────────────────────────────────────────
document.getElementById("lead-select").addEventListener("change", () => {
  state.leadIdx = parseInt(document.getElementById("lead-select").value, 10);
  if (state.signal) renderSignal();
});

// ── Signal chart ───────────────────────────────────────────────────────────
function renderSignal() {
  const container = document.getElementById("signal-container");
  container.innerHTML = "";

  const W = sigWidth();
  const H = SIG_HEIGHT;
  const { left: ml, right: mr, top: mt, bottom: mb } = SIG_MARGIN;

  signalSvg = d3.select(container)
    .append("svg")
    .attr("width",  W + ml + mr)
    .attr("height", H + mt + mb);

  signalG = signalSvg.append("g")
    .attr("transform", `translate(${ml},${mt})`);

  const lead   = state.leadIdx;
  const raw    = state.signal[lead];          // (T_ds,)
  const ds     = state.downsample;
  const T      = state.nSamples;
  const T_ds   = state.nSamplesDs;
  const tFixed = state.tFixed;

  // X: full-res samples → pixels
  _xScaleSig = d3.scaleLinear().domain([0, T]).range([0, W]);
  _winWpx    = _xScaleSig(tFixed) - _xScaleSig(0);

  // Y: amplitude
  const yMin = d3.min(raw), yMax = d3.max(raw);
  const yPad = (yMax - yMin) * 0.1 || 0.1;
  const yScale = d3.scaleLinear().domain([yMin - yPad, yMax + yPad]).range([H, 0]);

  // Grid
  signalG.append("g").attr("class", "grid")
    .call(d3.axisLeft(yScale).tickSize(-W).tickFormat("").ticks(5));

  // Axes
  signalG.append("g").attr("class", "axis")
    .attr("transform", `translate(0,${H})`)
    .call(d3.axisBottom(_xScaleSig).ticks(8).tickFormat(d => `${(d / 500).toFixed(1)}s`));
  signalG.append("g").attr("class", "axis")
    .call(d3.axisLeft(yScale).ticks(5));

  // ECG line (downsampled coords → full-res x via ds)
  const lineGen = d3.line()
    .x((_, i) => _xScaleSig(i * ds))
    .y(d => yScale(d))
    .curve(d3.curveLinear);

  signalG.append("path")
    .datum(raw)
    .attr("class", "ecg-line")
    .attr("d", lineGen);

  // ── Draggable window ─────────────────────────────────────────────────
  function startFromX(px) {
    const s = Math.round(_xScaleSig.invert(px));
    return Math.max(0, Math.min(T - tFixed, s));
  }

  windowRect = signalG.append("rect")
    .attr("class", "window-rect")
    .attr("y", 0)
    .attr("height", H)
    .attr("width", _winWpx);

  const handleH = 18, handleW = 8;
  windowHandleL = signalG.append("rect").attr("class", "window-handle")
    .attr("width", handleW).attr("height", handleH).attr("y", H / 2 - handleH / 2);
  windowHandleR = signalG.append("rect").attr("class", "window-handle")
    .attr("width", handleW).attr("height", handleH).attr("y", H / 2 - handleH / 2);

  updateWindowPosition(state.windowStart);

  // Drag on the window rect — update mini-plot live; fetch inference on end
  const dragWindow = d3.drag()
    .on("drag", function (event) {
      updateWindowPosition(startFromX(event.x - _winWpx / 2));
      renderWindowPlot();
    })
    .on("end", async function () { await updatePipeline(); });

  // Drag on left handle
  const dragHandleL = d3.drag()
    .on("drag", function (event) {
      updateWindowPosition(startFromX(event.x));
      renderWindowPlot();
    })
    .on("end", async function () { await updatePipeline(); });

  // Drag on right handle
  const dragHandleR = d3.drag()
    .on("drag", function (event) {
      updateWindowPosition(startFromX(event.x - _winWpx));
      renderWindowPlot();
    })
    .on("end", async function () { await updatePipeline(); });

  windowRect.call(dragWindow);
  windowHandleL.call(dragHandleL);
  windowHandleR.call(dragHandleR);

  // Click anywhere on chart to move window centre
  signalSvg.on("click", async function (event) {
    const [mx] = d3.pointer(event, signalG.node());
    updateWindowPosition(startFromX(mx - _winWpx / 2));
    await updatePipeline();
  });
}

function updateWindowLabel(start) {
  const sr = 500;
  const t0 = (start / sr).toFixed(2);
  const t1 = ((start + state.tFixed) / sr).toFixed(2);
  document.getElementById("window-pos-label").textContent =
    `window: ${start}–${start + state.tFixed} (${t0}s – ${t1}s)`;
}

// ── Pipeline update ───────────────────────────────────────────────────────
async function updatePipeline() {
  if (state.ecgId === null) return;

  renderWindowPlot();

  let probs, stds, taskProb, taskStd;

  if (state.sweepData) {
    const ni = findNearestSweepIndex(state.windowStart);
    probs    = state.sweepData.concept_probs[ni];
    stds     = state.sweepData.concept_stds[ni];
    taskProb = state.sweepData.task_probs[ni];
    taskStd  = state.sweepData.task_stds ? state.sweepData.task_stds[ni] : 0;
  } else {
    let result;
    try {
      const res = await fetch(`${API}/infer`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ ecg_id: state.ecgId, start: state.windowStart }),
      });
      if (!res.ok) throw new Error(await res.text());
      result = await res.json();
    } catch (e) {
      console.error("Infer error:", e);
      return;
    }
    probs    = result.concept_probs;
    stds     = result.concept_stds;
    taskProb = result.task_prob;
    taskStd  = result.task_std || 0;
  }

  renderHistogram(probs, stds);
  renderTaskPrediction(taskProb, taskStd);
}

// ── Window mini-plot ──────────────────────────────────────────────────────
function renderWindowPlot() {
  const container = document.getElementById("window-plot");
  container.innerHTML = "";
  if (!state.signal) return;

  const lead    = state.leadIdx;
  const raw     = state.signal[lead];
  const ds      = state.downsample;
  const startDs = Math.floor(state.windowStart / ds);
  const endDs   = Math.min(Math.ceil((state.windowStart + state.tFixed) / ds), raw.length);
  const slice   = raw.slice(startDs, endDs);
  if (!slice.length) return;

  const W = WIN_W, H = WIN_H;
  const { top: pt, right: pr, bottom: pb, left: pl } = WIN_PAD;
  const iW = W - pl - pr, iH = H - pt - pb;

  const svg = d3.select(container).append("svg").attr("width", W).attr("height", H);
  const g   = svg.append("g").attr("transform", `translate(${pl},${pt})`);

  const xScale = d3.scaleLinear().domain([0, slice.length - 1]).range([0, iW]);
  const yMin = d3.min(slice), yMax = d3.max(slice);
  const yPad = ((yMax - yMin) * 0.15) || 0.1;
  const yScale = d3.scaleLinear().domain([yMin - yPad, yMax + yPad]).range([iH, 0]);

  g.append("g").attr("class", "grid")
    .call(d3.axisLeft(yScale).tickSize(-iW).tickFormat("").ticks(3));

  g.append("g").attr("class", "axis")
    .attr("transform", `translate(0,${iH})`)
    .call(d3.axisBottom(xScale).ticks(4)
      .tickFormat(d => `${((state.windowStart + d * ds) / 500).toFixed(1)}s`));
  g.append("g").attr("class", "axis")
    .call(d3.axisLeft(yScale).ticks(3));

  // Filled area under the curve
  g.append("path")
    .datum(slice)
    .attr("fill", "rgba(34,211,238,0.06)")
    .attr("d", d3.area().x((_, i) => xScale(i)).y0(iH).y1(d => yScale(d)).curve(d3.curveMonotoneX));

  // ECG line in cyan
  g.append("path")
    .datum(slice)
    .attr("fill", "none")
    .attr("stroke", "#22d3ee")
    .attr("stroke-width", 1.5)
    .attr("d", d3.line().x((_, i) => xScale(i)).y(d => yScale(d)).curve(d3.curveMonotoneX));
}

// ── Concept histogram ────────────────────────────────────────────────────
function renderHistogram(probs, stds) {
  const container = document.getElementById("histogram-container");
  container.innerHTML = "";

  const names  = ["P", "Q", "R", "S", "T", "U", "PR", "QRS", "ST", "QT"];
  const K      = names.length;
  const W      = HIST_W;
  const H      = HIST_H;
  const { top: pt, right: pr, bottom: pb, left: pl } = HIST_PAD;

  const innerW = W - pl - pr;
  const innerH = H - pt - pb;

  const svg = d3.select(container).append("svg")
    .attr("width",  W)
    .attr("height", H);

  const g = svg.append("g").attr("transform", `translate(${pl},${pt})`);

  const xScale = d3.scaleBand()
    .domain(names)
    .range([0, innerW])
    .padding(0.28);

  const yScale = d3.scaleLinear()
    .domain([0, 1])
    .range([innerH, 0]);

  // Y-axis grid
  g.append("g").attr("class", "grid")
    .call(d3.axisLeft(yScale).tickSize(-innerW).tickFormat("").ticks(5));

  // Axes
  g.append("g").attr("class", "axis")
    .attr("transform", `translate(0,${innerH})`)
    .call(d3.axisBottom(xScale));

  g.append("g").attr("class", "axis")
    .call(d3.axisLeft(yScale).ticks(5).tickFormat(d3.format(".1f")));

  // Y-axis label
  g.append("text").attr("class", "hist-axis-label")
    .attr("transform", "rotate(-90)")
    .attr("y", -pl + 12).attr("x", -innerH / 2)
    .attr("text-anchor", "middle")
    .text("Activation probability");

  const barW = xScale.bandwidth();

  names.forEach((name, i) => {
    const p    = Math.max(0, Math.min(1, probs[i]));
    const s    = Math.min(Math.max(stds[i], 0), 0.5);
    const x    = xScale(name);
    const yTop = yScale(p);
    const barH = innerH - yTop;
    const col  = CONCEPT_COLORS[i % CONCEPT_COLORS.length];

    // Background track
    g.append("rect").attr("class", "hist-bar-bg")
      .attr("x", x).attr("y", 0)
      .attr("width", barW).attr("height", innerH)
      .attr("rx", 3);

    // ── Uncertainty band (±σ), drawn behind the value bar ─────────────
    const bandTop = Math.max(0,     yScale(Math.min(1, p + s)));
    const bandBot = Math.min(innerH, yScale(Math.max(0, p - s)));
    const bandH   = Math.max(1, bandBot - bandTop);
    g.append("rect")
      .attr("x", x).attr("y", bandTop)
      .attr("width", barW).attr("height", bandH)
      .attr("rx", 3)
      .attr("fill", "#f59e0b")   // amber
      .attr("opacity", 0.22);

    // Value bar
    g.append("rect").attr("class", "hist-bar-fill")
      .attr("x", x + 2).attr("y", yTop)
      .attr("width", barW - 4).attr("height", Math.max(0, barH))
      .attr("rx", 3)
      .attr("fill", col)
      .attr("opacity", 0.88);

    // ±σ tick lines (top and bottom of band, at bar centre)
    const cx = x + barW / 2;
    for (const ey of [bandTop, bandBot]) {
      g.append("line")
        .attr("x1", cx - 5).attr("x2", cx + 5)
        .attr("y1", ey).attr("y2", ey)
        .attr("stroke", "#f59e0b").attr("stroke-width", 1.5).attr("opacity", 0.8);
    }
    g.append("line")
      .attr("x1", cx).attr("x2", cx)
      .attr("y1", bandTop).attr("y2", bandBot)
      .attr("stroke", "#f59e0b").attr("stroke-width", 1).attr("opacity", 0.55);

    // Value text above bar
    g.append("text")
      .attr("x", cx).attr("y", Math.max(bandTop - 4, 9))
      .attr("text-anchor", "middle")
      .attr("fill", "#e2e8f0").attr("font-size", 9)
      .text(p.toFixed(2));

    // ±σ label below concept name (in the bottom axis area)
    g.append("text")
      .attr("x", cx).attr("y", innerH + 28)
      .attr("text-anchor", "middle")
      .attr("fill", "#f59e0b").attr("font-size", 7.5)
      .text(`±${(s * 100).toFixed(0)}%`);
  });

  // Legend note
  g.append("rect").attr("x", innerW - 70).attr("y", -18)
    .attr("width", 9).attr("height", 9).attr("rx", 1)
    .attr("fill", "#f59e0b").attr("opacity", 0.5);
  g.append("text")
    .attr("x", innerW - 57).attr("y", -11)
    .attr("fill", "#94a3b8").attr("font-size", 8)
    .text("±1σ uncertainty");

}

// ── Task prediction bar chart ────────────────────────────────────────────
function renderTaskPrediction(taskProb, taskStd) {
  const container = document.getElementById("task-pred-container");
  container.innerHTML = "";

  const W = TASK_W, H = TASK_H;
  const { top: pt, right: pr, bottom: pb, left: pl } = TASK_PAD;
  const iW = W - pl - pr, iH = H - pt - pb;

  const svg = d3.select(container).append("svg").attr("width", W).attr("height", H);
  const g   = svg.append("g").attr("transform", `translate(${pl},${pt})`);

  const cats  = ["Complete", "Incomplete"];
  const vals  = [taskProb, 1 - taskProb];
  const fills = [taskColour(taskProb), "#1e3a4f"];

  const xScale = d3.scaleBand().domain(cats).range([0, iW]).padding(0.32);
  const yScale = d3.scaleLinear().domain([0, 1]).range([iH, 0]);

  g.append("g").attr("class", "grid")
    .call(d3.axisLeft(yScale).tickSize(-iW).tickFormat("").ticks(4));

  g.append("g").attr("class", "axis")
    .attr("transform", `translate(0,${iH})`)
    .call(d3.axisBottom(xScale).tickSize(0))
    .selectAll("text").style("font-size", "8px");

  g.append("g").attr("class", "axis")
    .call(d3.axisLeft(yScale).ticks(4).tickFormat(d3.format(".0%")));

  // Bars
  cats.forEach((cat, i) => {
    const v  = vals[i];
    const x  = xScale(cat);
    const bW = xScale.bandwidth();
    const yT = yScale(v);
    const bH = Math.max(0, iH - yT);

    // Background
    g.append("rect").attr("x", x).attr("y", 0)
      .attr("width", bW).attr("height", iH)
      .attr("fill", "#1e2535").attr("rx", 3);

    // Value bar
    g.append("rect").attr("x", x).attr("y", yT)
      .attr("width", bW).attr("height", bH)
      .attr("fill", fills[i]).attr("rx", 3).attr("opacity", 0.85);

    // Percentage label
    g.append("text")
      .attr("x", x + bW / 2)
      .attr("y", Math.max(yT - 4, 9))
      .attr("text-anchor", "middle")
      .attr("fill", "#e2e8f0").attr("font-size", 9)
      .text(`${(v * 100).toFixed(0)}%`);
  });

  // ── Uncertainty band on "Complete" ───────────────────────────────────
  const cx = xScale("Complete") + xScale.bandwidth() / 2;
  const bW = xScale.bandwidth();

  if (taskStd > 0) {
    const eT = Math.max(0,  yScale(Math.min(1, taskProb + taskStd)));
    const eB = Math.min(iH, yScale(Math.max(0, taskProb - taskStd)));
    const bandH = Math.max(1, eB - eT);

    // Amber uncertainty rectangle
    g.append("rect")
      .attr("x", xScale("Complete")).attr("y", eT)
      .attr("width", bW).attr("height", bandH)
      .attr("rx", 3)
      .attr("fill", "#f59e0b").attr("opacity", 0.22);

    // Tick lines at ±σ boundaries
    for (const ey of [eT, eB]) {
      g.append("line")
        .attr("x1", cx - 8).attr("x2", cx + 8)
        .attr("y1", ey).attr("y2", ey)
        .attr("stroke", "#f59e0b").attr("stroke-width", 1.8).attr("opacity", 0.9);
    }
    g.append("line")
      .attr("x1", cx).attr("x2", cx).attr("y1", eT).attr("y2", eB)
      .attr("stroke", "#f59e0b").attr("stroke-width", 1).attr("opacity", 0.55);

    // ±σ numeric label
    g.append("text")
      .attr("x", cx).attr("y", iH + pb - 4)
      .attr("text-anchor", "middle")
      .attr("fill", "#f59e0b").attr("font-size", 8).attr("font-weight", "600")
      .text(`±${(taskStd * 100).toFixed(0)}%`);

    // Confidence interval annotation in top-right
    const ciLo = Math.max(0, (taskProb - taskStd) * 100).toFixed(0);
    const ciHi = Math.min(100, (taskProb + taskStd) * 100).toFixed(0);
    g.append("text")
      .attr("x", iW).attr("y", -8)
      .attr("text-anchor", "end")
      .attr("fill", "#f59e0b").attr("font-size", 7.5)
      .text(`±1σ  [${ciLo}%–${ciHi}%]`);
  }

  // Legend
  g.append("rect").attr("x", 0).attr("y", -18)
    .attr("width", 9).attr("height", 9).attr("rx", 1)
    .attr("fill", "#f59e0b").attr("opacity", 0.5);
  g.append("text")
    .attr("x", 13).attr("y", -11)
    .attr("fill", "#94a3b8").attr("font-size", 7.5)
    .text("uncertainty");
}

// ── Sweep ──────────────────────────────────────────────────────────────────
document.getElementById("sweep-btn").addEventListener("click", async () => {
  if (state.ecgId === null) return;
  const stride = parseInt(document.getElementById("stride-input").value, 10) || 25;
  const status = document.getElementById("sweep-status");
  status.textContent = "Running sweep…";
  status.className   = "status-label running";

  let data;
  try {
    const res = await fetch(`${API}/sweep`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ ecg_id: state.ecgId, stride }),
    });
    if (!res.ok) throw new Error(await res.text());
    data = await res.json();
  } catch (e) {
    status.textContent = `Error: ${e.message}`;
    status.className   = "status-label";
    return;
  }

  state.sweepData = data;
  status.textContent = `Done — ${data.starts.length} windows`;
  status.className   = "status-label done";

  renderHeatmap(data);
  document.getElementById("sweep-section").style.display = "";

  // Refresh pipeline from sweep data
  await updatePipeline();
});

function findNearestSweepIndex(windowStart) {
  const starts = state.sweepData.starts;
  let best = 0, bestDist = Infinity;
  starts.forEach((s, i) => {
    const d = Math.abs(s - windowStart);
    if (d < bestDist) { bestDist = d; best = i; }
  });
  return best;
}

// ── Sweep Heatmap ──────────────────────────────────────────────────────────
function renderHeatmap(data) {
  const container = document.getElementById("heatmap-container");
  container.innerHTML = "";

  const names     = data.concept_names;   // (K,)
  const starts    = data.starts;          // (N,)
  const cProbs    = data.concept_probs;   // (N, K)
  const K         = names.length;
  const N         = starts.length;

  const CELL_H    = 22;
  const CELL_W    = Math.max(2, Math.min(8, Math.floor(900 / N)));
  const LABEL_W   = 44;
  const MAP_TOP   = 10;
  const MAP_BOT   = 28;
  const TOTAL_W   = LABEL_W + CELL_W * N + 20;
  const TOTAL_H   = MAP_TOP + CELL_H * K + MAP_BOT;

  const color = d3.scaleSequential(d3.interpolateYlOrRd).domain([0, 1]);

  const svg = d3.select(container).append("svg")
    .attr("width",  TOTAL_W)
    .attr("height", TOTAL_H);

  const g = svg.append("g").attr("transform", `translate(${LABEL_W},${MAP_TOP})`);

  // Cells
  names.forEach((name, ki) => {
    starts.forEach((start, ni) => {
      g.append("rect")
        .attr("x",      ni * CELL_W)
        .attr("y",      ki * CELL_H)
        .attr("width",  CELL_W)
        .attr("height", CELL_H)
        .attr("fill",   color(cProbs[ni][ki]))
        .on("click", async () => {
          state.windowStart = start;
          updateWindowPosition(start);
          await updatePipeline();
        });
    });
    // Row label
    svg.append("text").attr("class", "heatmap-label")
      .attr("x", LABEL_W - 4)
      .attr("y", MAP_TOP + ki * CELL_H + CELL_H / 2 + 4)
      .attr("text-anchor", "end")
      .attr("fill", "#94a3b8")
      .attr("font-size", 10)
      .text(name);
  });

  // X-axis: time ticks
  const xScale = d3.scaleLinear()
    .domain([0, starts[starts.length - 1] + data.t_fixed])
    .range([0, CELL_W * N]);
  const xAxis = d3.axisBottom(xScale)
    .ticks(Math.min(10, N))
    .tickFormat(d => `${(d / 500).toFixed(1)}s`);
  g.append("g")
    .attr("class", "axis")
    .attr("transform", `translate(0,${CELL_H * K})`)
    .call(xAxis);

  // Cursor line (updated when window moves)
  const cursor = g.append("line").attr("class", "heatmap-cursor")
    .attr("y1", 0).attr("y2", CELL_H * K);

  // Expose so signal drag can update it
  window._heatmapCursorScale = xScale;
  window._heatmapCursor      = cursor;
  window._heatmapCellW       = CELL_W;
  window._heatmapN           = N;

  updateHeatmapCursor(state.windowStart);
}

function updateHeatmapCursor(windowStart) {
  if (!window._heatmapCursor) return;
  const px = window._heatmapCursorScale(windowStart + state.tFixed / 2);
  window._heatmapCursor.attr("x1", px).attr("x2", px);
}

// ── updateWindowPosition (module-level, uses shared _xScaleSig) ──────────
function updateWindowPosition(start) {
  state.windowStart = start;
  if (!signalG || !_xScaleSig) return;

  const px = _xScaleSig(start);
  windowRect.attr("x", px);
  windowHandleL.attr("x", px - 4);
  windowHandleR.attr("x", px + _winWpx - 4);

  updateWindowLabel(start);
  if (state.sweepData) updateHeatmapCursor(start);
}

// ── Start ─────────────────────────────────────────────────────────────────
init();
