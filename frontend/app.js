const fmtPct = (v) => `${(Number(v) * 100).toFixed(1)}%`;
const fmtNum = (v) => Number(v).toFixed(2);

const API_BASE = "/api";
const POLL_MS = 1000;
const POLL_TIMEOUT_MS = 90_000;

const REQUIRED_KEYS = [
  "run_id",
  "summary",
  "model_comparison",
  "equity_curve",
  "drawdown",
  "feature_importance",
  "failure_periods",
];

const DEMO_FALLBACK = {
  run_id: "precomputed_demo",
  summary: {
    best_model: "lightgbm",
    auc: 0.61,
    balanced_accuracy: 0.57,
    information_ratio: 0.73,
    cagr: 0.124,
    sharpe: 1.08,
    max_drawdown: -0.091,
    exposure: 0.43,
  },
  model_comparison: [
    { model_name: "logistic_regression", auc: 0.54, balanced_accuracy: 0.52, information_ratio: 0.31, sharpe: 0.44 },
    { model_name: "random_forest", auc: 0.57, balanced_accuracy: 0.55, information_ratio: 0.48, sharpe: 0.71 },
    { model_name: "lightgbm", auc: 0.61, balanced_accuracy: 0.57, information_ratio: 0.73, sharpe: 1.08 },
  ],
  equity_curve: [["2024-01-02", 1.0], ["2024-01-09", 1.01], ["2024-01-16", 1.03], ["2024-01-23", 1.02], ["2024-01-30", 1.04], ["2024-02-06", 1.05], ["2024-02-13", 1.07], ["2024-02-20", 1.06], ["2024-02-27", 1.08], ["2024-03-05", 1.11], ["2024-03-12", 1.1], ["2024-03-19", 1.13]],
  drawdown: [["2024-01-02", 0.0], ["2024-01-09", 0.0], ["2024-01-16", 0.0], ["2024-01-23", -0.01], ["2024-01-30", 0.0], ["2024-02-06", 0.0], ["2024-02-13", 0.0], ["2024-02-20", -0.01], ["2024-02-27", 0.0], ["2024-03-05", 0.0], ["2024-03-12", -0.01], ["2024-03-19", 0.0]],
  feature_importance: [["ret_lag_1", 0.18], ["vol_10", 0.14], ["mom_ma_5", 0.12], ["hdd_us_l1", 0.11], ["temp_anom_us_l1", 0.1], ["close_^VIX", 0.09], ["close_DX-Y.NYB", 0.08], ["cdd_delta_1d", 0.07]],
  failure_periods: [
    { window: "2024-02-14 to 2024-02-22", note: "Whipsaw around inventory surprise and risk-off rotation." },
    { window: "2024-04-03 to 2024-04-09", note: "Macro headline shock broke lag-based momentum signal." },
  ],
};

let lastFailedRunId = null;
let latestLiveRunId = null;

const tourState = {
  steps: [
    { target: '[data-tour="trace"]', title: "Run Trace", body: "This is live stage-by-stage execution from the backend worker queue." },
    { target: '[data-tour="provenance"]', title: "Data Provenance", body: "These fields prove what was computed: date window, row counts, selected model, and artifacts." },
    { target: '[data-tour="kpis"]', title: "Headline Metrics", body: "KPIs are populated from the completed live run payload." },
    { target: '[data-tour="table"]', title: "Model Selection", body: "Model comparison is computed in this run, then ranked by information ratio." },
    { target: '[data-tour="equity"]', title: "Equity Curve", body: "Net performance after transaction costs from the run's backtest." },
    { target: '[data-tour="drawdown"]', title: "Drawdown Control", body: "Risk profile from the same computed run." },
    { target: '[data-tour="shap"]', title: "Top Drivers", body: "Feature importance from the selected model in this run." },
    { target: '[data-tour="failures"]', title: "Failure Windows", body: "Weak windows extracted from live run backtest behavior." },
  ],
  idx: 0,
  active: false,
  timer: null,
  raf: null,
  suppressAutoAdvanceUntil: 0,
};

function sleep(ms) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

function setStatus(text, isError = false) {
  const el = document.getElementById("runMeta");
  if (!el) return;
  el.textContent = text;
  el.classList.toggle("error", Boolean(isError));
}

function updateLiveBadges({ modeLabel, runId, runtimeSec }) {
  const mode = document.getElementById("badgeMode");
  const rid = document.getElementById("badgeRunId");
  const rt = document.getElementById("badgeRuntime");
  if (mode) mode.textContent = modeLabel || "Awaiting run";
  if (rid) rid.textContent = `Run ID: ${runId || "-"}`;
  if (rt) rt.textContent = `Runtime: ${runtimeSec != null ? `${Number(runtimeSec).toFixed(1)}s` : "-"}`;
}

function linePath(points, width, height, pad = 24) {
  const xs = points.map((_, i) => i);
  const ys = points.map((p) => Number(p[1]));
  const minY = Math.min(...ys);
  const maxY = Math.max(...ys);
  const spanY = maxY - minY || 1;
  const spanX = Math.max(...xs) || 1;
  return points
    .map((p, i) => {
      const x = pad + (i / spanX) * (width - pad * 2);
      const y = height - pad - ((Number(p[1]) - minY) / spanY) * (height - pad * 2);
      return `${i === 0 ? "M" : "L"}${x},${y}`;
    })
    .join(" ");
}

function renderLineChart(svgId, points, color, fill = false) {
  const svg = document.getElementById(svgId);
  if (!svg) return;
  if (!Array.isArray(points) || points.length === 0) {
    svg.innerHTML = '<text x="24" y="34" fill="#5f6d7f" font-size="13" font-family="IBM Plex Mono">No data available</text>';
    return;
  }
  const width = 600;
  const height = 280;
  const path = linePath(points, width, height);
  const ys = points.map((p) => Number(p[1]));
  const minY = Math.min(...ys);
  const maxY = Math.max(...ys);
  const axis = `M24,${height - 24} L${width - 24},${height - 24}`;
  const fillPath = fill ? `<path d="${path} L${width - 24},${height - 24} L24,${height - 24} Z" fill="${color}1f"></path>` : "";

  svg.innerHTML = `
    <path d="${axis}" stroke="#d7e1eb" stroke-width="1"></path>
    ${fillPath}
    <path d="${path}" stroke="${color}" stroke-width="3" fill="none" stroke-linecap="round"></path>
    <text x="26" y="20" fill="#667487" font-size="11" font-family="IBM Plex Mono">max ${fmtNum(maxY)}</text>
    <text x="26" y="${height - 8}" fill="#667487" font-size="11" font-family="IBM Plex Mono">min ${fmtNum(minY)}</text>
  `;
}

function renderBarChart(svgId, data) {
  const svg = document.getElementById(svgId);
  if (!svg) return;
  if (!Array.isArray(data) || data.length === 0) {
    svg.innerHTML = '<text x="24" y="34" fill="#5f6d7f" font-size="13" font-family="IBM Plex Mono">No feature data</text>';
    return;
  }
  const rows = data.slice(0, 10);
  const width = 600;
  const height = 320;
  const pad = 26;
  const maxV = Math.max(...rows.map((d) => Number(d[1]))) || 1;
  const band = (height - pad * 2) / rows.length;

  const bars = rows
    .map((d, i) => {
      const name = String(d[0]);
      const val = Number(d[1]);
      const barW = ((width - 220) * val) / maxV;
      const y = pad + i * band;
      return `
        <text x="14" y="${y + band * 0.65}" fill="#496180" font-size="12" font-family="IBM Plex Mono">${name}</text>
        <rect x="190" y="${y + 2}" width="${barW}" height="${Math.max(8, band - 6)}" rx="6" fill="#1f6e74"></rect>
        <text x="${195 + barW}" y="${y + band * 0.65}" fill="#1f6e74" font-size="11" font-family="IBM Plex Mono">${fmtNum(val)}</text>
      `;
    })
    .join("");

  svg.innerHTML = bars;
}

function renderHeadline(summary) {
  const host = document.getElementById("headlineMetrics");
  if (!host) return;
  const cards = [
    ["Best Model", String(summary.best_model || "-").toUpperCase()],
    ["AUC", fmtNum(summary.auc || 0)],
    ["Information Ratio", fmtNum(summary.information_ratio || 0)],
    ["Sharpe", fmtNum(summary.sharpe || 0)],
  ];
  host.innerHTML = cards
    .map(([label, value]) => `<article class="headline-card"><div class="headline-card__label">${label}</div><div class="headline-card__value">${value}</div></article>`)
    .join("");
}

function renderKPIs(summary) {
  const host = document.getElementById("kpis");
  if (!host) return;
  const kpis = [
    ["Best Model", String(summary.best_model || "-").toUpperCase()],
    ["AUC", fmtNum(summary.auc || 0)],
    ["Info Ratio", fmtNum(summary.information_ratio || 0)],
    ["CAGR", fmtPct(summary.cagr || 0)],
    ["Sharpe", fmtNum(summary.sharpe || 0)],
    ["Max Drawdown", fmtPct(summary.max_drawdown || 0)],
    ["Balanced Acc", fmtNum(summary.balanced_accuracy || 0)],
    ["Exposure", fmtPct(summary.exposure || 0)],
  ];
  host.innerHTML = kpis
    .map(([label, value]) => `<article class="kpi"><div class="kpi__label">${label}</div><div class="kpi__value">${value}</div></article>`)
    .join("");
}

function renderModelTable(rows) {
  const tbody = document.querySelector("#modelTable tbody");
  if (!tbody) return;
  if (!Array.isArray(rows) || rows.length === 0) {
    tbody.innerHTML = '<tr><td colspan="5">No model comparison data</td></tr>';
    return;
  }
  const sorted = [...rows].sort((a, b) => Number(b.information_ratio || 0) - Number(a.information_ratio || 0));
  const best = sorted.length ? sorted[0].model_name : "";
  tbody.innerHTML = rows
    .map((r) => `<tr class="${r.model_name === best ? "is-best" : ""}"><td>${r.model_name}</td><td>${fmtNum(r.auc || 0)}</td><td>${fmtNum(r.balanced_accuracy || 0)}</td><td>${fmtNum(r.information_ratio || 0)}</td><td>${fmtNum(r.sharpe || 0)}</td></tr>`)
    .join("");
}

function renderFailures(items) {
  const host = document.getElementById("failures");
  if (!host) return;
  const list = Array.isArray(items) ? items : [];
  host.innerHTML =
    list.map((i) => `<li><b>${i.window || "Unknown window"}</b><span>${i.note || "No note provided."}</span></li>`).join("") ||
    '<li><span>No failure windows logged.</span></li>';
}

function validateDataShape(raw) {
  if (!raw || typeof raw !== "object") {
    throw new Error("JSON payload must be an object.");
  }
  const missing = REQUIRED_KEYS.filter((k) => !(k in raw));
  return { valid: missing.length === 0, missing };
}

function sanitizeData(raw) {
  const safe = { ...DEMO_FALLBACK, ...raw };
  safe.summary = { ...DEMO_FALLBACK.summary, ...(raw.summary || {}) };
  safe.model_comparison = Array.isArray(raw.model_comparison) ? raw.model_comparison : DEMO_FALLBACK.model_comparison;
  safe.equity_curve = Array.isArray(raw.equity_curve) ? raw.equity_curve : DEMO_FALLBACK.equity_curve;
  safe.drawdown = Array.isArray(raw.drawdown) ? raw.drawdown : DEMO_FALLBACK.drawdown;
  safe.feature_importance = Array.isArray(raw.feature_importance) ? raw.feature_importance : DEMO_FALLBACK.feature_importance;
  safe.failure_periods = Array.isArray(raw.failure_periods) ? raw.failure_periods : DEMO_FALLBACK.failure_periods;
  return safe;
}

function renderDashboard(rawData, { source = "precomputed" } = {}) {
  const data = sanitizeData(rawData);
  renderHeadline(data.summary || {});
  renderKPIs(data.summary || {});
  renderModelTable(data.model_comparison || []);
  renderLineChart("equityChart", data.equity_curve || [], "#d0622c", true);
  renderLineChart("drawdownChart", data.drawdown || [], "#1f6e74", true);
  renderBarChart("featureChart", data.feature_importance || []);
  renderFailures(data.failure_periods || []);

  const stamp = new Date().toLocaleTimeString();
  setStatus(`Loaded ${source} run: ${data.run_id || "unknown"} | rendered at ${stamp}`);

  document.body.classList.remove("flash-update");
  void document.body.offsetWidth;
  document.body.classList.add("flash-update");
}

function renderTrace(runStatus) {
  const stage = document.getElementById("traceStage");
  const bar = document.getElementById("traceProgressBar");
  const label = document.getElementById("traceProgressLabel");
  const list = document.getElementById("traceList");
  const err = document.getElementById("traceError");

  if (stage) stage.textContent = `Stage: ${runStatus.current_stage || runStatus.status || "idle"}`;
  if (bar) bar.style.width = `${Math.max(0, Math.min(100, Number(runStatus.progress_pct || 0)))}%`;
  if (label) label.textContent = `${Math.round(Number(runStatus.progress_pct || 0))}%`;

  const trace = Array.isArray(runStatus.trace) ? runStatus.trace : [];
  if (list) {
    list.innerHTML = trace
      .map((s) => {
        const stats = s.stats && typeof s.stats === "object"
          ? Object.entries(s.stats)
              .slice(0, 4)
              .map(([k, v]) => `${k}: ${v}`)
              .join(" | ")
          : "";
        return `<li><div class="trace-stage"><span>${s.stage}</span><span>${Math.round(Number(s.progress_pct || 0))}%</span></div><div class="trace-stats">${stats}</div></li>`;
      })
      .join("");
  }

  if (err) {
    if (runStatus.status === "failed") {
      err.hidden = false;
      err.textContent = runStatus.error && runStatus.error.message ? runStatus.error.message : "Run failed with unknown error.";
    } else {
      err.hidden = true;
      err.textContent = "";
    }
  }
}

function renderProvenance(prov = {}) {
  const meta = document.getElementById("provenanceMeta");
  const artifacts = document.getElementById("artifactList");

  const dateRange = prov.date_range || {};
  const rows = prov.row_counts || {};
  const params = prov.model_params || {};

  const items = [
    ["Date Range", `${dateRange.start || "-"} to ${dateRange.end || "-"}`],
    ["Selected Model", prov.selected_model || "-"],
    ["Rows (prices/weather)", `${rows.prices ?? "-"} / ${rows.weather ?? "-"}`],
    ["Rows (model/trades)", `${rows.model_table ?? "-"} / ${rows.trades ?? "-"}`],
    ["Threshold", params.probability_threshold ?? "-"],
    ["Cost (bps)", params.transaction_cost_bps ?? "-"],
  ];

  if (meta) {
    meta.innerHTML = items
      .map(([k, v]) => `<div class="provenance-item"><div class="provenance-item__k">${k}</div><div class="provenance-item__v">${v}</div></div>`)
      .join("");
  }

  if (artifacts) {
    const list = Array.isArray(prov.artifacts) ? prov.artifacts : [];
    artifacts.innerHTML = list.map((a) => `<li>${a}</li>`).join("") || "<li>No artifacts recorded.</li>";
  }
}

function scrollToDemo() {
  const section = document.getElementById("live-demo");
  if (section) section.scrollIntoView({ behavior: "smooth", block: "start" });
}

function clearTraceView() {
  renderTrace({ status: "queued", current_stage: "queued", progress_pct: 0, trace: [] });
  renderProvenance({});
}

async function postCreateRun() {
  const resp = await fetch(`${API_BASE}/runs`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ mode: "demo_quick" }),
  });
  if (!resp.ok) {
    throw new Error(`Failed to create run: HTTP ${resp.status}`);
  }
  return resp.json();
}

async function fetchRunStatus(runId) {
  const resp = await fetch(`${API_BASE}/runs/${encodeURIComponent(runId)}`, { cache: "no-store" });
  if (!resp.ok) {
    throw new Error(`Failed to fetch run status: HTTP ${resp.status}`);
  }
  return resp.json();
}

async function fetchRunResults(runId) {
  const resp = await fetch(`${API_BASE}/runs/${encodeURIComponent(runId)}/results`, { cache: "no-store" });
  if (!resp.ok) {
    throw new Error(`Failed to fetch run results: HTTP ${resp.status}`);
  }
  return resp.json();
}

async function pollUntilDone(runId) {
  const started = Date.now();
  while (Date.now() - started < POLL_TIMEOUT_MS) {
    const status = await fetchRunStatus(runId);
    renderTrace(status);
    updateLiveBadges({ modeLabel: status.status === "completed" ? "Live computed" : "Live computing", runId, runtimeSec: status.runtime_sec });

    if (status.status === "completed") return status;
    if (status.status === "failed") throw new Error(status.error && status.error.message ? status.error.message : "Run failed");

    await sleep(POLL_MS);
  }
  throw new Error("Run timed out before completion.");
}

async function runDemoFlow(startTourAfter = true) {
  try {
    clearTraceView();
    setStatus("Submitting live run...");
    scrollToDemo();

    const created = await postCreateRun();
    const runId = created.run_id;
    latestLiveRunId = runId;
    updateLiveBadges({ modeLabel: "Live computing", runId, runtimeSec: null });
    setStatus(`Run created: ${runId}. Executing...`);

    await pollUntilDone(runId);

    const payload = await fetchRunResults(runId);
    renderDashboard(payload.results, { source: "live computed" });
    renderProvenance(payload.provenance || {});
    updateLiveBadges({ modeLabel: "Live computed", runId, runtimeSec: payload.runtime_sec });

    if (startTourAfter) {
      setTimeout(() => startTour(), 450);
    }
  } catch (err) {
    const message = err instanceof Error ? err.message : "Unknown run error";
    lastFailedRunId = latestLiveRunId;
    setStatus(`Live run failed: ${message}`, true);
    updateLiveBadges({ modeLabel: "Run failed", runId: latestLiveRunId, runtimeSec: null });
    const errNode = document.getElementById("traceError");
    if (errNode) {
      errNode.hidden = false;
      errNode.textContent = message;
    }
  }
}

async function loadPrecomputedDemo() {
  try {
    const resp = await fetch("./data/demo_run.json", { cache: "no-store" });
    if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
    const data = await resp.json();
    const check = validateDataShape(data);
    if (!check.valid) {
      setStatus(`Precomputed demo missing keys: ${check.missing.join(", ")}. Using fallbacks.`, true);
    }
    renderDashboard(sanitizeData(data), { source: "precomputed" });
  } catch (err) {
    renderDashboard(DEMO_FALLBACK, { source: "fallback precomputed" });
    setStatus("Could not load precomputed demo file; fallback loaded.", true);
  }
}

async function handleUpload(file) {
  try {
    const text = await file.text();
    const parsed = JSON.parse(text);
    const check = validateDataShape(parsed);
    const safe = sanitizeData(parsed);
    renderDashboard(safe, { source: "uploaded precomputed" });
    scrollToDemo();
    updateLiveBadges({ modeLabel: "Uploaded precomputed", runId: safe.run_id || "uploaded", runtimeSec: null });

    if (!check.valid) {
      setStatus(`Upload loaded with fallbacks. Missing keys: ${check.missing.join(", ")}.`, true);
    } else {
      setStatus(`Uploaded precomputed run: ${safe.run_id || "unknown"}`);
    }
  } catch (err) {
    setStatus(`Upload failed: ${err instanceof Error ? err.message : "Unknown error"}`, true);
  }
}

function stopTour() {
  tourState.active = false;
  if (tourState.timer) clearTimeout(tourState.timer);
  tourState.timer = null;
  if (tourState.raf) cancelAnimationFrame(tourState.raf);
  tourState.raf = null;
  const overlay = document.getElementById("tourOverlay");
  if (overlay) overlay.hidden = true;
}

function placeTourHole(target) {
  const hole = document.getElementById("tourHole");
  if (!hole) return;
  const rect = target.getBoundingClientRect();

  const pad = 8;
  const left = Math.max(6, rect.left - pad);
  const top = Math.max(6, rect.top - pad);
  const width = Math.max(40, rect.width + pad * 2);
  const height = Math.max(40, rect.height + pad * 2);

  hole.style.left = `${left}px`;
  hole.style.top = `${top}px`;
  hole.style.width = `${width}px`;
  hole.style.height = `${height}px`;

  const topMask = document.getElementById("tourMaskTop");
  const leftMask = document.getElementById("tourMaskLeft");
  const rightMask = document.getElementById("tourMaskRight");
  const bottomMask = document.getElementById("tourMaskBottom");
  if (!topMask || !leftMask || !rightMask || !bottomMask) return;

  const viewportW = window.innerWidth;
  const viewportH = window.innerHeight;

  topMask.style.left = "0px";
  topMask.style.top = "0px";
  topMask.style.width = `${viewportW}px`;
  topMask.style.height = `${Math.max(0, top)}px`;

  leftMask.style.left = "0px";
  leftMask.style.top = "0px";
  leftMask.style.width = `${Math.max(0, left)}px`;
  leftMask.style.height = `${viewportH}px`;

  rightMask.style.left = `${Math.min(viewportW, left + width)}px`;
  rightMask.style.top = "0px";
  rightMask.style.width = `${Math.max(0, viewportW - (left + width))}px`;
  rightMask.style.height = `${viewportH}px`;

  bottomMask.style.left = "0px";
  bottomMask.style.top = `${Math.min(viewportH, top + height)}px`;
  bottomMask.style.width = `${viewportW}px`;
  bottomMask.style.height = `${Math.max(0, viewportH - (top + height))}px`;
}

function positionTourStep(shouldScroll = true) {
  if (!tourState.active) return;
  const step = tourState.steps[tourState.idx];
  if (!step) return;

  const target = document.querySelector(step.target);
  if (!target) {
    nextTourStep();
    return;
  }

  if (shouldScroll) {
    target.scrollIntoView({ behavior: "smooth", block: "center", inline: "nearest" });
    tourState.suppressAutoAdvanceUntil = Date.now() + 700;
  }

  const stepEl = document.getElementById("tourStep");
  const titleEl = document.getElementById("tourTitle");
  const bodyEl = document.getElementById("tourBody");
  const backBtn = document.getElementById("tourBack");
  const nextBtn = document.getElementById("tourNext");

  if (stepEl) stepEl.textContent = `Step ${tourState.idx + 1} / ${tourState.steps.length}`;
  if (titleEl) titleEl.textContent = step.title;
  if (bodyEl) bodyEl.textContent = step.body;
  if (backBtn) backBtn.disabled = tourState.idx === 0;
  if (nextBtn) nextBtn.textContent = tourState.idx === tourState.steps.length - 1 ? "Finish" : "Next";

  if (tourState.raf) cancelAnimationFrame(tourState.raf);
  const updateHole = () => {
    if (!tourState.active) return;
    const liveTarget = document.querySelector(step.target);
    if (liveTarget) placeTourHole(liveTarget);
    tourState.raf = requestAnimationFrame(updateHole);
  };
  tourState.raf = requestAnimationFrame(updateHole);
}

function queueAutoAdvance() {
  if (tourState.timer) clearTimeout(tourState.timer);
  tourState.timer = setTimeout(() => {
    if (Date.now() < tourState.suppressAutoAdvanceUntil) {
      queueAutoAdvance();
      return;
    }
    if (tourState.active) nextTourStep();
  }, 3200);
}

function nextTourStep() {
  if (!tourState.active) return;
  if (tourState.idx >= tourState.steps.length - 1) {
    stopTour();
    setStatus("Tour complete. Live run evidence shown above.");
    return;
  }
  tourState.idx += 1;
  positionTourStep(true);
  queueAutoAdvance();
}

function prevTourStep() {
  if (!tourState.active) return;
  tourState.idx = Math.max(0, tourState.idx - 1);
  positionTourStep(true);
  queueAutoAdvance();
}

function startTour() {
  const overlay = document.getElementById("tourOverlay");
  if (!overlay) return;
  overlay.hidden = false;
  tourState.active = true;
  tourState.idx = 0;
  positionTourStep(true);
  queueAutoAdvance();
}

function wireEvents() {
  const runHero = document.getElementById("runDemoHero");
  const runInline = document.getElementById("runDemoInline");
  const replay = document.getElementById("replayTourBtn");
  const retry = document.getElementById("retryRunBtn");
  const input = document.getElementById("jsonUpload");

  if (runHero) runHero.addEventListener("click", () => runDemoFlow(true));
  if (runInline) runInline.addEventListener("click", () => runDemoFlow(true));
  if (retry) retry.addEventListener("click", () => runDemoFlow(false));
  if (replay) replay.addEventListener("click", () => startTour());

  if (input) {
    input.addEventListener("change", (e) => {
      const files = e.target && e.target.files ? e.target.files : [];
      const file = files[0];
      if (!file) return;
      handleUpload(file);
    });
  }

  const tourBack = document.getElementById("tourBack");
  const tourNext = document.getElementById("tourNext");
  const tourSkip = document.getElementById("tourSkip");
  if (tourBack) tourBack.addEventListener("click", prevTourStep);
  if (tourNext) tourNext.addEventListener("click", nextTourStep);
  if (tourSkip) tourSkip.addEventListener("click", stopTour);

  const liveLink = document.querySelector('#heroLinks a[href="#live-demo"]');
  if (liveLink) {
    liveLink.addEventListener("click", (e) => {
      e.preventDefault();
      scrollToDemo();
    });
  }

  window.addEventListener("resize", () => {
    if (tourState.active) positionTourStep(false);
  });
  window.addEventListener("scroll", () => {
    if (tourState.active) positionTourStep(false);
  });
}

function init() {
  wireEvents();
  clearTraceView();
  renderProvenance({});
  updateLiveBadges({ modeLabel: "Precomputed preview", runId: "-", runtimeSec: null });
  loadPrecomputedDemo();
}

window.addEventListener("error", (e) => {
  setStatus(`Frontend error: ${e.message}`, true);
});

if (document.readyState === "loading") {
  document.addEventListener("DOMContentLoaded", init);
} else {
  init();
}
