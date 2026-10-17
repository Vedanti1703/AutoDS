// AutoDS-Agent 2.0 Modern Frontend Logic
// Vanilla JS — reactive state, tab navigation, dynamic forms, lightbox, and async polling

const state = {
  jobId: null,
  columns: [],
  pollTimer: null,
  inputColumns: [],
  sampleRecord: {},
  baselinePrediction: null,
  targetCol: null,
  cleanedData: null,
  engineeredData: null,
};

const el = (id) => document.getElementById(id);

// ---------------------------------------------------------------------------
// View & State Transitions
// ---------------------------------------------------------------------------

function showView(name) {
  ["upload", "configure", "running", "results"].forEach((v) => {
    const node = el(`view-${v}`);
    if (node) node.hidden = v !== name;
  });
  const rail = el("rail");
  if (rail) rail.hidden = (name === "upload");
  const topReset = el("topbarResetBtn");
  if (topReset) topReset.hidden = (name === "upload");
}

function setStatus(stateName, text) {
  const pill = el("statusPill");
  if (!pill) return;
  pill.dataset.state = stateName;
  el("statusText").textContent = text;
}

const STEP_ORDER = [
  "profile", "eda", "preprocess", "anomaly", "feature_eng", "feature_sel",
  "imbalance", "model_select", "tuning", "evaluation", "explain", "export", "report"
];

function updateRailFromRecords(records) {
  const seenStages = new Set(records.map((r) => r.stage_key));
  let lastSeenIdx = -1;
  let completedCount = 0;

  STEP_ORDER.forEach((step, idx) => {
    if (seenStages.has(step)) {
      lastSeenIdx = idx;
      completedCount++;
    }
  });

  const badge = el("railProgressBadge");
  if (badge) badge.textContent = `${completedCount}/${STEP_ORDER.length}`;

  STEP_ORDER.forEach((step, idx) => {
    const li = document.querySelector(`.rail-steps li[data-step="${step}"]`);
    if (!li) return;
    li.classList.remove("active", "done");
    if (idx < lastSeenIdx) li.classList.add("done");
    else if (idx === lastSeenIdx) li.classList.add("active");
  });
}

// ---------------------------------------------------------------------------
// Tabs Navigation
// ---------------------------------------------------------------------------

function setupTabs() {
  const tabBtns = document.querySelectorAll(".tab-btn");
  tabBtns.forEach((btn) => {
    btn.addEventListener("click", () => {
      switchTab(btn.dataset.tab);
    });
  });

  // Jump to cleaned data tab button
  const jumpBtn = el("jumpToCleanedTabBtn");
  if (jumpBtn) {
    jumpBtn.addEventListener("click", () => {
      switchTab("tab-prep");
    });
  }
}

function switchTab(targetId) {
  const tabBtns = document.querySelectorAll(".tab-btn");
  tabBtns.forEach((b) => b.classList.remove("active"));
  document.querySelectorAll(".tab-content").forEach((tc) => tc.classList.remove("active"));

  const targetBtn = document.querySelector(`.tab-btn[data-tab="${targetId}"]`);
  if (targetBtn) targetBtn.classList.add("active");

  const targetContent = el(targetId);
  if (targetContent) targetContent.classList.add("active");

  // Auto-load registry if user clicks registry tab
  if (targetId === "tab-registry") {
    fetchRegistry();
  }
}

// ---------------------------------------------------------------------------
// Upload & Dataset Handling
// ---------------------------------------------------------------------------

function showUploadError(msg) {
  const box = el("uploadError");
  box.textContent = msg;
  box.hidden = false;
}

function clearUploadError() {
  const box = el("uploadError");
  box.textContent = "";
  box.hidden = true;
}

function onUploadSuccess(data) {
  clearUploadError();
  state.jobId = data.job_id;
  state.columns = data.columns;

  el("confFilename").textContent = data.filename || "Uploaded CSV";
  el("confMeta").innerHTML = `
    <span class="badge">📊 ${data.n_rows.toLocaleString()} rows</span>
    <span class="badge">🏷️ ${data.n_cols} columns</span>
    <span class="badge">⚡ ${data.size_mb} MB</span>
  `;

  const sel = el("targetSelect");
  sel.innerHTML = "";
  data.columns.forEach((col) => {
    const opt = document.createElement("option");
    opt.value = col;
    opt.textContent = col;
    sel.appendChild(opt);
  });

  if (data.suggested_target && data.columns.includes(data.suggested_target)) {
    sel.value = data.suggested_target;
  } else {
    sel.value = data.columns[data.columns.length - 1];
  }

  renderPreviewTable("previewTableWrap", data.preview);
  showView("configure");
  setStatus("idle", "Dataset Ready");
}

function renderPreviewTable(wrapperId, rows) {
  const wrap = el(wrapperId);
  if (!wrap) return;
  if (!rows || !rows.length) {
    wrap.innerHTML = `<div class="empty-state" style="padding: 14px;">No preview available</div>`;
    return;
  }
  const cols = Object.keys(rows[0]);
  let html = `<table><thead><tr>${cols.map((c) => `<th>${c}</th>`).join("")}</tr></thead><tbody>`;
  rows.forEach((r) => {
    html += `<tr>${cols.map((c) => `<td>${r[c] === null ? '<span style="color:var(--text-faint); font-style:italic;">null</span>' : r[c]}</td>`).join("")}</tr>`;
  });
  html += `</tbody></table>`;
  wrap.innerHTML = html;
}

function uploadFile(file) {
  clearUploadError();
  setStatus("running", "Uploading CSV...");
  const form = new FormData();
  form.append("file", file);

  fetch("/api/upload", { method: "POST", body: form })
    .then((r) => r.json().then((d) => ({ status: r.status, body: d })))
    .then(({ status, body }) => {
      if (status !== 200) throw new Error(body.error || "Upload failed");
      onUploadSuccess(body);
    })
    .catch((err) => {
      setStatus("error", "Upload failed");
      showUploadError(err.message);
    });
}

function loadSample(type = "cancer") {
  clearUploadError();
  setStatus("running", "Loading benchmark dataset...");
  fetch("/api/sample", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ type: type })
  })
    .then((r) => r.json().then((d) => ({ status: r.status, body: d })))
    .then(({ status, body }) => {
      if (status !== 200) throw new Error(body.error || "Sample load failed");
      onUploadSuccess(body);
    })
    .catch((err) => {
      setStatus("error", "Sample failed");
      showUploadError(err.message);
    });
}

// ---------------------------------------------------------------------------
// Run Launch & Real-time Polling
// ---------------------------------------------------------------------------

function launchRun() {
  const targetCol = el("targetSelect").value;
  const distMb = parseFloat(el("distMbInput").value) || 500;
  const distRows = parseInt(el("distRowsInput").value, 10) || 1000000;

  state.targetCol = targetCol;
  setStatus("running", "Agent Active");
  showView("running");
  el("runningRecords").innerHTML = "";

  fetch("/api/run", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      job_id: state.jobId,
      target_col: targetCol,
      distributed_threshold_mb: distMb,
      distributed_threshold_rows: distRows,
    }),
  })
    .then((r) => r.json())
    .then((data) => {
      if (data.error) throw new Error(data.error);
      startPolling();
    })
    .catch((err) => {
      setStatus("error", "Start failed");
      alert("Failed to start run: " + err.message);
    });
}

function startPolling() {
  if (state.pollTimer) clearInterval(state.pollTimer);
  state.pollTimer = setInterval(pollProgress, 1000);
}

function pollProgress() {
  fetch(`/api/progress/${state.jobId}`)
    .then((r) => r.json())
    .then((data) => {
      if (data.records) {
        renderRunningRecords(data.records);
        updateRailFromRecords(data.records);
      }

      if (data.status === "done") {
        clearInterval(state.pollTimer);
        setStatus("done", "Complete");
        renderResults(data.result);
        showView("results");
      } else if (data.status === "error") {
        clearInterval(state.pollTimer);
        setStatus("error", "Failed");
        alert("Pipeline run failed: " + (data.error || "Unknown error"));
      }
    })
    .catch((err) => {
      console.warn("Poll error:", err);
    });
}

function renderRunningRecords(records) {
  const stream = el("runningRecords");
  stream.innerHTML = "";
  records.forEach((r) => {
    const item = document.createElement("div");
    item.className = "record-item";
    item.innerHTML = `
      <div class="record-top">
        <span class="record-stage">${escapeHtml(r.stage)}</span>
        ${r.duration_sec ? `<span style="font-family:var(--font-mono);font-size:11px;color:var(--text-faint)">${r.duration_sec.toFixed(2)}s</span>` : ''}
      </div>
      <div class="record-decision">${escapeHtml(r.decision)}</div>
      <div class="record-reasoning">${escapeHtml(r.reasoning)}</div>
    `;
    stream.appendChild(item);
  });
  stream.scrollTop = stream.scrollHeight;
  const counter = el("streamCounter");
  if (counter) counter.textContent = `${records.length} stages recorded`;
}

// ---------------------------------------------------------------------------
// Render Results (Tabs Population & Cleaned CSV Wiring)
// ---------------------------------------------------------------------------

function renderResults(res) {
  state.cleanedData = res.cleaned_data;
  state.engineeredData = res.engineered_data;

  el("resTitle").textContent = `${res.profile.target_col} • Autonomous Pipeline Output`;
  el("resBackendBadge").textContent = `${res.profile.backend.toUpperCase()} BACKEND`;

  // Wire up all Download Buttons
  const cleanUrl = `/api/download/cleaned/${state.jobId}`;
  const engUrl = `/api/download/engineered/${state.jobId}`;
  const reportUrl = `/api/report/${state.jobId}`;

  // Top Action Bar
  const dlCleanBtn = el("downloadCleanedBtn");
  if (dlCleanBtn) dlCleanBtn.href = cleanUrl;

  const dlEngBtn = el("downloadEngBtn");
  if (dlEngBtn) dlEngBtn.href = engUrl;

  const dlRepBtn = el("downloadReportBtn");
  if (dlRepBtn) dlRepBtn.href = reportUrl;

  // Overview Callout Card
  const calloutCleanBtn = el("calloutCleanedBtn");
  if (calloutCleanBtn) calloutCleanBtn.href = cleanUrl;

  // Preprocessing Tab Buttons
  const tabCleanBtn = el("tabPrepCleanedDownload");
  if (tabCleanBtn) tabCleanBtn.href = cleanUrl;

  const tabEngBtn = el("tabPrepEngDownload");
  if (tabEngBtn) tabEngBtn.href = engUrl;

  // Wire up Download Results Section Buttons
  const csvExports = res.csv_exports || {};
  const zipUrl = `/download/${state.jobId}/csv/all`;

  const btnDlAll = el("btnDownloadAllCsvs");
  if (btnDlAll) btnDlAll.href = zipUrl;

  const btnReport = el("btnDlFullReport");
  if (btnReport) btnReport.href = reportUrl;

  const wireCsvBtn = (btnId, key, defaultFilename) => {
    const btn = el(btnId);
    if (!btn) return;
    const info = csvExports[key];
    if (info && info.exists) {
      btn.href = info.url || `/download/${state.jobId}/csv/${defaultFilename}`;
      btn.classList.remove("disabled");
      btn.removeAttribute("disabled");
      btn.hidden = false;
    } else {
      btn.classList.add("disabled");
      btn.setAttribute("disabled", "true");
      btn.href = "#";
      if (key === "anomalies") {
        btn.hidden = true;
      }
    }
  };

  wireCsvBtn("btnDlModelResults", "model_comparison", "model_comparison.csv");
  wireCsvBtn("btnDlPredictions", "predictions", "predictions.csv");
  wireCsvBtn("btnDlFeatureImportance", "feature_importance", "feature_importance.csv");
  wireCsvBtn("btnDlDatasetProfile", "dataset_profile", "dataset_profile.csv");
  wireCsvBtn("btnDlPreprocessingLog", "preprocessing_log", "preprocessing_log.csv");
  wireCsvBtn("btnDlDecisionLog", "decision_log", "decision_log.csv");
  wireCsvBtn("btnDlAnomalies", "anomalies", "anomalies.csv");

  // 1. Overview Tab
  renderOverview(res);

  // 2. Preprocessing & Cleaned Data Tab
  renderCleanedDataAndPrep(res);

  // 3. EDA Tab
  renderEDA(res.eda);

  // 4. Models Tab
  renderModels(res.tuning_results, res.final_model_name, res.scores);

  // 5. Evaluation Tab
  renderEvaluation(res.metrics_detail, res.final_model_name, res.eval_plots, res.explainability);

  // 6. What-If Form
  setupWhatIf(res.input_columns, res.sample_record);

  // 7. Initial Assistant Context
  initAssistant();

  // 8. Fetch Registry
  fetchRegistry();

  // 9. Setup Drift Tab
  setupDrift();

  // 10. Setup Lightbox
  setupLightbox();
}

function renderOverview(res) {
  const bestScore = res.scores ? res.scores[res.final_model_name] : "N/A";
  const scoreLabel = res.profile.task_type === "classification" ? "Test Accuracy" : "Test R²";
  const cleanStats = res.cleaned_data?.stats || {};

  el("overviewMetrics").innerHTML = `
    <div class="metric-card" style="border-top-color:var(--emerald);">
      <div class="m-label">Selected Best Model</div>
      <div class="m-val" style="color:var(--emerald-light); font-size:22px;">${res.final_model_name}</div>
      <div class="m-sub">Tuned via Optuna Bayesian Search</div>
    </div>
    <div class="metric-card" style="border-top-color:var(--cyan);">
      <div class="m-label">Held-out ${scoreLabel}</div>
      <div class="m-val" style="color:var(--cyan-light);">${typeof bestScore === 'number' ? bestScore.toFixed(4) : bestScore}</div>
      <div class="m-sub">Tested on unseen partition</div>
    </div>
    <div class="metric-card" style="border-top-color:var(--purple);">
      <div class="m-label">Cleaned Dataset Export</div>
      <div class="m-val">${(cleanStats.n_rows || res.profile.n_rows).toLocaleString()}</div>
      <div class="m-sub">${cleanStats.n_cols || res.profile.n_cols} columns • Ready to Download</div>
    </div>
    <div class="metric-card" style="border-top-color:var(--amber);">
      <div class="m-label">Task & Backend</div>
      <div class="m-val" style="text-transform:capitalize; font-size:20px;">${res.profile.task_type}</div>
      <div class="m-sub">${res.profile.backend} execution routing</div>
    </div>
  `;

  // Update Callout Text
  const calloutDesc = el("cleanedCalloutDesc");
  if (calloutDesc && cleanStats.n_rows) {
    calloutDesc.textContent = `Cleaned dataset with ${cleanStats.n_rows.toLocaleString()} rows and ${cleanStats.n_cols} sanitized columns ready for immediate download.`;
  }

  // Leaderboard table
  let lbHtml = `<table><thead><tr><th>Rank</th><th>Candidate Model</th><th>Test Score</th><th>Optuna Best CV</th><th>Trials</th><th>Status</th></tr></thead><tbody>`;
  const sortedModels = Object.keys(res.scores || {}).sort((a, b) => (res.scores[b] || 0) - (res.scores[a] || 0));
  sortedModels.forEach((m, idx) => {
    const isWinner = m === res.final_model_name;
    const tuning = res.tuning_results[m] || {};
    lbHtml += `<tr style="${isWinner ? 'background:rgba(16,185,129,0.08); font-weight:600;' : ''}">
      <td>${idx === 0 ? '🥇' : idx === 1 ? '🥈' : idx === 2 ? '🥉' : `#${idx + 1}`}</td>
      <td><code>${m}</code></td>
      <td><strong style="color:${isWinner ? 'var(--emerald-light)' : 'inherit'}">${res.scores[m] ? res.scores[m].toFixed(4) : 'N/A'}</strong></td>
      <td>${tuning.best_score ? tuning.best_score.toFixed(4) : 'N/A'}</td>
      <td>${tuning.n_trials || '—'}</td>
      <td><span class="badge" style="color:${isWinner ? 'var(--emerald-light)' : 'var(--text-dim)'}">${isWinner ? 'Winner 🏆' : 'Candidate'}</span></td>
    </tr>`;
  });
  lbHtml += `</tbody></table>`;
  el("overviewLeaderboardTable").innerHTML = lbHtml;

  // Copy records stream to overview feed
  el("overviewAuditFeed").innerHTML = el("runningRecords").innerHTML;
}

function renderCleanedDataAndPrep(res) {
  const clean = res.cleaned_data || {};
  const stats = clean.stats || {};
  const prep = res.preprocessing || {};
  const anomaly = res.anomaly;
  const imbalance = res.imbalance;

  // Render Cleaned Data Stats Pills
  const pillsWrap = el("cleanedStatsPills");
  if (pillsWrap && stats.n_rows) {
    pillsWrap.innerHTML = `
      <div class="cleaned-stat-pill">
        <span>Rows:</span> <strong>${stats.n_rows.toLocaleString()}</strong>
      </div>
      <div class="cleaned-stat-pill">
        <span>Columns:</span> <strong>${stats.n_cols}</strong>
      </div>
      <div class="cleaned-stat-pill">
        <span>Missing Values:</span> <strong>0 (100% Imputed)</strong>
      </div>
      <div class="cleaned-stat-pill">
        <span>Memory Size:</span> <strong>${stats.size_kb} KB</strong>
      </div>
      <div class="cleaned-stat-pill">
        <span>Format:</span> <strong>Standard CSV (.csv)</strong>
      </div>
    `;
  }

  // Render Cleaned Data Preview Table
  renderPreviewTable("cleanedTableWrap", clean.preview || []);

  // Preprocessing Explanations
  const list = el("prepExplanationsList");
  list.innerHTML = "";
  (prep.explanations || []).forEach((exp) => {
    const li = document.createElement("li");
    li.textContent = exp;
    list.appendChild(li);
  });

  // Anomaly Summary
  let anomHtml = "";
  if (anomaly) {
    anomHtml = `
      <p>Isolation Forest flagged <strong>${(anomaly.if_pct || 0).toFixed(2)}%</strong> of records as anomalous outliers.</p>
      <p>Local Outlier Factor (LOF) flagged <strong>${(anomaly.lof_pct || 0).toFixed(2)}%</strong> of records as density outliers.</p>
      ${anomaly.top_indices && anomaly.top_indices.length ? `<p class="form-hint">Top flagged outlier row indices: <code>${anomaly.top_indices.slice(0, 10).join(', ')}</code></p>` : ''}
    `;
  } else {
    anomHtml = `<p class="muted">Anomaly detection completed. No critical anomalies flagged.</p>`;
  }
  el("anomalySummaryContent").innerHTML = anomHtml;

  // Imbalance Summary
  let imbHtml = "";
  if (imbalance) {
    imbHtml = `
      <p>Status: <strong>${imbalance.is_imbalanced ? '⚠️ Class Imbalance Detected' : '✓ Balanced Distribution'}</strong></p>
      <p>Minority class ratio: <strong>${(imbalance.minority_ratio * 100).toFixed(1)}%</strong></p>
      ${imbalance.is_imbalanced ? `<p>Optimal Strategy Selected: <strong><code>${imbalance.best_strategy}</code></strong></p>` : ''}
    `;
    if (imbalance.strategy_comparison && Object.keys(imbalance.strategy_comparison).length > 0) {
      imbHtml += `<table><thead><tr><th>Strategy</th><th>Macro F1</th><th>Recall</th></tr></thead><tbody>`;
      for (const [st, m] of Object.entries(imbalance.strategy_comparison)) {
        const isBest = st === imbalance.best_strategy;
        imbHtml += `<tr style="${isBest ? 'background:rgba(16,185,129,0.08);font-weight:600;' : ''}">
          <td><code>${st}</code> ${isBest ? '⭐' : ''}</td>
          <td>${(m.macro_f1 || 0).toFixed(4)}</td>
          <td>${(m.recall || 0).toFixed(4)}</td>
        </tr>`;
      }
      imbHtml += `</tbody></table>`;
    }
  } else {
    imbHtml = `<p class="muted">Not applicable (Regression task).</p>`;
  }
  el("imbalanceSummaryContent").innerHTML = imbHtml;
}

function renderEDA(eda) {
  if (!eda) return;
  const ext = eda.extended_profile || {};
  const outliers = eda.outlier_summary || {};

  let sumHtml = `<div>
    <h4 class="subheading">Dataset Characteristics</h4>
    <ul class="clean-list">
      <li>Numerical Features: <strong>${ext.n_numeric || 0}</strong></li>
      <li>Categorical Features: <strong>${ext.n_categorical || 0}</strong></li>
      <li>Duplicate Records Detected: <strong>${ext.n_duplicates || 0}</strong></li>
      <li>Constant / Zero-Variance Signals: <strong>${ext.n_constant || 0}</strong></li>
      <li>Inferred Target Subtype: <strong>${ext.target_subtype || 'Standard'}</strong></li>
    </ul>
  </div>
  <div>
    <h4 class="subheading">IQR Outlier Distribution</h4>`;

  if (Object.keys(outliers).length > 0) {
    sumHtml += `<table><thead><tr><th>Feature</th><th>Outlier Count</th><th>% of Total</th></tr></thead><tbody>`;
    for (const [col, info] of Object.entries(outliers).slice(0, 6)) {
      sumHtml += `<tr><td><code>${col}</code></td><td>${info.n_outliers}</td><td>${info.pct.toFixed(1)}%</td></tr>`;
    }
    sumHtml += `</tbody></table>`;
  } else {
    sumHtml += `<p class="muted">No significant univariate outliers flagged by IQR.</p>`;
  }
  sumHtml += `</div>`;
  el("edaSummaryGrid").innerHTML = sumHtml;

  // Diagnostic Plots gallery
  let plotsHtml = "";
  for (const [name, path] of Object.entries(eda.plots || {})) {
    if (!path) continue;
    plotsHtml += `
      <div class="plot-card" data-plot="${path}" data-title="${name.replace(/_/g, ' ')}">
        <div class="plot-header">
          <div class="plot-title">${name.replace(/_/g, " ")}</div>
          <span class="plot-zoom-hint">🔍 Click to expand</span>
        </div>
        <img src="${path}" alt="${name}" loading="lazy">
      </div>
    `;
  }
  el("edaPlotGallery").innerHTML = plotsHtml || `<p class="muted">No EDA plots generated.</p>`;
}

function renderModels(tuning, finalModel, scores) {
  let html = `<table><thead><tr><th>Model Architecture</th><th>Optuna CV Score</th><th>Trials Evaluated</th><th>Best Parameter Configuration</th></tr></thead><tbody>`;
  for (const [m, res] of Object.entries(tuning || {})) {
    const isWinner = m === finalModel;
    html += `<tr style="${isWinner ? 'background:rgba(16,185,129,0.08); font-weight:600;' : ''}">
      <td><code>${m}</code> ${isWinner ? '🏆 (Final Winner)' : ''}</td>
      <td><strong style="color:${isWinner ? 'var(--emerald-light)' : 'inherit'}">${res.best_score ? res.best_score.toFixed(4) : 'N/A'}</strong></td>
      <td>${res.n_trials}</td>
      <td><pre style="margin:0;font-size:11.5px;color:var(--text-dim);">${JSON.stringify(res.best_params, null, 1)}</pre></td>
    </tr>`;
  }
  html += `</tbody></table>`;
  el("tuningDetailTable").innerHTML = html;
}

function renderEvaluation(metricsDetail, finalModel, evalPlots, explain) {
  // Metrics Table
  let html = `<table><thead><tr><th>Model Candidate</th><th>Held-out Test Performance Metrics</th></tr></thead><tbody>`;
  for (const [m, mObj] of Object.entries(metricsDetail || {})) {
    const isWinner = m === finalModel;
    const mStr = Object.entries(mObj).map(([k, v]) => `<strong>${k}:</strong> <span style="color:var(--text);">${typeof v === 'number' ? v.toFixed(4) : v}</span>`).join(" &nbsp;•&nbsp; ");
    html += `<tr style="${isWinner ? 'background:rgba(16,185,129,0.08);' : ''}">
      <td><code>${m}</code> ${isWinner ? '<strong>🏆 (Winner)</strong>' : ''}</td>
      <td>${mStr}</td>
    </tr>`;
  }
  html += `</tbody></table>`;
  el("evalMetricsTable").innerHTML = html;

  // Evaluation Diagnostic Plots
  let plotsHtml = "";
  for (const [name, path] of Object.entries(evalPlots || {})) {
    if (!path) continue;
    plotsHtml += `
      <div class="plot-card" data-plot="${path}" data-title="${name.replace(/_/g, ' ')}">
        <div class="plot-header">
          <div class="plot-title">${name.replace(/_/g, " ")}</div>
          <span class="plot-zoom-hint">🔍 Expand</span>
        </div>
        <img src="${path}" alt="${name}" loading="lazy">
      </div>
    `;
  }
  el("evalPlotGallery").innerHTML = plotsHtml || `<p class="muted">No evaluation plots generated.</p>`;

  // SHAP ranked table
  let shapHtml = `<table><thead><tr><th>Rank</th><th>Feature Name</th><th>Mean |SHAP| Attribution</th></tr></thead><tbody>`;
  (explain.ranked_importance || []).forEach(([f, imp], idx) => {
    shapHtml += `<tr>
      <td>#${idx + 1}</td>
      <td><code>${f}</code></td>
      <td><strong>${imp.toFixed(4)}</strong></td>
    </tr>`;
  });
  shapHtml += `</tbody></table>`;
  el("shapRankedTable").innerHTML = shapHtml;

  // SHAP plots
  let expPlotsHtml = "";
  for (const [name, path] of Object.entries(explain.plots || {})) {
    if (!path) continue;
    expPlotsHtml += `
      <div class="plot-card" data-plot="${path}" data-title="${name.replace(/_/g, ' ')}">
        <div class="plot-header">
          <div class="plot-title">${name.replace(/_/g, " ")}</div>
          <span class="plot-zoom-hint">🔍 Expand</span>
        </div>
        <img src="${path}" alt="${name}" loading="lazy">
      </div>
    `;
  }
  el("shapPlotGallery").innerHTML = expPlotsHtml;
}

// ---------------------------------------------------------------------------
// Lightbox Modal for Full-Res Diagnostic Plots
// ---------------------------------------------------------------------------

function setupLightbox() {
  document.addEventListener("click", (e) => {
    const card = e.target.closest(".plot-card");
    if (card) {
      const src = card.dataset.plot;
      const title = card.dataset.title || "Diagnostic Plot";
      if (src) openLightbox(src, title);
    }
  });

  const modal = el("plotLightbox");
  const closeBtn = el("lightboxCloseBtn");
  const backdrop = el("lightboxBackdrop");

  if (closeBtn) closeBtn.onclick = closeLightbox;
  if (backdrop) backdrop.onclick = closeLightbox;

  document.addEventListener("keydown", (e) => {
    if (e.key === "Escape" && modal && !modal.hidden) {
      closeLightbox();
    }
  });

  // Ensure closed on initialization
  closeLightbox();
}

function openLightbox(imgSrc, title) {
  const modal = el("plotLightbox");
  if (!modal) return;
  const img = el("lightboxImg");
  const ttl = el("lightboxTitle");
  if (img) img.src = imgSrc;
  if (ttl) ttl.textContent = title;
  modal.hidden = false;
  modal.style.display = "flex";
}

function closeLightbox() {
  const modal = el("plotLightbox");
  if (modal) {
    modal.hidden = true;
    modal.style.display = "none";
  }
}

// ---------------------------------------------------------------------------
// What-If Counterfactual Analysis
// ---------------------------------------------------------------------------

function setupWhatIf(inputColumns, sampleRecord) {
  state.inputColumns = inputColumns || [];
  state.sampleRecord = sampleRecord || {};

  const form = el("whatifForm");
  form.innerHTML = "";

  state.inputColumns.forEach((col) => {
    const row = document.createElement("div");
    row.className = "form-row";
    row.innerHTML = `
      <label for="wi_${col.name}">${escapeHtml(col.name)}</label>
      <input class="input-styled" id="wi_${col.name}" name="${col.name}" type="${col.type}" value="${col.default}" step="${col.step || 'any'}">
    `;
    form.appendChild(row);
  });

  el("whatifPredictBtn").onclick = runWhatIfPredict;
  el("whatifResetBtn").onclick = () => {
    setupWhatIf(state.inputColumns, state.sampleRecord);
    el("whatifCompareBox").hidden = true;
  };
}

function getWhatIfRecord() {
  const record = {};
  state.inputColumns.forEach((col) => {
    const input = el(`wi_${col.name}`);
    if (input) {
      record[col.name] = col.type === "number" ? parseFloat(input.value) : input.value;
    }
  });
  return record;
}

function runWhatIfPredict() {
  const currentRecord = getWhatIfRecord();
  el("whatifResultBox").innerHTML = `<div class="spinner-radar" style="margin:20px auto;"><div class="radar-ping"></div><div class="radar-core"></div></div>`;

  fetch("/api/whatif/predict", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ job_id: state.jobId, record: currentRecord }),
  })
    .then((r) => r.json())
    .then((res) => {
      if (res.error) throw new Error(res.error);

      let probHtml = "";
      if (res.probability) {
        probHtml = `<div class="prob-bar-container">
          <div style="font-size:12px;font-weight:600;margin-bottom:8px;">Predicted Class Probabilities:</div>`;
        for (const [cls, p] of Object.entries(res.probability)) {
          const pct = (p * 100).toFixed(1);
          probHtml += `
            <div class="prob-bar-row">
              <div class="prob-bar-label">
                <span>Class ${cls}</span>
                <span>${pct}%</span>
              </div>
              <div class="prob-bar-track">
                <div class="prob-bar-fill" style="width:${pct}%"></div>
              </div>
            </div>
          `;
        }
        probHtml += `</div>`;
      }

      let shapHtml = "";
      if (res.shap_contributors && res.shap_contributors.length) {
        shapHtml = `<div class="mt-3">
          <div style="font-size:12px;font-weight:600;margin-bottom:6px;">Top Contributing Features (SHAP Local Attribution):</div>
          <ul class="clean-list mt-2">`;
        res.shap_contributors.forEach((sc) => {
          shapHtml += `<li><code>${sc.feature}</code>: <span style="font-weight:600; color:${sc.direction === 'positive' ? 'var(--emerald-light)' : 'var(--error)'}">${sc.direction === 'positive' ? '+' : ''}${sc.shap_value}</span></li>`;
        });
        shapHtml += `</ul></div>`;
      }

      el("whatifResultBox").innerHTML = `
        <div class="m-label">Simulated Prediction</div>
        <div class="whatif-pred-val">${res.prediction}</div>
        ${probHtml}
        ${shapHtml}
      `;

      // Compare against baseline
      if (!state.baselinePrediction) {
        state.baselinePrediction = res;
      } else {
        runWhatIfCompare(state.sampleRecord, currentRecord);
      }
    })
    .catch((err) => {
      el("whatifResultBox").innerHTML = `<div class="upload-error">${err.message}</div>`;
    });
}

function runWhatIfCompare(original, modified) {
  fetch("/api/whatif/compare", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ job_id: state.jobId, original: original, modified: modified }),
  })
    .then((r) => r.json())
    .then((res) => {
      if (res.error) return;
      const compBox = el("whatifCompareBox");
      compBox.hidden = false;

      const changed = res.changed_features || [];
      el("compareDetails").innerHTML = `
        <p><strong>Baseline Prediction:</strong> <code>${res.original_prediction?.prediction}</code> &nbsp;→&nbsp; <strong>Modified:</strong> <code>${res.modified_prediction?.prediction}</code></p>
        <p><strong>Modified Inputs (${changed.length}):</strong> ${changed.map((c) => `<span class="badge" style="color:var(--cyan-light);">${c}</span>`).join(' ') || 'None'}</p>
      `;
    });
}

// ---------------------------------------------------------------------------
// AI Data Scientist Assistant
// ---------------------------------------------------------------------------

function initAssistant() {
  document.querySelectorAll(".qq-chip").forEach((chip) => {
    chip.onclick = () => {
      const q = chip.dataset.q;
      el("chatInput").value = q;
      submitAssistantQuestion(q);
    };
  });

  el("chatForm").onsubmit = (e) => {
    e.preventDefault();
    const q = el("chatInput").value.trim();
    if (!q) return;
    el("chatInput").value = "";
    submitAssistantQuestion(q);
  };
}

function submitAssistantQuestion(question) {
  const container = el("chatMessages");

  // User Message
  const userMsg = document.createElement("div");
  userMsg.className = "chat-msg user";
  userMsg.innerHTML = `<div class="msg-bubble">${escapeHtml(question)}</div>`;
  container.appendChild(userMsg);

  // Bot Placeholder
  const botMsg = document.createElement("div");
  botMsg.className = "chat-msg bot";
  botMsg.innerHTML = `
    <div class="msg-avatar">🤖</div>
    <div class="msg-bubble"><span class="spinner-radar" style="width:18px;height:18px;display:inline-block;vertical-align:middle;margin-right:6px;"><div class="radar-ping"></div><div class="radar-core"></div></span> Analyzing run metrics & context...</div>
  `;
  container.appendChild(botMsg);
  container.scrollTop = container.scrollHeight;

  fetch("/api/assistant/ask", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ job_id: state.jobId, question: question }),
  })
    .then((r) => r.json())
    .then((data) => {
      botMsg.querySelector(".msg-bubble").innerHTML = data.answer || "I could not find an answer in the run context.";
      container.scrollTop = container.scrollHeight;
    })
    .catch((err) => {
      botMsg.querySelector(".msg-bubble").textContent = "Error: " + err.message;
      container.scrollTop = container.scrollHeight;
    });
}

function escapeHtml(str) {
  if (str === null || str === undefined) return "";
  return String(str).replace(/[&<>'"]/g, (tag) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', "'": '&#39;', '"': '&quot;' }[tag] || tag));
}

// ---------------------------------------------------------------------------
// Model Registry
// ---------------------------------------------------------------------------

function fetchRegistry() {
  fetch("/api/registry/list")
    .then((r) => r.json())
    .then((data) => {
      const models = data.models || [];
      const wrap = el("registryTableWrap");
      if (!models.length) {
        wrap.innerHTML = `<div class="empty-state" style="padding:16px;">No registered models found in registry store.</div>`;
        return;
      }

      let html = `<table><thead><tr><th>Version</th><th>Model Type</th><th>Dataset</th><th>Task Type</th><th>Held-out Metrics</th><th>Date Saved</th></tr></thead><tbody>`;
      models.forEach((m) => {
        const mStr = Object.entries(m.metrics || {}).map(([k, v]) => `${k}=${typeof v === 'number' ? v.toFixed(3) : v}`).join(", ");
        html += `<tr>
          <td><span class="badge" style="color:var(--emerald-light);">v${m.version}</span></td>
          <td><strong>${m.model_type}</strong></td>
          <td>${m.dataset_name}</td>
          <td><span class="badge">${m.task_type}</span></td>
          <td>${mStr}</td>
          <td style="font-family:var(--font-mono);font-size:12px;color:var(--text-dim);">${m.training_date}</td>
        </tr>`;
      });
      html += `</tbody></table>`;
      wrap.innerHTML = html;
    });
}

// ---------------------------------------------------------------------------
// Data Drift Monitoring & Auto-Retraining
// ---------------------------------------------------------------------------

function setupDrift() {
  el("driftCheckBtn").onclick = () => {
    const file = el("driftFileInput").files[0];
    if (!file) {
      alert("Please select a new CSV file to check for covariate drift.");
      return;
    }

    const form = new FormData();
    form.append("job_id", state.jobId);
    form.append("file", file);

    el("driftCheckBtn").disabled = true;
    el("driftCheckBtn").textContent = "Analyzing Statistical Drift...";

    fetch("/api/drift/check", { method: "POST", body: form })
      .then((r) => r.json())
      .then((res) => {
        el("driftCheckBtn").disabled = false;
        el("driftCheckBtn").textContent = "Run Statistical Drift Analysis";
        renderDriftResults(res);
      })
      .catch((err) => {
        el("driftCheckBtn").disabled = false;
        el("driftCheckBtn").textContent = "Run Statistical Drift Analysis";
        alert("Drift analysis failed: " + err.message);
      });
  };

  el("retrainBtn").onclick = () => {
    el("retrainBtn").disabled = true;
    el("retrainBtn").textContent = "Retraining in progress...";
    el("retrainLog").hidden = false;
    el("retrainLog").textContent = "Fitting candidate model on combined reference + drifted data with performance tolerance gating...";

    fetch("/api/drift/retrain", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ job_id: state.jobId }),
    })
      .then((r) => r.json())
      .then((res) => {
        el("retrainBtn").disabled = false;
        el("retrainBtn").textContent = "Trigger Auto-Retraining";
        if (res.accepted) {
          el("retrainLog").innerHTML = `<span style="color:var(--emerald-light); font-weight:bold;">✓ Model Retraining Accepted & Registered!</span> Version: <strong>${res.new_version}</strong> (Old Score: ${res.old_score?.toFixed(4)} → New Score: ${res.new_score?.toFixed(4)})`;
        } else {
          el("retrainLog").innerHTML = `<span style="color:var(--amber); font-weight:bold;">⚠️ Retraining Rejected:</span> ${res.reason || 'Candidate did not meet tolerance criteria.'}`;
        }
      })
      .catch((err) => {
        el("retrainBtn").disabled = false;
        el("retrainBtn").textContent = "Trigger Auto-Retraining";
        el("retrainLog").textContent = "Retraining error: " + err.message;
      });
  };
}

function renderDriftResults(res) {
  const box = el("driftResultsBox");
  box.hidden = false;

  const isDrifted = res.retrain_recommended;
  el("driftSummaryBadge").innerHTML = isDrifted ?
    `<div class="drift-badge drifted">⚠️ Significant Covariate Drift Detected (${res.n_drifted_features}/${res.total_features} features drifted, Mean PSI: ${res.overall_psi})</div>` :
    `<div class="drift-badge ok">✓ Data Distribution Stable (${res.n_drifted_features}/${res.total_features} features, Mean PSI: ${res.overall_psi})</div>`;

  let html = `<table><thead><tr><th>Feature</th><th>Statistical Test</th><th>Test Statistic</th><th>p-value / PSI</th><th>Drift Status</th></tr></thead><tbody>`;
  for (const [col, info] of Object.entries(res.feature_drift || {})) {
    const isDrift = info.drifted;
    html += `<tr style="${isDrift ? 'background:rgba(245,158,11,0.06);' : ''}">
      <td><code>${col}</code></td>
      <td>${info.test || 'KS-test'}</td>
      <td>${info.ks_stat !== undefined ? `KS=${info.ks_stat.toFixed(3)}` : `Chi²=${info.chi2_stat?.toFixed(2) || '—'}`}</td>
      <td>PSI: ${info.psi || '0.00'} (p=${info.p_value !== undefined ? info.p_value.toFixed(4) : '—'})</td>
      <td><span class="badge" style="color:${isDrift ? 'var(--amber)' : 'var(--emerald-light)'}">${isDrift ? 'DRIFTED' : 'STABLE'}</span></td>
    </tr>`;
  }
  html += `</tbody></table>`;
  el("driftFeaturesTableWrap").innerHTML = html;

  el("retrainSection").hidden = !isDrifted;
}

// ---------------------------------------------------------------------------
// Init & Event Listeners
// ---------------------------------------------------------------------------

document.addEventListener("DOMContentLoaded", () => {
  setupTabs();

  // Dropzone setup
  const dz = el("dropzone");
  const fileInput = el("fileInput");

  if (dz && fileInput) {
    dz.onclick = () => fileInput.click();
    fileInput.onchange = (e) => {
      if (e.target.files.length) uploadFile(e.target.files[0]);
    };

    dz.ondragover = (e) => {
      e.preventDefault();
      dz.classList.add("dragover");
    };
    dz.ondragleave = () => dz.classList.remove("dragover");
    dz.ondrop = (e) => {
      e.preventDefault();
      dz.classList.remove("dragover");
      if (e.dataTransfer.files.length) uploadFile(e.dataTransfer.files[0]);
    };
  }

  // Sample cards click setup
  document.querySelectorAll(".sample-card").forEach((card) => {
    card.onclick = () => {
      const sampleType = card.dataset.sample || "cancer";
      loadSample(sampleType);
    };
  });

  const launchBtn = el("launchBtn");
  if (launchBtn) launchBtn.onclick = launchRun;

  const resetAll = () => {
    if (state.pollTimer) clearInterval(state.pollTimer);
    showView("upload");
    setStatus("idle", "System Ready");
    const badge = el("railProgressBadge");
    if (badge) badge.textContent = "0/13";
    document.querySelectorAll(".rail-steps li").forEach((li) => li.classList.remove("active", "done"));
  };

  const resetBtn = el("resetBtn");
  if (resetBtn) resetBtn.onclick = resetAll;

  const topReset = el("topbarResetBtn");
  if (topReset) topReset.onclick = resetAll;

  const confCancel = el("confCancelBtn");
  if (confCancel) confCancel.onclick = resetAll;
});
