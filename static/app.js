/**
 * Jev System One — Modern Corporate Startup Client Controller
 * Features:
 * - Dual-Mode Workflow Creation: Interactive Visual Builder + JSON Schema Sync
 * - Live Native CUDA Hardware Telemetry & Calibrated Logits Inspection
 * - Rich Decision Primitive Visualizers (Noul gauges, Choice logit distributions, Score meters)
 * - 1-Click Code Exporter (cURL & Python SDK)
 * - Global Keyboard Shortcuts (⌘+Enter / Ctrl+Enter)
 */

const state = {
  examples: [],
  activeExample: null,
  workflow: [],
  lastResponse: null,
  activeTab: "visual",
  activeResultTab: "decisions",
};

const $ = (id) => document.getElementById(id);

// Elements
const contextInput = $("context-input");
const workflowInput = $("workflow-input");
const exampleSelect = $("example-select");
const modeSelect = $("mode-select");
const modelInput = $("model-input");
const temperatureInput = $("temperature-input");
const tempVal = $("temp-val");
const runButton = $("run-button");
const visualStepsList = $("visual-steps-list");
const bundledNativeModel = "models\\Qwen3-1.7B-Q8_0.gguf";

/* ==========================================================================
   Helper Utilities
   ========================================================================== */

function escapeHtml(value) {
  return String(value ?? "").replace(/[&<>'"]/g, (char) => ({
    "&": "&amp;", "<": "&lt;", ">": "&gt;", "'": "&#39;", '"': "&quot;"
  }[char]));
}

function pretty(value) {
  if (typeof value === "boolean") return value ? "true" : "false";
  if (value === null || value === undefined) return "—";
  return typeof value === "string" ? value : JSON.stringify(value);
}

function updateCharAndTokenCount() {
  const chars = contextInput.value.length;
  $("char-count").textContent = chars.toLocaleString();
  const approxTokens = Math.max(1, Math.round(chars / 4));
  $("token-estimate").textContent = chars ? `~${approxTokens.toLocaleString()} tokens` : "~0 tokens";
}

function setRunStatus(message, tone = "") {
  const target = $("run-message");
  target.textContent = message;
  target.className = `run-status-text ${tone}`;
}

/* ==========================================================================
   Visual Workflow Builder <-> JSON Bidirectional Sync
   ========================================================================== */

function syncWorkflowToVisual() {
  try {
    const parsed = JSON.parse(workflowInput.value);
    if (Array.isArray(parsed)) {
      state.workflow = parsed;
      renderVisualSteps();
    }
  } catch (_) {}
}

function syncVisualToWorkflow() {
  workflowInput.value = JSON.stringify(state.workflow, null, 2);
  updateExportSnippets();
}

function renderVisualSteps() {
  if (!state.workflow.length) {
    visualStepsList.innerHTML = `
      <div style="padding: 24px; text-align: center; color: var(--text-muted); font-size: 12px;">
        No decision steps configured. Click "+ Add Decision Step" below.
      </div>
    `;
    return;
  }

  visualStepsList.innerHTML = state.workflow.map((step, index) => {
    const kind = step.kind || "noul";
    const isChoice = kind === "choice";
    const isScore = kind === "score";
    const optionsStr = Array.isArray(step.options) ? step.options.join(", ") : "";

    return `
      <div class="visual-step-card" data-index="${index}">
        <div class="step-card-header">
          <div class="step-card-meta">
            <span class="step-order-badge">0${index + 1}</span>
            <input class="step-id-input" type="text" value="${escapeHtml(step.id || `step_${index + 1}`)}" 
                   placeholder="step_id" data-field="id" title="Unique identifier for this decision step">
            <select class="step-kind-select" data-field="kind">
              <option value="noul" ${kind === "noul" ? "selected" : ""}>Noul (Boolean)</option>
              <option value="choice" ${kind === "choice" ? "selected" : ""}>Choice (Categorical)</option>
              <option value="score" ${kind === "score" ? "selected" : ""}>Score (Metric)</option>
              <option value="text" ${kind === "text" ? "selected" : ""}>Text (Generation)</option>
            </select>
          </div>
          <div class="step-actions">
            ${index > 0 ? `<button class="step-btn-icon move-up-btn" type="button" title="Move Up">↑</button>` : ""}
            ${index < state.workflow.length - 1 ? `<button class="step-btn-icon move-down-btn" type="button" title="Move Down">↓</button>` : ""}
            <button class="step-btn-icon delete-step-btn" type="button" title="Delete Step">
              <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M3 6h18M19 6v14a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2V6M8 6V4a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2"/></svg>
            </button>
          </div>
        </div>

        <input class="step-prompt-input" type="text" value="${escapeHtml(step.prompt || "")}" 
               placeholder="Question or evaluation rubric for Jev..." data-field="prompt">

        <div class="step-extra-row">
          <input class="step-when-input" type="text" value="${escapeHtml(step.when || "")}" 
                 placeholder='Condition (e.g. "urgent == true")' data-field="when">

          ${isChoice ? `
            <input class="step-options-input" type="text" value="${escapeHtml(optionsStr)}" 
                   placeholder="Options (comma-separated, e.g. billing, technical, sales)" data-field="options">
          ` : ""}

          ${isScore ? `
            <div style="display: flex; align-items: center; gap: 6px;">
              <span style="font-size: 10px; color: var(--text-muted);">Min:</span>
              <input class="step-bounds-input" type="number" value="${step.min ?? 0}" data-field="min">
              <span style="font-size: 10px; color: var(--text-muted);">Max:</span>
              <input class="step-bounds-input" type="number" value="${step.max ?? 10}" data-field="max">
            </div>
          ` : ""}
        </div>
      </div>
    `;
  }).join("");

  attachVisualStepListeners();
}

function attachVisualStepListeners() {
  visualStepsList.querySelectorAll(".visual-step-card").forEach((card) => {
    const index = parseInt(card.dataset.index, 10);

    card.querySelectorAll("input, select").forEach((input) => {
      input.addEventListener("input", (e) => {
        const field = e.target.dataset.field;
        let val = e.target.value;
        if (field === "options") {
          val = val.split(",").map((s) => s.trim()).filter(Boolean);
        } else if (field === "min" || field === "max") {
          val = Number(val);
        }
        state.workflow[index][field] = val;
        syncVisualToWorkflow();
        if (field === "kind") renderVisualSteps();
      });
    });

    const deleteBtn = card.querySelector(".delete-step-btn");
    if (deleteBtn) {
      deleteBtn.addEventListener("click", () => {
        state.workflow.splice(index, 1);
        renderVisualSteps();
        syncVisualToWorkflow();
      });
    }

    const moveUpBtn = card.querySelector(".move-up-btn");
    if (moveUpBtn) {
      moveUpBtn.addEventListener("click", () => {
        if (index > 0) {
          const temp = state.workflow[index];
          state.workflow[index] = state.workflow[index - 1];
          state.workflow[index - 1] = temp;
          renderVisualSteps();
          syncVisualToWorkflow();
        }
      });
    }

    const moveDownBtn = card.querySelector(".move-down-btn");
    if (moveDownBtn) {
      moveDownBtn.addEventListener("click", () => {
        if (index < state.workflow.length - 1) {
          const temp = state.workflow[index];
          state.workflow[index] = state.workflow[index + 1];
          state.workflow[index + 1] = temp;
          renderVisualSteps();
          syncVisualToWorkflow();
        }
      });
    }
  });
}

function addNewStep(kind = "noul") {
  const count = state.workflow.length + 1;
  const newStep = {
    id: `step_${count}`,
    kind,
    prompt: kind === "noul" ? "Does this condition hold true?" : "Evaluate this input:",
  };
  if (kind === "choice") newStep.options = ["option_a", "option_b", "option_c"];
  if (kind === "score") { newStep.min = 1; newStep.max = 10; }

  state.workflow.push(newStep);
  renderVisualSteps();
  syncVisualToWorkflow();

  // Scroll to bottom of visual steps
  visualStepsList.scrollTop = visualStepsList.scrollHeight;
}

/* ==========================================================================
   Example Selection & Preset Loading
   ========================================================================== */

function selectExample(example) {
  state.activeExample = example;
  contextInput.value = example.context;
  state.workflow = JSON.parse(JSON.stringify(example.workflow));
  workflowInput.value = JSON.stringify(example.workflow, null, 2);
  $("example-description").textContent = example.description;

  updateCharAndTokenCount();
  renderVisualSteps();
  clearResults();
  setRunStatus("Ready to evaluate.");
  updateExportSnippets();
}

function clearResults() {
  state.lastResponse = null;
  $("result-mode").textContent = "IDLE";
  $("result-mode").className = "runtime-badge";
  $("telemetry-ribbon").classList.add("hidden");
  $("result-empty").style.display = "flex";
  $("outputs-list").innerHTML = "";
  $("trace-list").innerHTML = "";
}

/* ==========================================================================
   Output Rendering & Telemetry
   ========================================================================== */

function renderResponse(response) {
  state.lastResponse = response;
  const isError = response.status === "error";
  const mode = (response.meta?.mode || modeSelect.value).toUpperCase();

  // Update badge and telemetry ribbon
  $("result-mode").textContent = mode;
  $("result-mode").className = `runtime-badge ${isError ? "" : "active-mode"}`;

  $("telemetry-ribbon").classList.remove("hidden");
  $("metric-latency").textContent = `${response.meta?.elapsed_ms ?? 0} ms`;
  $("metric-steps").textContent = `${response.meta?.step_count ?? state.workflow.length}`;
  $("metric-engine").textContent = mode;

  $("result-empty").style.display = "none";

  if (isError) {
    $("outputs-list").innerHTML = `
      <div style="padding: 16px; background: rgba(244, 63, 94, 0.08); border: 1px solid rgba(244, 63, 94, 0.25); border-radius: var(--radius-md);">
        <strong style="color: var(--brand-rose); display: block; margin-bottom: 4px;">Evaluation Failed</strong>
        <p style="font-size: 12px; color: var(--text-secondary);">${escapeHtml(response.error || "The workflow encountered an execution error.")}</p>
      </div>
    `;
    return;
  }

  // Render Decisions
  const outputEntries = Object.entries(response.outputs || {});
  $("outputs-list").innerHTML = outputEntries.length
    ? outputEntries.map(([key, value]) => {
      const trace = (response.trace || []).find((item) => item.id === key);
      const kind = trace?.kind || "output";
      const isNoul = kind === "noul";
      const isChoice = kind === "choice";
      const isScore = kind === "score";

      let confidencePct = Math.round(Number(isNoul ? (trace?.noul ?? 0) : (trace?.confidence ?? 0)) * 100);
      let gaugeHtml = "";

      // 1. Noul Gauge
      if (isNoul && trace?.noul != null) {
        const noulPct = Math.round(trace.noul * 100);
        gaugeHtml = `
          <div style="margin-top: 10px;">
            <div style="display: flex; justify-content: space-between; font-size: 11px; margin-bottom: 4px;">
              <span style="color: var(--text-muted);">Calibrated Yes Probability</span>
              <span style="font-family: var(--font-mono); font-weight: 700; color: var(--brand-green);">${noulPct}%</span>
            </div>
            <div class="noul-gauge-bar">
              <div class="noul-gauge-fill" style="width: ${noulPct}%;"></div>
            </div>
          </div>
        `;
      }

      // 2. Choice Candidates Distribution
      if (isChoice && trace?.candidates?.length) {
        gaugeHtml = `
          <div class="candidate-table">
            <span class="sub-heading" style="margin-bottom: 4px;">Candidate Distribution (Logits)</span>
            ${trace.candidates.map((cand) => {
              const candScore = Math.round(Number(cand.score || 0) * 100);
              const isWinner = value === cand.value;
              const rawScore = cand.raw_score != null ? `${Math.round(cand.raw_score * 100)}% raw` : "";
              const label = cand.label && cand.label !== pretty(cand.value) ? `${pretty(cand.value)} (${cand.label})` : pretty(cand.value);

              return `
                <div class="candidate-row ${isWinner ? "winner" : ""}">
                  <div class="candidate-row-top">
                    <span class="candidate-name">${escapeHtml(label)}</span>
                    <div class="candidate-score-block">
                      ${rawScore ? `<span class="candidate-raw-tag">${rawScore}</span>` : ""}
                      <span class="candidate-pct">${candScore}%</span>
                    </div>
                  </div>
                  <div class="candidate-bar-bg">
                    <div class="candidate-bar-fill" style="width: ${candScore}%;"></div>
                  </div>
                </div>
              `;
            }).join("")}
          </div>
        `;
      }

      // 3. Score Metric Gauge
      if (isScore) {
        const stepDef = state.workflow.find((s) => s.id === key);
        const minVal = stepDef?.min ?? 0;
        const maxVal = stepDef?.max ?? 10;
        const currentVal = Number(value);
        const range = maxVal - minVal;
        const scorePct = range > 0 ? Math.min(100, Math.max(0, Math.round(((currentVal - minVal) / range) * 100))) : 50;

        gaugeHtml = `
          <div style="margin-top: 10px;">
            <div style="display: flex; justify-content: space-between; font-size: 11px; margin-bottom: 4px;">
              <span style="color: var(--text-muted);">Scale (${minVal} to ${maxVal})</span>
              <span style="font-family: var(--font-mono); font-weight: 700; color: var(--brand-amber);">${currentVal}</span>
            </div>
            <div class="noul-gauge-bar">
              <div class="noul-gauge-fill" style="width: ${scorePct}%; background: linear-gradient(90deg, #F59E0B, #10B981);"></div>
            </div>
          </div>
        `;
      }

      return `
        <div class="decision-card">
          <div class="decision-top">
            <span class="decision-id">${escapeHtml(key)}</span>
            <span class="decision-kind-pill ${kind}-badge">${escapeHtml(kind)}</span>
          </div>
          <div class="decision-value-row">
            <span class="decision-val">${escapeHtml(pretty(value))}</span>
            ${trace?.confidence != null ? `<span class="decision-confidence-tag">${confidencePct}% conf</span>` : ""}
          </div>
          ${gaugeHtml}
        </div>
      `;
    }).join("")
    : '<div style="padding: 20px; color: var(--text-muted); text-align: center;">No decision outputs produced.</div>';

  // Render Trace Timeline
  $("trace-list").innerHTML = (response.trace || []).map((item) => {
    const isSkipped = item.status === "skipped";
    const isErr = item.status === "error";
    const statusClass = isSkipped ? "skipped" : (isErr ? "error" : "");

    let detailStr = item.status;
    if (item.status === "completed") {
      detailStr = item.kind === "noul" && item.noul != null
        ? `${Math.round(item.noul * 100)}% yes probability`
        : `${Math.round((item.confidence || 0) * 100)}% confidence`;
    } else if (item.detail) {
      detailStr = item.detail;
    }

    return `
      <div class="trace-card">
        <div class="trace-left">
          <span class="trace-bullet ${statusClass}"></span>
          <div>
            <div class="trace-step-id">${escapeHtml(item.id)}</div>
            <div class="trace-step-detail">${escapeHtml(detailStr)}</div>
          </div>
        </div>
        <span class="trace-time-badge">${item.elapsed_ms || 0} ms</span>
      </div>
    `;
  }).join("") || '<div style="padding: 20px; color: var(--text-muted); text-align: center;">No trace items.</div>';

  updateExportSnippets();
}

/* ==========================================================================
   Code Exporter Generator (cURL & Python SDK)
   ========================================================================== */

function updateExportSnippets() {
  const currentContext = contextInput.value;
  const currentMode = modeSelect.value;
  const currentModelPath = modelInput.value.trim();

  const payload = {
    context: currentContext,
    workflow: state.workflow,
    mode: currentMode,
    model: currentMode === "native" ? null : currentModelPath,
    model_path: currentMode === "native" ? currentModelPath : null,
    temperature: parseFloat(temperatureInput.value) || 0.0,
  };

  const jsonPayload = JSON.stringify(payload, null, 2);

  // 1. cURL
  $("curl-code").textContent = `curl -X POST http://127.0.0.1:8000/api/run \\
  -H "Content-Type: application/json" \\
  -d '${jsonPayload.replace(/'/g, "'\\''")}'`;

  // 2. Python SDK
  $("python-code").textContent = `import httpx

payload = ${jsonPayload}

response = httpx.post("http://127.0.0.1:8000/api/run", json=payload, timeout=30.0)
result = response.json()

print("Status:", result["status"])
print("Outputs:", result["outputs"])
print("Latency:", result["meta"]["elapsed_ms"], "ms")`;
}

/* ==========================================================================
   Execution & Engine Runner
   ========================================================================== */

async function runWorkflow() {
  if (!contextInput.value.trim()) {
    setRunStatus("Please provide input context before evaluating.", "error");
    contextInput.focus();
    return;
  }

  if (!state.workflow.length) {
    setRunStatus("Add at least one decision step to the workflow.", "error");
    return;
  }

  runButton.disabled = true;
  runButton.innerHTML = `
    <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" class="spinner">
      <circle cx="12" cy="12" r="10" stroke-opacity="0.25"/>
      <path d="M12 2a10 10 0 0 1 10 10"/>
    </svg>
    <span>Evaluating...</span>
  `;

  setRunStatus("Evaluating native CUDA logits via llama.cpp...");

  const t0 = performance.now();

  try {
    const payload = {
      context: contextInput.value,
      workflow: state.workflow,
      mode: "native",
      model: null,
      model_path: modelInput.value.trim() || null,
      temperature: parseFloat(temperatureInput.value) || 0.0,
    };

    const res = await fetch("/api/run", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });

    const data = await res.json();
    if (!res.ok) throw new Error(data.detail || "API rejected the request.");

    renderResponse(data);
    const ms = Math.round(performance.now() - t0);
    setRunStatus(`Completed in ${data.meta?.elapsed_ms || ms} ms`, "success");

  } catch (err) {
    setRunStatus(err.message || "Failed to execute workflow.", "error");
    renderResponse({ status: "error", error: err.message, meta: { mode: "native", elapsed_ms: Math.round(performance.now() - t0) } });
  } finally {
    runButton.disabled = false;
    runButton.innerHTML = `
      <svg width="15" height="15" viewBox="0 0 24 24" fill="currentColor"><polygon points="5 3 19 12 5 21 5 3"/></svg>
      <span>Execute Workflow</span>
    `;
  }
}

/* ==========================================================================
   Initialization & Event Wiring
   ========================================================================== */

async function loadHealth() {
  const pill = $("health-pill");
  try {
    const res = await fetch("/api/health");
    const health = await res.json();
    const nativeReady = health.native?.available && health.native?.cuda;

    pill.className = `telemetry-pill ${nativeReady ? "ready" : "warn"}`;
    $("health-label").textContent = nativeReady
      ? "CUDA Native Active (Qwen 1.7B)"
      : health.native?.available
        ? "Native CPU Active"
        : "Engine Initializing";

    if (health.native?.default_model) {
      modelInput.value = health.native.default_model;
    }
  } catch (_) {
    pill.className = "telemetry-pill warn";
    $("health-label").textContent = "API offline";
  }
}

async function loadExamples() {
  try {
    const res = await fetch("/api/examples");
    const data = await res.json();
    state.examples = data.examples || [];
    exampleSelect.innerHTML = state.examples.map((ex) => 
      `<option value="${escapeHtml(ex.id)}">${escapeHtml(ex.name)}</option>`
    ).join("");

    if (state.examples[0]) selectExample(state.examples[0]);
  } catch (err) {
    setRunStatus("Failed to load workflow examples.", "error");
  }
}

// Event Listeners
exampleSelect.addEventListener("change", () => {
  const found = state.examples.find((ex) => ex.id === exampleSelect.value);
  if (found) selectExample(found);
});

contextInput.addEventListener("input", () => {
  updateCharAndTokenCount();
  updateExportSnippets();
});

workflowInput.addEventListener("input", () => {
  syncWorkflowToVisual();
  updateExportSnippets();
});

$("format-button").addEventListener("click", () => {
  try {
    workflowInput.value = JSON.stringify(JSON.parse(workflowInput.value), null, 2);
    syncWorkflowToVisual();
    setRunStatus("Workflow JSON formatted.", "success");
  } catch (_) {
    setRunStatus("Invalid JSON syntax in workflow editor.", "error");
  }
});

$("add-step-btn").addEventListener("click", () => addNewStep("noul"));

document.querySelectorAll(".quick-add-chip").forEach((chip) => {
  chip.addEventListener("click", () => {
    addNewStep(chip.dataset.kind || "noul");
  });
});

$("clear-context-btn").addEventListener("click", () => {
  contextInput.value = "";
  updateCharAndTokenCount();
  contextInput.focus();
});

$("reset-btn").addEventListener("click", () => {
  if (state.examples[0]) selectExample(state.examples[0]);
});

temperatureInput.addEventListener("input", (e) => {
  tempVal.textContent = parseFloat(e.target.value) === 0 ? "0.0 (Argmax)" : parseFloat(e.target.value).toFixed(2);
  updateExportSnippets();
});



// Tab Switchers: Visual Builder vs JSON
$("tab-visual").addEventListener("click", () => {
  $("tab-visual").classList.add("active");
  $("tab-json").classList.remove("active");
  $("view-visual").classList.add("active");
  $("view-json").classList.remove("active");
  syncWorkflowToVisual();
});

$("tab-json").addEventListener("click", () => {
  $("tab-json").classList.add("active");
  $("tab-visual").classList.remove("active");
  $("view-json").classList.add("active");
  $("view-visual").classList.remove("active");
  syncVisualToWorkflow();
});

// Output Tab Switchers: Decisions vs Trace vs Export
$("res-tab-decisions").addEventListener("click", () => {
  $("res-tab-decisions").classList.add("active");
  $("res-tab-trace").classList.remove("active");
  $("res-tab-export").classList.remove("active");
  $("res-view-decisions").classList.add("active");
  $("res-view-trace").classList.remove("active");
  $("res-view-export").classList.remove("active");
});

$("res-tab-trace").addEventListener("click", () => {
  $("res-tab-trace").classList.add("active");
  $("res-tab-decisions").classList.remove("active");
  $("res-tab-export").classList.remove("active");
  $("res-view-trace").classList.add("active");
  $("res-view-decisions").classList.remove("active");
  $("res-view-export").classList.remove("active");
});

$("res-tab-export").addEventListener("click", () => {
  $("res-tab-export").classList.add("active");
  $("res-tab-decisions").classList.remove("active");
  $("res-tab-trace").classList.remove("active");
  $("res-view-export").classList.add("active");
  $("res-view-decisions").classList.remove("active");
  $("res-view-trace").classList.remove("active");
  updateExportSnippets();
});

// Copy Chips
$("copy-curl-btn").addEventListener("click", () => {
  navigator.clipboard.writeText($("curl-code").textContent).then(() => {
    $("copy-curl-btn").textContent = "Copied!";
    setTimeout(() => { $("copy-curl-btn").textContent = "Copy cURL"; }, 1500);
  });
});

$("copy-py-btn").addEventListener("click", () => {
  navigator.clipboard.writeText($("python-code").textContent).then(() => {
    $("copy-py-btn").textContent = "Copied!";
    setTimeout(() => { $("copy-py-btn").textContent = "Copy Python"; }, 1500);
  });
});

// Keyboard Shortcut: Cmd/Ctrl + Enter runs workflow
document.addEventListener("keydown", (e) => {
  if ((e.metaKey || e.ctrlKey) && e.key === "Enter") {
    e.preventDefault();
    runWorkflow();
  }
});

runButton.addEventListener("click", runWorkflow);

// Spinner keyframe animation injection
const styleEl = document.createElement("style");
styleEl.innerHTML = `@keyframes spin { from { transform: rotate(0deg); } to { transform: rotate(360deg); } } .spinner { animation: spin 0.8s linear infinite; }`;
document.head.appendChild(styleEl);

// Init
Promise.all([loadExamples(), loadHealth()]);
