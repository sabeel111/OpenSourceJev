/**
 * OpenSourceJev — Client Controller
 * Pixel-accurate UI Controller matching the reference SaaS interface
 */

const state = {
  examples: [],
  activeExample: null,
  workflow: [],
  lastResponse: null,
  activeProfile: "fast",
  modelPath: "models\\Qwen3-1.7B-Q8_0.gguf",
  temperature: 0.0,
  logs: [],
  currentExampleIndex: 0,
  stepMode: "visual",
};

const $ = (id) => document.getElementById(id);

// Core Elements
const contextInput = $("context-input");
const presetSelect = $("preset-select");
const visualStepsList = $("visual-steps-list");
const runButton = $("run-btn");
const clearButton = $("clear-btn");
const addContextButton = $("add-context-btn");
const contextSnippetsMenu = $("context-snippets-menu");
const chainOptionsButton = $("chain-options-btn");
const chainOptionsMenu = $("chain-options-menu");
const themeToggleButton = $("theme-toggle-btn");
const themeSunIcon = $("theme-sun-icon");
const themeMoonIcon = $("theme-moon-icon");

/* ==========================================================================
   Utilities
   ========================================================================== */

function escapeHtml(value) {
  return String(value ?? "").replace(/[&<>'"]/g, (char) => ({
    "&": "&amp;", "<": "&lt;", ">": "&gt;", "'": "&#39;", '"': "&quot;"
  }[char]));
}

function loadTheme() {
  const saved = localStorage.getItem("jev_theme") || "light";
  if (saved === "dark") {
    document.documentElement.classList.add("dark");
    themeSunIcon.style.display = "none";
    themeMoonIcon.style.display = "block";
  } else {
    document.documentElement.classList.remove("dark");
    themeSunIcon.style.display = "block";
    themeMoonIcon.style.display = "none";
  }
}

function toggleTheme() {
  const isDark = document.documentElement.classList.toggle("dark");
  localStorage.setItem("jev_theme", isDark ? "dark" : "light");
  themeSunIcon.style.display = isDark ? "none" : "block";
  themeMoonIcon.style.display = isDark ? "block" : "none";
}

function loadLogsFromStorage() {
  try {
    state.logs = JSON.parse(localStorage.getItem("jev_logs") || "[]");
  } catch (_) {
    state.logs = [];
  }
}

function saveLogRecord(record) {
  state.logs.unshift(record);
  if (state.logs.length > 50) state.logs = state.logs.slice(0, 50);
  try {
    localStorage.setItem("jev_logs", JSON.stringify(state.logs));
  } catch (_) {}
}

/* ==========================================================================
   Decision Chain Steps Rendering & Drag/Reorder/Delete
   ========================================================================== */

function renderVisualSteps() {
  if (!state.workflow.length) {
    visualStepsList.innerHTML = `
      <div style="padding: 28px; text-align: center; color: var(--text-muted); font-size: 13px; background: var(--bg-card); border: 1px dashed var(--border-card); border-radius: var(--radius-lg);">
        No decision steps configured. Click "+ Add Step" to define an evaluation condition.
      </div>
    `;
    return;
  }

  visualStepsList.innerHTML = state.workflow.map((step, index) => {
    const kind = step.kind || "noul";
    const isChoice = kind === "choice";
    const isScore = kind === "score";
    const optionsStr = Array.isArray(step.options) ? step.options.join(", ") : "";
    const orderStr = index < 9 ? `0${index + 1}` : `${index + 1}`;

    return `
      <div class="step-card" data-index="${index}">
        <div class="step-row-top">
          <span class="step-order-badge">${orderStr}</span>
          
          <input class="step-id-input" type="text" value="${escapeHtml(step.id || `step_${index + 1}`)}" 
                 data-field="id" placeholder="step_id" title="Unique identifier for this step">

          <select class="step-kind-select" data-field="kind">
            <option value="noul" ${kind === "noul" ? "selected" : ""}>Noul (Boolean)</option>
            <option value="choice" ${kind === "choice" ? "selected" : ""}>Choice (Categorical)</option>
            <option value="score" ${kind === "score" ? "selected" : ""}>Score (Metric)</option>
            <option value="text" ${kind === "text" ? "selected" : ""}>Text (Generation)</option>
          </select>

          <div class="step-row-actions">
            ${index > 0 ? `<button class="step-action-btn move-up-btn" type="button" title="Move Up">
              <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5"><path d="m18 15-6-6-6 6"/></svg>
            </button>` : ""}
            ${index < state.workflow.length - 1 ? `<button class="step-action-btn move-down-btn" type="button" title="Move Down">
              <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5"><path d="m6 9 6 6 6-6"/></svg>
            </button>` : ""}
            <button class="step-action-btn delete delete-step-btn" type="button" title="Delete Step">
              <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                <path d="M3 6h18M19 6v14a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2V6M8 6V4a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2"/>
              </svg>
            </button>
          </div>
        </div>

        <input class="step-prompt-input" type="text" value="${escapeHtml(step.prompt || "")}" 
               placeholder="Question or evaluation rubric for OpenSourceJev..." data-field="prompt">

        <div class="step-extra-row">
          <input class="step-condition-input" type="text" value="${escapeHtml(step.when || "")}" 
                 placeholder='Condition (e.g. "urgent == true")' data-field="when">

          ${isChoice ? `
            <input class="step-options-input" type="text" value="${escapeHtml(optionsStr)}" 
                   placeholder="Options (comma-separated, e.g. billing, technical, account)" data-field="options">
          ` : ""}

          ${isScore ? `
            <div class="step-bounds-group">
              <span>Min</span>
              <input class="step-bound-input" type="number" value="${step.min ?? 1}" data-field="min">
              <span>Max</span>
              <input class="step-bound-input" type="number" value="${step.max ?? 5}" data-field="max">
            </div>
          ` : ""}
        </div>
      </div>
    `;
  }).join("");

  attachVisualStepListeners();
}

function attachVisualStepListeners() {
  visualStepsList.querySelectorAll(".step-card").forEach((card) => {
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
        syncVisualToJson();
        if (field === "kind") renderVisualSteps();
      });
    });

    const deleteBtn = card.querySelector(".delete-step-btn");
    if (deleteBtn) {
      deleteBtn.addEventListener("click", () => {
        state.workflow.splice(index, 1);
        renderVisualSteps();
        syncVisualToJson();
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
          syncVisualToJson();
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
          syncVisualToJson();
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
    prompt: kind === "noul" ? "Does this condition hold true?" : "Evaluate this state:",
  };
  if (kind === "choice") newStep.options = ["option_a", "option_b", "option_c"];
  if (kind === "score") { newStep.min = 1; newStep.max = 5; }

  state.workflow.push(newStep);
  renderVisualSteps();
  syncVisualToJson();

  // Scroll smoothly to newly added step
  visualStepsList.scrollTop = visualStepsList.scrollHeight;
}

/* ==========================================================================
   JSON Mode Synchronization & Utilities
   ========================================================================== */

function syncVisualToJson() {
  const jsonEditor = $("workflow-json-editor");
  if (jsonEditor) {
    jsonEditor.value = JSON.stringify(state.workflow, null, 2);
  }
  const jsonParseError = $("json-parse-error");
  if (jsonParseError) {
    jsonParseError.style.display = "none";
  }
}

function syncJsonToVisual() {
  const jsonEditor = $("workflow-json-editor");
  const jsonParseError = $("json-parse-error");
  if (!jsonEditor) return true;
  const text = jsonEditor.value.trim();
  if (!text) {
    state.workflow = [];
    renderVisualSteps();
    if (jsonParseError) jsonParseError.style.display = "none";
    return true;
  }
  try {
    const parsed = JSON.parse(text);
    if (!Array.isArray(parsed)) {
      throw new Error("Workflow steps must be a JSON array of step objects: [ { ... }, ... ]");
    }
    for (let i = 0; i < parsed.length; i++) {
      const item = parsed[i];
      if (!item || typeof item !== "object") {
        throw new Error(`Step at index ${i} is not a valid object.`);
      }
      if (!item.id) {
        item.id = `step_${i + 1}`;
      }
      if (!item.kind) {
        item.kind = "noul";
      }
    }
    state.workflow = parsed;
    if (jsonParseError) jsonParseError.style.display = "none";
    renderVisualSteps();
    return true;
  } catch (err) {
    if (jsonParseError) {
      jsonParseError.textContent = `JSON Error: ${err.message}`;
      jsonParseError.style.display = "block";
    }
    return false;
  }
}

function switchStepMode(mode) {
  const tabVisual = $("tab-step-visual");
  const tabJson = $("tab-step-json");
  const visualView = $("visual-steps-list");
  const jsonView = $("json-steps-view");
  const jsonEditor = $("workflow-json-editor");

  if (mode === "json") {
    state.stepMode = "json";
    if (tabVisual) tabVisual.classList.remove("active");
    if (tabJson) tabJson.classList.add("active");
    if (visualView) visualView.style.display = "none";
    if (jsonView) jsonView.style.display = "flex";
    syncVisualToJson();
    if (jsonEditor) jsonEditor.focus();
  } else {
    if (!syncJsonToVisual()) {
      return; // Do not switch if JSON is invalid
    }
    state.stepMode = "visual";
    if (tabJson) tabJson.classList.remove("active");
    if (tabVisual) tabVisual.classList.add("active");
    if (jsonView) jsonView.style.display = "none";
    if (visualView) visualView.style.display = "flex";
  }
}

function formatJson() {
  const jsonEditor = $("workflow-json-editor");
  const jsonParseError = $("json-parse-error");
  if (!jsonEditor) return;
  try {
    const parsed = JSON.parse(jsonEditor.value);
    jsonEditor.value = JSON.stringify(parsed, null, 2);
    if (jsonParseError) jsonParseError.style.display = "none";
    if (Array.isArray(parsed)) {
      state.workflow = parsed;
      renderVisualSteps();
    }
  } catch (err) {
    if (jsonParseError) {
      jsonParseError.textContent = `Cannot format invalid JSON: ${err.message}`;
      jsonParseError.style.display = "block";
    }
  }
}

function copyJson() {
  const jsonEditor = $("workflow-json-editor");
  const text = jsonEditor && state.stepMode === "json"
    ? jsonEditor.value
    : JSON.stringify(state.workflow, null, 2);
  navigator.clipboard.writeText(text).then(() => {
    const copyBtn = $("json-copy-btn");
    if (copyBtn) {
      const originalHtml = copyBtn.innerHTML;
      copyBtn.innerHTML = `
        <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
          <polyline points="20 6 9 17 4 12"/>
        </svg>
        <span>Copied!</span>
      `;
      setTimeout(() => { copyBtn.innerHTML = originalHtml; }, 1800);
    }
  }).catch(() => {
    alert("Workflow JSON copied to clipboard!");
  });
}

/* ==========================================================================
   Presets & Examples Loading
   ========================================================================== */

function selectExample(example) {
  state.activeExample = example;
  contextInput.value = example.context;
  state.workflow = JSON.parse(JSON.stringify(example.workflow));

  if (presetSelect.value !== example.id) {
    presetSelect.value = example.id;
  }

  renderVisualSteps();
  syncVisualToJson();
  resetOutputToIdle();
}

function resetOutputToIdle() {
  state.lastResponse = null;
  $("output-status-badge").textContent = "IDLE";
  $("output-status-badge").className = "status-badge idle";
  $("output-empty-state").style.display = "flex";
  $("output-results-wrapper").style.display = "none";
  $("output-decisions-list").innerHTML = "";
  $("trace-details-panel").style.display = "none";
}

async function loadExamples() {
  try {
    const res = await fetch("/api/examples");
    if (!res.ok) throw new Error("Could not load examples");
    const data = await res.json();
    state.examples = Array.isArray(data) ? data : (data.examples || []);

    presetSelect.innerHTML = state.examples.map((ex) => `
      <option value="${escapeHtml(ex.id)}">${escapeHtml(ex.name)}</option>
    `).join("");

    renderWorkflowsGallery();

    if (state.examples.length) {
      selectExample(state.examples[0]);
    }
  } catch (err) {
    console.error("Failed to load examples:", err);
  }
}

function renderWorkflowsGallery() {
  const gallery = $("workflows-gallery-list");
  if (!gallery) return;

  gallery.innerHTML = state.examples.map((ex) => {
    const kinds = ex.workflow.map((s) => s.kind || "noul");
    const pills = Array.from(new Set(kinds))
      .map((k) => `<span class="meta-tag steps">${escapeHtml(k)}</span>`)
      .join(" ");
    return `
      <div class="template-card" data-id="${escapeHtml(ex.id)}">
        <div style="display: flex; justify-content: space-between; align-items: flex-start; margin-bottom: 6px;">
          <h4 class="template-name" style="margin-bottom: 0;">${escapeHtml(ex.name)}</h4>
          <div style="display: flex; gap: 4px;">${pills}</div>
        </div>
        <p class="template-desc">${escapeHtml(ex.description)}</p>
        <div style="display: flex; justify-content: space-between; align-items: center; margin-top: auto; padding-top: 8px;">
          <span class="template-steps-count">${ex.workflow.length} evaluation steps &rarr;</span>
          <span style="font-size: 11px; font-weight: 600; color: var(--brand-green);">Load Preset</span>
        </div>
      </div>
    `;
  }).join("");

  gallery.querySelectorAll(".template-card").forEach((card) => {
    card.addEventListener("click", () => {
      const id = card.dataset.id;
      const found = state.examples.find((ex) => ex.id === id);
      if (found) {
        selectExample(found);
        closeAllModals();
      }
    });
  });
}

/* ==========================================================================
   Execution & Output Rendering
   ========================================================================== */

async function runWorkflow() {
  if (state.stepMode === "json") {
    if (!syncJsonToVisual()) {
      alert("Cannot run workflow: please resolve the JSON syntax error.");
      return;
    }
  }
  const context = contextInput.value.trim();
  if (!context) {
    alert("Please provide context or a command to evaluate.");
    contextInput.focus();
    return;
  }

  if (!state.workflow.length) {
    alert("Please add at least one step in the Decision Chain.");
    return;
  }

  // Set running state
  $("output-status-badge").textContent = "RUNNING";
  $("output-status-badge").className = "status-badge running";
  runButton.disabled = true;
  runButton.innerHTML = `
    <svg class="spinner" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
      <circle cx="12" cy="12" r="10" stroke-opacity="0.25"/>
      <path d="M12 2a10 10 0 0 1 10 10" stroke-opacity="0.9"/>
    </svg>
    <span>Evaluating...</span>
  `;

  const payload = {
    context,
    workflow: state.workflow,
    mode: "native",
    profile: state.activeProfile,
    model_path: state.modelPath,
    temperature: state.temperature,
  };

  const startTime = performance.now();

  try {
    const res = await fetch("/api/run", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });

    const elapsed = Math.round(performance.now() - startTime);
    const data = await res.json();

    if (!res.ok) {
      throw new Error(data.detail || data.error || "Workflow evaluation error");
    }

    renderResponse(data, elapsed);
    saveLogRecord({
      time: new Date().toLocaleTimeString(),
      context: context.length > 60 ? context.substring(0, 60) + "..." : context,
      status: "success",
      latency: data.meta?.elapsed_ms || elapsed,
      profile: state.activeProfile,
      outputs: data.outputs,
    });
  } catch (err) {
    console.error("Run error:", err);
    $("output-status-badge").textContent = "ERROR";
    $("output-status-badge").className = "status-badge idle";
    $("output-empty-state").style.display = "none";
    $("output-results-wrapper").style.display = "block";
    $("output-decisions-list").innerHTML = `
      <div style="padding: 14px; background: rgba(239, 68, 68, 0.08); border: 1px solid rgba(239, 68, 68, 0.25); border-radius: var(--radius-md);">
        <strong style="color: var(--brand-rose); font-size: 13px; display: block; margin-bottom: 4px;">Execution Error</strong>
        <p style="font-size: 12px; color: var(--text-secondary);">${escapeHtml(err.message)}</p>
      </div>
    `;
  } finally {
    runButton.disabled = false;
    runButton.innerHTML = `
      <svg class="run-play-icon" width="14" height="14" viewBox="0 0 24 24" fill="currentColor">
        <polygon points="6 3 20 12 6 21 6 3"/>
      </svg>
      <span>Run</span>
    `;
  }
}

function renderResponse(response, clientElapsed) {
  state.lastResponse = response;
  const elapsed = response.meta?.elapsed_ms || clientElapsed;

  $("output-status-badge").textContent = "COMPLETED";
  $("output-status-badge").className = "status-badge completed";
  $("pill-speed-text").textContent = `${elapsed}ms`;

  $("output-empty-state").style.display = "none";
  $("output-results-wrapper").style.display = "block";

  // Meta Tags
  $("meta-latency").textContent = `${elapsed} ms`;
  $("meta-profile").textContent = response.meta?.profile ? `${response.meta.profile} mode` : state.activeProfile;
  $("meta-steps").textContent = `${response.trace?.length || Object.keys(response.outputs || {}).length} decisions`;

  const decisionsList = $("output-decisions-list");
  const entries = Object.entries(response.outputs || {});

  if (!entries.length) {
    decisionsList.innerHTML = `
      <div style="padding: 16px; text-align: center; color: var(--text-muted); font-size: 12px;">
        All steps were skipped due to conditional criteria.
      </div>
    `;
    return;
  }

  decisionsList.innerHTML = entries.map(([key, value]) => {
    const trace = (response.trace || []).find((t) => t.id === key);
    const kind = trace?.kind || "noul";
    const isNoul = kind === "noul";
    const isChoice = kind === "choice";
    const isScore = kind === "score";

    let valueDisplay = String(value);
    let valueClass = "";
    if (isNoul) {
      valueDisplay = value ? "TRUE" : "FALSE";
      valueClass = value ? "true" : "false";
    }

    let extraHtml = "";

    // Noul probability gauge
    if (isNoul && trace?.noul != null) {
      const pct = Math.round(trace.noul * 100);
      extraHtml = `
        <div class="d-gauge-wrap">
          <div style="display: flex; justify-content: space-between; font-size: 10.5px; margin-bottom: 2px;">
            <span style="color: var(--text-muted);">Calibrated Yes Probability</span>
            <span style="font-weight: 700; color: var(--brand-green);">${pct}%</span>
          </div>
          <div class="d-gauge-bar">
            <div class="d-gauge-fill" style="width: ${pct}%;"></div>
          </div>
        </div>
      `;
    }

    // Choice probabilities distribution
    if (isChoice && Array.isArray(trace?.candidates)) {
      extraHtml = `
        <div style="margin-top: 8px;">
          ${trace.candidates.map((c) => {
            const cPct = Math.round((c.score || 0) * 100);
            const isWinner = c.value === value;
            return `
              <div class="d-candidate-row">
                <span style="color: ${isWinner ? 'var(--text-primary)' : 'var(--text-muted)'}; font-weight: ${isWinner ? '700' : '400'};">
                  ${escapeHtml(c.value)}
                </span>
                <span style="font-family: var(--font-mono); color: ${isWinner ? 'var(--brand-green)' : 'var(--text-dim)'}; font-weight: ${isWinner ? '700' : '400'};">
                  ${cPct}%
                </span>
              </div>
              <div class="d-candidate-bar">
                <div class="d-candidate-fill" style="width: ${cPct}%; opacity: ${isWinner ? '1' : '0.4'};"></div>
              </div>
            `;
          }).join("")}
        </div>
      `;
    }

    // Score range indicator
    if (isScore) {
      extraHtml = `
        <div style="display: flex; align-items: center; justify-content: space-between; margin-top: 6px; font-size: 10.5px; color: var(--text-muted);">
          <span>Rubric expectation: <strong>${value}</strong></span>
          <span style="font-family: var(--font-mono);">[${trace?.min ?? 1} &rarr; ${trace?.max ?? 5}]</span>
        </div>
      `;
    }

    return `
      <div class="decision-result-card">
        <div class="d-res-header">
          <span class="d-res-id">${escapeHtml(key)}</span>
          <span class="d-res-val ${valueClass}">${escapeHtml(valueDisplay)}</span>
        </div>
        ${extraHtml}
      </div>
    `;
  }).join("");

  // Populate raw JSON
  $("raw-json-output").textContent = JSON.stringify(response, null, 2);
}

/* ==========================================================================
   Modals & Drawers Management
   ========================================================================== */

function openModal(modalId) {
  closeAllModals();
  const modal = $(modalId);
  if (modal) modal.style.display = "flex";
}

function closeAllModals() {
  document.querySelectorAll(".modal-backdrop").forEach((m) => {
    m.style.display = "none";
  });
}

function renderLogs() {
  const container = $("logs-history-container");
  if (!container) return;

  if (!state.logs.length) {
    container.innerHTML = `
      <div style="padding: 24px; text-align: center; color: var(--text-muted); font-size: 12px;">
        No workflow executions recorded yet this session.
      </div>
    `;
    return;
  }

  container.innerHTML = state.logs.map((log) => `
    <div class="log-item">
      <div class="log-item-left">
        <span class="log-time">${escapeHtml(log.time)} &bull; ${escapeHtml(log.profile)} profile</span>
        <span class="log-context-snippet">${escapeHtml(log.context)}</span>
      </div>
      <div class="log-item-right">
        <span class="log-badge" style="background: var(--brand-green-dim); color: #047857;">${log.latency}ms</span>
      </div>
    </div>
  `).join("");
}

/* ==========================================================================
   Event Listeners Wire-Up
   ========================================================================== */

function attachEventListeners() {
  // Theme Toggle
  themeToggleButton.addEventListener("click", toggleTheme);

  // Nav item clicks
  $("nav-playground").addEventListener("click", () => {
    closeAllModals();
    $("nav-playground").classList.add("active");
  });

  $("nav-workflows").addEventListener("click", () => openModal("modal-workflows"));
  $("nav-templates").addEventListener("click", () => openModal("modal-workflows"));
  $("nav-models").addEventListener("click", () => openModal("modal-models"));
  $("nav-logs").addEventListener("click", () => {
    renderLogs();
    openModal("modal-logs");
  });
  $("nav-settings").addEventListener("click", () => openModal("modal-settings"));
  $("engine-status-card").addEventListener("click", () => openModal("modal-models"));

  // Modal Close buttons
  document.querySelectorAll("[data-close]").forEach((btn) => {
    btn.addEventListener("click", () => {
      const modalId = btn.dataset.close;
      const target = $(modalId);
      if (target) target.style.display = "none";
    });
  });

  document.querySelectorAll(".modal-backdrop").forEach((backdrop) => {
    backdrop.addEventListener("click", (e) => {
      if (e.target === backdrop) backdrop.style.display = "none";
    });
  });

  // Preset Select change
  presetSelect.addEventListener("change", () => {
    const found = state.examples.find((ex) => ex.id === presetSelect.value);
    if (found) selectExample(found);
  });

  // Context Actions
  clearButton.addEventListener("click", () => {
    contextInput.value = "";
    contextInput.focus();
  });

  addContextButton.addEventListener("click", (e) => {
    e.stopPropagation();
    contextSnippetsMenu.classList.toggle("show");
  });

  document.addEventListener("click", () => {
    contextSnippetsMenu.classList.remove("show");
    chainOptionsMenu.classList.remove("show");
  });

  contextSnippetsMenu.querySelectorAll(".dropdown-item").forEach((item) => {
    item.addEventListener("click", () => {
      contextInput.value = item.dataset.snippet;
      contextSnippetsMenu.classList.remove("show");
    });
  });

  // Decision Chain Header Actions
  $("add-step-btn").addEventListener("click", () => addNewStep("noul"));

  // Mode Switcher (Visual vs JSON)
  const tabStepVisual = $("tab-step-visual");
  const tabStepJson = $("tab-step-json");
  if (tabStepVisual) tabStepVisual.addEventListener("click", () => switchStepMode("visual"));
  if (tabStepJson) tabStepJson.addEventListener("click", () => switchStepMode("json"));

  // JSON Mode Buttons
  const jsonFormatBtn = $("json-format-btn");
  if (jsonFormatBtn) jsonFormatBtn.addEventListener("click", formatJson);

  const jsonCopyBtn = $("json-copy-btn");
  if (jsonCopyBtn) jsonCopyBtn.addEventListener("click", copyJson);

  // Live JSON validation & shortcut handling in JSON Editor
  const jsonEditor = $("workflow-json-editor");
  if (jsonEditor) {
    jsonEditor.addEventListener("input", () => {
      const text = jsonEditor.value.trim();
      const jsonParseError = $("json-parse-error");
      if (!text) {
        if (jsonParseError) jsonParseError.style.display = "none";
        return;
      }
      try {
        const parsed = JSON.parse(text);
        if (!Array.isArray(parsed)) {
          if (jsonParseError) {
            jsonParseError.textContent = "Workflow must be a JSON array: [ { ... } ]";
            jsonParseError.style.display = "block";
          }
          return;
        }
        if (jsonParseError) jsonParseError.style.display = "none";
        state.workflow = parsed;
      } catch (e) {
        if (jsonParseError) {
          jsonParseError.textContent = `JSON Syntax Error: ${e.message}`;
          jsonParseError.style.display = "block";
        }
      }
    });

    jsonEditor.addEventListener("keydown", (e) => {
      if ((e.ctrlKey && e.altKey && e.key.toLowerCase() === "f") ||
          (e.altKey && e.shiftKey && e.key.toLowerCase() === "f")) {
        e.preventDefault();
        formatJson();
      }
    });
  }

  chainOptionsButton.addEventListener("click", (e) => {
    e.stopPropagation();
    chainOptionsMenu.classList.toggle("show");
  });

  $("opt-reset-preset").addEventListener("click", () => {
    if (state.activeExample) selectExample(state.activeExample);
    chainOptionsMenu.classList.remove("show");
  });

  $("opt-clear-steps").addEventListener("click", () => {
    state.workflow = [];
    renderVisualSteps();
    syncVisualToJson();
    chainOptionsMenu.classList.remove("show");
  });

  const optFormatJson = $("opt-format-json");
  if (optFormatJson) {
    optFormatJson.addEventListener("click", () => {
      formatJson();
      chainOptionsMenu.classList.remove("show");
    });
  }

  $("opt-export-json").addEventListener("click", () => {
    copyJson();
    chainOptionsMenu.classList.remove("show");
  });

  // Run Button & Keyboard shortcut (Ctrl+Enter / Cmd+Enter)
  runButton.addEventListener("click", runWorkflow);

  document.addEventListener("keydown", (e) => {
    if ((e.metaKey || e.ctrlKey) && e.key === "Enter") {
      e.preventDefault();
      runWorkflow();
    }
  });

  // Trace / Raw JSON Toggle
  $("toggle-trace-btn").addEventListener("click", () => {
    const panel = $("trace-details-panel");
    const isHidden = panel.style.display === "none";
    panel.style.display = isHidden ? "block" : "none";
    $("toggle-trace-btn").textContent = isHidden ? "Hide Execution Trace & JSON" : "Show Execution Trace & JSON";
  });

  // Quick Action 1: Load Example
  $("qa-load-example").addEventListener("click", () => {
    if (!state.examples.length) return;
    state.currentExampleIndex = (state.currentExampleIndex + 1) % state.examples.length;
    selectExample(state.examples[state.currentExampleIndex]);
  });

  // Quick Action 2: Export Workflow
  $("qa-export-workflow").addEventListener("click", () => {
    const dataStr = "data:text/json;charset=utf-8," + encodeURIComponent(JSON.stringify(state.workflow, null, 2));
    const downloadAnchor = document.createElement("a");
    downloadAnchor.setAttribute("href", dataStr);
    downloadAnchor.setAttribute("download", `opensourcejev_workflow_${Date.now()}.json`);
    document.body.appendChild(downloadAnchor);
    downloadAnchor.click();
    downloadAnchor.remove();
  });

  // Quick Action 3: View Logs
  $("qa-view-logs").addEventListener("click", () => {
    renderLogs();
    openModal("modal-logs");
  });

  // Clear Logs
  $("clear-logs-btn").addEventListener("click", () => {
    state.logs = [];
    localStorage.removeItem("jev_logs");
    renderLogs();
  });

  // Model Profile Cards in Modal
  const cardFast = $("profile-card-fast");
  const cardAccuracy = $("profile-card-accuracy");

  cardFast.addEventListener("click", () => {
    cardFast.classList.add("selected");
    cardAccuracy.classList.remove("selected");
    state.activeProfile = "fast";
    state.modelPath = "models\\Qwen3-1.7B-Q8_0.gguf";
  });

  cardAccuracy.addEventListener("click", () => {
    cardAccuracy.classList.add("selected");
    cardFast.classList.remove("selected");
    state.activeProfile = "accuracy";
    state.modelPath = "models\\qwen35-4b-q4km\\Qwen3.5-4B-Q4_K_M.gguf";
  });

  $("apply-profile-btn").addEventListener("click", () => {
    $("sidebar-engine-desc").innerHTML = state.activeProfile === "fast" 
      ? `CUDA (Local)<br>Qwen3-1.7B Fast` 
      : `CUDA (Local)<br>Qwen3.5-4B Accuracy`;
    closeAllModals();
  });

  // Settings Modal controls
  const settingTemp = $("setting-temperature");
  const settingTempVal = $("setting-temp-val");

  settingTemp.addEventListener("input", (e) => {
    const val = parseFloat(e.target.value);
    settingTempVal.textContent = val === 0 ? "0.0 (Argmax)" : val.toFixed(2);
    state.temperature = val;
  });

  $("save-settings-btn").addEventListener("click", () => {
    const customPath = $("setting-model-path").value.trim();
    if (customPath) state.modelPath = customPath;
    closeAllModals();
  });
}

// Injected Spinner CSS
const styleEl = document.createElement("style");
styleEl.innerHTML = `@keyframes spin { from { transform: rotate(0deg); } to { transform: rotate(360deg); } } .spinner { animation: spin 0.8s linear infinite; }`;
document.head.appendChild(styleEl);

/* ==========================================================================
   Initialization
   ========================================================================== */

loadTheme();
loadLogsFromStorage();
attachEventListeners();
loadExamples();
