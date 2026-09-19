const state = { examples: [], activeExample: null, lastResponse: null };

const $ = (id) => document.getElementById(id);
const contextInput = $("context-input");
const workflowInput = $("workflow-input");
const exampleSelect = $("example-select");
const modeSelect = $("mode-select");
const modelInput = $("model-input");
const runButton = $("run-button");
const bundledNativeModel = "models\\Qwen3-1.7B-Q8_0.gguf";

function syncModeFields() {
  const native = modeSelect.value === "native";
  const ollama = modeSelect.value === "ollama";
  $("model-label").textContent = native ? "GGUF model path" : "Ollama model";
  $("model-helper").textContent = native
    ? "Point this at a local .gguf file. The native adapter reads logits directly through llama.cpp."
    : "Use the model tag shown by `ollama list`, for example qwen3:1.7b.";
  $("mode-helper").textContent = native
    ? "This is the genuine Jev path: candidate probabilities come from native logits, not model-reported confidence."
    : ollama
      ? "Ollama is useful for a product smoke test, but it does not expose full candidate logits."
      : "Demo mode is deterministic and needs no model download.";
  if (native) {
    if (!modelInput.value || modelInput.value === "qwen3:1.7b") modelInput.value = bundledNativeModel;
    modelInput.placeholder = "C:\\models\\Qwen3-1.7B-Q4_K_M.gguf";
  } else {
    modelInput.placeholder = "qwen3:1.7b";
    if (modelInput.value === bundledNativeModel) modelInput.value = "qwen3:1.7b";
    if (!modelInput.value) modelInput.value = "qwen3:1.7b";
  }
}

function pretty(value) {
  if (typeof value === "boolean") return value ? "true" : "false";
  if (value === null || value === undefined) return "—";
  return typeof value === "string" ? value : JSON.stringify(value);
}

function setRunMessage(message, tone = "") {
  const target = $("run-message");
  target.textContent = message;
  target.className = `run-message ${tone}`;
}

function updateStepCount() {
  try {
    const workflow = JSON.parse(workflowInput.value);
    $("step-count").textContent = Array.isArray(workflow) ? workflow.length : "—";
  } catch (_) {
    $("step-count").textContent = "—";
  }
}

function selectExample(example) {
  state.activeExample = example;
  contextInput.value = example.context;
  workflowInput.value = JSON.stringify(example.workflow, null, 2);
  $("example-description").textContent = example.description;
  updateStepCount();
  updateCharCount();
  clearResults();
  setRunMessage("Ready to evaluate.");
}

function updateCharCount() {
  $("char-count").textContent = `${contextInput.value.length.toLocaleString()} / 20,000`;
}

function clearResults() {
  state.lastResponse = null;
  $("result-mode").textContent = "IDLE";
  $("result-banner").className = "result-banner empty";
  $("result-banner").innerHTML = '<span class="banner-glyph" aria-hidden="true">◇</span><div><strong>Nothing run yet</strong><p>Your step outputs will appear here.</p></div>';
  $("outputs-list").innerHTML = "";
  $("trace-list").innerHTML = '<div class="trace-empty">Run a workflow to inspect timing, confidence, and skipped conditions.</div>';
}

function renderResponse(response) {
  state.lastResponse = response;
  const isError = response.status === "error";
  $("result-mode").textContent = (response.meta?.mode || modeSelect.value).toUpperCase();
  $("result-banner").className = `result-banner ${isError ? "error" : ""}`;
  $("result-banner").innerHTML = isError
    ? `<span class="banner-glyph" aria-hidden="true">!</span><div><strong>Run stopped</strong><p>${escapeHtml(response.error || "The workflow could not be completed.")}</p></div>`
    : `<span class="banner-glyph" aria-hidden="true">✓</span><div><strong>Workflow complete</strong><p>${response.meta.elapsed_ms} ms · ${response.meta.step_count} configured steps</p></div>`;

  const outputEntries = Object.entries(response.outputs || {});
  $("outputs-list").innerHTML = outputEntries.length
    ? outputEntries.map(([key, value]) => {
      const trace = (response.trace || []).find((item) => item.id === key);
      const isNoul = trace?.kind === "noul";
      const confidence = Math.round(Number(isNoul ? (trace?.noul || 0) : (trace?.confidence || 0)) * 100);
      const candidates = trace?.candidates || [];
      const candidateMarkup = candidates.length
        ? `<div class="candidate-list"><div class="candidate-heading">Candidate probabilities</div>${candidates.map((candidate) => {
          const score = Math.round(Number(candidate.score || 0) * 100);
          const raw = candidate.raw_score != null
            ? `<span class="candidate-raw">raw ${Math.round(Number(candidate.raw_score) * 100)}%</span>`
            : "";
          const label = candidate.label && candidate.label !== pretty(candidate.value)
            ? `${pretty(candidate.value)} · ${candidate.label}`
            : pretty(candidate.value);
          const isWinner = value === candidate.value || (typeof value === "number" && Math.abs(value - Number(candidate.value)) < 0.001);
          const isZero = score === 0;
          return `<div class="candidate-row ${isWinner ? "is-winner" : ""} ${isZero ? "is-zero" : ""}"><div class="candidate-meta"><div class="candidate-label"><span class="candidate-name" title="${escapeHtml(label)}">${escapeHtml(label)}</span>${raw}</div><span class="candidate-score ${isZero ? "zero" : ""}">${score}%</span></div><div class="candidate-bar"><span style="width:${score}%"></span></div></div>`;
        }).join("")}</div>`
        : "";
      const confidenceBar = isNoul && trace?.noul != null
        ? `<div class="confidence" title="${confidence}% yes probability"><span style="width:${confidence}%"></span></div>`
        : trace?.confidence != null
          ? `<div class="confidence" title="${confidence}% confidence"><span style="width:${confidence}%"></span></div>`
          : "";
      return `<div class="output-item"><div class="output-top"><span class="output-key">${escapeHtml(key)}</span><span class="output-type">${escapeHtml(trace?.kind || "output")}</span></div><div class="output-value">${escapeHtml(pretty(value))}</div>${confidenceBar}${candidateMarkup}</div>`;
    }).join("")
    : '<div class="trace-empty">No output values were produced.</div>';

  $("trace-list").innerHTML = (response.trace || []).map((item) => {
    const detail = item.status === "completed"
      ? item.kind === "noul" && item.noul != null
        ? `${Math.round(item.noul * 100)}% yes probability`
        : `${Math.round((item.confidence || 0) * 100)}% confidence`
      : (item.detail || item.status);
    return `<div class="trace-item"><span class="trace-dot ${item.status === "skipped" ? "skipped" : item.status === "error" ? "error" : ""}"></span><div><span class="trace-name">${escapeHtml(item.id)}</span><span class="trace-detail">${escapeHtml(item.status)} · ${escapeHtml(detail)}</span></div><span class="trace-time">${item.elapsed_ms} ms</span></div>`;
  }).join("") || '<div class="trace-empty">No trace entries.</div>';
}

function escapeHtml(value) {
  return String(value).replace(/[&<>'"]/g, (char) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", "'": "&#39;", '"': "&quot;" }[char]));
}

async function loadExamples() {
  const response = await fetch("/api/examples");
  if (!response.ok) throw new Error("Could not load examples.");
  const payload = await response.json();
  state.examples = payload.examples || [];
  exampleSelect.innerHTML = state.examples.map((example) => `<option value="${escapeHtml(example.id)}">${escapeHtml(example.name)}</option>`).join("");
  if (state.examples[0]) selectExample(state.examples[0]);
}

async function loadHealth() {
  const pill = $("health-pill");
  try {
    const response = await fetch("/api/health");
    const health = await response.json();
    const nativeReady = health.native?.available && health.native?.cuda;
    pill.className = `health-pill ${nativeReady || health.ollama.connected ? "ready" : "warn"}`;
    $("health-label").textContent = nativeReady
      ? "CUDA native ready"
      : health.native?.available
        ? "Native CPU ready"
        : health.ollama.connected
          ? "Ollama connected"
          : "Demo mode ready";
    if (health.native?.default_model && modeSelect.value === "native") {
      modelInput.value = health.native.default_model;
    } else if (health.default_model && !modelInput.value) {
      modelInput.value = health.default_model;
    }
  } catch (_) {
    pill.className = "health-pill warn";
    $("health-label").textContent = "API unavailable";
  }
}

async function runWorkflow() {
  let workflow;
  try {
    workflow = JSON.parse(workflowInput.value);
    if (!Array.isArray(workflow) || workflow.length === 0) throw new Error("Workflow must be a non-empty JSON array.");
  } catch (error) {
    setRunMessage(error.message || "Workflow JSON is invalid.", "error");
    workflowInput.focus();
    return;
  }
  if (!contextInput.value.trim()) {
    setRunMessage("Add some context before running the workflow.", "error");
    contextInput.focus();
    return;
  }
  runButton.disabled = true;
  runButton.innerHTML = '<span class="button-glyph" aria-hidden="true">◌</span> Running…';
  setRunMessage(
    modeSelect.value === "ollama"
      ? "Calling the local model…"
      : modeSelect.value === "native"
        ? "Evaluating native logits…"
        : "Evaluating deterministic demo…"
  );
  try {
    const mode = modeSelect.value;
    const response = await fetch("/api/run", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        context: contextInput.value,
        workflow,
        mode,
        model: mode === "native" ? null : (modelInput.value.trim() || null),
        model_path: mode === "native" ? (modelInput.value.trim() || null) : null,
      }),
    });
    const payload = await response.json();
    if (!response.ok) throw new Error(payload.detail || "The API rejected this workflow.");
    renderResponse(payload);
    setRunMessage(payload.status === "completed" ? "Run complete." : "Run stopped with an error.", payload.status === "completed" ? "success" : "error");
  } catch (error) {
    setRunMessage(error.message || "Could not reach the local API.", "error");
    $("result-mode").textContent = "ERROR";
  } finally {
    runButton.disabled = false;
    runButton.innerHTML = '<span class="button-glyph" aria-hidden="true">▶</span> Run workflow';
  }
}

exampleSelect.addEventListener("change", () => {
  const example = state.examples.find((item) => item.id === exampleSelect.value);
  if (example) selectExample(example);
});
contextInput.addEventListener("input", updateCharCount);
workflowInput.addEventListener("input", updateStepCount);
$("format-button").addEventListener("click", () => {
  try { workflowInput.value = JSON.stringify(JSON.parse(workflowInput.value), null, 2); updateStepCount(); setRunMessage("Workflow JSON formatted.", "success"); }
  catch (_) { setRunMessage("The workflow is not valid JSON yet.", "error"); }
});
runButton.addEventListener("click", runWorkflow);
modeSelect.addEventListener("change", syncModeFields);

Promise.all([loadExamples(), loadHealth()]).catch((error) => setRunMessage(error.message, "error"));
syncModeFields();
