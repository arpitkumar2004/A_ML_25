const SAMPLE = {
  sample_id: 1001,
  catalog_content: [
    "Item Name: Log Cabin Sugar Free Syrup, 24 FL OZ (Pack of 12)",
    "Bullet Point 1: Contains twelve (12) 24-ounce bottles of Log Cabin Sugar Free Syrup for Pancakes and Waffles",
    "Bullet Point 2: Indulge in thick, delicious syrup for pancakes, waffles, French toast and more",
    "Bullet Point 3: 90% fewer calories than our original syrup and no sugar or high fructose corn syrup",
    "Bullet Point 4: Amazing syrup that you can feel good about serving to your family and guests",
    "Bullet Point 5: Stock up on this breakfast staple for decadent pancakes and waffles anytime",
    "Value: 288.0",
    "Unit: Fl Oz"
  ].join("\n"),
  image_link: "https://m.media-amazon.com/images/I/71QD2OFXqDL.jpg",
  value: 288.0,
  unit: "Fl Oz",
  category: "Breakfast / Pantry",
  reference_price: 12.195
};

const node = (id) => document.getElementById(id);
const money = (value) => value === null || value === undefined || Number.isNaN(Number(value))
  ? "--"
  : Number(value).toLocaleString(undefined, { style: "currency", currency: "USD", minimumFractionDigits: 2, maximumFractionDigits: 3 });
const percent = (value) => `${(Number(value || 0) * 100).toFixed(2)}%`;
const widthPercent = (value) => `${Math.max(0, Math.min(100, Number(value || 0) * 100)).toFixed(1)}%`;

let telemetrySnapshot = null;
let activeTerminalRun = 0;
let serviceSnapshot = null;

function readForm() {
  const brokenToggle = node("simulateBroken").checked;
  const rawImage = (node("imageUrl").value || "").trim();
  return {
    sample_id: Number(node("sampleId").value || "1"),
    catalog_content: (node("catalogContent").value || "").trim(),
    image_link: brokenToggle ? "https://invalid.example.com/broken-image.jpg" : rawImage,
    value: node("numericValue").value,
    unit: (node("unitText").value || "").trim(),
    category: (node("categoryText").value || "").trim(),
    reference_price: node("referencePrice").value,
    round: true
  };
}

function buildPayload() {
  const form = readForm();
  const record = {
    sample_id: form.sample_id,
    catalog_content: form.catalog_content,
    Description: form.catalog_content,
    image_link: form.image_link,
    image_path: form.image_link
  };

  if (form.value !== "") record.value = Number(form.value);
  if (form.unit) record.unit = form.unit;
  if (form.category) record.category = form.category;
  if (form.reference_price !== "") {
    const ref = Number(form.reference_price);
    if (!Number.isNaN(ref)) {
      record.price = ref;
      record.Price = ref;
    }
  }

  return {
    records: [record],
    text_col: "catalog_content",
    image_col: "image_link",
    id_col: "sample_id",
    pred_col: "predicted_price",
    min_value: 0.0,
    round: form.round
  };
}

function syncPreview() {
  const form = readForm();
  node("payloadPreview").textContent = JSON.stringify(buildPayload(), null, 2);
  node("referenceView").textContent = form.reference_price !== "" ? money(Number(form.reference_price)) : "--";

  const previewImage = node("imagePreview");
  const previewFallback = node("imageFallback");
  if (form.image_link) {
    previewImage.src = form.image_link;
    previewImage.style.display = "block";
    previewFallback.style.display = "none";
  } else {
    previewImage.removeAttribute("src");
    previewImage.style.display = "none";
    previewFallback.style.display = "block";
  }
}

function loadSample() {
  node("sampleId").value = SAMPLE.sample_id;
  node("catalogContent").value = SAMPLE.catalog_content;
  node("imageUrl").value = SAMPLE.image_link;
  node("numericValue").value = SAMPLE.value;
  node("unitText").value = SAMPLE.unit;
  node("categoryText").value = SAMPLE.category;
  node("referencePrice").value = SAMPLE.reference_price;
  node("simulateBroken").checked = false;
  syncPreview();
}

function setStatus(isHealthy, label) {
  const statusDot = node("statusDot");
  statusDot.className = `status-dot ${isHealthy ? "ok" : "err"}`;
  node("apiStatusText").textContent = `Status: ${label}`;
}

function setLink(id, href, fallback) {
  node(id).href = href || fallback;
}

function renderServiceInfo(service) {
  if (!service) return;
  serviceSnapshot = service;
  setStatus(Boolean(service.ready), service.ready ? "Healthy" : "Degraded");
  node("activeRunId").textContent = service.run_id || "unknown";
  node("environmentTag").textContent = service.environment || "PRODUCTION";
  node("serviceSnapshotBox").textContent = JSON.stringify(service, null, 2);

  const links = service.links || {};
  setLink("bundleLink", links.github_repo, "https://github.com/arpitkumar2004/A_ML_25");
  setLink("mlflowLink", links.mlflow_run, links.github_repo || "https://github.com/arpitkumar2004/A_ML_25");
  setLink("manifestLink", "/service/info", "/service/info");
}

function renderTelemetry(metrics) {
  telemetrySnapshot = metrics;
  const fallback = metrics.fallback || {};
  const dataQuality = metrics.data_quality || {};

  node("p95Latency").textContent = `${Math.round(Number(metrics.latency_ms && metrics.latency_ms.p95 || 0))} ms`;
  node("requestsCount").textContent = Number(metrics.request_count || 0).toLocaleString();
  node("errorRate").textContent = percent(metrics.error_rate || 0);
  node("fallbackRate").textContent = percent(Math.max(Number(fallback.image_rate || 0), Number(fallback.text_rate || 0)));
  node("dqPassRate").textContent = percent(dataQuality.pass_rate || 0);

  node("imageFallbackBar").style.width = widthPercent(fallback.image_rate || 0);
  node("textFallbackBar").style.width = widthPercent(fallback.text_rate || 0);
  node("errorRateBar").style.width = widthPercent(metrics.error_rate || 0);
  node("dqPassBar").style.width = widthPercent(dataQuality.pass_rate || 0);
}

async function pollSystem() {
  try {
    const [healthRes, infoRes, metricsRes] = await Promise.all([
      fetch("/healthz"),
      fetch("/service/info"),
      fetch("/metrics/json")
    ]);

    const health = await healthRes.json();
    const info = await infoRes.json();
    const metrics = await metricsRes.json();

    renderServiceInfo(info);
    renderTelemetry(metrics);
    if (!healthRes.ok || !info.ready) {
      setStatus(false, "Degraded");
    }
  } catch (error) {
    setStatus(false, "Unreachable");
  }
}

function pushTerminalLine(runId, seconds, text, kind = "") {
  if (runId !== activeTerminalRun) return;
  const line = document.createElement("div");
  line.className = `terminal-line${kind ? ` ${kind}` : ""}`;
  line.textContent = `[${seconds}] ${text}`;
  node("terminalBody").appendChild(line);
  node("terminalBody").scrollTop = node("terminalBody").scrollHeight;
}

function startTerminal(runId) {
  node("terminalPanel").classList.remove("is-hidden");
  node("terminalBody").innerHTML = "";
  node("terminalStatus").textContent = "Execution in progress";
  const form = readForm();
  const steps = [
    ["0.01s", "Request ID generated and middleware engaged."],
    ["0.05s", "Schema normalized and alias columns matched."],
    ["0.12s", "Extracting parsed numeric signals from text."],
    ["0.45s", "Generating text embeddings for serving payload."],
    ["0.82s", form.image_link ? "Resolving image input and checking fallback safety." : "Image input missing. Zero-vector fallback path available."],
    ["1.12s", "Executing base models (LGBM, XGB, RF)." ],
    ["1.30s", "Awaiting ensemble stacker output." ]
  ];
  steps.forEach(([seconds, text], index) => {
    window.setTimeout(() => pushTerminalLine(runId, seconds, text), index * 170);
  });
}

function fillTable(targetId, rows) {
  const table = node(targetId);
  if (!rows.length) {
    table.innerHTML = '<div class="table-row"><span>No details available yet.</span><strong>--</strong></div>';
    return;
  }
  table.innerHTML = rows.map((row) => `<div class="table-row"><span>${row.label}</span><strong>${row.value}</strong></div>`).join("");
}

function renderTabs(trace) {
  const schema = trace.schema_alignment || {};
  const ensemble = trace.ensemble || {};
  const parsed = trace.parsed_signals || {};
  const matrix = trace.feature_matrix || {};
  const fx = trace.feature_extraction || {};
  const textDims = Number(fx.text && fx.text.dimensions || 0);
  const imageDims = Number(fx.image && fx.image.dimensions || 0);
  const numericDims = Number(fx.numeric && fx.numeric.dimensions || 0);
  const featureWidth = textDims + imageDims + numericDims;
  const renameEntries = Object.entries(schema.rename_map || {});

  const ensembleRows = Object.entries(ensemble.base_model_outputs || {}).map(([name, value]) => ({
    label: name,
    value: money(value)
  }));
  ensembleRows.push({
    label: ensemble.stacker_enabled ? "Stacker Output" : "Ensemble Mean",
    value: money(ensemble.final_prediction)
  });
  fillTable("ensembleTable", ensembleRows);

  fillTable("schemaTable", [
    { label: "Rename Map", value: renameEntries.length ? renameEntries.map(([from, to]) => `${from} -> ${to}`).join(", ") : "Already canonical" },
    { label: "Resolved Text Column", value: schema.resolved_text_col || "catalog_content" },
    { label: "Resolved Image Column", value: schema.resolved_image_col || "image_link" },
    { label: "Original Columns", value: (schema.original_columns || []).join(", ") || "--" },
    { label: "Normalized Columns", value: (schema.normalized_columns || []).join(", ") || "--" },
    { label: "Raw Feature Matrix", value: Array.isArray(matrix.raw_shape) ? `(${matrix.raw_shape.join(", ")})` : "--" },
    { label: "Final Feature Matrix", value: Array.isArray(matrix.final_shape) ? `(${matrix.final_shape.join(", ")})` : "--" },
    { label: "Feature Selection", value: matrix.selection && matrix.selection.applied ? "applied" : "not applied" },
    { label: "Post Log Transform", value: matrix.post_log_transform && matrix.post_log_transform.applied ? "applied" : "not applied" }
  ]);

  fillTable("parsedTable", [
    { label: "parsed_value", value: `${parsed.parsed_value ?? 0}` },
    { label: "parsed_unit", value: parsed.parsed_unit || "none" },
    { label: "parsed_ounces", value: `${parsed.parsed_ounces ?? 0}` },
    { label: "parsed_quantity_mentions", value: `${parsed.quantity_mentions ?? 0}` },
    { label: "parsed_total_weight_g", value: `${parsed.total_weight_g ?? 0}` },
    { label: "parsed_total_volume_ml", value: `${parsed.total_volume_ml ?? 0}` },
    { label: "parsed_total_count_units", value: `${parsed.total_count_units ?? 0}` },
    { label: "text_blank_rows", value: `${fx.text && fx.text.blank_rows || 0}` },
    { label: "image_backend", value: fx.image && (fx.image.model_name || fx.image.backend) || "n/a" },
    { label: "numeric_columns", value: fx.numeric && (fx.numeric.columns || []).join(", ") || "--" }
  ]);

  node("schemaView").textContent = "sample_id, catalog_content, image_link";
  node("featureShapeView").textContent = `(1, ${featureWidth || "--"})`;
  node("schemaSummary").textContent = renameEntries.length ? "aliases remapped" : "already canonical";
  node("featureWidthView").textContent = featureWidth || "--";
  node("modelCountView").textContent = `${ensemble.base_model_count || 0} models`;
  node("fallbackStateView").textContent = Number(fx.image && fx.image.zero_rows || 0) > 0 ? "image fallback active" : "no image fallback";
  node("rawTraceBox").textContent = JSON.stringify(trace, null, 2);
}

function selectTab(name) {
  document.querySelectorAll(".tab").forEach((tab) => {
    tab.classList.toggle("is-active", tab.dataset.tab === name);
  });
  document.querySelectorAll(".tab-panel").forEach((panel) => {
    panel.classList.toggle("is-active", panel.dataset.panel === name);
  });
}

async function runInference() {
  const runId = Date.now();
  activeTerminalRun = runId;
  startTerminal(runId);
  node("predictionMeta").textContent = "Serving pipeline executing. Terminal trace is live.";
  node("predictedPrice").textContent = "$--";
  node("variantView").textContent = "--";
  node("canaryView").textContent = "--";
  node("deltaView").textContent = "--";
  node("schemaSummary").textContent = "processing";
  node("featureWidthView").textContent = "--";
  node("modelCountView").textContent = "--";
  node("fallbackStateView").textContent = "pending";
  node("rawTraceBox").textContent = "Diagnostics will appear after the model responds.";
  if (serviceSnapshot) {
    node("serviceSnapshotBox").textContent = JSON.stringify(serviceSnapshot, null, 2);
  }
  selectTab("schema");

  try {
    const response = await fetch("/v1/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(buildPayload())
    });
    const data = await response.json();
    const requestId = response.headers.get("x-request-id") || "n/a";
    pushTerminalLine(runId, "1.42s", `Request ID confirmed: ${requestId}.`);

    if (!response.ok) {
      pushTerminalLine(runId, "1.45s", `Serving pipeline failed: ${JSON.stringify(data.detail || data)}`, "bad");
      node("predictionMeta").textContent = `Execution failed (${response.status}).`;
      fillTable("ensembleTable", []);
      fillTable("parsedTable", []);
      return;
    }

    const trace = data.trace || {};
    const imageZeroRows = Number(trace.feature_extraction && trace.feature_extraction.image && trace.feature_extraction.image.zero_rows || 0);
    if (imageZeroRows > 0) {
      pushTerminalLine(runId, "0.85s", "Image URL failed -> Zero-vector fallback applied safely.", "warn");
    } else {
      pushTerminalLine(runId, "0.85s", "Image embedding step completed without fallback.");
    }
    pushTerminalLine(runId, "1.30s", "Stacker ensemble complete. Returning prediction.");

    const prediction = data.predictions && data.predictions[0] ? data.predictions[0].predicted_price : null;
    const reference = readForm().reference_price === "" ? null : Number(readForm().reference_price);
    const delta = prediction === null || reference === null || Number.isNaN(reference) ? null : Number(prediction) - reference;

    node("predictedPrice").textContent = money(prediction);
    node("predictionMeta").textContent = `Pipeline complete. Active model path: ${data.model_variant || "primary"}.`;
    node("variantView").textContent = data.model_variant || "--";
    node("canaryView").textContent = data.canary_divergence_mae === null || data.canary_divergence_mae === undefined ? "--" : Number(data.canary_divergence_mae).toFixed(4);
    node("deltaView").textContent = delta === null ? "--" : `${delta >= 0 ? "+" : ""}${delta.toFixed(3)}`;
    renderTabs(trace);

    await pollSystem();
  } catch (error) {
    pushTerminalLine(runId, "1.45s", `Network or runtime error: ${String(error)}`, "bad");
    node("predictionMeta").textContent = "Execution could not complete.";
  } finally {
    node("terminalStatus").textContent = "Execution finished";
  }
}

node("sampleBtn").addEventListener("click", loadSample);
node("runBtn").addEventListener("click", runInference);
node("simulateBroken").addEventListener("change", syncPreview);
node("imagePreview").addEventListener("error", () => {
  node("imagePreview").style.display = "none";
  node("imageFallback").style.display = "block";
});

["catalogContent", "imageUrl", "sampleId", "numericValue", "unitText", "categoryText", "referencePrice"].forEach((id) => {
  node(id).addEventListener("input", syncPreview);
});

document.querySelectorAll(".tab").forEach((tab) => {
  tab.addEventListener("click", () => selectTab(tab.dataset.tab));
});

loadSample();
node("rawTraceBox").textContent = "Diagnostics will appear after the model responds.";
node("serviceSnapshotBox").textContent = "Service snapshot will appear after polling /service/info.";
fillTable("ensembleTable", []);
fillTable("schemaTable", []);
fillTable("parsedTable", []);
pollSystem();
window.setInterval(pollSystem, 5000);
