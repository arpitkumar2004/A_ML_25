/* ==========================================================================
   NEURALIS Senior UI/UX Master Application Controller
   ========================================================================== */

const SAMPLE_ITEM = {
  sample_id: 1001,
  catalog_content: [
    "Item Name: Log Cabin Sugar Free Syrup, 24 FL OZ (Pack of 12)",
    "Bullet Point 1: Contains twelve (12) 24-ounce bottles of Log Cabin Sugar Free Syrup for Pancakes and Waffles",
    "Bullet Point 2: Indulge in thick, delicious syrup for pancakes, waffles, French toast and more",
    "Bullet Point 3: 90% fewer calories than our original syrup and no sugar or high fructose corn syrup",
    "Bullet Point 4: Amazing syrup that you can feel good about serving to your family and guests",
    "Bullet Point 5: Stock up on this breakfast staple for decadent pancakes and waffles anytime"
  ].join("\n"),
  image_link: "https://m.media-amazon.com/images/I/71QD2OFXqDL.jpg",
  category: "Pantry & Grocery",
  reference_price: 2.195
};

const node = (id) => document.getElementById(id);
const money = (val) => val === null || val === undefined || Number.isNaN(Number(val))
  ? "--"
  : Number(val).toLocaleString("en-US", { style: "currency", currency: "USD", minimumFractionDigits: 2, maximumFractionDigits: 2 });
const percent = (val) => `${(Number(val || 0) * 100).toFixed(0)}%`;

let currentPriceValue = null;
let uploadedImageDataUrl = "";
let valuationHistory = [];
let isCalculating = false;

// Initialize Session Valuation History from localStorage
function initHistory() {
  try {
    const saved = localStorage.getItem("neuralis_valuation_history");
    if (saved) valuationHistory = JSON.parse(saved);
  } catch (e) {
    valuationHistory = [];
  }
  renderHistoryTable();
}

function saveValuationHistory(item) {
  valuationHistory.unshift(item);
  if (valuationHistory.length > 25) valuationHistory.pop();
  try {
    localStorage.setItem("neuralis_valuation_history", JSON.stringify(valuationHistory));
  } catch (e) {}
  renderHistoryTable();
}

function renderHistoryTable() {
  const tbody = node("historyTableBody");
  const badge = node("historyCountBadge");
  if (!tbody || !badge) return;
  badge.textContent = valuationHistory.length;

  if (valuationHistory.length === 0) {
    tbody.innerHTML = `<tr class="empty-row"><td colspan="7">No saved valuation history yet. Run a calculation in the workspace!</td></tr>`;
    return;
  }

  tbody.innerHTML = valuationHistory.map((item, idx) => `
    <tr>
      <td><small>${item.time}</small></td>
      <td><strong>${escapeHtml(item.title)}</strong></td>
      <td><span class="brand-badge">${escapeHtml(item.category)}</span></td>
      <td><strong style="color: var(--accent-primary)">${money(item.predPrice)}</strong></td>
      <td>${item.refPrice ? money(item.refPrice) : "--"}</td>
      <td>${item.confidence}</td>
      <td>
        <button type="button" class="btn btn-ghost btn-xs" onclick="reloadHistoryItem(${idx})">Reload</button>
      </td>
    </tr>
  `).join("");
}

function reloadHistoryItem(idx) {
  const item = valuationHistory[idx];
  if (!item) return;
  
  switchTab("tabValuation");
  node("catalogContent").value = item.rawContent || "";
  node("imageUrl").value = item.imageUrl || "";
  node("categorySelect").value = item.category || "Pantry & Grocery";
  node("referencePrice").value = item.refPrice || "";
  uploadedImageDataUrl = item.uploadedImage || "";
  
  updateImagePreview();
  updateCharCount();
  handleCategoryChange();
}

window.reloadHistoryItem = reloadHistoryItem;

function clearHistory() {
  if (confirm("Are you sure you want to clear session valuation history?")) {
    valuationHistory = [];
    localStorage.removeItem("neuralis_valuation_history");
    renderHistoryTable();
  }
}

function exportHistoryCSV() {
  if (valuationHistory.length === 0) {
    alert("No history records available to export.");
    return;
  }
  const headers = ["Time", "Category", "Estimated Price", "Listed Price", "Confidence", "Description"];
  const rows = valuationHistory.map(h => [
    `"${h.time}"`,
    `"${h.category}"`,
    h.predPrice,
    h.refPrice || "",
    `"${h.confidence}"`,
    `"${(h.rawContent || "").replace(/"/g, '""')}"`
  ]);
  const csvContent = "data:text/csv;charset=utf-8," + [headers.join(","), ...rows.map(e => e.join(","))].join("\n");
  const encodedUri = encodeURI(csvContent);
  const link = document.createElement("a");
  link.setAttribute("href", encodedUri);
  link.setAttribute("download", `neuralis_valuations_${Date.now()}.csv`);
  document.body.appendChild(link);
  link.click();
  document.body.removeChild(link);
}

function escapeHtml(str) {
  return String(str || "").replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;").replace(/"/g, "&quot;");
}

// Price Animated Counter
function animatePredictionPrice(targetValue) {
  if (targetValue === null || targetValue === undefined || Number.isNaN(Number(targetValue))) {
    currentPriceValue = null;
    node("predictedPrice").textContent = "$--";
    return;
  }

  const to = Number(targetValue);
  const from = currentPriceValue === null ? Math.max(0, to * 0.85) : currentPriceValue;
  const durationMs = 600;
  const startTs = performance.now();

  function step(now) {
    const t = Math.min(1, (now - startTs) / durationMs);
    const eased = 1 - Math.pow(1 - t, 3);
    const value = from + (to - from) * eased;
    node("predictedPrice").textContent = money(value);
    if (t < 1) {
      window.requestAnimationFrame(step);
    } else {
      currentPriceValue = to;
      node("predictedPrice").textContent = money(to);
    }
  }
  window.requestAnimationFrame(step);
}

// Image Preview Controller
function updateImagePreview() {
  const urlInput = (node("imageUrl").value || "").trim();
  const activeSource = uploadedImageDataUrl || urlInput;
  const previewImage = node("imagePreview");
  const previewFallback = node("imageFallback");
  const removeBtn = node("removeImageBtn");

  if (activeSource) {
    previewImage.src = activeSource;
    previewImage.style.display = "block";
    previewFallback.style.display = "none";
    if (removeBtn) removeBtn.classList.remove("is-hidden");
  } else {
    previewImage.removeAttribute("src");
    previewImage.style.display = "none";
    previewFallback.style.display = "flex";
    if (removeBtn) removeBtn.classList.add("is-hidden");
  }
}

function updateCharCount() {
  const len = (node("catalogContent").value || "").length;
  if (node("catalogCharCount")) node("catalogCharCount").textContent = `${len} chars`;
}

function handleCategoryChange() {
  const selectVal = node("categorySelect").value;
  const customGroup = node("customCategoryGroup");
  if (selectVal === "Other") {
    customGroup.classList.remove("is-hidden");
  } else {
    customGroup.classList.add("is-hidden");
  }
}

function getSelectedCategory() {
  const selectVal = node("categorySelect").value;
  if (selectVal === "Other") {
    return (node("categoryCustom").value || "").trim() || "General / Other";
  }
  return selectVal;
}

function readForm() {
  const urlInput = (node("imageUrl").value || "").trim();
  const imageSource = uploadedImageDataUrl || urlInput;
  const rawRef = node("referencePrice").value;

  return {
    catalog_content: (node("catalogContent").value || "").trim(),
    image_link: imageSource,
    category: getSelectedCategory(),
    reference_price: rawRef !== "" && !Number.isNaN(Number(rawRef)) ? Number(rawRef) : null,
    min_value: Number(node("optMinValue") ? node("optMinValue").value : 0.0),
    round: node("optRound") ? node("optRound").value === "true" : true
  };
}

function buildPayload() {
  const form = readForm();
  const record = {
    sample_id: 1001,
    catalog_content: form.catalog_content,
    Description: form.catalog_content,
    image_link: form.image_link || "",
    image_path: form.image_link || "",
    category: form.category
  };

  if (form.reference_price !== null) {
    record.price = form.reference_price;
    record.Price = form.reference_price;
  }

  return {
    records: [record],
    text_col: "catalog_content",
    image_col: "image_link",
    id_col: "sample_id",
    pred_col: "predicted_price",
    min_value: form.min_value,
    round: form.round
  };
}

// Render Results & Diagnostics
function renderResults(prediction, reference, trace, canaryMae) {
  if (prediction === null || prediction === undefined || Number.isNaN(Number(prediction))) {
    animatePredictionPrice(null);
    node("predictionMeta").textContent = "Valuation calculation failed.";
    node("confidenceBand").textContent = "$-- – $--";
    node("confidenceBar").style.width = "0%";
    node("confidenceNote").textContent = "Unable to estimate confidence interval.";
    node("deltaView").textContent = "--";
    node("confidenceRatingView").textContent = "--";
    node("deltaPill").classList.add("is-hidden");
    if (node("traceCodeBlock")) node("traceCodeBlock").textContent = JSON.stringify(trace || { error: "No trace details available" }, null, 2);
    return;
  }

  const predVal = Number(prediction);
  animatePredictionPrice(predVal);
  node("predictionMeta").textContent = "Multimodal valuation completed across vision, NLP & stacking ensembler.";

  // Reference & Delta Calculation
  if (reference !== null && !Number.isNaN(reference)) {
    const delta = predVal - reference;
    const pct = reference > 0 ? (delta / reference) * 100 : 0;
    const sign = delta >= 0 ? "+" : "";
    const deltaText = `${sign}${money(delta)} (${sign}${pct.toFixed(1)}%)`;

    node("referenceView").textContent = money(reference);
    node("deltaView").textContent = deltaText;

    const deltaPill = node("deltaPill");
    deltaPill.textContent = `${sign}${money(delta)} vs Listed`;
    deltaPill.className = `delta-pill ${delta >= 0 ? "positive" : "negative"}`;
    deltaPill.classList.remove("is-hidden");
  } else {
    node("referenceView").textContent = "--";
    node("deltaView").textContent = "--";
    node("deltaPill").classList.add("is-hidden");
  }

  // Confidence Band
  const mae = canaryMae === null || canaryMae === undefined || Number.isNaN(Number(canaryMae)) ? 0.04 : Number(canaryMae);
  const fx = trace.feature_extraction || {};
  const imageFallbackActive = Number(fx.image && fx.image.zero_rows || 0) > 0;

  let confidenceScore = 0.92 - Math.min(0.35, mae * 2.5);
  if (imageFallbackActive) confidenceScore -= 0.15;
  confidenceScore = Math.max(0.35, Math.min(0.98, confidenceScore));

  const spread = Math.max(0.06, (1 - confidenceScore) * 0.24);
  const lowRange = Math.max(0, predVal * (1 - spread));
  const highRange = predVal * (1 + spread);

  node("confidenceBand").textContent = `${money(lowRange)} – ${money(highRange)}`;
  node("confidenceBar").style.width = `${(confidenceScore * 100).toFixed(0)}%`;

  const ratingText = confidenceScore >= 0.8 ? "High" : confidenceScore >= 0.6 ? "Moderate" : "Low";
  node("confidenceRatingView").textContent = `${ratingText} (${(confidenceScore * 100).toFixed(0)}%)`;

  node("confidenceNote").textContent = imageFallbackActive
    ? "Visual fallback applied for missing image. Estimated price range is slightly wider."
    : confidenceScore >= 0.8
      ? "High confidence valuation based on rich text & visual alignment."
      : "Moderate confidence valuation based on extracted signals.";

  // Drivers Breakdown
  const textDims = Number(fx.text && fx.text.dimensions || 0);
  const imageDims = Number(fx.image && fx.image.dimensions || 0);
  const numericDims = Number(fx.numeric && fx.numeric.dimensions || 0);
  const total = Math.max(1, textDims + imageDims + numericDims);

  const textShare = textDims / total;
  const imageShare = imageDims / total;
  const numericShare = numericDims / total;

  node("textDriver").textContent = percent(textShare);
  node("imageDriver").textContent = percent(imageShare);
  node("numericDriver").textContent = percent(numericShare);

  node("textDriverBar").style.width = `${(textShare * 100).toFixed(0)}%`;
  node("imageDriverBar").style.width = `${(imageShare * 100).toFixed(0)}%`;
  node("numericDriverBar").style.width = `${(numericShare * 100).toFixed(0)}%`;

  node("textDriverNote").textContent = textDims > 0 ? `${textDims} active NLP features` : "Standard text signals";
  node("imageDriverNote").textContent = imageFallbackActive ? "Image fallback used" : imageDims > 0 ? `${imageDims} visual features` : "No image signals";
  node("numericDriverNote").textContent = numericDims > 0 ? `${numericDims} quantity features` : "Standard quantity signals";

  // Diagnostics Code Block
  if (node("traceCodeBlock")) node("traceCodeBlock").textContent = JSON.stringify(trace, null, 2);

  // Save Valuation Record to Session History
  const form = readForm();
  saveValuationHistory({
    time: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' }),
    title: (form.catalog_content.split("\n")[0] || "Product Item").substring(0, 45),
    category: form.category,
    predPrice: predVal,
    refPrice: reference,
    confidence: `${ratingText} (${(confidenceScore * 100).toFixed(0)}%)`,
    rawContent: form.catalog_content,
    imageUrl: node("imageUrl").value,
    uploadedImage: uploadedImageDataUrl
  });
}

function resetForm() {
  node("catalogContent").value = "";
  node("imageUrl").value = "";
  uploadedImageDataUrl = "";
  node("categorySelect").value = "Pantry & Grocery";
  node("categoryCustom").value = "";
  node("referencePrice").value = "";
  updateCharCount();
  handleCategoryChange();
  updateImagePreview();

  animatePredictionPrice(null);
  node("predictionMeta").textContent = "Enter catalog content on the left and click 'Calculate Valuation'.";
  node("confidenceBand").textContent = "$-- – $--";
  node("confidenceBar").style.width = "0%";
  node("confidenceNote").textContent = "Valuation confidence score updates automatically upon pipeline completion.";
  node("referenceView").textContent = "--";
  node("deltaView").textContent = "--";
  node("confidenceRatingView").textContent = "--";
  node("deltaPill").classList.add("is-hidden");

  ["textDriver", "imageDriver", "numericDriver"].forEach((id) => node(id).textContent = "--");
  ["textDriverBar", "imageDriverBar", "numericDriverBar"].forEach((id) => node(id).style.width = "0%");
  node("textDriverNote").textContent = "Awaiting calculation";
  node("imageDriverNote").textContent = "Awaiting calculation";
  node("numericDriverNote").textContent = "Awaiting calculation";
  if (node("traceCodeBlock")) node("traceCodeBlock").textContent = "Run a valuation to inspect pipeline diagnostics JSON...";
}

function loadSample() {
  resetForm();
  node("catalogContent").value = SAMPLE_ITEM.catalog_content;
  node("imageUrl").value = SAMPLE_ITEM.image_link;
  node("categorySelect").value = SAMPLE_ITEM.category;
  node("referencePrice").value = SAMPLE_ITEM.reference_price;
  updateCharCount();
  handleCategoryChange();
  updateImagePreview();
}

async function runValuation() {
  if (isCalculating) return;
  const form = readForm();

  if (!form.catalog_content) {
    node("catalogContent").focus();
    node("predictionMeta").textContent = "Please provide a catalog description before calculating.";
    return;
  }

  isCalculating = true;
  const runBtn = node("runBtn");
  const runBtnText = node("runBtnText");
  const progressCard = node("pipelineProgressCard");

  runBtn.disabled = true;
  if (runBtnText) runBtnText.textContent = "Calculating...";
  if (progressCard) progressCard.classList.remove("is-hidden");

  // Step 1 Animation
  const stepText = node("stepText");
  const stepVision = node("stepVision");
  const stepEnsemble = node("stepEnsemble");

  if (stepText) stepText.className = "step-item active";
  if (stepVision) stepVision.className = "step-item";
  if (stepEnsemble) stepEnsemble.className = "step-item";

  try {
    await new Promise(r => setTimeout(r, 200));
    if (stepText) stepText.className = "step-item done";
    if (stepVision) stepVision.className = "step-item active";

    const response = await fetch("/v1/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(buildPayload())
    });

    if (stepVision) stepVision.className = "step-item done";
    if (stepEnsemble) stepEnsemble.className = "step-item active";

    const data = await response.json();
    await new Promise(r => setTimeout(r, 200));
    if (stepEnsemble) stepEnsemble.className = "step-item done";

    if (!response.ok) {
      node("predictionMeta").textContent = `Server Error: ${data.detail || "Unable to compute valuation."}`;
      renderResults(null, null, data, null);
      return;
    }

    const prediction = data.predictions && data.predictions[0] ? data.predictions[0].predicted_price : null;
    const trace = data.trace || {};
    renderResults(prediction, form.reference_price, trace, data.canary_divergence_mae);

  } catch (err) {
    node("predictionMeta").textContent = "Network Error: Could not connect to the FastAPI valuation server.";
    renderResults(null, null, {}, null);
  } finally {
    isCalculating = false;
    runBtn.disabled = false;
    if (runBtnText) runBtnText.textContent = "Calculate Valuation";
    if (progressCard) setTimeout(() => progressCard.classList.add("is-hidden"), 1000);
  }
}

// Copy Valuation Summary
function copyValuationSummary() {
  const price = node("predictedPrice").textContent;
  const band = node("confidenceBand").textContent;
  const delta = node("deltaView").textContent;
  const desc = node("catalogContent").value.substring(0, 100);

  const summary = `NEURALIS Valuation Summary:\nEstimated Market Price: ${price}\nConfidence Band (95%): ${band}\nDelta vs Listed: ${delta}\nProduct: ${desc}...`;
  navigator.clipboard.writeText(summary).then(() => {
    alert("Valuation summary copied to clipboard!");
  }).catch(() => {
    alert(summary);
  });
}

// Tab Switching System
function switchTab(tabId) {
  const tabs = ["tabValuation", "tabTelemetry", "tabHistory"];
  const panels = ["viewValuation", "viewTelemetry", "viewHistory"];

  tabs.forEach((tId, idx) => {
    const tabEl = node(tId);
    const panelEl = node(panels[idx]);
    if (!tabEl || !panelEl) return;
    if (tId === tabId) {
      tabEl.classList.add("active");
      tabEl.setAttribute("aria-selected", "true");
      panelEl.classList.add("active");
    } else {
      tabEl.classList.remove("active");
      tabEl.setAttribute("aria-selected", "false");
      panelEl.classList.remove("active");
    }
  });

  if (tabId === "tabTelemetry") pollTelemetry();
}

// File Dropzone Handler
function initFileDropzone() {
  const dropzone = node("dropzone");
  const fileInput = node("imageFileInput");
  const browseBtn = node("browseFilesBtn");
  if (!dropzone || !fileInput || !browseBtn) return;

  browseBtn.addEventListener("click", () => fileInput.click());

  fileInput.addEventListener("change", (e) => {
    const file = e.target.files && e.target.files[0];
    if (file) processImageFile(file);
  });

  ['dragenter', 'dragover'].forEach(eventName => {
    dropzone.addEventListener(eventName, (e) => {
      e.preventDefault();
      e.stopPropagation();
      dropzone.classList.add('is-dragover');
    }, false);
  });

  ['dragleave', 'drop'].forEach(eventName => {
    dropzone.addEventListener(eventName, (e) => {
      e.preventDefault();
      e.stopPropagation();
      dropzone.classList.remove('is-dragover');
    }, false);
  });

  dropzone.addEventListener('drop', (e) => {
    const dt = e.dataTransfer;
    const file = dt.files && dt.files[0];
    if (file) processImageFile(file);
  });
}

function processImageFile(file) {
  if (!file.type.startsWith('image/')) {
    alert('Please select a valid image file (PNG, JPG, WEBP).');
    return;
  }
  const reader = new FileReader();
  reader.onload = (e) => {
    uploadedImageDataUrl = e.target.result;
    node("imageUrl").value = "";
    updateImagePreview();
  };
  reader.readAsDataURL(file);
}

// Accordion Toggles
function initAccordions() {
  const setupAccordion = (toggleId, bodyId) => {
    const toggle = node(toggleId);
    const body = node(bodyId);
    if (!toggle || !body) return;

    toggle.addEventListener("click", () => {
      const expanded = toggle.getAttribute("aria-expanded") === "true";
      toggle.setAttribute("aria-expanded", !expanded);
      if (expanded) {
        body.classList.add("is-hidden");
      } else {
        body.classList.remove("is-hidden");
      }
    });
  };

  setupAccordion("accordionToggle", "accordionBody");
  setupAccordion("traceToggle", "traceBody");
}

// Telemetry & Health Polling
async function pollHealth() {
  const statusDot = node("statusDot");
  const statusText = node("apiStatusText");
  if (!statusDot || !statusText) return;
  try {
    const res = await fetch("/healthz");
    if (res.ok) {
      statusDot.className = "status-dot ok";
      statusText.textContent = "Status: Online";
    } else {
      statusDot.className = "status-dot warning";
      statusText.textContent = "Status: Degraded";
    }
  } catch (e) {
    statusDot.className = "status-dot err";
    statusText.textContent = "Status: Offline";
  }
}

async function pollTelemetry() {
  try {
    const res = await fetch("/metrics/json");
    if (!res.ok) return;
    const data = await res.json();

    if (node("telRequestCount")) node("telRequestCount").textContent = (data.request_count || 0).toLocaleString();
    if (node("telP50")) node("telP50").textContent = `${(data.latency_ms?.p50 || 0).toFixed(1)} ms`;
    if (node("telP95")) node("telP95").textContent = `${(data.latency_ms?.p95 || 0).toFixed(1)} ms`;
    if (node("telErrorRate")) node("telErrorRate").textContent = `${((data.error_rate || 0) * 100).toFixed(1)}%`;

    const s = data.service || {};
    if (node("telRunId")) node("telRunId").textContent = s.run_id || "Legacy Bundle";
    if (node("telBundlePath")) node("telBundlePath").textContent = s.bundle_path || "Default Path";
    if (node("telEnvStage")) node("telEnvStage").textContent = s.environment || "PRODUCTION";

    if (s.links?.github_repo && node("linkGithub")) node("linkGithub").href = s.links.github_repo;
    if (s.links?.dagshub_repo && node("linkDagshub")) node("linkDagshub").href = s.links.dagshub_repo;
  } catch (e) {}
}

// Keyboard Shortcut Listeners
document.addEventListener("keydown", (e) => {
  if ((e.ctrlKey || e.metaKey) && e.key === "Enter") {
    e.preventDefault();
    runValuation();
  } else if (e.key === "Escape") {
    resetForm();
  } else if (e.altKey && (e.key === "s" || e.key === "S")) {
    e.preventDefault();
    loadSample();
  }
});

// Event Binding
if (node("tabValuation")) node("tabValuation").addEventListener("click", () => switchTab("tabValuation"));
if (node("tabTelemetry")) node("tabTelemetry").addEventListener("click", () => switchTab("tabTelemetry"));
if (node("tabHistory")) node("tabHistory").addEventListener("click", () => switchTab("tabHistory"));

if (node("sampleBtn")) node("sampleBtn").addEventListener("click", loadSample);
if (node("runBtn")) node("runBtn").addEventListener("click", runValuation);
if (node("resetFormBtn")) node("resetFormBtn").addEventListener("click", resetForm);
if (node("copySummaryBtn")) node("copySummaryBtn").addEventListener("click", copyValuationSummary);
if (node("exportHistoryBtn")) node("exportHistoryBtn").addEventListener("click", exportHistoryCSV);
if (node("clearHistoryBtn")) node("clearHistoryBtn").addEventListener("click", clearHistory);

if (node("catalogContent")) node("catalogContent").addEventListener("input", updateCharCount);
if (node("categorySelect")) node("categorySelect").addEventListener("change", handleCategoryChange);
if (node("imageUrl")) node("imageUrl").addEventListener("input", () => {
  uploadedImageDataUrl = "";
  updateImagePreview();
});

if (node("removeImageBtn")) {
  node("removeImageBtn").addEventListener("click", () => {
    uploadedImageDataUrl = "";
    node("imageUrl").value = "";
    updateImagePreview();
  });
}

// App Startup Initialization
initFileDropzone();
initAccordions();
initHistory();
resetForm();
pollHealth();
window.setInterval(pollHealth, 10000);

