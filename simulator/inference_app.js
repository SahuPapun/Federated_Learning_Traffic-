/**
 * inference_app.js – Frontend logic for the Inference Dashboard
 *
 * Talks to the REST API at API_BASE (default: http://localhost:8002).
 */

const API_BASE = 'http://localhost:8002';

// ---- DOM refs (populated after DOMContentLoaded) ----
let uploadZone, fileInput, fileInfo;
let clientSelect, predictBtn;
let statusBar, statusMsg, statusSpinner;
let resultsPanel, emptyState;
let metricsSection, mseVal, maeVal, rmseVal, r2Val, numPredVal;
let chartCanvas, errorChartCanvas;
let tableBody;
let downloadBtn, predInfoText;
let chartTabMain, chartTabError;
let mainChartWrap, errorChartWrap;

let currentPredictionId = null;
let currentRows = [];

// ---- Initialise ----
document.addEventListener('DOMContentLoaded', () => {
  uploadZone    = document.getElementById('uploadZone');
  fileInput     = document.getElementById('fileInput');
  fileInfo      = document.getElementById('fileInfo');
  clientSelect  = document.getElementById('clientSelect');
  predictBtn    = document.getElementById('predictBtn');
  statusBar     = document.getElementById('statusBar');
  statusMsg     = document.getElementById('statusMsg');
  statusSpinner = document.getElementById('statusSpinner');
  resultsPanel  = document.getElementById('resultsPanel');
  emptyState    = document.getElementById('emptyState');
  metricsSection = document.getElementById('metricsSection');
  mseVal        = document.getElementById('mseVal');
  maeVal        = document.getElementById('maeVal');
  rmseVal       = document.getElementById('rmseVal');
  r2Val         = document.getElementById('r2Val');
  numPredVal    = document.getElementById('numPredVal');
  chartCanvas      = document.getElementById('mainChart');
  errorChartCanvas = document.getElementById('errorChart');
  tableBody     = document.getElementById('tableBody');
  downloadBtn   = document.getElementById('downloadBtn');
  predInfoText  = document.getElementById('predInfoText');
  chartTabMain  = document.getElementById('chartTabMain');
  chartTabError = document.getElementById('chartTabError');
  mainChartWrap = document.getElementById('mainChartWrap');
  errorChartWrap = document.getElementById('errorChartWrap');

  // Wire events
  uploadZone.addEventListener('dragover', onDragOver);
  uploadZone.addEventListener('dragleave', onDragLeave);
  uploadZone.addEventListener('drop', onDrop);
  fileInput.addEventListener('change', onFileChange);
  predictBtn.addEventListener('click', onPredict);
  downloadBtn.addEventListener('click', onDownload);
  chartTabMain.addEventListener('click', () => switchChart('main'));
  chartTabError.addEventListener('click', () => switchChart('error'));

  // Load available models into dropdown
  loadAvailableModels();
});

// ---- Drag-and-drop ----
function onDragOver(e) {
  e.preventDefault();
  uploadZone.classList.add('drag-over');
}

function onDragLeave() {
  uploadZone.classList.remove('drag-over');
}

function onDrop(e) {
  e.preventDefault();
  uploadZone.classList.remove('drag-over');
  const file = e.dataTransfer.files[0];
  if (file) setFile(file);
}

function onFileChange() {
  if (fileInput.files[0]) setFile(fileInput.files[0]);
}

function setFile(file) {
  if (!file.name.toLowerCase().endsWith('.csv')) {
    showStatus('error', '⚠ Only CSV files are supported.');
    return;
  }
  // Attach file to a DataTransfer so fileInput.files reflects the drag-dropped file
  const dt = new DataTransfer();
  dt.items.add(file);
  fileInput.files = dt.files;

  fileInfo.textContent = `📄 ${file.name}  (${(file.size / 1024).toFixed(1)} KB)`;
  fileInfo.style.display = 'block';
  showStatus('info', 'File ready. Select a client and click Predict.');
}

// ---- Load Models ----
async function loadAvailableModels() {
  try {
    const resp = await fetch(`${API_BASE}/api/inference/available-models`);
    if (!resp.ok) throw new Error('Server error');
    const data = await resp.json();

    clientSelect.innerHTML = '';
    data.models.forEach(m => {
      const opt = document.createElement('option');
      opt.value = m.client_id;
      const badge = m.available ? '✅' : '❌';
      opt.textContent = `${badge} Client ${m.client_id} – ${m.name}`;
      opt.disabled = !m.available;
      clientSelect.appendChild(opt);
    });

    const anyAvailable = data.models.some(m => m.available);
    if (!anyAvailable) {
      showStatus('error', '⚠ No trained models found. Please run python run.py first.');
    }
  } catch {
    showStatus('error', '⚠ Cannot connect to Inference API at ' + API_BASE +
      '. Make sure to start it with: python api_inference_server.py');
  }
}

// ---- Predict ----
async function onPredict() {
  if (!fileInput.files || !fileInput.files[0]) {
    showStatus('error', '⚠ Please upload a CSV file first.');
    return;
  }

  const clientId = clientSelect.value;
  predictBtn.disabled = true;
  showStatus('info', '⏳ Running predictions…', true);
  ChartUtils.destroyAll();
  clearResults();

  const formData = new FormData();
  formData.append('file', fileInput.files[0]);
  formData.append('client_id', clientId);

  try {
    const resp = await fetch(`${API_BASE}/api/inference/predict`, {
      method: 'POST',
      body: formData,
    });
    const data = await resp.json();

    if (!resp.ok) {
      showStatus('error', `⚠ ${data.error || 'Prediction failed.'}`);
      return;
    }

    currentPredictionId = data.prediction_id;
    currentRows = data.rows;

    renderMetrics(data.metrics);
    renderCharts(data.rows);
    renderTable(data.rows);

    predInfoText.textContent =
      `Client ${data.client_id} · ${data.client_name} · ${data.rows.length} predictions`;
    downloadBtn.style.display = 'inline-flex';
    emptyState.style.display = 'none';
    metricsSection.style.display = 'block';
    showStatus('success', `✅ Done! ${data.rows.length} predictions generated.`);
  } catch (err) {
    showStatus('error', `⚠ Network error: ${err.message}`);
  } finally {
    predictBtn.disabled = false;
  }
}

// ---- Render Metrics ----
function renderMetrics(m) {
  mseVal.textContent    = m.mse.toFixed(4);
  maeVal.textContent    = m.mae.toFixed(4);
  rmseVal.textContent   = m.rmse.toFixed(4);
  r2Val.textContent     = m.r2.toFixed(4);
  numPredVal.textContent = m.num_predictions;
}

// ---- Render Charts ----
function renderCharts(rows) {
  const labels    = rows.map(r => r.index);
  const actual    = rows.map(r => r.actual);
  const predicted = rows.map(r => r.predicted);
  const errors    = rows.map(r => r.error);

  ChartUtils.renderMainChart(chartCanvas, labels, actual, predicted);
  ChartUtils.renderErrorChart(errorChartCanvas, labels, errors);

  switchChart('main');
}

// ---- Render Table ----
function renderTable(rows) {
  const MAX_ROWS = 500;
  const displayed = rows.slice(0, MAX_ROWS);
  tableBody.innerHTML = displayed.map(r => {
    const errClass = r.error > 0 ? 'positive-error' : 'negative-error';
    return `
      <tr>
        <td>${r.index}</td>
        <td>${r.actual.toFixed(4)}</td>
        <td>${r.predicted.toFixed(4)}</td>
        <td class="${errClass}">${r.error.toFixed(4)}</td>
      </tr>`;
  }).join('');

  if (rows.length > MAX_ROWS) {
    const tr = document.createElement('tr');
    tr.innerHTML = `<td colspan="4" style="text-align:center;color:var(--muted);padding:12px">
      … showing first ${MAX_ROWS} of ${rows.length} rows. Download CSV for full results.
    </td>`;
    tableBody.appendChild(tr);
  }
}

// ---- Download ----
function onDownload() {
  if (!currentPredictionId) return;
  window.location.href = `${API_BASE}/api/inference/export/${currentPredictionId}`;
}

// ---- Chart Tabs ----
function switchChart(which) {
  if (which === 'main') {
    mainChartWrap.style.display  = 'block';
    errorChartWrap.style.display = 'none';
    chartTabMain.classList.add('active');
    chartTabError.classList.remove('active');
  } else {
    mainChartWrap.style.display  = 'none';
    errorChartWrap.style.display = 'block';
    chartTabMain.classList.remove('active');
    chartTabError.classList.add('active');
  }
}

// ---- Helpers ----
function showStatus(type, msg, spinner = false) {
  statusBar.className = `status-bar ${type}`;
  statusBar.style.display = 'flex';
  statusMsg.textContent = msg;
  statusSpinner.style.display = spinner ? 'block' : 'none';
}

function clearResults() {
  tableBody.innerHTML = '';
  currentPredictionId = null;
  currentRows = [];
  downloadBtn.style.display = 'none';
  predInfoText.textContent = '';
  metricsSection.style.display = 'none';
  emptyState.style.display = 'flex';
}
