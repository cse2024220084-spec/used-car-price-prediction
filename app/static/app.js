const el = (id) => document.getElementById(id);

let featureData = null; // cached response from GET /feature-names

// ---------------------------------------------------------------
// Tab Loading & Theme
// ---------------------------------------------------------------
const loadedTabs = {};

async function loadTab(tabId) {
  // Update button active states
  document.querySelectorAll('.tab-btn').forEach(btn => btn.classList.remove('active'));
  const activeBtn = el('btn-' + tabId);
  const mobileActiveBtn = el('mobile-btn-' + tabId);
  if (activeBtn) activeBtn.classList.add('active');
  if (mobileActiveBtn) mobileActiveBtn.classList.add('active');

  // Hide all tab containers
  document.querySelectorAll('.tab-container').forEach(container => {
    container.style.display = 'none';
  });

  const container = el(`${tabId}-container`);
  if (!container) return;

  // Load HTML if not already loaded
  if (!loadedTabs[tabId]) {
    try {
      const response = await fetch(`${tabId}.html`);
      if (!response.ok) throw new Error("Failed to load tab HTML");
      const html = await response.text();
      container.innerHTML = html;
      loadedTabs[tabId] = true;

      // Initialize tab specific logic
      if (tabId === 'prediction') {
        initPredictionTab();
      } else if (tabId === 'dataset') {
        initDatasetTab();
      }
    } catch (err) {
      console.error("Error loading tab:", err);
      container.innerHTML = `<p style="color:red; padding: 20px;">Failed to load ${tabId} tab.</p>`;
    }
  }

  // Show the requested container
  container.style.display = 'block';
}

function initTheme() {
  const savedTheme = localStorage.getItem("theme") || "dark";
  document.documentElement.setAttribute("data-theme", savedTheme);
  updateThemeButton(savedTheme);
}

function toggleTheme() {
  const currentTheme = document.documentElement.getAttribute("data-theme") || "dark";
  const newTheme = currentTheme === "dark" ? "light" : "dark";
  document.documentElement.setAttribute("data-theme", newTheme);
  localStorage.setItem("theme", newTheme);
  updateThemeButton(newTheme);
}

function toggleMobileMenu() {
  const overlay = el('mobileMenuOverlay');
  if (!overlay) return;
  if (overlay.classList.contains('active')) {
    overlay.classList.remove('active');
    document.body.style.overflow = '';
  } else {
    overlay.classList.add('active');
    document.body.style.overflow = 'hidden';
  }
}

function updateThemeButton(theme) {
  const icon = el("themeIcon");
  const label = el("themeLabel");
  if (!icon || !label) return;
  if (theme === "dark") {
    icon.textContent = "☀️";
    label.textContent = "Light Mode";
  } else {
    icon.textContent = "🌙";
    label.textContent = "Dark Mode";
  }
}

// ---------------------------------------------------------------
// API Endpoints & Utility
// ---------------------------------------------------------------
const BASE_URLS = [
  window.location.protocol.startsWith("http") ? window.location.origin : null,
  "http://127.0.0.1:8000",
  "http://localhost:8000"
].filter(Boolean);

function apiUrl(path) {
  return BASE_URLS.map(base => `${base}${path}`);
}

function formatUSD(n) {
  return "$" + Math.round(n).toLocaleString("en-US");
}

// ---------------------------------------------------------------
// Prediction Tab Logic
// ---------------------------------------------------------------
async function initPredictionTab() {
  // Attach event listener
  const makeSelect = el("make");
  if (makeSelect) {
    makeSelect.addEventListener("change", updateSubDropdowns);
  }
  const predictBtn = el("predictBtn");
  if (predictBtn) {
    predictBtn.addEventListener("click", handlePrediction);
  }

  // Load feature names if not loaded
  if (!featureData) {
    await loadFeatureNames();
  } else {
    populateManufacturers();
  }
}

async function loadFeatureNames() {
  const urls = apiUrl("/feature-names");
  for (const url of urls) {
    try {
      const res = await fetch(url);
      if (res.ok) {
        featureData = await res.json();
        populateManufacturers();
        return;
      }
    } catch (e) {
      // try next
    }
  }
  console.error("Failed to load feature names from API.");
}

function populateManufacturers() {
  const makeSelect = el("make");
  if (!makeSelect) return;
  makeSelect.innerHTML = "";
  if (!featureData || !featureData.manufacturers) return;

  featureData.manufacturers.forEach((mfr, idx) => {
    const opt = document.createElement("option");
    opt.value = mfr.name;
    opt.textContent = mfr.name;
    if (idx === 0) opt.selected = true;
    makeSelect.appendChild(opt);
  });

  updateSubDropdowns();
}

function updateSubDropdowns() {
  if (!featureData) return;
  const makeSelect = el("make");
  if (!makeSelect) return;
  const selectedMake = makeSelect.value;
  const mfr = featureData.manufacturers.find(m => m.name === selectedMake);
  if (!mfr) return;

  fillSelect("model", mfr.car_names);
  fillSelect("carType", mfr.car_types);
  fillSelect("year", mfr.years.map(String));
  fillSelect("fuel", mfr.energy);
  fillSelect("transmission", mfr.gearbox);
}

function fillSelect(id, values) {
  const select = el(id);
  if (!select) return;
  const prevValue = select.value;
  select.innerHTML = "";
  values.forEach((val) => {
    const opt = document.createElement("option");
    opt.value = val;
    opt.textContent = val;
    select.appendChild(opt);
  });
  if (values.includes(prevValue)) {
    select.value = prevValue;
  }
}

function buildPayload(modelName) {
  return {
    make: el("make").value,
    model: el("model").value,
    year: parseInt(el("year").value, 10),
    car_type: el("carType").value,
    fuel_type: el("fuel").value,
    transmission: el("transmission").value,
    model_name: modelName
  };
}

async function sendPredictionRequest(payload) {
  let lastError = null;
  const urls = apiUrl("/predict");

  for (const url of urls) {
    try {
      const response = await fetch(url, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload)
      });
      if (!response.ok) {
        const errData = await response.json().catch(() => ({}));
        throw new Error(errData.detail || `Server returned ${response.status}`);
      }
      return await response.json();
    } catch (err) {
      lastError = err;
    }
  }
  throw lastError || new Error("Failed to reach prediction backend.");
}

async function handlePrediction() {
  const btn = el("predictBtn");
  const yearSelect = el("year");
  if(!yearSelect) return;
  const year = parseInt(yearSelect.value, 10);

  if (isNaN(year) || year < 2015 || year > 2024) {
    showStatus("Please select a valid manufactured year (2015–2024).", true);
    return;
  }

  clearStatus();
  btn.disabled = true;
  btn.innerHTML = '<span class="loading-spinner"></span>Calculating...';
  el("comparisonBanner").style.display = "none";

  try {
    const payloadRF = buildPayload("random_forest");
    const [rfData, xgbData] = await Promise.all([
      sendPredictionRequest(payloadRF),
      sendPredictionRequest(buildPayload("xgboost"))
    ]);

    const rfPrice = rfData.predicted_price ?? rfData.prediction;
    const xgbPrice = xgbData.predicted_price ?? xgbData.prediction;
    const actualBenchmarkPrice = rfData.actual_benchmark ?? xgbData.actual_benchmark;

    el("actualPriceValue").textContent = formatUSD(actualBenchmarkPrice);
    el("rfPriceValue").textContent = formatUSD(rfPrice);
    el("xgbPriceValue").textContent = formatUSD(xgbPrice);

    const rfDiff = rfPrice - actualBenchmarkPrice;
    const xgbDiff = xgbPrice - actualBenchmarkPrice;

    const rfPct = ((rfDiff / actualBenchmarkPrice) * 100).toFixed(1);
    const xgbPct = ((xgbDiff / actualBenchmarkPrice) * 100).toFixed(1);

    el("rfVarianceBadge").textContent = `${rfDiff >= 0 ? '+' : ''}${formatUSD(rfDiff)} (${rfPct}%)`;
    el("rfVarianceBadge").className = `variance-badge ${rfDiff >= 0 ? 'pos' : 'neg'}`;

    el("xgbVarianceBadge").textContent = `${xgbDiff >= 0 ? '+' : ''}${formatUSD(xgbDiff)} (${xgbPct}%)`;
    el("xgbVarianceBadge").className = `variance-badge ${xgbDiff >= 0 ? 'pos' : 'neg'}`;

    if (Array.isArray(rfData.contributions) && rfData.contributions.length > 0) {
      renderContributions(rfData.contributions, "rfContribList");
    }
    if (Array.isArray(xgbData.contributions) && xgbData.contributions.length > 0) {
      renderContributions(xgbData.contributions, "xgbContribList");
    }

    const rfErr = Math.abs(rfDiff);
    const xgbErr = Math.abs(xgbDiff);
    const diffBetweenModels = Math.abs(rfPrice - xgbPrice);

    let accuracySummary = "";
    if (rfErr < xgbErr) {
      const delta = xgbErr - rfErr;
      accuracySummary = `🏆 <strong>Random Forest</strong> is closer to the Actual Dataset Price (${formatUSD(actualBenchmarkPrice)}) by <strong>${formatUSD(delta)}</strong> (${Math.abs(rfPct)}% error vs ${Math.abs(xgbPct)}% error for XGBoost).`;
    } else if (xgbErr < rfErr) {
      const delta = rfErr - xgbErr;
      accuracySummary = `🏆 <strong>XGBoost</strong> is closer to the Actual Dataset Price (${formatUSD(actualBenchmarkPrice)}) by <strong>${formatUSD(delta)}</strong> (${Math.abs(xgbPct)}% error vs ${Math.abs(rfPct)}% error for Random Forest).`;
    } else {
      accuracySummary = `🤝 Both models have identical precision relative to the Actual Dataset Price (${formatUSD(actualBenchmarkPrice)}).`;
    }

    const banner = el("comparisonBanner");
    banner.style.display = "block";
    banner.innerHTML = `
      <div>${accuracySummary}</div>
      <div class="metric-line">
        Model Gap: Random Forest and XGBoost predictions differ by <strong>${formatUSD(diffBetweenModels)}</strong> on this valuation.
      </div>
    `;

  } catch (err) {
    console.warn("Prediction error:", err);
    showStatus(`API Error: ${err.message}`, true);
  } finally {
    btn.disabled = false;
    btn.textContent = "Predict & Compare";
  }
}

function renderContributions(items, listId) {
  const list = el(listId);
  if (!list) return;
  list.innerHTML = "";

  const maxAbs = Math.max(...items.map(c => Math.abs(c.value)), 1);

  items.forEach(c => {
    const isZero = c.value === 0;
    const pct = isZero ? 0 : Math.min(100, (Math.abs(c.value) / maxAbs) * 100);
    const cls = isZero ? "neutral" : (c.value > 0 ? "pos" : "neg");
    const valDisplay = isZero ? "$0" : (c.value > 0 ? "+" + formatUSD(c.value) : "-" + formatUSD(Math.abs(c.value)));

    const row = document.createElement("div");
    row.className = "contrib-row";
    row.innerHTML = `
      <div class="contrib-name">${c.name}</div>
      <div class="contrib-bar-track"><div class="contrib-bar ${cls}" style="width:${pct}%"></div></div>
      <div class="contrib-val">${valDisplay}</div>
    `;
    list.appendChild(row);
  });
}

function showStatus(message, isError = false) {
  const banner = el("statusBanner");
  if (!banner) return;
  banner.textContent = message;
  banner.className = `status-banner ${isError ? "error" : "success"}`;
  banner.style.display = "block";
}

function clearStatus() {
  const banner = el("statusBanner");
  if (!banner) return;
  banner.style.display = "none";
  banner.className = "status-banner";
}

// ---------------------------------------------------------------
// Dataset Tab Logic
// ---------------------------------------------------------------
async function initDatasetTab() {
  await loadDatasetSamples();
}

async function loadDatasetSamples() {
  const urls = apiUrl("/dataset-head");
  
  let data = null;
  for (const url of urls) {
    try {
      const res = await fetch(url);
      if (res.ok) {
        data = await res.json();
        break;
      }
    } catch (e) {
      // ignore
    }
  }

  const tbody = el("datasetTableBody");
  const matchCount = el("totalMatchCount");
  
  if (!tbody) return;

  if (!data || !data.rows || data.rows.length === 0) {
    tbody.innerHTML = '<tr><td colspan="7" style="text-align: center; color: var(--text-dim);">Failed to load dataset rows.</td></tr>';
    if(matchCount) matchCount.textContent = '0';
    return;
  }
  
  if(matchCount) matchCount.textContent = data.total_matches;
  tbody.innerHTML = '';
  
  data.rows.forEach(r => {
    const tr = document.createElement('tr');
    tr.innerHTML = `
      <td>${r.manufacturer}</td>
      <td>${r.car_name}</td>
      <td>${r.year}</td>
      <td>${r.car_type}</td>
      <td>${r.fuel_type}</td>
      <td>${r.transmission}</td>
      <td style="color: var(--teal); font-weight: 600;">${formatUSD(r.price)}</td>
    `;
    tbody.appendChild(tr);
  });
}

// ---------------------------------------------------------------
// Initialization
// ---------------------------------------------------------------
initTheme();
loadTab('prediction');