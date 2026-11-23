const chartNode = document.querySelector("#price-chart");
const noticeNode = document.querySelector("#notice");
const predictButton = document.querySelector("#predict-button");
let dashboardData = null;
let chartRange = 252;
let displayCurrency = "USD";
let forecastValueUsd = null;
let dashboardRequestId = 0;
let isTraining = false;

function setText(selector, value) {
  const node = document.querySelector(selector);
  if (node) node.textContent = value;
}

function formatPrice(value) {
  if (!Number.isFinite(Number(value))) return "—";
  const converted = displayCurrency === "INR"
    ? Number(value) * (dashboardData?.usd_inr_rate ?? 1)
    : Number(value);
  return new Intl.NumberFormat(displayCurrency === "INR" ? "en-IN" : "en-US", {
    style: "currency",
    currency: displayCurrency,
    minimumFractionDigits: 2,
    maximumFractionDigits: 2,
  }).format(converted);
}

function convertValue(value) {
  return displayCurrency === "INR"
    ? value * (dashboardData?.usd_inr_rate ?? 1)
    : value;
}

function setNotice(message) {
  noticeNode.textContent = message;
  noticeNode.classList.toggle("hidden", !message);
}

async function readJson(response) {
  let body;
  try {
    body = await response.json();
  } catch {
    throw new Error(`Server returned an invalid response (${response.status}).`);
  }
  if (!response.ok) {
    const detail = typeof body.detail === "string" ? body.detail : JSON.stringify(body.detail || body);
    throw new Error(detail || `Request failed (${response.status}).`);
  }
  return body;
}

function showChart(points) {
  const visible = points.slice(-chartRange);
  if (!visible.length) {
    chartNode.innerHTML = '<div class="chart-loading">No market history is available.</div>';
    return;
  }

  const width = 760;
  const height = 190;
  const left = 47;
  const right = 10;
  const top = 12;
  const bottom = 25;
  const values = visible.map((point) => convertValue(point.close));
  const low = Math.min(...values);
  const high = Math.max(...values);
  const span = Math.max(high - low, Math.abs(high) * 0.015, 1);
  const min = low - span * 0.12;
  const max = high + span * 0.12;
  const plotWidth = width - left - right;
  const plotHeight = height - top - bottom;
  const x = (index) => left + (visible.length === 1 ? 0 : (index / (visible.length - 1)) * plotWidth);
  const y = (value) => top + ((max - value) / (max - min)) * plotHeight;
  const line = visible.map((point, index) => `${index ? "L" : "M"}${x(index).toFixed(1)},${y(convertValue(point.close)).toFixed(1)}`).join(" ");
  const area = `${line} L${x(visible.length - 1).toFixed(1)},${(height - bottom).toFixed(1)} L${left},${(height - bottom).toFixed(1)} Z`;
  const svg = document.createElementNS("http://www.w3.org/2000/svg", "svg");
  svg.setAttribute("viewBox", `0 0 ${width} ${height}`);
  svg.setAttribute("preserveAspectRatio", "none");
  svg.setAttribute("aria-hidden", "true");

  const defs = document.createElementNS(svg.namespaceURI, "defs");
  const gradient = document.createElementNS(svg.namespaceURI, "linearGradient");
  gradient.setAttribute("id", "chartFill");
  gradient.setAttribute("x1", "0");
  gradient.setAttribute("x2", "0");
  gradient.setAttribute("y1", "0");
  gradient.setAttribute("y2", "1");
  for (const [offset, opacity] of [["0%", "0.22"], ["100%", "0.01"]]) {
    const stop = document.createElementNS(svg.namespaceURI, "stop");
    stop.setAttribute("offset", offset);
    stop.setAttribute("stop-color", "#1a9b79");
    stop.setAttribute("stop-opacity", opacity);
    gradient.append(stop);
  }
  defs.append(gradient);
  svg.append(defs);

  for (let tick = 0; tick <= 4; tick += 1) {
    const value = max - ((max - min) * tick) / 4;
    const lineNode = document.createElementNS(svg.namespaceURI, "line");
    lineNode.setAttribute("x1", left);
    lineNode.setAttribute("x2", width - right);
    lineNode.setAttribute("y1", y(value));
    lineNode.setAttribute("y2", y(value));
    lineNode.setAttribute("class", "chart-grid");
    svg.append(lineNode);

    const label = document.createElementNS(svg.namespaceURI, "text");
    label.setAttribute("x", left - 8);
    label.setAttribute("y", y(value) + 3);
    label.setAttribute("text-anchor", "end");
    label.setAttribute("class", "chart-y-label");
    const symbol = displayCurrency === "INR" ? "₹" : "$";
    const locale = displayCurrency === "INR" ? "en-IN" : "en-US";
    label.textContent = `${symbol}${Math.round(value).toLocaleString(locale)}`;
    svg.append(label);
  }

  const areaPath = document.createElementNS(svg.namespaceURI, "path");
  areaPath.setAttribute("d", area);
  areaPath.setAttribute("class", "chart-area");
  svg.append(areaPath);
  const linePath = document.createElementNS(svg.namespaceURI, "path");
  linePath.setAttribute("d", line);
  linePath.setAttribute("class", "chart-line");
  svg.append(linePath);

  for (const index of [0, Math.floor((visible.length - 1) / 2), visible.length - 1]) {
    const label = document.createElementNS(svg.namespaceURI, "text");
    label.setAttribute("x", x(index));
    label.setAttribute("y", height - 5);
    label.setAttribute("text-anchor", index === 0 ? "start" : index === visible.length - 1 ? "end" : "middle");
    label.setAttribute("class", "chart-x-label");
    label.textContent = new Date(`${visible[index].date}T00:00:00`).toLocaleDateString("en-US", {
      month: "short",
      year: "2-digit",
    });
    svg.append(label);
  }

  const last = visible[visible.length - 1];
  const point = document.createElementNS(svg.namespaceURI, "circle");
  point.setAttribute("cx", x(visible.length - 1));
  point.setAttribute("cy", y(convertValue(last.close)));
  point.setAttribute("r", "4");
  point.setAttribute("class", "chart-point");
  svg.append(point);
  chartNode.replaceChildren(svg);
}
function setStageStatus(stage, complete) {
  const node = document.querySelector(`#stage-${stage}`);
  if (!node) return;
  node.textContent = complete ? "Ready" : "Pending";
  node.className = `step-status ${complete ? "done" : "pending"}`;
}

function activateSection(link) {
  document.querySelectorAll(".side-nav .nav-item").forEach((item) => {
    const active = item === link;
    item.classList.toggle("active", active);
    if (active) {
      item.setAttribute("aria-current", "page");
    } else {
      item.removeAttribute("aria-current");
    }
  });
  setText(".crumb strong", link.dataset.sectionLabel);
}

function renderDashboard(data) {
  if (dashboardData && dashboardData.ticker !== data.ticker) {
    forecastValueUsd = null;
  }
  dashboardData = data;
  const isModelSupported = data.model_supported;
  const tickerBadge = document.querySelector("#ticker-badge");
  tickerBadge.innerHTML = `<span class="apple-mark">${data.ticker.slice(0, 1)}</span> NASDAQ: ${data.ticker}`;
  setText("#latest-ticker", data.ticker);
  setText("#chart-title", `${data.ticker} closing price`);
  setText("#chart-subtitle", `Historical daily close · ${displayCurrency}`);
  document.querySelector("#price-chart").setAttribute("aria-label", `${data.company} historical closing price chart`);
  document.querySelector("#latest-close").closest(".metric-card").querySelector(".metric-icon").textContent =
    displayCurrency === "INR" ? "₹" : "$";
  setText("#latest-close", formatPrice(data.latest_close));
  setText("#chart-latest", formatPrice(data.latest_close));
  setText("#exchange-rate", `1 USD = ₹${data.usd_inr_rate.toFixed(2)}`);
  setText("#latest-date", `As of ${new Date(`${data.latest_date}T00:00:00`).toLocaleDateString("en-US", { year: "numeric", month: "short", day: "numeric" })}`);
  setText("#updated-at", `${data.observations.toLocaleString("en-US")} daily observations · Updated ${new Date().toLocaleTimeString("en-US", { hour: "numeric", minute: "2-digit" })}`);
  setText("#observation-count", `${data.observations.toLocaleString("en-US")} total observations`);
  setText("#metric-rmse", isModelSupported && data.metrics ? formatPrice(data.metrics.rmse) : "—");
  setText("#metric-mae", isModelSupported && data.metrics ? formatPrice(data.metrics.mae) : "—");
  const mape = Number(data.metrics?.mape_percent);
setText("#metric-mape", Number.isFinite(mape) ? `${mape.toFixed(2)}%` : "—");
  setText("#test-samples", isModelSupported
    ? data.metrics?.test_sequences?.toLocaleString("en-US") || "—"
    : "Not trained");
  setText("#registered-model-name", `${data.ticker}_LSTM`);
  setText("#model-description", isModelSupported
    ? "Two-layer recurrent neural network"
    : `Model trained for AAPL · viewing ${data.ticker} market data`);
  setText("#model-version", isModelSupported
    ? data.metrics?.model_version || (data.ticker === "AAPL" ? "Loading" : "local")
    : "—");
  setText("#prediction-value", isModelSupported && forecastValueUsd !== null
    ? formatPrice(forecastValueUsd)
    : "—");
  setText("#prediction-label", `PREDICTED NEXT CLOSE · ${displayCurrency}`);
  setText("#prediction-version", isModelSupported
    ? "Champion model · next trading day estimate"
    : `No ${data.ticker} model trained yet. Train this ticker to enable forecasts.`);
  setText("#prediction-message", "");

  predictButton.disabled = isTraining || (!isModelSupported && !data.ticker) ||
    (isModelSupported && (!data.window || data.window.length !== 60));
  predictButton.querySelector("span").textContent = isTraining
    ? "Training..."
    : isModelSupported
      ? "Run prediction"
      : `Train ${data.ticker} model`;

  const epochs = isModelSupported ? data.history?.loss?.length : null;
  setText("#training-epochs", epochs ? `${epochs} epochs` : "—");
  for (const stage of ["ingest", "preprocess", "train", "evaluate", "monitor"]) {
    setStageStatus(stage, Boolean(data.pipeline.stages[stage]));
  }
  const dvcStatus = document.querySelector("#dvc-status");
  dvcStatus.textContent = data.pipeline.dvc_initialized ? "DVC initialized" : "DVC not initialized";
  dvcStatus.classList.toggle("enabled", data.pipeline.dvc_initialized);

  document.querySelector("#prediction-report").classList.toggle("hidden", !isModelSupported || !data.reports.prediction_plot);
  document.querySelector("#loss-report").classList.toggle("hidden", !isModelSupported || !data.reports.loss_plot);
  document.querySelector("#drift-report").classList.toggle("hidden", !isModelSupported || !data.reports.drift_report);
  showChart(data.points);
  if (!isTraining) setNotice("");
}

async function loadHealth() {
  const badge = document.querySelector("#api-status");
  const modelStatus = document.querySelector("#model-status");
  try {
    const health = await readJson(await fetch("/health", { cache: "no-store" }));
    if (health.status === "ok") {
      badge.className = "status-pill status-online";
      badge.innerHTML = "<i></i>API & model online";
      modelStatus.className = dashboardData && !dashboardData.model_supported ? "tag tag-muted" : "tag tag-good";
      modelStatus.textContent = dashboardData && !dashboardData.model_supported ? "Model not trained" : "Champion ready";
      if (!dashboardData || dashboardData.ticker === "AAPL") {
        setText("#model-version", health.model_version);
      }
      if (!dashboardData || dashboardData.model_supported) {
        const modelVersion = dashboardData?.metrics?.model_version || health.model_version;
        setText("#prediction-version", `Champion model · v${modelVersion}`);
      }
    } else {
      badge.className = "status-pill status-offline";
      badge.innerHTML = "<i></i>Model unavailable";
      modelStatus.className = "tag tag-warn";
      modelStatus.textContent = "Model unavailable";
      setText("#model-version", "—");
      setText("#prediction-version", "Train a model to enable predictions");
    }
  } catch {
    badge.className = "status-pill status-offline";
    badge.innerHTML = "<i></i>API unavailable";
    modelStatus.className = "tag tag-warn";
    modelStatus.textContent = "API unavailable";
    setText("#prediction-version", "Could not connect to prediction API");
  }
}

async function loadDashboard() {
  const refresh = document.querySelector("#refresh-button");
  const tickerSelect = document.querySelector("#ticker-select");
  const ticker = tickerSelect.value;
  const requestId = ++dashboardRequestId;
  if (!dashboardData || dashboardData.ticker !== ticker) {
    dashboardData = null;
    forecastValueUsd = null;
    setText("#ticker-badge", `NASDAQ: ${ticker}`);
    setText("#latest-ticker", ticker);
    setText("#latest-close", "—");
    setText("#chart-latest", "—");
    setText("#chart-title", `${ticker} closing price`);
    setText("#chart-subtitle", `Historical daily close · ${displayCurrency}`);
    setText("#exchange-rate", "Loading USD/INR exchange rate…");
    setText("#metric-rmse", "—");
    setText("#metric-mae", "—");
    setText("#metric-mape", "—");
    setText("#test-samples", "—");
    setText("#model-version", "Loading");
    setText("#prediction-value", "—");
    setText("#prediction-version", "Loading model status…");
    setText("#training-epochs", "—");
    chartNode.innerHTML = '<div class="chart-loading">Loading market data…</div>';
  }
  refresh.disabled = true;
  const refreshLabel = refresh.querySelector("span");
  if (refreshLabel) refreshLabel.textContent = "Refreshing…";
  if (!isTraining) setNotice("");
  try {
    const response = await fetch(`/api/dashboard?ticker=${encodeURIComponent(ticker)}`, {
      cache: "no-store",
    });
    const data = await readJson(response);
    if (requestId !== dashboardRequestId) return;
    renderDashboard(data);
  } catch (error) {
    if (requestId !== dashboardRequestId) return;
    setNotice(`Could not load dashboard: ${error.message}`);
    chartNode.innerHTML = '<div class="chart-loading">Unable to load market data.</div>';
  } finally {
    if (requestId === dashboardRequestId) {
      refresh.disabled = isTraining;
      const refreshLabel = refresh.querySelector("span");
if (refreshLabel) {
  refreshLabel.textContent = isTraining ? "Training…" : "Refresh";
}
    }
  }
}
async function trainTickerModel(ticker) {
  if (isTraining) return;

  isTraining = true;
  const tickerSelect = document.querySelector("#ticker-select");
  const refreshButton = document.querySelector("#refresh-button");
  predictButton.disabled = true;
  tickerSelect.disabled = true;
  refreshButton.disabled = true;
  predictButton.querySelector("span").textContent = "Training...";
  setNotice(`Training ${ticker} LSTM model. This may take several minutes…`);

  try {
    const response = await fetch("/train", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({ ticker }),
    });

    const result = await readJson(response);

    setNotice(
      `${ticker} training completed successfully. ` +
      `RMSE: ${Number(result.metrics.rmse).toFixed(4)} · ` +
      `MAE: ${Number(result.metrics.mae).toFixed(4)} · ` +
      `Epochs: ${result.metrics.epochs}`
    );

    await loadDashboard();
  } catch (error) {
    setNotice(`Training failed for ${ticker}: ${error.message}`);
  } finally {
    isTraining = false;
    tickerSelect.disabled = false;
    refreshButton.disabled = false;
    predictButton.disabled = false;
    predictButton.querySelector("span").textContent =
      dashboardData?.model_supported ? "Run prediction" : `Train ${ticker} model`;
    refreshButton.querySelector("span").textContent = "Refresh";
  }
}

async function runPrediction() {
  const ticker = dashboardData?.ticker;
  if (!ticker) return;

  predictButton.disabled = true;
  predictButton.querySelector("span").textContent = "Predicting…";
  setText("#prediction-message", "Running the trained LSTM model…");

  try {
    const response = await fetch("/predict", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
  ticker: ticker,
  close: dashboardData.window
}),
    });
    const result = await readJson(response);
    forecastValueUsd = result.predicted_close;
    setText("#prediction-value", formatPrice(forecastValueUsd));
    setText("#prediction-message", `Prediction generated for ${ticker}.`);
  } catch (error) {
    setText("#prediction-message", `Prediction failed: ${error.message}`);
  } finally {
    predictButton.disabled = false;
    predictButton.querySelector("span").textContent = dashboardData?.model_supported
      ? "Run prediction"
      : `Train ${ticker} model`;
  }
}

document.querySelector("#predict-button").addEventListener("click", async () => {
  if (!dashboardData) return;
  if (!dashboardData.model_supported) {
    await trainTickerModel(dashboardData.ticker);
  } else {
    await runPrediction();
  }
});
const trainButton = document.querySelector("#train-button");

if (trainButton) {
    trainButton.addEventListener("click", async () => {
        if (!dashboardData || isTraining) return;
        await trainTickerModel(dashboardData.ticker);
    });
}
document.querySelector("#refresh-button").addEventListener("click", loadDashboard);
document.querySelector("#ticker-select").addEventListener("change", loadDashboard);

document.querySelectorAll(".chart-range").forEach((button) => {
  button.addEventListener("click", () => {
    console.log("Chart range clicked:", button.dataset.range);
    chartRange = Number(button.dataset.range);
    document.querySelectorAll(".chart-range").forEach((item) => {
    item.classList.remove("selected");
    item.setAttribute("aria-pressed", "false");
});

button.classList.add("selected");
button.setAttribute("aria-pressed", "true");
    if (dashboardData) {
  console.log("Updating chart to range:", chartRange);
  showChart(dashboardData.points);

  const count = dashboardData.points.slice(-chartRange).length;
  document.querySelector("#observation-count").textContent =
    `${count.toLocaleString()} observations`;
}
  });
});

document.querySelector("#currency-select").addEventListener("change", (event) => {
  displayCurrency = event.target.value;
  if (dashboardData) renderDashboard(dashboardData);
});

document.querySelectorAll(".side-nav .nav-item").forEach((link) => {
  link.addEventListener("click", (event) => {
    event.preventDefault();
    activateSection(link);
    const section = document.querySelector(`#${link.dataset.section}`);
    if (section) section.scrollIntoView({ behavior: "smooth", block: "start" });
  });
});

loadDashboard();
loadHealth();