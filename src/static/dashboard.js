const chartNode = document.querySelector("#price-chart");
const noticeNode = document.querySelector("#notice");
const predictButton = document.querySelector("#predict-button");
let dashboardData = null;
let chartRange = 252;
let displayCurrency = "USD";
let forecastValueUsd = null;
let dashboardRequestId = 0;

function setText(selector, value) {
  document.querySelector(selector).textContent = value;
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
  const body = await response.json();
  if (!response.ok) {
    throw new Error(body.detail || `Request failed (${response.status}).`);
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
  setText("#metric-mape", data.metrics ? `${Number(data.metrics.mape_percent).toFixed(2)}%` : "—");
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
    : `No ${data.ticker} model trained yet. Run the ticker training command to enable forecasts.`);
  setText("#prediction-message", "");
  predictButton.disabled = !isModelSupported || !data.window || data.window.length !== 60;
  predictButton.querySelector("span").textContent = isModelSupported ? "Run prediction" : `Train ${data.ticker} model`;

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
  setNotice("");
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
  const ticker = document.querySelector("#ticker-select").value;
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
    setText("#training-epochs", "—");
    setText("#model-description", "Loading selected company data…");
    setText("#prediction-value", "—");
    setText("#prediction-label", `PREDICTED NEXT CLOSE · ${displayCurrency}`);
    setText("#prediction-version", "Loading selected company data…");
    predictButton.disabled = true;
    chartNode.innerHTML = '<div class="chart-loading">Loading selected company data…</div>';
    setNotice("");
  }
  refresh.disabled = true;
  try {
    const data = await readJson(await fetch(`/api/dashboard?ticker=${encodeURIComponent(ticker)}`, { cache: "no-store" }));
    if (requestId !== dashboardRequestId || ticker !== document.querySelector("#ticker-select").value) {
      return;
    }
    renderDashboard(data);
  } catch (error) {
    if (requestId !== dashboardRequestId) return;
    dashboardData = null;
    forecastValueUsd = null;
    setNotice(`${error.message} Start the API with "python -m uvicorn src.api:app --reload" and make sure the pipeline data is available.`);
    chartNode.innerHTML = '<div class="chart-loading">Market history is not available.</div>';
    setText("#exchange-rate", "USD/INR exchange rate unavailable");
    setText("#prediction-version", "Market data could not be loaded");
  } finally {
    if (requestId === dashboardRequestId) refresh.disabled = false;
  }
  if (requestId === dashboardRequestId) await loadHealth();
}

document.querySelector("#refresh-button").addEventListener("click", loadDashboard);
document.querySelector("#ticker-select").addEventListener("change", loadDashboard);
document.querySelector("#currency-select").addEventListener("change", (event) => {
  displayCurrency = event.target.value;
  if (dashboardData) renderDashboard(dashboardData);
});
document.querySelectorAll(".side-nav .nav-item").forEach((link) => {
  link.addEventListener("click", (event) => {
    const target = document.querySelector(link.hash);
    if (!target) return;
    event.preventDefault();
    history.replaceState(null, "", link.hash);
    activateSection(link);
    target.scrollIntoView({
      behavior: window.matchMedia("(prefers-reduced-motion: reduce)").matches ? "auto" : "smooth",
      block: "start",
    });
  });
});
document.querySelectorAll(".range-switch button").forEach((button) => {
  button.addEventListener("click", () => {
    chartRange = Number(button.dataset.range);
    document.querySelectorAll(".range-switch button").forEach((item) => {
      item.classList.toggle("selected", item === button);
      item.setAttribute("aria-pressed", item === button ? "true" : "false");
    });
    if (dashboardData) showChart(dashboardData.points);
  });
});

predictButton.addEventListener("click", async () => {
  if (!dashboardData) return;
  predictButton.disabled = true;
  setText("#prediction-value", "…");
  setText("#prediction-version", "Requesting forecast…");
  const message = document.querySelector("#prediction-message");
  message.textContent = "";
  try {
    const response = await fetch("/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ ticker: dashboardData.ticker, close: dashboardData.window }),
    });
    const result = await readJson(response);
    forecastValueUsd = result.predicted_close;
    setText("#prediction-value", formatPrice(forecastValueUsd));
    setText("#prediction-label", `PREDICTED NEXT CLOSE · ${displayCurrency}`);
    setText("#prediction-version", `Model version ${result.model_version} · Next trading day estimate`);
  } catch (error) {
    forecastValueUsd = null;
    setText("#prediction-value", "Unavailable");
    setText("#prediction-version", "Forecast could not be generated");
    message.textContent = error.message;
  } finally {
    predictButton.disabled = !dashboardData.model_supported || dashboardData.window.length !== 60;
  }
});

loadDashboard();
