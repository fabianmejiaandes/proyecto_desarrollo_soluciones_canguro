const PHASE_COLORS = {
  F0_Prenatal_Parto: { color: "#D9A441", soft: "#F5E8C9" },
  F1_Nacimiento: { color: "#D85B47", soft: "#F8E6E2" },
  F2_Hospitalizacion: { color: "#3E6FB5", soft: "#E6EEF8" },
  F3_40semanas: { color: "#3F8A5C", soft: "#E7F2EC" },
  F4_3meses: { color: "#8B6FB5", soft: "#EEE7F5" },
  F5_6meses: { color: "#A1276F", soft: "#F5E4EE" },
  F6_9meses: { color: "#2F4858", soft: "#E5EDF0" },
};

const OUTCOME_COLUMNS = [
  ["Stunting", "Talla baja"],
  ["Bajo_peso", "Bajo peso"],
  ["Wasting", "Aguda"],
  ["Mixta", "Mixta"],
];

function formatNumber(value) {
  return new Intl.NumberFormat("es-CO").format(value);
}

function formatDecimal(value, digits = 1) {
  return new Intl.NumberFormat("es-CO", {
    minimumFractionDigits: digits,
    maximumFractionDigits: digits,
  }).format(value);
}

function formatMetric(value, digits = 3) {
  return formatDecimal(value, digits);
}

function createStatCard(label, value) {
  const card = document.createElement("article");
  card.className = "stat-card";
  card.innerHTML = `
    <div class="stat-label">${label}</div>
    <div class="stat-value">${value}</div>
  `;
  return card;
}

function renderStats(data) {
  const container = document.getElementById("project-stats");
  if (!container) return;
  container.innerHTML = "";
  [
    ["Variables acumuladas", data.project.variables],
    ["Modelos", data.project.models],
    ["Seguimiento", "7 fases clínicas acumulativas"],
  ].forEach(([label, value]) => container.appendChild(createStatCard(label, value)));
}

function renderOutcomes(data) {
  const container = document.getElementById("outcome-cards");
  if (!container) return;
  container.innerHTML = "";
  data.outcomes.forEach((outcome) => {
    const card = document.createElement("article");
    card.className = "outcome-card";
    card.innerHTML = `
      <h3>${outcome.label}</h3>
      <p>${outcome.clinical}</p>
      <dl>
        <div><dt>Niños con Z &lt; −2 DE</dt><dd>${formatNumber(outcome.positives)} de ${formatNumber(outcome.n)} evaluados</dd></div>
        <div><dt>% de niños afectados</dt><dd>${formatDecimal(outcome.prevalence, 1)}%</dd></div>
        <div><dt>AUC de F0 a F6</dt><dd>${formatMetric(outcome.aucF0)} a ${formatMetric(outcome.aucF6)}</dd></div>
        <div><dt>Sens. / Esp. F6</dt><dd>${formatMetric(outcome.sensF6)} / ${formatMetric(outcome.specF6)}</dd></div>
      </dl>
    `;
    container.appendChild(card);
  });
}

function renderPhases(data) {
  const container = document.getElementById("phase-track");
  if (!container) return;
  container.innerHTML = "";
  data.phases.forEach((phase) => {
    const phaseColors = PHASE_COLORS[phase.id] || PHASE_COLORS.F2_Hospitalizacion;
    const card = document.createElement("article");
    card.className = "phase-card";
    card.dataset.phase = phase.short;
    card.style.setProperty("--phase-color", phaseColors.color);
    card.style.setProperty("--phase-soft", phaseColors.soft);
    card.innerHTML = `
      <div class="phase-chip">${phase.short}</div>
      <h3 class="phase-title">${phase.label}</h3>
      <p class="phase-meta">${phase.features} variables acumuladas</p>
      <p class="phase-time">${phase.time}</p>
    `;
    card.addEventListener("click", () => highlightPhase(phase.short));
    container.appendChild(card);
  });
}

function renderMetricsTable(data) {
  const container = document.getElementById("metrics-table");
  if (!container) return;
  container.innerHTML = "";
  const table = document.createElement("table");
  const headers = OUTCOME_COLUMNS.map(([, label]) => `<th>${label}</th>`).join("");
  table.innerHTML = `
    <thead>
      <tr>
        <th>Fase</th>
        ${headers}
      </tr>
    </thead>
    <tbody></tbody>
  `;

  const tbody = table.querySelector("tbody");
  data.metricsMatrix.forEach((row) => {
    const cells = OUTCOME_COLUMNS.map(([key]) => `<td>${formatMetric(row[key])}</td>`).join("");
    const tr = document.createElement("tr");
    tr.dataset.phase = row.short;
    tr.innerHTML = `<td>${row.phase}</td>${cells}`;
    tbody.appendChild(tr);
  });

  container.appendChild(table);
}

function renderFeatureList(data) {
  const container = document.getElementById("feature-list");
  if (!container) return;
  container.innerHTML = "";
  data.topFeatures.slice(0, 10).forEach((feature) => {
    const item = document.createElement("li");
    item.innerHTML = `<strong>${feature.label}</strong><span>${feature.earliestPhase}. Importancia SHAP media: ${formatMetric(feature.importance)}.</span>`;
    container.appendChild(item);
  });
}

function renderMissingness(data) {
  const container = document.getElementById("missingness-summary");
  if (!container || !data.missingness) return;

  const phases = data.missingness.phases;
  const f0 = phases[0];
  const f6 = phases[phases.length - 1];
  container.innerHTML = `
    <div class="missing-grid">
      <article class="missing-card">
        <h3>Ausencia promedio por variable</h3>
        <p>Entre ${formatDecimal(f6.meanMissing, 1)}% y ${formatDecimal(f0.meanMissing, 1)}%, según la fase. Esto confirma que la solución no podía depender de registros completos.</p>
      </article>
      <article class="missing-card">
        <h3>Disponibilidad por paciente</h3>
        <p>En F6, el paciente mediano tiene ${formatDecimal(f6.medianPatientAvailable, 1)}% de las variables disponibles; ${formatDecimal(f6.patientsWith75Available, 1)}% de pacientes tienen al menos tres cuartas partes de los datos.</p>
      </article>
    </div>
    <div class="missing-table-wrap">
      <table>
        <thead>
          <tr>
            <th>Fase</th>
            <th>Variables</th>
            <th>Ausencia promedio</th>
            <th>Variables con &gt;50% ausencia</th>
            <th>Disponibilidad mediana por paciente</th>
          </tr>
        </thead>
        <tbody>
          ${phases.map((phase) => `
            <tr>
              <td>${phase.phase}</td>
              <td>${phase.features}</td>
              <td>${formatDecimal(phase.meanMissing, 1)}%</td>
              <td>${formatDecimal(phase.featuresOver50, 1)}%</td>
              <td>${formatDecimal(phase.medianPatientAvailable, 1)}%</td>
            </tr>
          `).join("")}
        </tbody>
      </table>
    </div>
  `;
}

function renderTechnicalSummary(data) {
  const container = document.getElementById("technical-summary");
  if (!container) return;
  const tech = data.technical;
  container.innerHTML = `
    <article class="technical-card">
      <h3>Diseño del experimento</h3>
      <p>Se entrenaron ${tech.reportedModels} clasificadores binarios: cuatro desenlaces por siete ventanas clínicas acumulativas.</p>
    </article>
    <article class="technical-card">
      <h3>Evaluación</h3>
      <p>${tech.metricMethod}. Las métricas se calculan sobre predicciones fuera de la partición de entrenamiento.</p>
    </article>
    <article class="technical-card">
      <h3>Modelo principal</h3>
      <p>${tech.modelFamily}, con ${tech.roundRange}. El entrenamiento usa balanceo por clase positiva.</p>
    </article>
    <article class="technical-card">
      <h3>Comparador lineal</h3>
      <p>${tech.baseline.model}: AUC ${formatMetric(tech.baseline.auc, 4)}. LightGBM en F6 para talla baja: ${formatMetric(tech.baseline.lgbmAuc, 4)}.</p>
    </article>
  `;
}

function renderKeyValueTable(containerId, headers, rows) {
  const container = document.getElementById(containerId);
  if (!container) return;
  container.innerHTML = `
    <table>
      <thead><tr>${headers.map((header) => `<th>${header}</th>`).join("")}</tr></thead>
      <tbody>
        ${rows.map((row) => `<tr>${row.map((cell) => `<td>${cell}</td>`).join("")}</tr>`).join("")}
      </tbody>
    </table>
  `;
}

function renderTechnicalTables(data) {
  const tech = data.technical;
  renderKeyValueTable(
    "technical-params-table",
    ["Parámetro", "Valor", "Por qué se usó"],
    tech.params.map((row) => [row[0], row[1], row[2]])
  );
  renderKeyValueTable(
    "technical-phase-table",
    ["Fase", "Variables acumuladas"],
    tech.phaseFeatureCounts.map((row) => [row.phase, row.features])
  );
  renderKeyValueTable(
    "technical-balance-table",
    ["Desenlace", "N evaluable", "Casos positivos", "Prevalencia", "scale_pos_weight"],
    tech.outcomeBalance.map((row) => [
      row.outcome,
      formatNumber(row.nTotal),
      formatNumber(row.nPositive),
      `${formatDecimal(row.prevalence, 1)}%`,
      formatDecimal(row.scalePosWeight, 2),
    ])
  );
  renderKeyValueTable(
    "technical-artifacts-table",
    ["Artefacto", "Uso dentro del proyecto"],
    tech.artifacts
  );
}

function setupTabs() {
  const tabs = document.querySelectorAll("[data-tab-target]");
  const panels = document.querySelectorAll("[data-tab-panel]");
  tabs.forEach((tab) => {
    tab.addEventListener("click", () => {
      const target = tab.dataset.tabTarget;
      tabs.forEach((item) => {
        const selected = item === tab;
        item.classList.toggle("is-active", selected);
        item.setAttribute("aria-selected", selected ? "true" : "false");
      });
      panels.forEach((panel) => {
        const selected = panel.dataset.tabPanel === target;
        panel.classList.toggle("is-active", selected);
        panel.hidden = !selected;
      });
    });
  });
}

function highlightPhase(shortPhase) {
  document.querySelectorAll(".phase-card").forEach((card) => {
    card.classList.toggle("is-active", card.dataset.phase === shortPhase);
  });
}

function setupFigureModal() {
  const modal = document.getElementById("figure-modal");
  const modalImage = document.getElementById("figure-modal-image");
  const modalCaption = document.getElementById("figure-modal-caption");
  const closeButton = document.getElementById("figure-modal-close");
  if (!modal || !modalImage || !modalCaption || !closeButton) return;

  const openModal = (img, captionHtml) => {
    modalImage.src = img.src;
    modalImage.alt = img.alt || "Figura ampliada";
    modalCaption.innerHTML = captionHtml || "";
    modal.classList.add("is-open");
    modal.setAttribute("aria-hidden", "false");
    document.body.style.overflow = "hidden";
  };

  const closeModal = () => {
    modal.classList.remove("is-open");
    modal.setAttribute("aria-hidden", "true");
    modalImage.src = "";
    modalCaption.innerHTML = "";
    document.body.style.overflow = "";
  };

  document.querySelectorAll(".figure").forEach((figure, index) => {
    const card = figure.querySelector(".figure-card");
    const image = figure.querySelector(".chart-image");
    const caption = figure.querySelector("figcaption");
    if (!card || !image || card.querySelector(".figure-expand-btn")) return;

    const button = document.createElement("button");
    button.type = "button";
    button.className = "figure-expand-btn";
    button.setAttribute("aria-label", `Ampliar figura ${index + 1}`);
    button.textContent = "Ampliar";
    button.addEventListener("click", () => openModal(image, caption ? caption.innerHTML : ""));
    card.appendChild(button);
  });

  closeButton.addEventListener("click", closeModal);
  modal.addEventListener("click", (event) => {
    if (event.target instanceof HTMLElement && event.target.dataset.closeModal === "true") closeModal();
  });
  document.addEventListener("keydown", (event) => {
    if (event.key === "Escape" && modal.classList.contains("is-open")) closeModal();
  });
}

function initArticle() {
  if (typeof document === "undefined") return;

  setupTabs();
  setupFigureModal();

  if (!window.ARTICLE_DATA) return;
  renderStats(window.ARTICLE_DATA);
  renderOutcomes(window.ARTICLE_DATA);
  renderPhases(window.ARTICLE_DATA);
  renderMetricsTable(window.ARTICLE_DATA);
  renderFeatureList(window.ARTICLE_DATA);
  renderMissingness(window.ARTICLE_DATA);
  renderTechnicalSummary(window.ARTICLE_DATA);
  renderTechnicalTables(window.ARTICLE_DATA);
  highlightPhase("F4");
}

initArticle();
