(function () {
  function normalizeData(raw) {
    if (!raw || !Array.isArray(raw.patients)) return null;
    if (raw.patients[0] && !Array.isArray(raw.patients[0])) return raw;
    return {
      meta: raw.meta,
      featured: raw.featured,
      patients: raw.patients.map((row) => ({
        id: row[0],
        sede: row[1],
        periodo: row[2],
        birthGroup: row[3],
        summaryValues: row[4],
        similarityValues: row[5],
        outcomes: Object.fromEntries(Object.entries(row[6] || {}).map(([key, value]) => [
          key,
          {
            outcomeReal: value[0],
            finalRisk: value[1],
            probs: value[2],
            cluster: value[3],
            clusterLabel: value[4],
          },
        ])),
      })),
    };
  }

  const DATA = normalizeData(window.TRAJECTORY_EXPLORER_LITE || window.TRAJECTORY_EXPLORER_DATA);
  if (!DATA) return;

  // Remap outcome labels to clinical-standard terminology
  const LABEL_REMAP = {
    Stunting:  { label: "Talla para la edad (T/E)", shortLabel: "T/E", positiveLabel: "con T/E baja", negativeLabel: "sin T/E baja" },
    Bajo_peso: { label: "Peso para la edad (P/E)", shortLabel: "P/E", positiveLabel: "con P/E bajo", negativeLabel: "sin P/E bajo" },
    Wasting:   { label: "Peso para la talla (P/T)", shortLabel: "P/T", positiveLabel: "con P/T bajo", negativeLabel: "sin P/T bajo" },
    Mixta:     { label: "Mixta (P/T + T/E)", shortLabel: "Mixta", positiveLabel: "con condición mixta", negativeLabel: "sin condición mixta" },
  };
  Object.entries(LABEL_REMAP).forEach(([key, overrides]) => {
    if (DATA.meta.outcomes[key]) Object.assign(DATA.meta.outcomes[key], overrides);
  });

  // birthGroup ↔ RCIU mapping: 1,3 = con RCIU; 2,4 = sin RCIU
  const RCIU_GROUPS = { con: [1, 3], sin: [2, 4] };

  const BIRTH_GROUP_LABELS = {
    1: "G1 · Prematuro + RCIU",
    2: "G2 · Prematuro sin RCIU",
    3: "G3 · Término + RCIU",
    4: "G4 · Término sin RCIU",
  };

  const PHASE_COLORS = ["#D9A441", "#D85B47", "#3E6FB5", "#3F8A5C", "#8B6FB5", "#A1276F", "#2F4858"];
  const CLUSTER_COLORS = ["#A1276F", "#3E6FB5", "#3F8A5C"];
  const fmt = new Intl.NumberFormat("es-CO");
  const pct = new Intl.NumberFormat("es-CO", { maximumFractionDigits: 1, minimumFractionDigits: 1 });
  const dec = new Intl.NumberFormat("es-CO", { maximumFractionDigits: 3, minimumFractionDigits: 3 });

  function debounce(fn, ms) {
    let timer;
    return function (...args) {
      clearTimeout(timer);
      timer = setTimeout(() => fn.apply(this, args), ms);
    };
  }

  // ── State ──
  const state = {
    outcome: DATA.meta.defaultOutcome || "Stunting",
    phaseIndex: 2,
    target: "all",
    site: "all",
    egGroup: "all",
    rciu: "all",
    yearFrom: 1993,
    yearTo: 2023,
    egTolerance: 1,
    zTolerance: 0.2,
    selectedId: null,
    dirty: false,
    chartMode: "probability",
  };

  // ── DOM references ──
  const els = {
    outcome: document.getElementById("outcome-select"),
    phase: document.getElementById("phase-select"),
    target: document.getElementById("target-select"),
    site: document.getElementById("site-select"),
    egGroup: document.getElementById("eg-group-select"),
    rciu: document.getElementById("rciu-select"),
    yearFrom: document.getElementById("year-from"),
    yearTo: document.getElementById("year-to"),
    egTolerance: document.getElementById("eg-tolerance"),
    zTolerance: document.getElementById("z-tolerance"),
    search: document.getElementById("patient-search"),
    searchButton: document.getElementById("patient-search-button"),
    exampleRow: document.getElementById("example-row"),
    stats: document.getElementById("stat-strip"),
    chart: document.getElementById("trajectory-chart"),
    chartMode: document.getElementById("chart-mode"),
    equivalence: document.getElementById("equivalence-table"),
    neighbors: document.getElementById("neighbor-table"),
    selectedSummary: document.getElementById("selected-summary"),
    selectedFields: document.getElementById("selected-fields"),
    clusterReading: document.getElementById("cluster-reading"),
    periodReading: document.getElementById("period-reading"),
    clusterGrid: document.getElementById("cluster-grid"),
    legendRow: document.getElementById("legend-row"),
    calculateBtn: document.getElementById("calculate-btn"),
    calculateBar: document.getElementById("calculate-bar"),
    calculateHint: document.getElementById("calculate-hint"),
    resultsArea: document.getElementById("results-area"),
  };

  // ── Utilities ──
  function isNumber(value) {
    return typeof value === "number" && Number.isFinite(value);
  }

  function parseInputNumber(value) {
    const normalized = String(value).replace(",", ".");
    const parsed = Number(normalized);
    return Number.isFinite(parsed) ? parsed : 0;
  }

  function riskLabel(value) {
    return isNumber(value) ? `${pct.format(value * 100)}%` : "sin dato";
  }

  function valueLabel(value, unit) {
    if (value === null || value === undefined || value === "") return "sin dato";
    if (isNumber(value)) return `${fmt.format(value)}${unit ? ` ${unit}` : ""}`;
    return String(value);
  }

  function outcomeMeta() {
    return DATA.meta.outcomes[state.outcome];
  }

  function patientsWithOutcome() {
    return DATA.patients.filter((patient) => Boolean(patient.outcomes[state.outcome]));
  }

  function uniqueSorted(values) {
    return [...new Set(values.filter((value) => value !== null && value !== undefined && value !== ""))]
      .sort((a, b) => Number(a) - Number(b));
  }

  // ── Dirty / Calculate pattern ──
  function markDirty() {
    state.dirty = true;
    if (els.calculateBtn) {
      els.calculateBtn.disabled = false;
      els.calculateBtn.classList.add("is-active");
    }
    if (els.calculateBar) els.calculateBar.classList.add("is-dirty");
    if (els.calculateHint) els.calculateHint.innerHTML = "Los filtros cambiaron. Presione <strong>Calcular</strong> para actualizar los resultados.";
    if (els.resultsArea) els.resultsArea.classList.add("is-stale");
  }

  function markClean() {
    state.dirty = false;
    if (els.calculateBtn) {
      els.calculateBtn.disabled = true;
      els.calculateBtn.classList.remove("is-active");
    }
    if (els.calculateBar) els.calculateBar.classList.remove("is-dirty");
    if (els.calculateHint) els.calculateHint.innerHTML = "Los filtros están listos. Presione <strong>Calcular</strong> cada vez que cambie una condición.";
    if (els.resultsArea) els.resultsArea.classList.remove("is-stale");
  }

  // ── Z-score trajectory discovery ──
  let zscoreTrajectories = [];

  function discoverZscoreTrajectories() {
    const fields = DATA.meta.similarityFields;
    if (!fields || !fields.length) return [];
    const groups = {};
    fields.forEach(function (field, simIndex) {
      var key = field.key || "";
      var match = key.match(/^(zscore(?:pesotalla|talla|peso))/i);
      if (!match) return;
      var base = match[1].toLowerCase();
      if (!groups[base]) groups[base] = { fields: [] };
      groups[base].fields.push({ phaseIndex: field.phaseIndex, simIndex: simIndex, label: field.label });
    });

    var LABELS = {
      zscoretalla: "Z-score de talla (longitud/estatura)",
      zscorepeso: "Z-score de peso",
      zscorepesotalla: "Z-score de peso para la talla",
    };

    return Object.entries(groups)
      .filter(function (entry) { return entry[1].fields.length >= 3; })
      .map(function (entry) {
        return {
          key: entry[0],
          label: LABELS[entry[0]] || entry[0],
          fields: entry[1].fields.sort(function (a, b) { return a.phaseIndex - b.phaseIndex; }),
        };
      });
  }

  // ── Filters ──
  function patientPassesFilters(patient, includeTarget) {
    if (includeTarget === undefined) includeTarget = true;
    var outcome = patient.outcomes[state.outcome];
    if (!outcome) return false;
    if (includeTarget && state.target === "ok" && outcome.outcomeReal !== 0) return false;
    if (includeTarget && state.target === "not-ok" && outcome.outcomeReal !== 1) return false;
    if (state.site !== "all" && String(patient.sede) !== state.site) return false;

    // Year range filter
    if (Number.isFinite(state.yearFrom) && patient.periodo < state.yearFrom) return false;
    if (Number.isFinite(state.yearTo) && patient.periodo > state.yearTo) return false;

    // EG group filter
    if (state.egGroup !== "all") {
      var eg = patient.summaryValues ? patient.summaryValues[2] : null;
      if (!Number.isFinite(eg)) return false;
      if (state.egGroup === "lt32" && eg >= 32) return false;
      if (state.egGroup === "32-34" && (eg < 32 || eg > 34)) return false;
      if (state.egGroup === "35-37" && (eg < 35 || eg > 37)) return false;
      if (state.egGroup === "gt37" && eg <= 37) return false;
    }

    // RCIU filter (birthGroup 1,3 = con RCIU; 2,4 = sin RCIU)
    if (state.rciu === "con" && patient.birthGroup !== 1 && patient.birthGroup !== 3) return false;
    if (state.rciu === "sin" && patient.birthGroup !== 2 && patient.birthGroup !== 4) return false;

    return true;
  }

  function filteredPatients(includeTarget) {
    if (includeTarget === undefined) includeTarget = true;
    return patientsWithOutcome().filter(function (patient) { return patientPassesFilters(patient, includeTarget); });
  }

  function selectedPatient() {
    if (state.selectedId === null) return null;
    var current = DATA.patients.find(function (patient) { return patient.id === state.selectedId && patient.outcomes[state.outcome]; });
    return current || null;
  }

  // ── Similarity engine ──
  function fieldValue(patient, field) {
    if (!patient) return null;
    if (field.kind === "summary") return patient.summaryValues[field.index];
    if (field.kind === "probability") {
      var out = patient.outcomes[state.outcome];
      return out && out.probs ? out.probs[field.phaseIndex] : null;
    }
    if (patient.similarityValues && field.simIndex !== undefined) return patient.similarityValues[field.simIndex];
    var row = patient.phaseValues ? patient.phaseValues[field.phaseIndex] || [] : [];
    return row[field.index];
  }

  function comparableFields(reference, phaseIndex) {
    var fields = [];
    var summaryIndex = DATA.meta.summaryFields.findIndex(function (field) { return field.key === "gestasal"; });
    if (summaryIndex >= 0) {
      fields.push({ kind: "summary", key: "gestasal", label: "Edad gestacional", index: summaryIndex, tolerance: state.egTolerance });
    }

    if (Array.isArray(DATA.meta.similarityFields)) {
      DATA.meta.similarityFields.forEach(function (template, simIndex) {
        if (template.phaseIndex > phaseIndex) return;
        var selectedValue = reference ? fieldValue(reference, { kind: "phase", key: template.key, phaseIndex: template.phaseIndex, index: template.index, simIndex: simIndex }) : null;
        if (selectedValue === null || selectedValue === undefined || !Number.isFinite(Number(selectedValue))) return;
        fields.push({
          kind: "phase",
          key: template.key,
          label: template.label,
          phaseIndex: template.phaseIndex,
          index: template.index,
          simIndex: simIndex,
          tolerance: state.zTolerance,
        });
      });

      // Add probability at each phase ≤ phaseIndex as similarity field
      // This ensures visual coherence: patients look similar in the chart
      for (var p = 0; p <= phaseIndex; p++) {
        var refProb = reference ? fieldValue(reference, { kind: "probability", phaseIndex: p }) : null;
        if (Number.isFinite(refProb)) {
          fields.push({
            kind: "probability",
            key: "prob_" + DATA.meta.phases[p].id,
            label: "Probabilidad " + DATA.meta.phases[p].id,
            phaseIndex: p,
            tolerance: 0.15,
          });
        }
      }

      return fields;
    }

    var seen = new Set();
    for (var phase = 0; phase <= phaseIndex; phase += 1) {
      var phaseId = DATA.meta.phases[phase].id;
      var templates = DATA.meta.phaseFieldTemplates[phaseId] || [];
      templates.forEach(function (template, index) {
        var key = template.key || "";
        if (!key.toLowerCase().includes("zscore") || key.toLowerCase().endsWith("cat") || seen.has(key)) return;
        var selectedValue = reference ? fieldValue(reference, { kind: "phase", key: key, phaseIndex: phase, index: index }) : null;
        if (selectedValue === null || selectedValue === undefined || !Number.isFinite(Number(selectedValue))) return;
        seen.add(key);
        fields.push({
          kind: "phase",
          key: key,
          label: template.label,
          phaseIndex: phase,
          index: index,
          tolerance: state.zTolerance,
        });
      });
    }

    // Fallback also gets probability fields
    for (var pf = 0; pf <= phaseIndex; pf++) {
      var refProbFb = reference ? fieldValue(reference, { kind: "probability", phaseIndex: pf }) : null;
      if (Number.isFinite(refProbFb)) {
        fields.push({
          kind: "probability",
          key: "prob_" + DATA.meta.phases[pf].id,
          label: "Probabilidad " + DATA.meta.phases[pf].id,
          phaseIndex: pf,
          tolerance: 0.15,
        });
      }
    }

    return fields;
  }

  function similarityDistance(candidate, reference, fields) {
    var compared = 0;
    var total = 0;
    for (var i = 0; i < fields.length; i++) {
      var field = fields[i];
      var refValue = Number(fieldValue(reference, field));
      if (!Number.isFinite(refValue)) continue;
      var candidateValue = Number(fieldValue(candidate, field));
      if (!Number.isFinite(candidateValue)) return Infinity;
      var delta = Math.abs(candidateValue - refValue);
      if (delta > field.tolerance) return Infinity;
      compared += 1;
      total += delta / Math.max(field.tolerance, 0.00001);
    }
    return compared > 0 ? total / compared : Infinity;
  }

  function similarPatients(reference, phaseIndex, includeTarget) {
    if (includeTarget === undefined) includeTarget = true;
    if (!reference) return [];
    var fields = comparableFields(reference, phaseIndex);
    return filteredPatients(includeTarget)
      .filter(function (patient) { return patient.id !== reference.id; })
      .map(function (patient) {
        return { patient: patient, distance: similarityDistance(patient, reference, fields) };
      })
      .filter(function (item) { return Number.isFinite(item.distance); })
      .sort(function (a, b) { return a.distance - b.distance; })
      .map(function (item) { return item.patient; });
  }

  function trajectory(patient) {
    return patient.outcomes[state.outcome].probs;
  }

  function samplePatients(patients, limit) {
    if (patients.length <= limit) return patients;
    var step = patients.length / limit;
    return Array.from({ length: limit }, function (_, index) { return patients[Math.floor(index * step)]; });
  }

  // ── SVG helpers ──
  function svgEl(name, attrs) {
    var node = document.createElementNS("http://www.w3.org/2000/svg", name);
    if (attrs) Object.entries(attrs).forEach(function (pair) { node.setAttribute(pair[0], pair[1]); });
    return node;
  }

  function pathFor(values, xFor, yFor) {
    return values.map(function (value, index) { return (index === 0 ? "M" : "L") + " " + xFor(index) + " " + yFor(value); }).join(" ");
  }

  function pathForSparse(values, xFor, yFor) {
    var first = true;
    return values
      .map(function (value, index) {
        if (!Number.isFinite(value)) return null;
        var cmd = first ? "M" : "L";
        first = false;
        return cmd + " " + xFor(index) + " " + yFor(value);
      })
      .filter(Boolean)
      .join(" ");
  }

  // ── Chart: probability mode ──
  function drawProbabilityChart(reference, similar) {
    els.chart.innerHTML = "";
    var width = 760, height = 420;
    var margin = { top: 28, right: 26, bottom: 64, left: 62 };
    var innerW = width - margin.left - margin.right;
    var innerH = height - margin.top - margin.bottom;
    var xFor = function (index) { return margin.left + (innerW / (DATA.meta.phases.length - 1)) * index; };
    var yFor = function (value) { return margin.top + innerH - Number(value) * innerH; };

    // Grid lines + Y labels
    [0, 0.25, 0.5, 0.75, 1].forEach(function (tick) {
      els.chart.appendChild(svgEl("line", { x1: margin.left, x2: width - margin.right, y1: yFor(tick), y2: yFor(tick), class: "chart-grid" }));
      els.chart.appendChild(svgEl("text", { x: margin.left - 8, y: yFor(tick) + 4, class: "chart-label", "text-anchor": "end" })).textContent = Math.round(tick * 100) + "%";
    });

    // Y-axis label
    var yLabel = svgEl("text", { x: 16, y: margin.top + innerH / 2, class: "chart-label", "text-anchor": "middle", transform: "rotate(-90, 16, " + (margin.top + innerH / 2) + ")" });
    yLabel.textContent = "Probabilidad estimada de riesgo";
    els.chart.appendChild(yLabel);

    // Phase labels
    DATA.meta.phases.forEach(function (phase, index) {
      els.chart.appendChild(svgEl("line", { x1: xFor(index), x2: xFor(index), y1: margin.top, y2: height - margin.bottom, class: "chart-grid" }));
      els.chart.appendChild(svgEl("text", { x: xFor(index), y: height - 34, "text-anchor": "middle", class: "chart-label" })).textContent = phase.id;
      var time = els.chart.appendChild(svgEl("text", { x: xFor(index), y: height - 18, "text-anchor": "middle", class: "chart-label" }));
      time.textContent = phase.label.length > 13 ? phase.label.slice(0, 13) : phase.label;
    });

    // Cluster profiles (dashed)
    var clusterProfiles = outcomeMeta().clusterProfiles || [];
    clusterProfiles.forEach(function (profile, index) {
      var values = DATA.meta.phases.map(function (phase) { return profile["prob_media_" + phase.phaseKey]; });
      els.chart.appendChild(svgEl("path", {
        d: pathFor(values, xFor, yFor),
        class: "chart-line-cluster",
        stroke: CLUSTER_COLORS[index % CLUSTER_COLORS.length],
      }));
    });

    // Cohort lines — colored by outcome (already sampled in render)
    similar.forEach(function (patient) {
      var isNoOk = patient.outcomes[state.outcome].outcomeReal === 1;
      els.chart.appendChild(svgEl("path", {
        d: pathFor(trajectory(patient), xFor, yFor),
        class: isNoOk ? "chart-line-nook" : "chart-line-ok",
      }));
    });

    // Selected patient
    if (reference) {
      var values = trajectory(reference);
      els.chart.appendChild(svgEl("path", { d: pathFor(values, xFor, yFor), class: "chart-line-selected" }));
      values.forEach(function (value, index) {
        els.chart.appendChild(svgEl("circle", { cx: xFor(index), cy: yFor(value), r: 4.2, class: "chart-point" }));
      });
    }

    // Axes
    els.chart.appendChild(svgEl("line", { x1: margin.left, x2: width - margin.right, y1: height - margin.bottom, y2: height - margin.bottom, class: "chart-axis" }));
    els.chart.appendChild(svgEl("line", { x1: margin.left, x2: margin.left, y1: margin.top, y2: height - margin.bottom, class: "chart-axis" }));
  }

  // ── Chart: z-score mode ──
  function drawZscoreChart(reference, similar, zscoreKey) {
    var traj = zscoreTrajectories.find(function (t) { return t.key === zscoreKey; });
    if (!traj) return;

    els.chart.innerHTML = "";
    var width = 760, height = 420;
    var margin = { top: 28, right: 26, bottom: 64, left: 62 };
    var innerW = width - margin.left - margin.right;
    var innerH = height - margin.top - margin.bottom;

    // Build phase → simIndex mapping
    var phaseMap = {};
    traj.fields.forEach(function (f) { phaseMap[f.phaseIndex] = f.simIndex; });

    // Y-axis range
    var yMin = -5, yMax = 3;
    var yRange = yMax - yMin;
    var xFor = function (index) { return margin.left + (innerW / (DATA.meta.phases.length - 1)) * index; };
    var yFor = function (value) { return margin.top + innerH - ((value - yMin) / yRange) * innerH; };

    // Grid lines + Y labels
    [-4, -3, -2, -1, 0, 1, 2].forEach(function (tick) {
      var isThreshold = tick === -2;
      els.chart.appendChild(svgEl("line", {
        x1: margin.left, x2: width - margin.right, y1: yFor(tick), y2: yFor(tick),
        class: isThreshold ? "" : "chart-grid",
        stroke: isThreshold ? "#D85B47" : undefined,
        "stroke-dasharray": isThreshold ? "6 3" : undefined,
        "stroke-width": isThreshold ? "1.5" : undefined,
        opacity: isThreshold ? "0.6" : undefined,
      }));
      els.chart.appendChild(svgEl("text", {
        x: margin.left - 8, y: yFor(tick) + 4, class: "chart-label", "text-anchor": "end",
        fill: isThreshold ? "#D85B47" : undefined, "font-weight": isThreshold ? "600" : undefined,
      })).textContent = tick + (isThreshold ? " DE" : "");
    });

    // Y-axis label
    var yLabel = svgEl("text", { x: 16, y: margin.top + innerH / 2, class: "chart-label", "text-anchor": "middle", transform: "rotate(-90, 16, " + (margin.top + innerH / 2) + ")" });
    yLabel.textContent = traj.label;
    els.chart.appendChild(yLabel);

    // Phase labels
    DATA.meta.phases.forEach(function (phase, index) {
      var hasData = phaseMap[index] !== undefined;
      els.chart.appendChild(svgEl("line", { x1: xFor(index), x2: xFor(index), y1: margin.top, y2: height - margin.bottom, class: "chart-grid", opacity: hasData ? "1" : "0.3" }));
      els.chart.appendChild(svgEl("text", { x: xFor(index), y: height - 34, "text-anchor": "middle", class: "chart-label", opacity: hasData ? "1" : "0.4" })).textContent = phase.id;
      var time = els.chart.appendChild(svgEl("text", { x: xFor(index), y: height - 18, "text-anchor": "middle", class: "chart-label", opacity: hasData ? "1" : "0.4" }));
      time.textContent = phase.label.length > 13 ? phase.label.slice(0, 13) : phase.label;
    });

    // Helper: get z-score values for a patient
    function zscoreValues(patient) {
      return DATA.meta.phases.map(function (_, i) {
        if (phaseMap[i] === undefined) return null;
        var val = patient.similarityValues ? patient.similarityValues[phaseMap[i]] : null;
        return Number.isFinite(val) ? val : null;
      });
    }

    // Cohort lines — colored by outcome (already sampled in render)
    similar.forEach(function (patient) {
      var vals = zscoreValues(patient);
      var isNoOk = patient.outcomes[state.outcome].outcomeReal === 1;
      var path = pathForSparse(vals, xFor, yFor);
      if (path) {
        els.chart.appendChild(svgEl("path", { d: path, class: isNoOk ? "chart-line-nook" : "chart-line-ok" }));
      }
    });

    // Selected patient
    if (reference) {
      var vals = zscoreValues(reference);
      var path = pathForSparse(vals, xFor, yFor);
      if (path) {
        els.chart.appendChild(svgEl("path", { d: path, class: "chart-line-selected" }));
        vals.forEach(function (val, i) {
          if (Number.isFinite(val)) {
            els.chart.appendChild(svgEl("circle", { cx: xFor(i), cy: yFor(val), r: 4.2, class: "chart-point" }));
          }
        });
      }
    }

    // Axes
    els.chart.appendChild(svgEl("line", { x1: margin.left, x2: width - margin.right, y1: height - margin.bottom, y2: height - margin.bottom, class: "chart-axis" }));
    els.chart.appendChild(svgEl("line", { x1: margin.left, x2: margin.left, y1: margin.top, y2: height - margin.bottom, class: "chart-axis" }));
  }

  // ── Chart dispatch ──
  function drawChart(reference, similar) {
    if (state.chartMode === "probability") {
      drawProbabilityChart(reference, similar);
    } else {
      drawZscoreChart(reference, similar, state.chartMode);
    }
    updateLegend();
  }

  function updateLegend() {
    if (!els.legendRow) return;
    var isZscore = state.chartMode !== "probability";
    var html = '<span class="legend-item"><span class="legend-swatch"></span>Paciente seleccionado</span>';
    html += '<span class="legend-item"><span class="legend-swatch ok"></span>Parecidos OK</span>';
    html += '<span class="legend-item"><span class="legend-swatch nook"></span>Parecidos No OK</span>';
    if (!isZscore) {
      html += '<span class="legend-item"><span class="legend-swatch cluster"></span>Media de cluster</span>';
    } else {
      html += '<span class="legend-item"><span class="legend-swatch threshold"></span>Umbral -2 DE</span>';
    }
    els.legendRow.innerHTML = html;
  }

  // ── Render functions ──
  function renderStats(reference, similar, cohort) {
    var out = reference ? reference.outcomes[state.outcome] : null;
    var okCohort = cohort.filter(function (p) { return p.outcomes[state.outcome].outcomeReal === 0; }).length;
    var nokCohort = cohort.length - okCohort;
    var nokRate;
    var stats;
    if (reference) {
      var okSim = similar.filter(function (p) { return p.outcomes[state.outcome].outcomeReal === 0; }).length;
      var nokSim = similar.length - okSim;
      nokRate = similar.length ? nokSim / similar.length : 0;
      stats = [
        ["Muestra filtrada", fmt.format(cohort.length)],
        ["Parecidos", fmt.format(similar.length) + " (" + fmt.format(okSim) + " OK / " + fmt.format(nokSim) + " No OK)"],
        ["No OK en parecidos", pct.format(nokRate * 100) + "%"],
        ["Riesgo final paciente", out ? riskLabel(out.finalRisk) : "sin dato"],
      ];
    } else {
      nokRate = cohort.length ? nokCohort / cohort.length : 0;
      stats = [
        ["Muestra filtrada", fmt.format(cohort.length)],
        ["OK en muestra", fmt.format(okCohort)],
        ["No OK en muestra", fmt.format(nokCohort) + " (" + pct.format(nokRate * 100) + "%)"],
        ["Paciente", "Ninguno seleccionado"],
      ];
    }
    els.stats.innerHTML = stats.map(function (pair) {
      return '<article class="mini-stat"><span>' + pair[0] + '</span><strong>' + pair[1] + '</strong></article>';
    }).join("");
  }

  function renderSelected(reference) {
    if (!reference) {
      els.selectedSummary.textContent = "Haga clic en una trayectoria de la gráfica para seleccionar un paciente, o busque por ID.";
      els.selectedFields.innerHTML = "";
      return;
    }
    var out = reference.outcomes[state.outcome];
    var bgLabel = BIRTH_GROUP_LABELS[reference.birthGroup] || ("Grupo " + reference.birthGroup);
    els.selectedSummary.textContent = "ID " + reference.id + ". " + (out.outcomeReal === 1 ? "No OK" : "OK") + " a 12 meses para " + outcomeMeta().shortLabel.toLowerCase() + ". Cluster " + (out.cluster || "sin cluster") + ": " + (out.clusterLabel || "sin etiqueta") + ".";

    var summaryRows = DATA.meta.summaryFields.map(function (field, index) { return [field.label, valueLabel(reference.summaryValues[index], field.unit)]; });
    var extraRows = [
      ["Sede", valueLabel(reference.sede)],
      ["Año", valueLabel(reference.periodo)],
      ["Grupo al nacer", bgLabel],
      ["Riesgo F0", riskLabel(out.probs[0])],
      ["Riesgo F6", riskLabel(out.probs[out.probs.length - 1])],
    ];
    els.selectedFields.innerHTML = extraRows.concat(summaryRows).map(function (pair) {
      return '<div class="field-row"><span>' + pair[0] + '</span><strong>' + pair[1] + '</strong></div>';
    }).join("");
  }

  function renderEquivalence(reference) {
    var rows = DATA.meta.phases.map(function (phase, index) {
      var group = similarPatients(reference, index, false);
      var positives = group.filter(function (patient) { return patient.outcomes[state.outcome].outcomeReal === 1; }).length;
      var negatives = group.length - positives;
      var avgRisk = group.length
        ? group.reduce(function (sum, patient) { return sum + patient.outcomes[state.outcome].finalRisk; }, 0) / group.length
        : null;
      return { phase: phase, index: index, group: group, positives: positives, negatives: negatives, avgRisk: avgRisk };
    });

    els.equivalence.innerHTML = '<table><thead><tr>' +
      '<th>Fase usada</th><th>Campos comparables</th><th>Pacientes parecidos</th>' +
      '<th>OK</th><th>No OK</th><th>Prevalencia no OK</th><th>Riesgo final medio</th>' +
      '</tr></thead><tbody>' +
      rows.map(function (row) {
        var fieldCount = comparableFields(reference, row.index).length;
        var prev = row.group.length ? row.positives / row.group.length : 0;
        return '<tr data-selectable="true" data-phase-index="' + row.index + '">' +
          '<td>' + row.phase.id + ' · ' + row.phase.label + '</td>' +
          '<td>' + fieldCount + '</td>' +
          '<td>' + fmt.format(row.group.length) + '</td>' +
          '<td>' + fmt.format(row.negatives) + '</td>' +
          '<td>' + fmt.format(row.positives) + '</td>' +
          '<td>' + pct.format(prev * 100) + '%</td>' +
          '<td>' + (row.avgRisk === null ? "sin dato" : riskLabel(row.avgRisk)) + '</td>' +
          '</tr>';
      }).join("") +
      '</tbody></table>' +
      '<p class="note-text">La tabla respeta filtros de sede, grupo al nacer y ventana temporal, pero no aplica el filtro OK/no OK para poder mostrar la separación del target.</p>';

    els.equivalence.querySelectorAll("tr[data-phase-index]").forEach(function (row) {
      row.addEventListener("click", function () {
        state.phaseIndex = Number(row.dataset.phaseIndex);
        els.phase.value = String(state.phaseIndex);
        markDirty();
      });
    });
  }

  function renderNeighbors(reference, similar) {
    var rows = similar.slice(0, 10);
    els.neighbors.innerHTML = '<table><thead><tr>' +
      '<th>ID</th><th>Target</th><th>Cluster</th><th>Riesgo F0</th><th>Riesgo F6</th>' +
      '<th>Sede</th><th>Grupo</th><th>Año</th>' +
      '</tr></thead><tbody>' +
      rows.map(function (patient) {
        var out = patient.outcomes[state.outcome];
        var bgLabel = BIRTH_GROUP_LABELS[patient.birthGroup] || ("Grupo " + patient.birthGroup);
        return '<tr data-selectable="true" data-patient-id="' + patient.id + '">' +
          '<td>' + patient.id + '</td>' +
          '<td>' + (out.outcomeReal === 1 ? "No OK" : "OK") + '</td>' +
          '<td>' + (out.cluster || "") + ' · ' + (out.clusterLabel || "") + '</td>' +
          '<td>' + riskLabel(out.probs[0]) + '</td>' +
          '<td>' + riskLabel(out.finalRisk) + '</td>' +
          '<td>' + valueLabel(patient.sede) + '</td>' +
          '<td>' + bgLabel + '</td>' +
          '<td>' + valueLabel(patient.periodo) + '</td>' +
          '</tr>';
      }).join("") +
      '</tbody></table>' +
      '<p class="note-text">' + (rows.length ? "Al seleccionar una fila, ese paciente pasa a ser la referencia." : "No se encontraron sujetos con los umbrales actuales. Puede relajar EG o z-score.") + '</p>';

    els.neighbors.querySelectorAll("tr[data-patient-id]").forEach(function (row) {
      row.addEventListener("click", function () {
        state.selectedId = Number(row.dataset.patientId);
        els.search.value = state.selectedId;
        renderSelection();
      });
    });
  }

  function sparkline(profile) {
    var values = DATA.meta.phases.map(function (phase) { return profile["prob_media_" + phase.phaseKey]; });
    var width = 250, height = 58;
    var xFor = function (index) { return 8 + (234 / (values.length - 1)) * index; };
    var yFor = function (value) { return 50 - Number(value) * 42; };
    var path = pathFor(values, xFor, yFor);
    var color = CLUSTER_COLORS[(profile.cluster - 1) % CLUSTER_COLORS.length];
    return '<svg class="sparkline" viewBox="0 0 ' + width + ' ' + height + '" aria-hidden="true">' +
      '<line x1="8" x2="242" y1="' + yFor(0.5) + '" y2="' + yFor(0.5) + '" stroke="#d9d9d6" stroke-dasharray="3 4"></line>' +
      '<path d="' + path + '" fill="none" stroke="' + color + '" stroke-width="2"></path>' +
      values.map(function (value, index) {
        return '<circle cx="' + xFor(index) + '" cy="' + yFor(value) + '" r="2.7" fill="#fff" stroke="' + color + '" stroke-width="1.5"></circle>';
      }).join("") +
      '</svg>';
  }

  function renderClusters(reference) {
    var meta = outcomeMeta();
    var allFiltered = filteredPatients(false);
    var selectedCluster = reference ? reference.outcomes[state.outcome].cluster : null;

    els.clusterGrid.innerHTML = (meta.clusterProfiles || []).map(function (profile) {
      var probValues = DATA.meta.phases.map(function (phase) { return profile["prob_media_" + phase.phaseKey]; }).filter(Number.isFinite);
      var probMin = probValues.length ? Math.min.apply(null, probValues) : 0;
      var probMax = probValues.length ? Math.max.apply(null, probValues) : 0;

      var clusterPatients = allFiltered.filter(function (p) {
        var out = p.outcomes[state.outcome];
        return out && out.cluster === profile.cluster;
      });
      var noOk = clusterPatients.filter(function (p) { return p.outcomes[state.outcome].outcomeReal === 1; }).length;
      var ok = clusterPatients.length - noOk;
      var isCurrent = profile.cluster === selectedCluster;

      return '<article class="cluster-card' + (isCurrent ? ' is-current' : '') + '">' +
        '<h3>Cluster ' + profile.cluster + ': ' + profile.cluster_label + (isCurrent ? ' (actual)' : '') + '</h3>' +
        '<p class="cluster-meta">' + fmt.format(profile.n_pacientes) + ' pac. (global) · ' + pct.format(profile.prevalencia * 100) + '% no OK</p>' +
        '<p class="cluster-meta">Prob. media: ' + pct.format(probMin * 100) + '% &rarr; ' + pct.format(probMax * 100) + '%</p>' +
        '<p class="cluster-meta">En filtro actual: ' + fmt.format(ok) + ' OK · ' + fmt.format(noOk) + ' No OK</p>' +
        sparkline(profile) +
        '</article>';
    }).join("");

    var best = (meta.clusterProfiles || []).find(function (profile) { return profile.cluster === selectedCluster; });
    els.clusterReading.textContent = best
      ? "El paciente seleccionado pertenece al cluster " + best.cluster + " (" + best.cluster_label + "), que agrupa " + fmt.format(best.n_pacientes) + " pacientes con " + pct.format(best.prevalencia * 100) + "% de prevalencia No OK. Use las líneas punteadas en la gráfica para contrastar su trayectoria individual con el perfil medio de este cluster."
      : "Cada cluster agrupa pacientes con trayectorias de probabilidad similares a lo largo de las 7 fases. La prevalencia indica qué proporción terminó con el desenlace (No OK) a los 12 meses.";
  }

  function renderExamples() {
    var featured = DATA.featured[state.outcome] || {};
    var examples = [
      ["Riesgo alto desde F0", featured.earlyHigh],
      ["Riesgo bajo estable", featured.stableLow],
      ["Escalada tardía", featured.lateEscalation],
    ].filter(function (pair) { return pair[1]; });
    els.exampleRow.innerHTML = examples.map(function (pair) {
      return '<button class="example-button" type="button" data-patient-id="' + pair[1] + '">' + pair[0] + ' (ID ' + pair[1] + ')</button>';
    }).join("");
    els.exampleRow.querySelectorAll("[data-patient-id]").forEach(function (button) {
      button.addEventListener("click", function () {
        state.selectedId = Number(button.dataset.patientId);
        els.search.value = state.selectedId;
        renderSelection();
      });
    });
  }

  function renderPeriodNote() {
    var total = filteredPatients(false).length;
    els.periodReading.textContent = "Ventana temporal activa: " + state.yearFrom + "–" + state.yearTo + ". Hay " + fmt.format(total) + " pacientes en este rango con los filtros clínicos aplicados.";
  }

  // ── Full render (called by Calcular) ── recomputes visible lines ──
  function render() {
    state.egTolerance = parseInputNumber(els.egTolerance.value);
    state.zTolerance = parseInputNumber(els.zTolerance.value);
    state.yearFrom = parseInputNumber(els.yearFrom.value);
    state.yearTo = parseInputNumber(els.yearTo.value);
    state.chartMode = els.chartMode ? els.chartMode.value : "probability";
    var reference = selectedPatient();
    var cohort = filteredPatients(true);

    // Compute visible lines: if reference → similar patients, else → full cohort
    var pool;
    if (reference) {
      pool = similarPatients(reference, state.phaseIndex, true);
    } else {
      pool = cohort;
    }
    var visibleLines = samplePatients(pool, 140);
    state._visibleLines = visibleLines;
    state._displayedPatients = visibleLines;
    state._cohort = cohort;

    renderExamples();
    renderStats(reference, pool, cohort);
    renderSelected(reference);
    drawChart(reference, visibleLines);

    if (reference) {
      renderEquivalence(reference);
      renderNeighbors(reference, pool);
    } else {
      els.equivalence.innerHTML = '<p class="note-text">Haga clic en una trayectoria de la gráfica o busque un paciente por ID para ver las clases de equivalencia.</p>';
      els.neighbors.innerHTML = '';
    }

    renderClusters(reference);
    renderPeriodNote();
    markClean();
  }

  // ── Selection render (called by line click) ── keeps visible lines stable ──
  function renderSelection() {
    var reference = selectedPatient();
    var cohort = state._cohort || filteredPatients(true);
    var similar = reference ? similarPatients(reference, state.phaseIndex, true) : [];
    var visibleLines = state._visibleLines || [];

    renderStats(reference, similar, cohort);
    renderSelected(reference);
    drawChart(reference, visibleLines);

    if (reference) {
      renderEquivalence(reference);
      renderNeighbors(reference, similar);
    } else {
      els.equivalence.innerHTML = '<p class="note-text">Haga clic en una trayectoria de la gráfica o busque un paciente por ID para ver las clases de equivalencia.</p>';
      els.neighbors.innerHTML = '';
    }

    renderClusters(reference);
  }

  // ── Populate selects ──
  function populateSelects() {
    els.outcome.innerHTML = Object.values(DATA.meta.outcomes).map(function (outcome) {
      return '<option value="' + outcome.key + '">' + outcome.label + '</option>';
    }).join("");
    els.outcome.value = state.outcome;

    els.phase.innerHTML = DATA.meta.phases.map(function (phase, index) {
      return '<option value="' + index + '">' + phase.id + ' · ' + phase.label + '</option>';
    }).join("");
    els.phase.value = String(state.phaseIndex);

    var sites = uniqueSorted(DATA.patients.map(function (patient) { return patient.sede; }));
    els.site.innerHTML = '<option value="all">Todas las sedes</option>' + sites.map(function (site) { return '<option value="' + site + '">Sede ' + site + '</option>'; }).join("");

    // Year range from actual data
    var years = uniqueSorted(DATA.patients.map(function (p) { return p.periodo; })).map(Number).filter(Number.isFinite);
    var minYear = years.length ? Math.min.apply(null, years) : 1993;
    var maxYear = years.length ? Math.max.apply(null, years) : 2023;
    els.yearFrom.min = minYear;
    els.yearFrom.max = maxYear;
    els.yearFrom.value = minYear;
    els.yearTo.min = minYear;
    els.yearTo.max = maxYear;
    els.yearTo.value = maxYear;
    state.yearFrom = minYear;
    state.yearTo = maxYear;

    // Chart mode options (probability + z-score trajectories)
    zscoreTrajectories = discoverZscoreTrajectories();
    if (els.chartMode) {
      els.chartMode.innerHTML = '<option value="probability">Probabilidad de riesgo estimada</option>' +
        zscoreTrajectories.map(function (t) { return '<option value="' + t.key + '">' + t.label + '</option>'; }).join("");
    }
  }

  // ── Event binding ──
  function bindEvents() {
    var debouncedDirty = debounce(markDirty, 300);

    // ── Filter changes → mark dirty (don't auto-render) ──
    els.outcome.addEventListener("change", function () {
      state.outcome = els.outcome.value;
      state.selectedId = null;
      els.search.value = "";
      markDirty();
    });
    els.phase.addEventListener("change", function () {
      state.phaseIndex = Number(els.phase.value);
      markDirty();
    });
    els.target.addEventListener("change", function () {
      state.target = els.target.value;
      markDirty();
    });
    els.site.addEventListener("change", function () {
      state.site = els.site.value;
      markDirty();
    });
    els.egGroup.addEventListener("change", function () {
      state.egGroup = els.egGroup.value;
      markDirty();
    });
    els.rciu.addEventListener("change", function () {
      state.rciu = els.rciu.value;
      markDirty();
    });
    els.yearFrom.addEventListener("input", function () {
      state.yearFrom = parseInputNumber(els.yearFrom.value);
      debouncedDirty();
    });
    els.yearTo.addEventListener("input", function () {
      state.yearTo = parseInputNumber(els.yearTo.value);
      debouncedDirty();
    });
    els.egTolerance.addEventListener("input", debouncedDirty);
    els.zTolerance.addEventListener("input", debouncedDirty);

    // ── Chart mode → mark dirty ──
    if (els.chartMode) {
      els.chartMode.addEventListener("change", function () {
        state.chartMode = els.chartMode.value;
        markDirty();
      });
    }

    // ── Calculate button → full render ──
    if (els.calculateBtn) {
      els.calculateBtn.addEventListener("click", function () {
        render();
      });
    }

    // ── Patient selection → immediate render ──
    els.searchButton.addEventListener("click", function () {
      var id = Number(els.search.value);
      var patient = DATA.patients.find(function (item) { return item.id === id && item.outcomes[state.outcome]; });
      if (patient) {
        state.selectedId = id;
        renderSelection();
      }
    });
    els.search.addEventListener("keydown", function (event) {
      if (event.key === "Enter") els.searchButton.click();
    });

    // ── Click on chart line to select a patient ──
    els.chart.style.cursor = "crosshair";
    els.chart.addEventListener("click", function (evt) {
      var displayed = state._displayedPatients || [];
      if (!displayed.length) return;

      var pt = els.chart.createSVGPoint();
      pt.x = evt.clientX;
      pt.y = evt.clientY;
      var svgPt = pt.matrixTransform(els.chart.getScreenCTM().inverse());

      var width = 760, height = 420;
      var margin = { top: 28, right: 26, bottom: 64, left: 62 };
      var innerW = width - margin.left - margin.right;
      var innerH = height - margin.top - margin.bottom;
      var phases = DATA.meta.phases.length;

      // Ignore clicks outside chart area
      if (svgPt.x < margin.left || svgPt.x > width - margin.right || svgPt.y < margin.top || svgPt.y > height - margin.bottom) return;

      // Find closest phase column
      var bestPhase = 0;
      var bestPhaseDist = Infinity;
      for (var i = 0; i < phases; i++) {
        var px = margin.left + (innerW / (phases - 1)) * i;
        var dist = Math.abs(svgPt.x - px);
        if (dist < bestPhaseDist) { bestPhaseDist = dist; bestPhase = i; }
      }

      // Find nearest patient at that phase
      var isZscore = state.chartMode !== "probability";
      var zscorePhaseMap = null;
      if (isZscore) {
        var traj = zscoreTrajectories.find(function (t) { return t.key === state.chartMode; });
        if (traj) {
          zscorePhaseMap = {};
          traj.fields.forEach(function (f) { zscorePhaseMap[f.phaseIndex] = f.simIndex; });
        }
      }

      var bestPatient = null;
      var bestDist = Infinity;
      displayed.forEach(function (patient) {
        var val;
        if (isZscore && zscorePhaseMap) {
          if (zscorePhaseMap[bestPhase] === undefined) return;
          val = patient.similarityValues ? patient.similarityValues[zscorePhaseMap[bestPhase]] : null;
        } else {
          val = trajectory(patient)[bestPhase];
        }
        if (!Number.isFinite(val)) return;

        var patientY;
        if (isZscore) {
          patientY = margin.top + innerH - ((val - (-5)) / (3 - (-5))) * innerH;
        } else {
          patientY = margin.top + innerH - val * innerH;
        }
        var d = Math.abs(svgPt.y - patientY);
        if (d < bestDist) { bestDist = d; bestPatient = patient; }
      });

      if (bestPatient && bestDist < 30) {
        state.selectedId = bestPatient.id;
        els.search.value = bestPatient.id;
        renderSelection();
      }
    });
  }

  // ── Init ──
  function init() {
    populateSelects();
    state.selectedId = null;
    els.search.value = "";
    bindEvents();
    render(); // first render shows all cohort lines — no patient selected yet
  }

  init();
})();
