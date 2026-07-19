/* ==========================================================================
   PROJECT DATA
   -- Metrics and the GSTT2B finding below are the real, confirmed
   numbers from this project's actual pipeline run. genePanel (further
   down) is explicitly mock data past its first row -- see the comment
   there before mistaking it for a real 100-gene result set.
   ========================================================================== */

const projectMeta = {
  tagline:
    "Predicting vital status (Alive vs. Dead) from TCGA-BRCA RNA-Seq expression data.",
  overviewStats: [
    { label: "Raw genes", value: "60,660", note: "RNA-Seq, GDC Hub" },
    { label: "Cohort", value: "TCGA-BRCA", note: "UCSC Xena" },
    { label: "Target", value: "Alive / Dead", note: "vital_status.demographic" },
  ],
  funnel: [
    { stage: "Raw expression", genes: 60660 },
    { stage: "Preprocessed", genes: 5658 },
    { stage: "MAD-filtered", genes: 2000 },
    { stage: "Biomarker panel", genes: 100 },
  ],
};

const modelMetrics = [
  { label: "ROC-AUC", value: 0.6243, note: "vs. 0.50 no-skill baseline" },
  { label: "PR-AUC", value: 0.3102, note: "read against class prevalence" },
  { label: "Macro-F1", value: 0.5654, note: "equal weight, both classes" },
];

const topFinding = {
  symbol: "GSTT2B",
  ensemblId: "ENSG00000133433",
  fullName: "Glutathione S-transferase theta 2B",
  role:
    "A Phase II detoxification enzyme. Helps clear electrophilic compounds -- including several chemotherapy agents -- from cells.",
  meanAbsShap: 0.3835,
  meanShap: -0.1319,
};

const SHAP_SCALE_MAX = 0.4; // domain for the gradient-bar indicator

/* --------------------------------------------------------------------------
   MOCK GENE PANEL
   The real panel has 100 genes; only row 0 (GSTT2B) is real here. The
   rest are deterministic placeholder rows -- not Math.random(), so the
   page looks identical on every reload -- meant to show what a fully
   populated list looks like. Replace this function's mock loop with
   GenomicsFeatureSelector.get_selected_genes() output, or a fetch()
   against reports/shap_biomarker_table.csv, to make it live.
   -------------------------------------------------------------------------- */
function buildGenePanel(realTopGene, mockRowCount) {
  const rows = [
    {
      symbol: realTopGene.symbol,
      ensemblId: realTopGene.ensemblId,
      meanShap: realTopGene.meanShap,
      meanAbsShap: realTopGene.meanAbsShap,
    },
  ];

  for (let i = 1; i <= mockRowCount; i++) {
    const magnitude = Number((0.36 - i * 0.013).toFixed(4));
    const sign = i % 3 === 0 ? -1 : 1;
    rows.push({
      symbol: `GENE_${String(i).padStart(3, "0")}`,
      ensemblId: `ENSG00000${100200 + i * 41}`,
      meanShap: Number((magnitude * sign).toFixed(4)),
      meanAbsShap: magnitude,
    });
  }

  return rows;
}

const genePanel = buildGenePanel(topFinding, 23); // 24 rows, demo-sized

/* ==========================================================================
   HELPERS
   ========================================================================== */
// Generates placeholder text for mock genes with no real annotation
// -- only topFinding (GSTT2B) has real fullName/role values.
function getGeneAnnotation(gene) {
  const hasRealAnnotation = Boolean(gene.fullName && gene.role);

  if (hasRealAnnotation) {
    return { fullName: gene.fullName, role: gene.role };
  }

  return {
    fullName: "Full name not available for this gene yet.",
    role: "Mock demo entry -- swap in real annotation data once this panel is connected to actual pipeline output.",
  };
}
// Maps a signed SHAP value to a 0-100 position on the gradient bar:
// 0 = fully toward Alive, 100 = fully toward Dead. Clamped so one
// unusually large value can't push the marker off the track.
function shapToPercent(meanShap) {
  const clamped = Math.max(-SHAP_SCALE_MAX, Math.min(SHAP_SCALE_MAX, meanShap));
  return ((clamped + SHAP_SCALE_MAX) / (2 * SHAP_SCALE_MAX)) * 100;
}

// Builds the blue-to-coral direction bar used everywhere a signed SHAP
// value appears. Returns a real DOM node (not an HTML string) so it
// can be appended directly wherever it's needed -- a template literal
// can hold text, but not a live element reference.
function createShapBar(meanShap, { compact = false } = {}) {
  const bar = document.createElement("div");
  bar.className = compact ? "shap-bar shap-bar--sm" : "shap-bar";

  const track = document.createElement("div");
  track.className = "shap-bar-track";

  const marker = document.createElement("div");
  marker.className = "shap-bar-marker";
  marker.style.left = `${shapToPercent(meanShap)}%`;

  track.appendChild(marker);
  bar.appendChild(track);
  return bar;
}

/* ==========================================================================
   RENDER FUNCTIONS
   Two DOM-writing techniques are shown on purpose:
     1. createElement + appendChild (renderOverviewStats, createShapBar)
        -- explicit, one call per node. Worth reading closely if DOM
        methods are new to you.
     2. Build one HTML string with map().join(""), then set innerHTML
        once (renderGenePanel) -- faster to write for a long list, at
        the cost of one big string instead of individually-referenceable
        nodes.
   ========================================================================== */

function renderHeader() {
  document.getElementById("project-tagline").textContent = projectMeta.tagline;
  document.getElementById("key-finding-pill").textContent =
    `Key finding → ${topFinding.symbol}`;
}

function renderOverviewStats() {
  const container = document.getElementById("overview-stats");

  projectMeta.overviewStats.forEach((stat) => {
    const card = document.createElement("div");
    card.className = "stat-card";

    const value = document.createElement("p");
    value.className = "stat-value";
    value.textContent = stat.value;

    const label = document.createElement("p");
    label.className = "stat-label";
    label.textContent = stat.label;

    const note = document.createElement("p");
    note.className = "stat-note";
    note.textContent = stat.note;

    // .append() accepts multiple nodes at once -- shorthand for
    // three separate appendChild() calls.
    card.append(value, label, note);
    container.appendChild(card);
  });
}

function renderFunnel() {
  const container = document.getElementById("gene-funnel");
  const steps = projectMeta.funnel;

  steps.forEach((step, index) => {
    const stepEl = document.createElement("div");
    stepEl.className = "funnel-step";
    stepEl.innerHTML = `
      <span class="funnel-genes">${step.genes.toLocaleString()}</span>
      <span class="funnel-stage">${step.stage}</span>
    `;
    container.appendChild(stepEl);

    if (index < steps.length - 1) {
      const arrow = document.createElement("span");
      arrow.className = "funnel-arrow";
      arrow.textContent = "→";
      arrow.setAttribute("aria-hidden", "true");
      container.appendChild(arrow);
    }
  });
}

function renderMetrics() {
  const container = document.getElementById("metrics-container");

  modelMetrics.forEach((metric) => {
    const card = document.createElement("div");
    card.className = "metric-card";
    card.innerHTML = `
      <p class="metric-value">${metric.value.toFixed(4)}</p>
      <p class="metric-label">${metric.label}</p>
      <p class="metric-note">${metric.note}</p>
    `;
    container.appendChild(card);
  });
}

function renderGenePanel() {
  document.getElementById("panel-count").textContent =
    `${genePanel.length} genes shown`;

  const listHtml = genePanel
    .map((gene) => {
      const direction = gene.meanShap < 0 ? "alive" : "dead";
      return `
        <li class="gene-row" data-symbol="${gene.symbol}" tabindex="0" role="button" aria-label="View SHAP details for ${gene.symbol}">
          <div class="gene-id">
            <span class="gene-symbol">${gene.symbol}</span>
            <span class="gene-ensembl">${gene.ensemblId}</span>
          </div>
          <div class="gene-shap gene-shap--${direction}">
            <span class="gene-shap-value">${gene.meanShap.toFixed(4)}</span>
          </div>
        </li>
      `;
    })
    .join("");

  const list = document.getElementById("gene-list");
  list.innerHTML = listHtml;

  // The mini gradient bars are appended as real elements AFTER the
  // innerHTML write above, since createShapBar() returns a DOM node
  // and a template string can only ever hold text.
  list.querySelectorAll(".gene-row").forEach((row, i) => {
    const bar = createShapBar(genePanel[i].meanShap, { compact: true });
    row.querySelector(".gene-shap").prepend(bar);
  });
}

function renderFindingCard(gene = topFinding) {
  const card = document.getElementById("finding-card");
  const { fullName, role } = getGeneAnnotation(gene);
  const isTopFinding = gene.symbol === topFinding.symbol;
  card.classList.toggle("finding-card--protective", gene.meanShap < 0);

  card.innerHTML = `
    <p class="finding-eyebrow">${isTopFinding ? "Top biomarker" : "Selected gene"}</p>
    <h3 class="finding-symbol">${gene.symbol}</h3>
    <p class="finding-fullname">${fullName}</p>
    <p class="finding-role">${role}</p>
    <div class="finding-stats">
      <div>
        <span class="finding-stat-label">Mean |SHAP|</span>
        <span class="finding-stat-value">${gene.meanAbsShap}</span>
      </div>
      <div>
        <span class="finding-stat-label">Mean SHAP (signed)</span>
        <span class="finding-stat-value">${gene.meanShap}</span>
      </div>
    </div>
    <div class="finding-bar-wrap">
      <div class="finding-bar-labels">
        <span>← Toward Alive</span>
        <span>Toward Dead →</span>
      </div>
    </div>
  `;

  const bar = createShapBar(gene.meanShap);
  card.querySelector(".finding-bar-wrap").appendChild(bar);
}

// Swaps in a text fallback if the real beeswarm image can't be found
// -- e.g. this dashboard hasn't been dropped into the actual project
// folder yet. The `error` event fires on the <img> itself when its
// src fails to load; nothing fires at all if it loads successfully.
function wireImageFallback() {
  const img = document.getElementById("shap-plot-image");
  const fallback = document.getElementById("shap-plot-fallback");

  img.addEventListener("error", () => {
    img.hidden = true;
    fallback.hidden = false;
  });
}

// One listener on the <ul>, not one per row -- event delegation, per
// your requirement #1. Also works correctly if the list is ever
// re-rendered later (e.g. by a sort/filter feature), since the
// listener lives on the parent, not the individual <li>s.
function wireGeneRowClicks() {
  const list = document.getElementById("gene-list");
  const card = document.getElementById("finding-card");
  const prefersReducedMotion = window.matchMedia(
    "(prefers-reduced-motion: reduce)"
  ).matches;

  function activateRow(row) {
    const gene = genePanel.find((g) => g.symbol === row.dataset.symbol);
    if (!gene) return;

    renderFindingCard(gene);
    // Gene list and finding card sit in different sections -- without
    // this, a click below the fold updates the card with no visible
    // feedback, and looks like nothing happened.
    card.scrollIntoView({
      behavior: prefersReducedMotion ? "auto" : "smooth",
      block: "center",
    });
  }

  list.addEventListener("click", (event) => {
    const row = event.target.closest(".gene-row");
    if (row) activateRow(row);
  });

  // Enter/Space activate a focused row, matching native button
  // behavior -- works because of tabindex="0" + role="button" added
  // to each .gene-row above.
  list.addEventListener("keydown", (event) => {
    if (event.key !== "Enter" && event.key !== " ") return;
    const row = event.target.closest(".gene-row");
    if (!row) return;
    event.preventDefault();
    activateRow(row);
  });
}
/* ==========================================================================
   INIT
   This <script> tag sits at the very end of <body>, so every element
   referenced above has already been parsed into the DOM by the time
   this file runs -- no DOMContentLoaded listener needed.
   ========================================================================== */
renderHeader();
renderOverviewStats();
renderFunnel();
renderMetrics();
renderGenePanel();
renderFindingCard();
wireImageFallback();
wireGeneRowClicks();