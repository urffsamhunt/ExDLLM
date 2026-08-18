/* ────────────────────────────────────────────────────────────────────────
   DLLM Visualizer — app.js
   Handles: API calls · step rendering · token chips · timeline heatmap ·
            tag distribution chart · playback · token detail panel
──────────────────────────────────────────────────────────────────────── */

// ── State ─────────────────────────────────────────────────────────────────
let trajectory = [];
let totalSteps = 0;
let currentStep = 0;
let playTimer = null;
let chartInstance = null;
let selectedTokenPos = null;

// Tag → CSS class / colour
const TAG_COLOR = {
  KEEP:       '#3ecf8e',
  REPLACE:    '#f59e42',
  DELETE:     '#f24e4e',
  INSERT:     '#a78bfa',
  EXPAND:     '#38bdf8',
};
const TAG_ORDER = ['KEEP', 'REPLACE', 'DELETE', 'INSERT', 'EXPAND'];

// Position type → chip class
function chipClass(tok) {
  const t = tok.type;
  if (t === 'bos' || t === 'eos') return 'chip-structural';
  if (t === 'prompt_tok')         return 'chip-prompt';
  if (t === 'prompt_imask')       return 'chip-structural';
  if (t === 'response_imask')     return 'chip-structural';
  // Response tokens: colour by predicted tag
  return `chip-${tok.tag}`;
}

// ── Status Check ──────────────────────────────────────────────────────────
async function checkStatus() {
  try {
    const r = await fetch('/api/status');
    const d = await r.json();
    const dot  = document.getElementById('statusDot');
    const text = document.getElementById('statusText');
    if (d.loaded) {
      dot.className  = 'status-dot ok';
      text.textContent = `${d.config} · ${d.device}`;
    } else {
      dot.className  = 'status-dot err';
      text.textContent = 'Model not loaded';
    }
  } catch {
    document.getElementById('statusDot').className  = 'status-dot err';
    document.getElementById('statusText').textContent = 'Server offline';
  }
}

// ── Slider sync ───────────────────────────────────────────────────────────
function bindSlider(id, valId, suffix = '') {
  const el = document.getElementById(id);
  const vl = document.getElementById(valId);
  if (!el || !vl) return;
  el.addEventListener('input', () => { vl.textContent = el.value + suffix; });
}
bindSlider('temperature', 'tempVal');
bindSlider('topP',        'topPVal');
bindSlider('topK',        'topKVal');

// ── Generate ──────────────────────────────────────────────────────────────
async function runGenerate() {
  const prompt = document.getElementById('promptInput').value.trim();
  if (!prompt) return;

  const targetLen = document.getElementById('targetLen').value.trim();
  const body = {
    prompt,
    max_iterations: +document.getElementById('maxIter').value,
    target_length:  targetLen ? +targetLen : null,
    temperature:    +document.getElementById('temperature').value,
    top_k:          +document.getElementById('topK').value,
    top_p:          +document.getElementById('topP').value,
  };

  stopPlay();
  setLoading(true);

  try {
    const r = await fetch('/api/generate', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
    });
    const data = await r.json();
    if (data.error) { alert('Error: ' + data.error); return; }

    trajectory  = data.trajectory;
    totalSteps  = trajectory.length;
    currentStep = 0;

    // Populate result
    document.getElementById('finalResponse').textContent = data.response_only || '(empty)';
    document.getElementById('finalFull').textContent     = data.full_clean    || '(empty)';

    // Show panels
    show('navCard'); show('resultCard'); show('timelineCard'); show('chartCard');
    hide('tokenDetail');

    // Init navigator
    const slider = document.getElementById('stepSlider');
    slider.max   = totalSteps - 1;
    slider.value = 0;
    document.getElementById('totalStepNum').textContent = totalSteps - 1;

    // Build timeline and chart (once per result)
    buildTimeline();
    buildChart();

    // Render first step
    gotoStep(0);
  } catch (e) {
    alert('Request failed: ' + e.message);
  } finally {
    setLoading(false);
  }
}

// ── Step Navigation ───────────────────────────────────────────────────────
function gotoStep(n) {
  n = Math.max(0, Math.min(totalSteps - 1, +n));
  currentStep = n;
  document.getElementById('currentStepNum').textContent = n;
  document.getElementById('stepSlider').value = n;
  renderStep(trajectory[n]);
  highlightTimelineColumn(n);
}
function nextStep() { gotoStep(currentStep + 1); }
function prevStep() { gotoStep(currentStep - 1); }

// ── Playback ──────────────────────────────────────────────────────────────
function togglePlay() {
  if (playTimer) { stopPlay(); return; }
  document.getElementById('playBtn').classList.add('playing');
  document.getElementById('playIcon').textContent = '⏸ Pause';
  const speed = +document.getElementById('playSpeed').value;
  playTimer = setInterval(() => {
    if (currentStep >= totalSteps - 1) { stopPlay(); return; }
    nextStep();
  }, speed);
}
function stopPlay() {
  clearInterval(playTimer); playTimer = null;
  const btn = document.getElementById('playBtn');
  if (btn) {
    btn.classList.remove('playing');
    document.getElementById('playIcon').textContent = '▶ Play';
  }
}

// ── Render a single step ──────────────────────────────────────────────────
function renderStep(step) {
  // Badge
  const badge = document.getElementById('stepBadge');
  badge.textContent = step.step === 0 ? 'Initial Canvas'
    : `Step ${step.step}${step.converged ? ' ✓ Converged' : ''}`;
  if (step.converged) badge.style.color = 'var(--keep)';
  else badge.style.color = '';

  // Canvas chips
  const body = document.getElementById('canvasBody');
  body.innerHTML = '';

  step.tokens.forEach((tok, i) => {
    const chip = document.createElement('div');
    chip.className = `token-chip ${chipClass(tok)}`;
    chip.dataset.pos = i;
    chip.style.animationDelay = `${i * 8}ms`;

    // Confidence ring colour
    const confColor = confToColor(tok.confidence);
    chip.innerHTML = `
      <div class="chip-text">${escHtml(tok.token)}</div>
      <div class="chip-tag">${tok.tag}</div>
      <div class="chip-conf" style="border-color:${confColor};color:${confColor}">
        ${Math.round(tok.confidence * 100)}
      </div>`;

    chip.addEventListener('click', () => showTokenDetail(tok, i));
    body.appendChild(chip);
  });

  // Tag bar
  renderTagBar(step.tag_counts, step.tokens.length);

  // Highlight active step in chart
  if (chartInstance && step.step > 0) {
    chartInstance.data.datasets.forEach(ds => {
      ds.borderWidth = Array.from({ length: totalSteps }, (_, k) =>
        k === step.step - 1 ? 3 : 1
      );
    });
    chartInstance.update('none');
  }

  // Re-highlight selected token if still visible
  if (selectedTokenPos !== null && selectedTokenPos < step.tokens.length) {
    const chips = body.querySelectorAll('.token-chip');
    chips[selectedTokenPos]?.classList.add('active');
    showTokenDetail(step.tokens[selectedTokenPos], selectedTokenPos);
  } else {
    hide('tokenDetail');
  }
}

// ── Tag Bar ───────────────────────────────────────────────────────────────
function renderTagBar(counts, total) {
  const wrap = document.getElementById('tagBar');
  wrap.innerHTML = '';
  TAG_ORDER.forEach(tag => {
    const n = counts[tag] || 0;
    const pct = total > 0 ? (n / total * 100).toFixed(1) : 0;
    wrap.innerHTML += `
      <div class="tag-bar-row">
        <span class="tag-bar-label" style="color:${TAG_COLOR[tag]}">${tag}</span>
        <div class="tag-bar-track">
          <div class="tag-bar-fill" style="width:${pct}%;background:${TAG_COLOR[tag]}"></div>
        </div>
        <span class="tag-bar-count">${n}</span>
      </div>`;
  });
}

// ── Token Detail Panel ────────────────────────────────────────────────────
function showTokenDetail(tok, posIdx) {
  selectedTokenPos = posIdx;
  show('tokenDetail');

  // Mark active chip
  document.querySelectorAll('.token-chip').forEach(c => c.classList.remove('active'));
  const chips = document.getElementById('canvasBody').querySelectorAll('.token-chip');
  chips[posIdx]?.classList.add('active');

  const grid = document.getElementById('detailGrid');
  grid.innerHTML = `
    <div class="detail-item"><div class="detail-key">Token</div><div class="detail-val">${escHtml(tok.token)}</div></div>
    <div class="detail-item"><div class="detail-key">ID</div><div class="detail-val">${tok.token_id}</div></div>
    <div class="detail-item"><div class="detail-key">Type</div><div class="detail-val">${tok.type}</div></div>
    <div class="detail-item"><div class="detail-key">Tag</div><div class="detail-val" style="color:${TAG_COLOR[tok.tag]||'inherit'}">${tok.tag}</div></div>
  `;

  const bars = document.getElementById('probBars');
  bars.innerHTML = '';
  TAG_ORDER.forEach(tag => {
    const p = tok.probs[tag] ?? 0;
    const pct = (p * 100).toFixed(1);
    bars.innerHTML += `
      <div class="prob-row">
        <span class="prob-name" style="color:${TAG_COLOR[tag]}">${tag}</span>
        <div class="prob-track">
          <div class="prob-fill" style="width:${pct}%;background:${TAG_COLOR[tag]}"></div>
        </div>
        <span class="prob-pct">${pct}%</span>
      </div>`;
  });
}

// ── Timeline Heatmap ──────────────────────────────────────────────────────
function buildTimeline() {
  const wrap = document.getElementById('timelineWrap');
  wrap.innerHTML = '';
  if (trajectory.length < 2) return;

  // Find max number of tokens across all steps
  const maxLen = Math.max(...trajectory.map(s => s.tokens.length));
  const steps  = trajectory.length;

  const grid = document.createElement('div');
  grid.className = 'timeline-grid';
  grid.style.gridTemplateColumns = `repeat(${steps}, 14px)`;
  grid.style.gridTemplateRows    = `repeat(${maxLen}, 14px)`;

  for (let row = 0; row < maxLen; row++) {
    for (let col = 0; col < steps; col++) {
      const cell = document.createElement('div');
      cell.className = 'timeline-cell';
      const step = trajectory[col];
      if (row < step.tokens.length) {
        const tok = step.tokens[row];
        cell.style.background = tagToTimelineColor(tok);
        cell.title = `Step ${col} · pos ${row} · ${tok.token} [${tok.tag} ${(tok.confidence*100).toFixed(0)}%]`;
        const c = col, r = row;
        cell.addEventListener('click', () => { gotoStep(c); showTokenDetail(trajectory[c].tokens[r], r); });
      } else {
        cell.style.background = 'transparent';
      }
      grid.appendChild(cell);
    }
  }
  wrap.appendChild(grid);
}

function highlightTimelineColumn(stepIdx) {
  const grid = document.querySelector('.timeline-grid');
  if (!grid) return;
  const cells = grid.querySelectorAll('.timeline-cell');
  const maxLen = Math.max(...trajectory.map(s => s.tokens.length));
  const steps  = trajectory.length;
  cells.forEach((cell, i) => {
    const col = i % steps;
    cell.style.outline = col === stepIdx ? '1.5px solid #fff8' : 'none';
    cell.style.zIndex  = col === stepIdx ? '5' : '1';
  });
}

function tagToTimelineColor(tok) {
  const alpha = 0.3 + tok.confidence * 0.7;
  const c = TAG_COLOR[tok.tag] || '#636b82';
  // Parse hex and add alpha
  const r = parseInt(c.slice(1,3),16), g = parseInt(c.slice(3,5),16), b = parseInt(c.slice(5,7),16);
  return `rgba(${r},${g},${b},${alpha.toFixed(2)})`;
}

// ── Tag Distribution Chart ────────────────────────────────────────────────
function buildChart() {
  const canvas = document.getElementById('tagChart');
  const ctx = canvas.getContext('2d');

  if (chartInstance) { chartInstance.destroy(); chartInstance = null; }

  // Data: one dataset per tag, one point per non-zero step
  const labels = trajectory.slice(1).map(s => `S${s.step}`);
  const datasets = TAG_ORDER.map(tag => ({
    label: tag,
    data: trajectory.slice(1).map(s => s.tag_counts[tag] || 0),
    borderColor: TAG_COLOR[tag],
    backgroundColor: TAG_COLOR[tag] + '22',
    borderWidth: 1.5,
    tension: 0.4,
    fill: false,
    pointRadius: 3,
    pointHoverRadius: 5,
  }));

  chartInstance = new Chart(ctx, {
    type: 'line',
    data: { labels, datasets },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      resizeDelay: 0,
      animation: { duration: 200 },
      interaction: { mode: 'index', intersect: false },
      plugins: {
        legend: {
          labels: {
            color: '#9ba3b8', font: { family: 'Inter', size: 11 },
            boxWidth: 12, padding: 12,
          },
        },
        tooltip: {
          backgroundColor: '#1a1e28',
          borderColor: '#2a2f3e',
          borderWidth: 1,
          titleColor: '#e8eaf0',
          bodyColor: '#9ba3b8',
        },
      },
      scales: {
        x: {
          ticks: { color: '#636b82', font: { size: 10 } },
          grid:  { color: '#1a1e28' },
        },
        y: {
          ticks: { color: '#636b82', font: { size: 10 } },
          grid:  { color: '#1a1e28' },
          beginAtZero: true,
        },
      },
    },
  });
}

// ── Utilities ─────────────────────────────────────────────────────────────
function confToColor(c) {
  // green at high confidence → orange → red at low
  if (c > 0.75) return '#3ecf8e';
  if (c > 0.5)  return '#f59e42';
  return '#f24e4e';
}

function escHtml(s) {
  return String(s)
    .replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;')
    .replace(/"/g,'&quot;');
}

function show(id) { const el = document.getElementById(id); if(el) el.style.display = ''; }
function hide(id) { const el = document.getElementById(id); if(el) el.style.display = 'none'; }

function setLoading(on) {
  document.getElementById('spinner').style.display = on ? '' : 'none';
  document.getElementById('generateBtn').disabled  = on;
}

// ── Load Chart.js dynamically ─────────────────────────────────────────────
(function loadChartJS() {
  const s = document.createElement('script');
  s.src = 'https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js';
  s.onload = () => console.log('Chart.js loaded');
  document.head.appendChild(s);
})();

// ── Init ──────────────────────────────────────────────────────────────────
checkStatus();
setInterval(checkStatus, 30000);
