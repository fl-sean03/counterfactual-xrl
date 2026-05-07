/* Counterfactual XRL viewer — synced side-by-side PPO vs MCTS step-through. */

const ACTION_NAMES = { 0: "turn_left", 1: "turn_right", 2: "move_forward" };
const DIR_NAMES = { 0: "east (+x)", 1: "south (+y)", 2: "west (-x)", 3: "north (-y)" };
const AGENT_IDS = ["ppo_tuned", "mcts_baseline"];
const PREFIX = { ppo_tuned: "ppo", mcts_baseline: "mcts" };
const FIDELITY_TOL = 0.1;
const GOAL = [6, 6];

let DATA = null;
let state = {
  seed: null,
  step: 0,
  playing: false,
  playTimer: null,
};

/* ---- Bootstrap ---- */
async function main() {
  const res = await fetch("./data.json", { cache: "no-cache" });
  DATA = await res.json();

  populateMetricsSummary();
  populateTaskTags();
  populateSeedSelect();
  bindControls();

  setSeed(firstSeed());
  setStep(0);
}
window.addEventListener("DOMContentLoaded", main);

/* ---- Seeds + steps ---- */
function firstSeed() {
  // Use the union of seeds across both agents, sorted.
  const seeds = new Set();
  for (const aid of AGENT_IDS) {
    for (const s of Object.keys(DATA.agents[aid].trajectories)) seeds.add(s);
  }
  return Array.from(seeds).sort()[0];
}

function maxStepsForSeed(seed) {
  let n = 0;
  for (const aid of AGENT_IDS) {
    const traj = DATA.agents[aid].trajectories[seed] || [];
    if (traj.length > n) n = traj.length;
  }
  return n;
}

function trajectoryAt(agentId, seed, step) {
  const traj = DATA.agents[agentId].trajectories[seed] || [];
  if (step >= traj.length) return null;  // ended earlier
  return traj[step];
}

/* ---- Top bar ---- */
function populateMetricsSummary() {
  const root = document.getElementById("metrics-summary");
  root.innerHTML = "";
  const m = DATA.metrics_summary;
  if (!m) return;
  const order = [
    ["policy_rollout", "PPO rollout"],
    ["mcts_tree", "MCTS tree"],
  ];
  for (const [src, label] of order) {
    if (!m[src]) continue;
    const card = document.createElement("div");
    card.className = "metric-card";
    const fid = m[src].fidelity, snd = m[src].soundness, inf = m[src].inferability;
    card.innerHTML = `
      <span class="name">${label} (n=${m[src].n})</span>
      <div class="row"><span>fidelity</span><span class="val">${fid.mean.toFixed(3)}</span></div>
      <div class="row"><span>soundness</span><span class="val">${snd.mean.toFixed(3)}</span></div>
      <div class="row"><span>inferability</span><span class="val">${inf.mean.toFixed(3)}</span></div>
    `;
    root.appendChild(card);
  }
}

function populateTaskTags() {
  const ts = DATA.task_summary || {};
  setTaskTag("ppo-task-tag", ts.ppo_tuned);
  setTaskTag("mcts-task-tag", ts.mcts_baseline);
}
function setTaskTag(id, summary) {
  const el = document.getElementById(id);
  if (!summary) { el.textContent = ""; return; }
  el.textContent =
    `task-success ${summary.success.mean.toFixed(3)} · ` +
    `return ${signed(summary.return_.mean)} · ` +
    `${summary.steps.mean.toFixed(1)} steps`;
}
function signed(x) {
  if (Number.isNaN(x)) return "—";
  return (x >= 0 ? "+" : "") + x.toFixed(3);
}

/* ---- Trajectory selector + step controls ---- */
function populateSeedSelect() {
  const sel = document.getElementById("seed-select");
  sel.innerHTML = "";
  const seeds = new Set();
  for (const aid of AGENT_IDS) {
    for (const s of Object.keys(DATA.agents[aid].trajectories)) seeds.add(s);
  }
  for (const s of Array.from(seeds).sort()) {
    const o = document.createElement("option");
    o.value = s;
    o.textContent = s;
    sel.appendChild(o);
  }
}

function bindControls() {
  document.getElementById("seed-select").addEventListener("change", (e) => {
    setSeed(e.target.value);
    setStep(0);
  });
  document.getElementById("step-slider").addEventListener("input", (e) => {
    pause();
    setStep(parseInt(e.target.value, 10));
  });
  document.getElementById("prev-btn").addEventListener("click", () => {
    pause();
    setStep(state.step - 1);
  });
  document.getElementById("next-btn").addEventListener("click", () => {
    pause();
    setStep(state.step + 1);
  });
  document.getElementById("play-btn").addEventListener("click", togglePlay);
  document.addEventListener("keydown", (e) => {
    if (e.target.tagName === "INPUT" || e.target.tagName === "SELECT") return;
    if (e.key === "ArrowLeft") { pause(); setStep(state.step - 1); }
    else if (e.key === "ArrowRight") { pause(); setStep(state.step + 1); }
    else if (e.key === " ") { e.preventDefault(); togglePlay(); }
  });
}

function setSeed(seed) {
  state.seed = seed;
  document.getElementById("seed-select").value = seed;
  const max = Math.max(0, maxStepsForSeed(seed) - 1);
  const slider = document.getElementById("step-slider");
  slider.max = max;
  if (state.step > max) state.step = max;
}

function setStep(step) {
  const max = Math.max(0, maxStepsForSeed(state.seed) - 1);
  state.step = clamp(step, 0, max);
  document.getElementById("step-slider").value = state.step;
  document.getElementById("step-label").textContent =
    `step ${state.step} / ${max}`;
  for (const aid of AGENT_IDS) renderAgent(aid);
}

function togglePlay() {
  if (state.playing) pause();
  else play();
}
function play() {
  state.playing = true;
  document.getElementById("play-btn").textContent = "⏸";
  state.playTimer = setInterval(() => {
    const max = Math.max(0, maxStepsForSeed(state.seed) - 1);
    if (state.step >= max) { pause(); return; }
    setStep(state.step + 1);
  }, 900);
}
function pause() {
  state.playing = false;
  document.getElementById("play-btn").textContent = "▶";
  if (state.playTimer) { clearInterval(state.playTimer); state.playTimer = null; }
}
function clamp(v, lo, hi) { return Math.max(lo, Math.min(hi, v)); }

/* ---- Per-agent rendering ---- */
function renderAgent(agentId) {
  const obj = trajectoryAt(agentId, state.seed, state.step);
  const p = PREFIX[agentId];
  if (!obj) {
    renderEnded(agentId);
    return;
  }
  renderGrid(`${p}-grid`, obj.agent_pos, obj.agent_dir, obj.obstacle_positions, agentId);
  renderChosen(`${p}-chosen`, obj.chosen_action);
  renderExplanation(p, obj, agentId);
  renderEvidence(`${p}-evidence`, obj.per_action_stats, obj.chosen_action, agentId);
  renderMetadata(p, obj.agent_metadata, agentId);
}

function renderEnded(agentId) {
  const p = PREFIX[agentId];
  const grid = document.getElementById(`${p}-grid`);
  grid.innerHTML = `<g><rect x="0" y="0" width="800" height="800" fill="#161b22" />
    <text x="400" y="380" fill="#8b95a3" text-anchor="middle" font-size="36"
      font-family="SFMono-Regular, Menlo, Consolas, monospace">trajectory ended</text>
    <text x="400" y="430" fill="#8b95a3" text-anchor="middle" font-size="22"
      font-family="SFMono-Regular, Menlo, Consolas, monospace">no record at this step</text>
  </g>`;
  const chosen = document.getElementById(`${p}-chosen`);
  chosen.classList.add("terminated");
  chosen.innerHTML = `<span class="label">trajectory ended before this step</span>`;
  document.getElementById(`${p}-rationale`).textContent = "—";
  document.getElementById(`${p}-counterfactual`).textContent = "—";
  document.getElementById(`${p}-confidence`).textContent = "";
  document.getElementById(`${p}-cost`).textContent = "";
  document.getElementById(`${p}-evidence`).innerHTML = "";
  if (document.getElementById(`${p}-metadata-body`)) {
    document.getElementById(`${p}-metadata-body`).innerHTML = "";
  }
  if (p === "mcts" && document.getElementById("mcts-tree-body")) {
    document.getElementById("mcts-tree-body").innerHTML = "";
  }
}

function renderChosen(id, action) {
  const el = document.getElementById(id);
  el.classList.remove("terminated");
  el.innerHTML = `<span class="label">chosen action: </span>` +
    `<span class="name">${ACTION_NAMES[action]} (action ${action})</span>`;
}

function renderExplanation(prefix, obj, agentId) {
  const exp = obj.explanation;
  const ratEl = document.getElementById(`${prefix}-rationale`);
  const cfEl = document.getElementById(`${prefix}-counterfactual`);
  const confEl = document.getElementById(`${prefix}-confidence`);
  const costEl = document.getElementById(`${prefix}-cost`);
  if (!exp) {
    ratEl.textContent = "(no explanation cached)";
    cfEl.textContent = "";
    confEl.textContent = "";
    costEl.textContent = "";
    return;
  }
  const claimsResolved = (exp.claims || []).map((c) => ({
    ...c,
    _ok: claimMatches(c, obj),
  }));
  ratEl.innerHTML = highlightClaims(exp.rationale || "", claimsResolved);
  cfEl.innerHTML = highlightClaims(exp.counterfactual || "", claimsResolved);
  confEl.textContent = `confidence ${(exp.confidence ?? 0).toFixed(2)}`;
  if (typeof exp.cost_usd === "number" && exp.cost_usd > 0) {
    costEl.textContent = `gen-cost $${exp.cost_usd.toFixed(4)}`;
  } else {
    costEl.textContent = "";
  }
}

function claimMatches(claim, obj) {
  if (claim == null || claim.value == null || claim.action == null) return false;
  const val = Number(claim.value);
  if (!Number.isFinite(val)) return false;
  const stat = (obj.per_action_stats || []).find((s) => s.action === Number(claim.action));
  if (!stat) return false;
  const lookup = {
    discounted_return: stat.discounted_return,
    std_return: stat.std_return,
    mean_steps_to_end: stat.mean_steps_to_end,
    n_rollouts: stat.n_rollouts,
    visits: stat.n_rollouts,
    visit_count: stat.n_rollouts,
    root_child_visits: stat.n_rollouts,
    mean_value: stat.discounted_return,
    mean_return: stat.discounted_return,
    root_child_discounted_return: stat.discounted_return,
  };
  // MCTS-only tree fields
  const diag = (obj.agent_metadata || {}).tree_diagnostics || {};
  if (diag.value_variance_by_action && diag.value_variance_by_action[String(claim.action)] != null) {
    lookup.value_variance = diag.value_variance_by_action[String(claim.action)];
  }
  for (const pv of diag.principal_variations || []) {
    if (pv.first_action === Number(claim.action)) {
      if (pv.root_child_visits != null) lookup.root_child_visits = pv.root_child_visits;
      if (pv.root_child_discounted_return != null) {
        lookup.root_child_discounted_return = pv.root_child_discounted_return;
      }
    }
  }
  const ref = lookup[claim.metric];
  if (ref == null || !Number.isFinite(Number(ref))) return false;
  return Math.abs(Number(ref) - val) <= FIDELITY_TOL;
}

function highlightClaims(text, claims) {
  // Highlight every numeric token in ``text`` that matches a claim's value
  // (within fidelity tolerance). Mark green if the claim's value matches the
  // record, red otherwise. Numbers not associated with any claim are left
  // alone.
  if (!text || !claims.length) return escapeHTML(text);
  // Split text into tokens that are either numbers or non-numbers, then
  // wrap matching numbers.
  const tokenRe = /-?\d+(?:\.\d+)?/g;
  let out = "";
  let last = 0;
  let m;
  while ((m = tokenRe.exec(text)) !== null) {
    out += escapeHTML(text.slice(last, m.index));
    const numStr = m[0];
    const num = Number(numStr);
    let cls = null;
    for (const claim of claims) {
      const cv = Number(claim.value);
      if (Number.isFinite(cv) && Math.abs(cv - num) <= 1e-3) {
        cls = claim._ok ? "claim-ok" : "claim-bad";
        break;
      }
    }
    out += cls ? `<span class="${cls}" title="${claim_label(claims, num)}">${escapeHTML(numStr)}</span>`
               : escapeHTML(numStr);
    last = m.index + numStr.length;
  }
  out += escapeHTML(text.slice(last));
  return out;
}

function claim_label(claims, num) {
  for (const c of claims) {
    if (Math.abs(Number(c.value) - num) <= 1e-3) {
      const status = c._ok ? "matches evidence" : "does not match evidence";
      const meta = `metric=${c.metric ?? "?"} action=${c.action ?? "?"}`;
      return `${status} (${meta})`;
    }
  }
  return "";
}

function escapeHTML(s) {
  return String(s ?? "")
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;")
    .replace(/'/g, "&#39;");
}

/* ---- Per-action evidence table ---- */
function renderEvidence(id, stats, chosen, agentId) {
  const el = document.getElementById(id);
  if (!stats || !stats.length) { el.innerHTML = ""; return; }
  const rows = stats.map((s) => {
    const ci = (s.return_ci || []).map((v) => v.toFixed(3)).join(", ");
    const steps = s.mean_steps_to_end == null ? "—" : Number(s.mean_steps_to_end).toFixed(1);
    const cls = s.action === chosen ? "chosen-row" : "";
    return `<tr class="${cls}">
      <td>${s.action}&nbsp;<span style="color:var(--text-dim)">(${ACTION_NAMES[s.action]})</span></td>
      <td>${num(s.discounted_return, 3)}</td>
      <td>[${ci}]</td>
      <td>${num(s.std_return, 3)}</td>
      <td>${steps}</td>
      <td>${s.n_rollouts}</td>
    </tr>`;
  }).join("");
  el.innerHTML = `<table>
    <thead>
      <tr>
        <th>action</th>
        <th>discounted return</th>
        <th>95% CI</th>
        <th>std</th>
        <th>steps</th>
        <th>n</th>
      </tr>
    </thead>
    <tbody>${rows}</tbody>
  </table>`;
}
function num(x, k) { return x == null || !Number.isFinite(Number(x)) ? "—" : Number(x).toFixed(k); }

/* ---- Agent metadata + tree diagnostics ---- */
function renderMetadata(prefix, meta, agentId) {
  const body = document.getElementById(`${prefix}-metadata-body`);
  if (!body) return;
  if (!meta) { body.innerHTML = ""; return; }
  // For MCTS, separately render tree_diagnostics in its own section.
  let copy = meta;
  if (agentId === "mcts_baseline") {
    copy = { ...meta };
    delete copy.tree_diagnostics;
    renderTreeDiagnostics(meta.tree_diagnostics);
  }
  body.innerHTML = "<pre>" + escapeHTML(JSON.stringify(copy, null, 2)) + "</pre>";
}

function renderTreeDiagnostics(diag) {
  const root = document.getElementById("mcts-tree-body");
  if (!root) return;
  if (!diag) { root.innerHTML = "<p style=\"color:var(--text-dim)\">no tree diagnostics</p>"; return; }
  let html = "";

  if (diag.principal_variations && diag.principal_variations.length) {
    html += "<h4 style=\"margin:8px 0 4px;font-size:12px;color:var(--text-dim)\">principal variations</h4>";
    for (const pv of diag.principal_variations) {
      const actions = (pv.principal_variation || []).join(" → ");
      html += `<div class="pv-line">
        <span class="pv-actions">first=${pv.first_action} (${ACTION_NAMES[pv.first_action]}); next: ${actions || "—"}</span>
        <span class="pv-meta">visits=${pv.root_child_visits}, dr=${num(pv.root_child_discounted_return, 4)}</span>
      </div>`;
    }
  }
  if (diag.depth2_visit_distribution) {
    html += "<h4 style=\"margin:12px 0 4px;font-size:12px;color:var(--text-dim)\">depth-2 visit distribution</h4>";
    html += "<pre>" + escapeHTML(JSON.stringify(diag.depth2_visit_distribution, null, 2)) + "</pre>";
  }
  if (diag.value_variance_by_action) {
    html += "<h4 style=\"margin:12px 0 4px;font-size:12px;color:var(--text-dim)\">per-child variance</h4>";
    html += "<pre>" + escapeHTML(JSON.stringify(diag.value_variance_by_action, null, 2)) + "</pre>";
  }
  root.innerHTML = html;
}

/* ---- 8x8 grid renderer ---- */
function renderGrid(svgId, agent_pos, agent_dir, obstacle_positions, agentId) {
  const svg = document.getElementById(svgId);
  // Width/height = 800 (matches viewBox); 8 cells with 1-cell-wide outer wall
  // matches the MiniGrid convention: wall on cells 0 and 7, playable area 1..6.
  const N = 8;
  const cell = 800 / N;
  const lines = [];
  for (let i = 0; i <= N; i++) {
    lines.push(`<line x1="0" y1="${i * cell}" x2="800" y2="${i * cell}" stroke="var(--grid-line)" stroke-width="1" />`);
    lines.push(`<line x1="${i * cell}" y1="0" x2="${i * cell}" y2="800" stroke="var(--grid-line)" stroke-width="1" />`);
  }
  // Cells
  let cells = "";
  for (let y = 0; y < N; y++) {
    for (let x = 0; x < N; x++) {
      const fill = ((x + y) % 2 === 0) ? "#1a2028" : "#20272f";
      cells += `<rect x="${x * cell}" y="${y * cell}" width="${cell}" height="${cell}" fill="${fill}" />`;
    }
  }
  // Walls (cells 0 and N-1 on every edge)
  let walls = "";
  for (let i = 0; i < N; i++) {
    walls += `<rect x="0" y="${i * cell}" width="${cell}" height="${cell}" fill="#0b0e12" />`;
    walls += `<rect x="${(N - 1) * cell}" y="${i * cell}" width="${cell}" height="${cell}" fill="#0b0e12" />`;
    walls += `<rect x="${i * cell}" y="0" width="${cell}" height="${cell}" fill="#0b0e12" />`;
    walls += `<rect x="${i * cell}" y="${(N - 1) * cell}" width="${cell}" height="${cell}" fill="#0b0e12" />`;
  }
  // Goal
  const [gx, gy] = GOAL;
  const goalCell = `<rect x="${gx * cell + 6}" y="${gy * cell + 6}" width="${cell - 12}" height="${cell - 12}"
    fill="var(--goal)" rx="6" ry="6" />`;
  // Obstacles (red filled circles)
  const obstacles = (obstacle_positions || []).map(([ox, oy]) => {
    const cx = ox * cell + cell / 2;
    const cy = oy * cell + cell / 2;
    return `<circle cx="${cx}" cy="${cy}" r="${cell * 0.32}" fill="var(--obstacle)" stroke="#1a1313" stroke-width="2" />`;
  }).join("");
  // Agent (triangle pointing in agent_dir)
  const agentColor = (agentId === "mcts_baseline") ? "var(--accent-2)" : "var(--accent)";
  const ax = agent_pos[0] * cell + cell / 2;
  const ay = agent_pos[1] * cell + cell / 2;
  const r = cell * 0.36;
  const trianglePoints = trianglePointsFor(ax, ay, r, agent_dir);
  const agent = `<polygon points="${trianglePoints}" fill="${agentColor}" stroke="#0b1018" stroke-width="2" />`;

  svg.innerHTML = cells + walls + goalCell + lines.join("") + obstacles + agent;
}

function trianglePointsFor(cx, cy, r, dir) {
  // dir 0=east, 1=south, 2=west, 3=north
  const angles = { 0: 0, 1: Math.PI / 2, 2: Math.PI, 3: -Math.PI / 2 };
  const theta = angles[dir] ?? 0;
  // Tip at cx + r*cos, cy + r*sin; base 60° behind on each side
  const tip = [cx + r * Math.cos(theta), cy + r * Math.sin(theta)];
  const left = [cx + r * Math.cos(theta + 2.5), cy + r * Math.sin(theta + 2.5)];
  const right = [cx + r * Math.cos(theta - 2.5), cy + r * Math.sin(theta - 2.5)];
  return [tip, left, right].map((p) => `${p[0].toFixed(1)},${p[1].toFixed(1)}`).join(" ");
}
