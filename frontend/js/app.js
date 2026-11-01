/* Sepsis Watch UI: hash-routed single page app that talks to the FastAPI backend.
   Clinical Command Center edition: light/dark theme, SVG threat gauge, spline telemetry, XAI meters.
   API routes, payload keys and state handling are unchanged. */
(() => {
  const $ = (s, r = document) => r.querySelector(s);
  const esc = Charts.esc;
  const view = $("#view");
  const COLORS = ["#6366F1", "#22D3EE", "#F59E0B", "#A78BFA", "#34D399", "#FB7185"];
  const DASH = [null, "6 3", "2 3", "8 3 2 3", "10 4", "3 3"];
  const pct = (v, d = 1) => (v * 100).toFixed(d) + "%";
  const num = (v, d = 3) => Number(v).toFixed(d);
  const risk = v => (v < 0.01 ? (v * 100).toFixed(2) : (v * 100).toFixed(1)) + "%";
  const api = async (url, opts) => {
    const r = await fetch(url, opts);
    if (!r.ok) { let d = r.statusText; try { d = (await r.json()).detail || d; } catch (e) {} throw new Error(typeof d === "string" ? d : JSON.stringify(d)); }
    return r.json();
  };

  const S = { model: null, schema: null, samples: null, rows: [], age: 65, firstHour: 1, result: null, labels: null, sel: null, ref: null };
  const PAGES = {
    monitor: ["Patient monitor", "Enter or load a patient's hourly measurements. The model scores every hour and shows which fuzzy rules caused the risk."],
    model: ["How the model works", "Genetic Algorithm → ANFIS fuzzy rules → Particle Swarm Optimization, all trained on PhysioNet ICU data."],
    performance: ["Performance", "Results on patients the model never saw, at the real (rare) sepsis rate."],
    about: ["About & safety", "What this project is, what it is not, and how to reproduce it."],
  };

  /* ---------------- THEME (light / dark, persisted in localStorage) ---------------- */
  const THEME_KEY = "sepsis-theme";
  const getTheme = () => (document.documentElement.getAttribute("data-theme") === "light" ? "light" : "dark");
  function applyTheme(t) {
    document.documentElement.setAttribute("data-theme", t);
    const b = $("#themeToggle");
    if (b) {
      b.setAttribute("aria-checked", t === "dark" ? "true" : "false");
      b.setAttribute("aria-label", t === "dark" ? "Switch to light theme" : "Switch to dark theme");
    }
  }
  function initTheme() {
    let saved = null;
    try { saved = localStorage.getItem(THEME_KEY); } catch (e) {}
    applyTheme(saved === "dark" || saved === "light" ? saved : getTheme());
    const b = $("#themeToggle");
    if (!b) return;
    b.addEventListener("click", () => {
      const next = getTheme() === "dark" ? "light" : "dark";
      const root = document.documentElement;
      root.classList.add("theme-anim");
      applyTheme(next);
      try { localStorage.setItem(THEME_KEY, next); } catch (e) {}
      setTimeout(() => root.classList.remove("theme-anim"), 450);
    });
  }

  /* ---------------- shell ---------------- */
  async function boot() {
    initTheme();
    window.addEventListener("hashchange", route);
    $("#menuBtn").addEventListener("click", () => $("#sidebar").classList.toggle("open"));
    try {
      const hl = await api("/api/health");
      if (hl.status === "no_model") { S.noModel = hl.detail; $("#modelChip").textContent = "No model loaded"; route(); return; }
      [S.model, S.schema, S.samples] = await Promise.all([api("/api/model"), api("/api/schema"), api("/api/samples")]);
    } catch (e) {
      view.innerHTML = `<div class="err">Could not reach the backend: ${esc(e.message)}</div>`; return;
    }
    $("#modelChip").innerHTML = (S.model.is_demo ? "Synthetic TEST model" : "Trained model") + `<br><span class="tiny" style="color:inherit;opacity:.8">${S.model.selected_features.length} features · ${S.model.rules.length} fuzzy rules</span>`;
    $("#demoBanner").hidden = !S.model.is_demo;
    route();
  }

  function route() {
    const page = (location.hash.replace("#", "") || "monitor");
    const p = PAGES[page] ? page : "monitor";
    document.querySelectorAll("#nav a").forEach(a => a.classList.toggle("active", a.dataset.page === p));
    $("#pageTitle").textContent = PAGES[p][0]; $("#pageSub").textContent = PAGES[p][1];
    $("#sidebar").classList.remove("open"); window.scrollTo(0, 0);
    if (S.noModel && p !== "about") return pageNoModel();
    ({ monitor: pageMonitor, model: pageModel, performance: pagePerformance, about: pageAbout })[p]();
  }

  function pageNoModel() {
    view.innerHTML = `<div class="card"><h2>No trained model found</h2>
      <p style="margin-top:8px">For safety this app never shows results from a stand-in model. It needs the model trained in the project notebook.</p>
      <ol class="list">
        <li>Open the Colab notebook where the model was trained (same session, all cells already run).</li>
        <li>Add a new last cell, paste the contents of <span class="mono">scripts/colab_export_model_cell.py</span> and run it. It downloads <span class="mono">model.json</span>.</li>
        <li>Put the file at <span class="mono">artifacts/model.json</span> in this project and restart the app.</li>
      </ol>
      <p class="note">Details: <span class="mono">docs/EXPORT_FROM_COLAB.md</span>. Alternatively run <span class="mono">make train</span> to train on PhysioNet data locally.</p>
      <div class="err">${esc(S.noModel)}</div></div>`;
  }

  /* ---------------- Monitor ---------------- */
  function pageMonitor() {
    const used = new Set(S.model.required_inputs);
    view.innerHTML = `
    <div class="card">
      <div class="card-head"><div><h2>ICU vital registration matrix</h2><p>Hourly records in time order. Leave a cell empty if it was not measured. Columns marked <span style="color:var(--accent)">●</span> are used by the loaded model.</p></div></div>
      <div class="toolbar">
        <label class="field">Example patient<select id="sampleSel">${S.samples.map(s => `<option value="${s.id}">${esc(s.title)}</option>`).join("")}</select></label>
        <button class="btn" id="loadSample">Load example</button>
        <label class="field">Age (years)<input id="age" type="number" min="0" max="120" value="${S.age}" style="width:90px"></label>
        <label class="field">First ICU hour<input id="firstHour" type="number" min="0" value="${S.firstHour}" style="width:100px"></label>
        <span class="sp"></span>
        <button class="btn" id="upload">Upload .psv file</button><input id="file" type="file" accept=".psv,.txt,.csv" hidden>
        <button class="btn" id="addRow">+ Add hour</button><button class="btn" id="delRow">Remove last</button><button class="btn" id="clear">Clear</button>
        <button class="btn btn-primary" id="assess">Assess risk</button>
      </div>
      <p class="note" id="sampleNote" style="margin:10px 0 0"></p>
      <div id="msg"></div>
      <div class="scroll" style="margin-top:12px;max-height:340px"><table class="data-table matrix" id="tbl"></table></div>
    </div>
    <div id="result"></div>`;
    const sel = $("#sampleSel");
    const note = () => { const s = S.samples.find(x => x.id === sel.value); $("#sampleNote").textContent = s ? `Synthetic illustration: ${s.description}` : ""; };
    sel.addEventListener("change", note); note();
    $("#loadSample").onclick = () => loadSample(sel.value);
    $("#age").oninput = e => S.age = e.target.value === "" ? null : +e.target.value;
    $("#firstHour").oninput = e => S.firstHour = e.target.value === "" ? 1 : +e.target.value;
    $("#addRow").onclick = () => { S.rows.push({}); renderTable(used); };
    $("#delRow").onclick = () => { S.rows.pop(); renderTable(used); };
    $("#clear").onclick = () => { S.rows = [{}]; S.result = null; S.labels = null; renderTable(used); renderResult(); $("#msg").innerHTML = ""; };
    $("#assess").onclick = () => assess();
    $("#upload").onclick = () => $("#file").click();
    $("#file").onchange = e => uploadPsv(e.target.files[0], used);
    if (!S.rows.length) { loadSample("deteriorating", true); } else { renderTable(used); renderResult(); }
  }

  function renderTable(used) {
    const f = S.schema;
    $("#tbl").innerHTML = `<thead><tr><th class="num">Hour</th>${f.map(c => `<th class="num ${used.has(c.name) ? "used" : ""}" title="${esc(c.label)}">${c.name}<small>${esc(c.unit)}</small></th>`).join("")}</tr></thead>
      <tbody>${S.rows.map((r, i) => `<tr><td class="num"><b>${i + 1}</b></td>${f.map(c => `<td class="num"><input class="cell ${r[c.name] == null ? "na" : ""}" type="number" step="any" data-i="${i}" data-k="${c.name}" value="${r[c.name] ?? ""}" placeholder="–"></td>`).join("")}</tr>`).join("")}</tbody>`;
    $("#tbl").querySelectorAll(".cell").forEach(inp => inp.addEventListener("input", e => {
      const v = e.target.value; S.rows[+e.target.dataset.i][e.target.dataset.k] = v === "" ? null : Number(v);
      e.target.classList.toggle("na", v === "");
    }));
  }

  async function loadSample(id, quiet) {
    try {
      const s = await api("/api/samples/" + id);
      S.rows = s.hours; S.age = s.age; S.labels = s.labels; $("#age").value = S.age;
      renderTable(new Set(S.model.required_inputs)); await assess(quiet);
    } catch (e) { showMsg(e.message); }
  }

  function showMsg(t) { const m = $("#msg"); if (m) m.innerHTML = t ? `<div class="err">${esc(t)}</div>` : ""; }

  async function assess() {
    showMsg("");
    if (!S.rows.length) return showMsg("Add at least one hourly record.");
    const btn = $("#assess"); btn.disabled = true; btn.textContent = "Scoring…";
    try {
      S.result = await api("/api/predict", { method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ hours: S.rows, age: S.age, first_icu_hour: S.firstHour }) });
      S.sel = S.result.hours.length - 1; renderResult();
    } catch (e) { showMsg(e.message); } finally { btn.disabled = false; btn.textContent = "Assess risk"; }
  }

  async function uploadPsv(file, used) {
    if (!file) return; showMsg("");
    const fd = new FormData(); fd.append("file", file);
    try {
      const out = await api("/api/predict/psv", { method: "POST", body: fd });
      S.rows = out.hours_raw.map(h => { const r = {}; S.schema.forEach(c => r[c.name] = h[c.name] ?? null); return r; });
      if (out.age != null) { S.age = out.age; $("#age").value = S.age; }
      S.labels = out.true_labels; S.result = out; S.sel = out.hours.length - 1;
      renderTable(used); renderResult();
    } catch (e) { showMsg(e.message); }
    $("#file").value = "";
  }

  function renderResult() {
    const box = $("#result"); if (!box) return;
    const R = S.result;
    if (!R) { box.innerHTML = `<div class="card empty">Load an example or enter values, then press <b>Assess risk</b>.</div>`; return; }
    const sm = R.summary, last = R.hours[R.hours.length - 1], thr = R.threshold, alert = sm.last_status === "alert";
    box.innerHTML = `
    <div class="hero">
      <div class="card gauge-card">
        <div class="card-head"><div><h2>Sepsis threat gauge</h2><p id="gaugeSub">Click a point or row to move the dial.</p></div></div>
        <div id="gauge"></div>
        <div class="gauge-legend"><span><i style="background:var(--ok)"></i>Below threshold</span><span><i style="background:var(--danger)"></i>Alert zone</span><span><i class="t"></i>Threshold ${risk(thr)}</span></div>
      </div>
      <div class="kpis">
        <div class="tile ${alert ? "alert" : "ok"}"><div class="k">Current risk (hour ${last.hour})</div><div class="v">${risk(last.risk)}</div><div class="n"><span class="chip ${alert ? "chip-alert" : "chip-ok"}">${alert ? "ALERT" : "Normal"}</span> threshold ${risk(thr)}</div></div>
        <div class="tile"><div class="k">Peak risk</div><div class="v">${risk(sm.max_risk)}</div><div class="n">over ${sm.n_hours} hours</div></div>
        <div class="tile"><div class="k">Hours in alert</div><div class="v">${sm.hours_in_alert}<span style="font-size:1rem;color:var(--muted)"> / ${sm.n_hours}</span></div><div class="n">first alert: ${sm.first_alert_hour ? "hour " + sm.first_alert_hour : "none"}</div></div>
        <div class="tile"><div class="k">Sustained alert (≥${R.sustained_hours} h in a row)</div><div class="v">${sm.first_sustained_alert_hour ? "hour " + sm.first_sustained_alert_hour : "none"}</div><div class="n">extension, not part of the evaluated model</div></div>
      </div>
    </div>
    <div class="split">
      <div class="card"><div class="card-head"><div><h2>Medical telemetry · risk waveform</h2><p>Click a node to see why the model reacted at that hour.</p></div></div><div class="chart" id="riskChart"></div></div>
      <div class="card" id="whyCard"></div>
    </div>
    <div class="card"><div class="card-head"><div><h2>Hour by hour</h2><p>Click a row to inspect it.</p></div></div>
      <div class="scroll" style="max-height:320px"><table id="hrTbl" class="matrix"><thead><tr><th>Hour</th><th class="num">ICU hour</th><th class="num">Risk</th><th>Status</th><th>Main rule</th>${S.model.selected_features.map(f => `<th class="num" title="${esc(f.label)}">${esc(f.name)}</th>`).join("")}</tr></thead><tbody>
      ${R.hours.map((h, i) => `<tr data-i="${i}" class="${i === S.sel ? "hl" : ""}" style="cursor:pointer"><td>${h.hour}</td><td class="num">${h.iculos}</td><td class="num risk-cell">${risk(h.risk)}</td><td><span class="chip ${h.alert ? "chip-alert" : "chip-ok"}">${h.alert ? "Alert" : "Normal"}</span>${h.sustained_alert ? ' <span class="chip chip-warn">sustained</span>' : ""}</td><td>Rule ${h.top_rule}</td>${S.model.selected_features.map(f => `<td class="num">${h.inputs[f.name]}</td>`).join("")}</tr>`).join("")}</tbody></table></div></div>`;
    drawGauge(); drawRisk(); drawWhy();
    $("#hrTbl tbody").querySelectorAll("tr").forEach(tr => tr.addEventListener("click", () => select(+tr.dataset.i)));
  }

  function select(i) {
    S.sel = i; drawGauge(); drawRisk(); drawWhy();
    document.querySelectorAll("#hrTbl tbody tr").forEach(tr => tr.classList.toggle("hl", +tr.dataset.i === i));
  }

  /* Semi-circle threat gauge. The arc length L is animated with CSS stroke-dashoffset (see .gauge .gv in index.html).
     Scale maximum is derived from the threshold and the peak risk so rare-event probabilities fill the dial usefully. */
  function drawGauge() {
    const R = S.result, box = $("#gauge"); if (!box) return;
    const h = R.hours[S.sel], thr = R.threshold, L = Math.PI * 80;
    const gMax = Math.min(1, Math.max(0.05, Math.ceil(Math.max(thr * 2.5, R.summary.max_risk * 1.15) / 0.05 - 1e-9) * 0.05));
    const f = Math.min(1, h.risk / gMax), ft = Math.min(1, thr / gMax);
    let built = false;
    if (!box.querySelector(".gv")) {
      built = true;
      const arc = "M 20 100 A 80 80 0 0 1 180 100", ang = Math.PI * ft;
      const px = r => (100 - r * Math.cos(ang)).toFixed(2), py = r => (100 - r * Math.sin(ang)).toFixed(2);
      box.innerHTML = `<svg class="gauge" viewBox="0 0 200 124" role="img" aria-label="Sepsis risk gauge">
          <path class="g-z1" d="${arc}" style="stroke-dasharray:${L * ft} ${L}"/>
          <path class="g-z2" d="${arc}" style="stroke-dasharray:${L * (1 - ft)} ${L};stroke-dashoffset:${-L * ft}"/>
          <path class="gv" d="${arc}" style="stroke-dasharray:${L} ${L};stroke-dashoffset:${L};opacity:0"/>
          <line class="g-tick" x1="${px(63)}" y1="${py(63)}" x2="${px(97)}" y2="${py(97)}"/>
          <text class="g-val" x="100" y="86" text-anchor="middle"></text>
          <text class="g-sub" x="100" y="102" text-anchor="middle"></text>
          <text class="g-end" x="20" y="118" text-anchor="middle">0%</text>
          <text class="g-end" x="180" y="118" text-anchor="middle">${pct(gMax, 0)}</text>
        </svg><div class="gauge-status"></div>`;
    }
    const gv = box.querySelector(".gv");
    const apply = () => { gv.style.strokeDashoffset = L * (1 - f); gv.style.opacity = f > 0 ? 1 : 0; gv.classList.toggle("alert", !!h.alert); };
    if (built) requestAnimationFrame(() => requestAnimationFrame(apply)); else apply();
    box.querySelector(".g-val").textContent = risk(h.risk);
    box.querySelector(".g-sub").textContent = "hour " + h.hour;
    box.querySelector(".gauge-status").innerHTML = `<span class="chip ${h.alert ? "chip-alert" : "chip-ok"}">${h.alert ? "Alert" : "Normal"}</span>`;
    const sub = $("#gaugeSub"); if (sub) sub.textContent = `Hour ${h.hour} · dial scale 0–${pct(gMax, 0)}`;
  }

  function drawRisk() {
    const R = S.result, x = R.hours.map(h => h.hour), y = R.hours.map(h => h.risk * 100), thr = R.threshold * 100;
    const bands = []; let start = null;
    R.hours.forEach((h, i) => { if (h.alert && start === null) start = h.hour; if ((!h.alert || i === R.hours.length - 1) && start !== null) { bands.push({ x0: start, x1: h.alert ? h.hour : R.hours[i - 1].hour, color: "rgba(225,29,72,.16)" }); start = null; } });
    const ymax = Math.max(Math.max(...y), thr) * 1.25 || 1;
    const lab = S.labels ? x.filter((_, i) => S.labels[i] === 1) : [];
    Charts.line($("#riskChart"), {
      title: "Sepsis risk per hour", series: [{ name: "Risk", x, y, color: "#6366F1", points: true, width: 2.6 }],
      smooth: true, area: true,
      xMin: Math.min(...x), xMax: Math.max(...x), yMin: 0, yMax: ymax, xLabel: "Hour", yLabel: "Risk (%)", yFmt: v => (v < 1 ? v.toFixed(2) : v.toFixed(0)) + "%",
      hlines: [{ y: thr, label: "alert threshold " + risk(R.threshold), color: "#f43f5e" }], bands, selected: S.sel, onPoint: select,
      strip: lab.length ? { x: lab, color: "#F59E0B" } : null, legend: false,
      pointTitle: i => `Hour ${R.hours[i].hour}: ${risk(R.hours[i].risk)}`,
      legendExtra: `<span><i style="background:#6366F1"></i>Risk</span><span><i style="background:rgba(225,29,72,.5)"></i>Alert hours</span>${lab.length ? '<span><i style="background:#F59E0B"></i>Sepsis label in the data (from 6 h before onset)</span>' : ""}`,
    });
  }

  function drawWhy() {
    const R = S.result, h = R.hours[S.sel], rule = S.model.rules[h.top_rule - 1];
    const acts = h.rule_activation
      .map((v, i) => ({ label: "Rule " + (i + 1), value: v * 100, text: (v * 100).toFixed(1) + "%", sub: "w " + v.toFixed(3), desc: S.model.rules[i] ? S.model.rules[i].text : "", top: i === h.top_rule - 1 }))
      .sort((a, b) => b.value - a.value);
    const box = $("#whyCard");
    box.innerHTML = `<div class="card-head"><div><h2>Explainable AI · hour ${h.hour}</h2><p>Risk ${risk(h.risk)} · <span class="chip ${h.alert ? "chip-alert" : "chip-ok"}">${h.alert ? "Alert" : "Normal"}</span></p></div></div>
      <div class="explain"><b>Rule ${rule.id} is strongest.</b> It describes patients where ${esc(rule.text)}. Patients that look like this have a risk of about ${risk(rule.centre_risk)}.</div>
      <p class="note" style="margin:12px 0 2px">Rule firing strength (normalised weights, sum = 100%)</p>
      <div id="rb"></div>
      <p class="note" style="margin:10px 0 4px">Model inputs at this hour</p>
      <div class="pills">${S.model.selected_features.map(f => `<span class="pill">${esc(f.label)} <b>${h.inputs[f.name]}</b> <small>${esc(f.unit)}</small></span>`).join("")}</div>`;
    Charts.bars($("#rb"), acts);
  }

  /* ---------------- Model ---------------- */
  function pageModel() {
    const M = S.model, d = M.dataset, st = M.metrics, h = M.history, t = M.timing;
    view.innerHTML = `
    <div class="steps">
      <div class="step"><div class="no">1</div><h3>Data &amp; split</h3><div class="big">${d.patients.toLocaleString()} patients</div><p>${d.records.toLocaleString()} hourly records. Split <b>by patient</b> into train / validation / test so no patient is in two parts. Test sepsis rate ${pct(d.test_sepsis_rate, 2)}.</p></div>
      <div class="step"><div class="no">2</div><h3>Feature engineering</h3><div class="big">37 candidates</div><p>Vitals, labs, how often each lab was ordered, and 6-hour trends (change, mean, variability) of the vital signs.</p></div>
      <div class="step"><div class="no">3</div><h3>Genetic Algorithm</h3><div class="big">${M.selected_features.length} features picked</div><p>Chromosome = one bit per feature. Fitness = AUROC of a quick classifier minus a small penalty per feature. ${M.config.ga_gen} generations, population ${M.config.ga_pop}. Took ${t.ga_s.toFixed(1)} s.</p></div>
      <div class="step"><div class="no">4</div><h3>ANFIS fuzzy rules</h3><div class="big">${M.rules.length} rules</div><p>k-means finds patient groups; each becomes an IF-THEN rule with bell-shaped (Gaussian) membership curves. Rule outputs are blended by how strongly each rule fires.</p></div>
      <div class="step"><div class="no">5</div><h3>Particle Swarm Optimization</h3><div class="big">loss ${num(h.pso_start, 3)} → ${num(h.pso[h.pso.length - 1], 3)}</div><p>${M.config.pso_particles} particles tune the curve centres and widths for ${M.config.pso_iters} rounds, starting from the k-means solution. Took ${t.pso_s.toFixed(1)} s. Validation AUROC ${num(st.val_auc_kmeans_only, 3)} → ${num(st.val_auc_after_pso, 3)}.</p></div>
      <div class="step"><div class="no">6</div><h3>Calibration &amp; alert threshold</h3><div class="big">${risk(M.alert.threshold)}</div><p>A sigmoid (Platt scaling) turns the score into a probability. The alert threshold is chosen on validation patients so about ${pct(M.alert.target_sensitivity, 0)} of sepsis hours are caught.</p></div>
    </div>

    <div class="grid g2" style="margin-top:18px">
      <div class="card"><div class="card-head"><div><h2>GA convergence</h2><p>Best and average fitness of the population per generation.</p></div></div><div class="chart" id="gaChart"></div></div>
      <div class="card"><div class="card-head"><div><h2>PSO convergence</h2><p>Best training loss per round (lower is better).</p></div></div><div class="chart" id="psoChart"></div></div>
    </div>

    <div class="card"><div class="card-head"><div><h2>Features chosen by the Genetic Algorithm</h2><p>Quick check: a simple model on these ${M.selected_features.length} features scores AUROC ${num(st.ga_quick_auc_selected, 3)}, against ${num(st.ga_quick_auc_all, 3)} with all 37.</p></div></div>
      <div class="scroll"><table class="wrap"><thead><tr><th>Feature</th><th>Group</th><th>Unit</th><th>What it means</th></tr></thead><tbody>
      ${M.selected_features.map(f => `<tr><td><b>${esc(f.label)}</b><br><span class="mono" style="color:var(--muted)">${esc(f.name)}</span></td><td>${esc(f.group)}</td><td>${esc(f.unit)}</td><td>${esc(f.desc)}</td></tr>`).join("")}</tbody></table></div></div>

    <div class="card"><div class="card-head"><div><h2>The fuzzy rules</h2><p>Each rule is a patient profile. Centres are shown in real units. "Risk at centre" is the risk the model gives to a patient sitting exactly at the rule's centre.</p></div></div>
      <div class="scroll"><table class="wrap"><thead><tr><th>Rule</th>${M.selected_features.map(f => `<th class="num">${esc(f.name)}</th>`).join("")}<th class="num">Risk at centre</th><th>Reads as</th></tr></thead><tbody>
      ${M.rules.map(r => `<tr><td><b>Rule ${r.id}</b></td>${M.selected_features.map(f => `<td class="num">${r.centres[f.name]}</td>`).join("")}<td class="num">${risk(r.centre_risk)}</td><td>IF ${esc(r.text)}</td></tr>`).join("")}</tbody></table></div></div>

    <div class="card"><div class="card-head"><div><h2>Why fuzzy rules?</h2></div></div>
      <ul class="list">
        <li><b>Graded, not on/off.</b> “Heart rate is high” is a degree from 0 to 1 instead of a hard limit like 90 bpm.</li>
        <li><b>Readable.</b> For every alert you can see which rules fired and what profile each rule stands for.</li>
        <li><b>Small and fast.</b> ${M.rules.length} rules × ${M.selected_features.length} inputs; one prediction takes about ${t.predict_us_per_record.toFixed(1)} microseconds.</li>
        <li><b>Hybrid training.</b> The linear formula inside each rule is solved exactly by least squares; PSO only searches the curve shapes, which gradient descent handles poorly.</li>
      </ul></div>`;
    Charts.line($("#gaChart"), { series: [
      { name: "Best", x: h.ga_best.map((_, i) => i + 1), y: h.ga_best, color: COLORS[0] },
      { name: "Population mean", x: h.ga_mean.map((_, i) => i + 1), y: h.ga_mean, color: COLORS[1], dash: "6 3" }],
      smooth: true, xLabel: "Generation", yLabel: "Fitness", yFmt: v => v.toFixed(3) });
    Charts.line($("#psoChart"), { series: [{ name: "Best loss", x: h.pso.map((_, i) => i), y: h.pso, color: COLORS[0] }],
      smooth: true, area: true,
      xLabel: "PSO round", yLabel: "Loss", yFmt: v => v.toFixed(3), hlines: [{ y: h.pso_start, label: "k-means start", color: "#F59E0B" }],
      yMax: Math.max(h.pso_start, ...h.pso) * 1.02, yMin: Math.min(...h.pso) * 0.98 });
  }

  /* ---------------- Performance ---------------- */
  function modelsTable(models) {
    const best = Math.max(...models.map(m => m.auroc));
    return `<div class="scroll"><table><thead><tr><th>Model</th><th class="num">AUROC</th><th class="num">AUPRC</th><th class="num">Sensitivity</th><th class="num">Specificity</th><th class="num">Precision</th><th class="num">Accuracy</th><th class="num">False alarms per true alert</th></tr></thead><tbody>
    ${models.map((m, i) => `<tr class="${m.auroc === best ? "best" : ""}" ${i === 0 ? 'style="background:var(--teal-l)"' : ""}><td>${esc(m.model)}</td><td class="num">${num(m.auroc, 4)}</td><td class="num">${num(m.auprc, 4)}</td><td class="num">${pct(m.sensitivity)}</td><td class="num">${pct(m.specificity)}</td><td class="num">${pct(m.precision)}</td><td class="num">${pct(m.accuracy)}</td><td class="num">${num(m.false_alarms_per_true_alert, 1)}</td></tr>`).join("")}</tbody></table></div>`;
  }
  function confusion(c) {
    return `<div class="cm"><span></span><span class="h">Predicted: no sepsis</span><span class="h">Predicted: sepsis</span>
      <span class="r">Actual: no sepsis</span><div class="c good"><b>${c.tn.toLocaleString()}</b>true negatives</div><div class="c bad"><b>${c.fp.toLocaleString()}</b>false alarms</div>
      <span class="r">Actual: sepsis</span><div class="c mid"><b>${c.fn.toLocaleString()}</b>missed</div><div class="c good"><b>${c.tp.toLocaleString()}</b>caught</div></div>`;
  }

  async function pagePerformance() {
    const M = S.model, a = M.metrics.anfis;
    view.innerHTML = `
    ${M.is_demo ? `<div class="banner banner-warn"><b>Synthetic test model.</b> These numbers do not describe the project's trained model.</div>` : ""}
    <div class="card"><div class="card-head"><div><h2>Loaded model on its test patients</h2><p>${M.dataset.test_patients.toLocaleString()} patients · ${M.dataset.test_rows.toLocaleString()} hourly rows · sepsis rate ${pct(M.dataset.test_sepsis_rate, 2)}. The alert threshold (${risk(M.alert.threshold)}) was chosen on separate validation patients.</p></div></div>
      <div class="grid g6">
        <div class="tile"><div class="k">AUROC</div><div class="v">${num(a.auroc, 3)}</div><div class="n">0.5 = random</div></div>
        <div class="tile"><div class="k">AUPRC</div><div class="v">${num(a.auprc, 3)}</div><div class="n">random = ${num(M.curves.test_prevalence, 3)}</div></div>
        <div class="tile"><div class="k">Sensitivity</div><div class="v">${pct(a.sensitivity, 0)}</div><div class="n">sepsis hours caught</div></div>
        <div class="tile"><div class="k">Specificity</div><div class="v">${pct(a.specificity, 0)}</div><div class="n">normal hours cleared</div></div>
        <div class="tile"><div class="k">Precision</div><div class="v">${pct(a.precision, 1)}</div><div class="n">alerts that were real</div></div>
        <div class="tile"><div class="k">False alarms</div><div class="v">${num(a.false_alarms_per_true_alert, 1)}</div><div class="n">per true alert</div></div>
      </div></div>
    <div class="grid g2">
      <div class="card"><div class="card-head"><div><h2>ROC curve</h2><p>Higher and further left is better.</p></div></div><div class="chart" id="roc"></div></div>
      <div class="card"><div class="card-head"><div><h2>Precision–recall curve</h2><p>The dashed line is what a random model scores.</p></div></div><div class="chart" id="pr"></div></div>
    </div>
    <div class="card"><div class="card-head"><div><h2>Comparison with baselines</h2><p>Same split, same rule for the threshold (catch about ${pct(M.alert.target_sensitivity, 0)} of sepsis hours on validation data).</p></div></div>${modelsTable(M.metrics.models)}</div>
    <div class="card"><div class="card-head"><div><h2>Confusion matrix</h2><p>${esc(M.name)} on the test rows.</p></div></div>${confusion(M.metrics.confusion)}</div>
    <div class="card"><div class="card-head"><div><h2>Reference run from the project report</h2><p id="refNote"></p></div></div><div id="ref"><div class="empty">Loading…</div></div></div>
    <div class="card"><h2>How to read these numbers</h2><ul class="list">
      <li><b>Accuracy is misleading.</b> Only about 2% of hours are sepsis. A model that always says “normal” would score 98% accuracy and save nobody.</li>
      <li><b>AUROC</b> measures how well the model ranks sick hours above healthy ones, independent of any threshold.</li>
      <li><b>Low precision is expected.</b> With rare sepsis and hourly checks, most alerts are false alarms. The model is a ranking and support tool, not a diagnosis.</li></ul></div>`;
    const rocS = Object.entries(M.curves.roc).map(([n, c], i) => ({ name: n, x: c.fpr, y: c.tpr, color: COLORS[i % 6], dash: DASH[i % 6], width: i === 0 ? 3 : 1.6 }));
    Charts.line($("#roc"), { series: rocS, smooth: false, xMin: 0, xMax: 1, yMin: 0, yMax: 1, xLabel: "False positive rate", yLabel: "Sensitivity", diagonal: true, xFmt: v => v.toFixed(1), yFmt: v => v.toFixed(1) });
    const prS = Object.entries(M.curves.pr).map(([n, c], i) => ({ name: n, x: c.recall, y: c.precision, color: COLORS[i % 6], dash: DASH[i % 6], width: i === 0 ? 3 : 1.6 }));
    Charts.line($("#pr"), { series: prS, smooth: false, xMin: 0, xMax: 1, yMin: 0, yMax: 1, xLabel: "Recall", yLabel: "Precision", hlines: [{ y: M.curves.test_prevalence, label: "random", color: "#94a3b8" }], xFmt: v => v.toFixed(1), yFmt: v => v.toFixed(1) });
    try {
      S.ref = S.ref || await api("/api/reference"); const R = S.ref;
      $("#refNote").textContent = R.note;
      $("#ref").innerHTML = `<div class="grid g4" style="margin-bottom:14px">
        <div class="tile"><div class="k">Patients</div><div class="v">${R.dataset.patients.toLocaleString()}</div><div class="n">${R.dataset.test_rows.toLocaleString()} test rows</div></div>
        <div class="tile"><div class="k">Selected features</div><div class="v" style="font-size:1rem">${R.selected_features.join(", ")}</div></div>
        <div class="tile"><div class="k">Training time</div><div class="v">${(R.timing.ga_s + R.timing.pso_s).toFixed(0)} s</div><div class="n">GA ${R.timing.ga_s} s + PSO ${R.timing.pso_s} s</div></div>
        <div class="tile"><div class="k">Alert threshold</div><div class="v">${risk(R.threshold)}</div></div></div>
        ${modelsTable(R.models)}<div style="height:14px"></div>${confusion(R.confusion)}
        <h3 style="margin-top:16px">Known limitations of this run</h3><ul class="list">${R.limitations.map(l => `<li>${esc(l)}</li>`).join("")}</ul>`;
    } catch (e) { $("#ref").innerHTML = `<div class="err">${esc(e.message)}</div>`; }
  }

  /* ---------------- About ---------------- */
  function pageAbout() {
    view.innerHTML = `
    <div class="card"><h2>What is sepsis and why early warning?</h2>
      <p style="margin-top:8px">Sepsis is the body's over-reaction to an infection that starts to damage its own organs. It is a leading cause of death in intensive care. Each hour of delay in treatment lowers the chance of survival by about 7.6%, so a warning a few hours earlier can matter.</p>
      <p>This project builds a transparent <b>fuzzy early-warning model</b>: a Genetic Algorithm chooses the measurements, an ANFIS turns them into readable IF-THEN rules, and Particle Swarm Optimization tunes the rules. It was built for the <b>Soft Computing AAT (MMCE30C3)</b> at B.M.S. College of Engineering, Bengaluru.</p></div>
    <div class="grid g2">
      <div class="card"><h2>Safety &amp; limits</h2><ul class="list">
        <li><b>Not a medical device.</b> Do not use it for clinical decisions.</li>
        <li><b>Many false alarms.</b> Precision is low (about 3% in the reference run).</li>
        <li><b>Time proxy.</b> The reference model selected “hours in ICU”, which reflects how sepsis risk grows with stay length, not an abnormal body measurement.</li>
        <li><b>One dataset.</b> Trained on PhysioNet 2019 Training Set A (US hospitals); results may not transfer to other hospitals.</li>
        <li><b>Sustained-alert rule</b> on the monitor page is a usability extension and was not part of the evaluation.</li></ul></div>
      <div class="card"><h2>Data</h2><p style="margin-top:8px">PhysioNet / Computing in Cardiology Challenge 2019, Training Set A: 20,336 ICU stays with hourly vital signs and lab values. The label <span class="mono">SepsisLabel</span> is 1 from 6 hours before sepsis onset.</p>
        <p class="note">Reyna et al., <i>Early Prediction of Sepsis from Clinical Data</i>, Critical Care Medicine 48(2), 2020. The example patients in this app are synthetic illustrations, not real people.</p></div>
    </div>
    <div class="card"><h2>Run it yourself</h2>
      <pre class="mono">make install        # once
# put the Colab-exported model.json into artifacts/ (see docs/EXPORT_FROM_COLAB.md)
make run            # http://localhost:8000
make train          # alternative: retrain locally on all 20,336 patients (about 10-15 minutes)
make test           # run the test-suite</pre>
      <p class="note">The API is documented interactively at <a href="/docs" target="_blank" rel="noopener">/docs</a>. The README and the <span class="mono">docs/</span> folder describe the method, API and architecture in detail.</p></div>
    <div class="card"><h2>Team</h2><dl class="kv"><dt>Students</dt><dd>Lohith S (1BM25MC052) · Manjunath H (1BM25MC054)</dd><dt>Guide</dt><dd>Dr. K Vijayakumar, Professor</dd><dt>Department</dt><dd>Computer Applications, B.M.S. College of Engineering, Bengaluru</dd></dl></div>`;
  }

  boot();
})();