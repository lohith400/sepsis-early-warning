/* Dependency-free SVG charts for the Clinical Command Center.
   - line(): smooth monotone cubic-spline waveform, optional indigo gradient under-fill, crosshair + glowing
     active node + tooltip for the selected point, theme-aware grid (CSS variables --grid, --grid-v, --axis, --diag).
   - bars(): animated progress meters with rule weights (XAI panel).
   All theme colours come from CSS variables, so a theme switch needs no re-render. */
const Charts = (() => {
  let uid = 0;
  const esc = s => String(s).replace(/[&<>"]/g, c => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));

  function niceTicks(min, max, n = 5) {
    if (max === min) { max = min + 1; }
    const span = max - min, step0 = span / n, mag = Math.pow(10, Math.floor(Math.log10(step0)));
    const f = step0 / mag, step = (f < 1.5 ? 1 : f < 3 ? 2 : f < 7 ? 5 : 10) * mag;
    const out = [];
    for (let v = Math.ceil(min / step) * step; v <= max + 1e-9; v += step) out.push(+v.toFixed(10));
    return out;
  }

  /* Monotone cubic Hermite spline (Fritsch-Carlson) converted to cubic Bezier segments.
     Smooth like a telemetry trace, but never overshoots the data (no fake peaks below 0). */
  function spline(P) {
    const n = P.length, f = v => v.toFixed(1);
    if (n < 3) return P.map((p, i) => `${i ? "L" : "M"}${f(p[0])} ${f(p[1])}`).join("");
    const dx = [], m = [], t = new Array(n);
    for (let i = 0; i < n - 1; i++) {
      dx[i] = P[i + 1][0] - P[i][0];
      m[i] = dx[i] ? (P[i + 1][1] - P[i][1]) / dx[i] : 0;
    }
    t[0] = m[0]; t[n - 1] = m[n - 2];
    for (let i = 1; i < n - 1; i++) t[i] = m[i - 1] * m[i] <= 0 ? 0 : (m[i - 1] + m[i]) / 2;
    for (let i = 0; i < n - 1; i++) {
      if (m[i] === 0) { t[i] = 0; t[i + 1] = 0; continue; }
      const a = t[i] / m[i], b = t[i + 1] / m[i], s = a * a + b * b;
      if (s > 9) { const tau = 3 / Math.sqrt(s); t[i] = tau * a * m[i]; t[i + 1] = tau * b * m[i]; }
    }
    let d = `M${f(P[0][0])} ${f(P[0][1])}`;
    for (let i = 0; i < n - 1; i++) {
      const h = dx[i] / 3;
      d += `C${f(P[i][0] + h)} ${f(P[i][1] + t[i] * h)} ${f(P[i + 1][0] - h)} ${f(P[i + 1][1] - t[i + 1] * h)} ${f(P[i + 1][0])} ${f(P[i + 1][1])}`;
    }
    return d;
  }

  function line(el, o) {
    const id = "cg" + (++uid);
    const W = o.width || 640, H = o.height || 280, m = { l: 54, r: 16, t: 16, b: o.xLabel ? 44 : 30 };
    const allx = o.series.flatMap(s => s.x), ally = o.series.flatMap(s => s.y);
    const xMin = o.xMin ?? Math.min(...allx), xMax = o.xMax ?? Math.max(...allx);
    let yMin = o.yMin ?? Math.min(...ally), yMax = o.yMax ?? Math.max(...ally);
    if (yMax === yMin) yMax = yMin + 1;
    const sx = v => m.l + (v - xMin) / ((xMax - xMin) || 1) * (W - m.l - m.r);
    const sy = v => H - m.b - (v - yMin) / (yMax - yMin) * (H - m.t - m.b);
    const xf = o.xFmt || (v => v), yf = o.yFmt || (v => v);
    const base = H - m.b, smooth = o.smooth !== false;
    const yt = niceTicks(yMin, yMax, 5), xt = o.xTicks || niceTicks(xMin, xMax, 6);

    let defs = "", b = "";
    // theme-aware gridlines: horizontal (stronger) + vertical (faint)
    b += `<g>` + yt.map(v => `<line x1="${m.l}" x2="${W - m.r}" y1="${sy(v)}" y2="${sy(v)}" style="stroke:var(--grid)" stroke-width="1"/>`).join("")
      + xt.map(v => `<line x1="${sx(v)}" x2="${sx(v)}" y1="${m.t}" y2="${base}" style="stroke:var(--grid-v)" stroke-width="1" opacity=".7"/>`).join("") + `</g>`;
    (o.bands || []).forEach(bd => { b += `<rect x="${sx(bd.x0)}" y="${m.t}" width="${Math.max(2, sx(bd.x1) - sx(bd.x0))}" height="${H - m.t - m.b}" fill="${bd.color}" rx="2"/>`; });
    b += `<g class="axis">` + yt.map(v => `<text x="${m.l - 8}" y="${sy(v) + 4}" text-anchor="end">${esc(yf(v))}</text>`).join("")
      + xt.map(v => `<text x="${sx(v)}" y="${H - m.b + 16}" text-anchor="middle">${esc(xf(v))}</text>`).join("");
    if (o.xLabel) b += `<text x="${(m.l + W - m.r) / 2}" y="${H - 6}" text-anchor="middle">${esc(o.xLabel)}</text>`;
    if (o.yLabel) b += `<text transform="translate(14 ${(m.t + H - m.b) / 2}) rotate(-90)" text-anchor="middle">${esc(o.yLabel)}</text>`;
    b += `</g><line x1="${m.l}" x2="${W - m.r}" y1="${base}" y2="${base}" style="stroke:var(--axis)"/>`;
    (o.hlines || []).forEach(h => {
      if (h.y < yMin || h.y > yMax) return;
      b += `<line x1="${m.l}" x2="${W - m.r}" y1="${sy(h.y)}" y2="${sy(h.y)}" stroke="${h.color || "#f43f5e"}" stroke-dasharray="6 4" stroke-width="1.5"/>`
        + (h.label ? `<text x="${W - m.r - 4}" y="${sy(h.y) - 6}" text-anchor="end" font-size="11" font-weight="600" fill="${h.color || "#f43f5e"}">${esc(h.label)}</text>` : "");
    });
    if (o.diagonal) b += `<line x1="${sx(xMin)}" y1="${sy(yMin)}" x2="${sx(xMax)}" y2="${sy(yMax)}" stroke-dasharray="4 4" style="stroke:var(--diag)"/>`;
    if (o.strip) o.strip.x.forEach(x => { b += `<rect x="${sx(x) - 3}" y="${base - 7}" width="6" height="7" fill="${o.strip.color}" opacity=".9" rx="1"/>`; });

    let hits = "", sel = "";
    o.series.forEach((s, si) => {
      const P = s.x.map((x, i) => [sx(x), sy(s.y[i])]);
      const d = smooth ? spline(P) : P.map((p, i) => `${i ? "L" : "M"}${p[0].toFixed(1)} ${p[1].toFixed(1)}`).join("");
      if (o.area && si === 0 && P.length > 1) {
        defs += `<linearGradient id="${id}a" x1="0" y1="0" x2="0" y2="1"><stop offset="0%" style="stop-color:${s.color};stop-opacity:.50"/><stop offset="55%" style="stop-color:${s.color};stop-opacity:.16"/><stop offset="100%" style="stop-color:${s.color};stop-opacity:0"/></linearGradient>`;
        b += `<path d="${d}L${P[P.length - 1][0].toFixed(1)} ${base}L${P[0][0].toFixed(1)} ${base}Z" fill="url(#${id}a)" stroke="none"/>`;
      }
      b += `<path d="${d}" fill="none" stroke="${s.color}" stroke-width="${s.width || 2}" ${s.dash ? `stroke-dasharray="${s.dash}"` : ""} stroke-linejoin="round" stroke-linecap="round" style="filter:drop-shadow(0 0 5px ${s.color})"/>`;
      if (s.points) {
        P.forEach((p, i) => {
          if (o.selected !== i) b += `<circle cx="${p[0]}" cy="${p[1]}" r="3" style="fill:var(--surface-solid)" stroke="${s.color}" stroke-width="1.6" pointer-events="none"/>`;
        });
        const k = o.selected;
        if (k != null && P[k]) {
          const [px, py] = P[k];
          sel += `<line x1="${px}" x2="${px}" y1="${m.t}" y2="${base}" stroke="${s.color}" stroke-width="1.2" stroke-dasharray="3 4" opacity=".7" pointer-events="none"/>`
            + `<circle class="halo" cx="${px}" cy="${py}" r="11" fill="${s.color}" opacity=".35" pointer-events="none"/>`
            + `<circle cx="${px}" cy="${py}" r="6.5" style="fill:var(--surface-solid);filter:drop-shadow(0 0 6px ${s.color})" stroke="${s.color}" stroke-width="3" pointer-events="none"/>`;
          const txt = o.pointTitle ? o.pointTitle(k) : `${xf(s.x[k])}: ${yf(s.y[k])}`;
          const tw = txt.length * 6.3 + 18;
          const tx = Math.max(m.l, Math.min(px - tw / 2, W - m.r - tw));
          let ty = py - 38; if (ty < 2) ty = py + 16;
          sel += `<g pointer-events="none"><rect class="tip-bg" x="${tx}" y="${ty}" width="${tw}" height="24" rx="7"/><text class="tip-tx" x="${tx + tw / 2}" y="${ty + 16}" text-anchor="middle">${esc(txt)}</text></g>`;
        }
        P.forEach((p, i) => {
          hits += `<circle class="pt" data-i="${i}" cx="${p[0]}" cy="${p[1]}" r="9" fill="transparent"><title>${esc(o.pointTitle ? o.pointTitle(i) : `${xf(s.x[i])}: ${yf(s.y[i])}`)}</title></circle>`;
        });
      }
    });

    const svg = `<svg viewBox="0 0 ${W} ${H}" role="img" aria-label="${esc(o.title || "chart")}"><defs>${defs}</defs>${b}${sel}${hits}</svg>`;
    let leg = "";
    if (o.legend !== false && o.series.length > 1 || o.legendExtra) {
      leg = `<div class="legend">` + o.series.map(s => `<span><i style="background:${s.color}"></i>${esc(s.name)}</span>`).join("") + (o.legendExtra || "") + `</div>`;
    }
    el.innerHTML = svg + leg;
    if (o.onPoint) el.querySelectorAll(".pt").forEach(c => c.addEventListener("click", () => o.onPoint(+c.dataset.i)));
  }

  /* Animated progress meters. items = [{label, value (0-100), text, sub, desc, top}] */
  function bars(el, items) {
    el.innerHTML = `<div class="meters">` + items.map(it =>
      `<div class="meter ${it.top ? "top" : ""}">
         <div class="m-head"><span class="m-name">${esc(it.label)}</span><span class="m-val">${esc(it.text)}${it.sub ? `<small>${esc(it.sub)}</small>` : ""}</span></div>
         <div class="m-track"><div class="m-fill" data-w="${Math.max(0.5, it.value)}"></div></div>
         ${it.desc ? `<div class="m-desc" title="${esc(it.desc)}">${esc(it.desc)}</div>` : ""}
       </div>`).join("") + `</div>`;
    requestAnimationFrame(() => requestAnimationFrame(() => {
      el.querySelectorAll(".m-fill").forEach(f => { f.style.width = f.dataset.w + "%"; });
    }));
  }

  return { line, bars, esc };
})();