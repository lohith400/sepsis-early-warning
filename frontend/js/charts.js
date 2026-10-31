/* Tiny dependency-free SVG charts (line, bars). */
const Charts = (() => {
  const NS = "http://www.w3.org/2000/svg";
  const esc = s => String(s).replace(/[&<>"]/g, c => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));

  function niceTicks(min, max, n = 5) {
    if (max === min) { max = min + 1; }
    const span = max - min, step0 = span / n, mag = Math.pow(10, Math.floor(Math.log10(step0)));
    const f = step0 / mag, step = (f < 1.5 ? 1 : f < 3 ? 2 : f < 7 ? 5 : 10) * mag;
    const out = [];
    for (let v = Math.ceil(min / step) * step; v <= max + 1e-9; v += step) out.push(+v.toFixed(10));
    return out;
  }

  function line(el, o) {
    const W = o.width || 640, H = o.height || 280, m = { l: 54, r: 16, t: 14, b: o.xLabel ? 44 : 30 };
    const allx = o.series.flatMap(s => s.x), ally = o.series.flatMap(s => s.y);
    const xMin = o.xMin ?? Math.min(...allx), xMax = o.xMax ?? Math.max(...allx);
    let yMin = o.yMin ?? Math.min(...ally), yMax = o.yMax ?? Math.max(...ally);
    if (yMax === yMin) yMax = yMin + 1;
    const sx = v => m.l + (v - xMin) / ((xMax - xMin) || 1) * (W - m.l - m.r);
    const sy = v => H - m.b - (v - yMin) / (yMax - yMin) * (H - m.t - m.b);
    const xf = o.xFmt || (v => v), yf = o.yFmt || (v => v);
    let g = `<svg viewBox="0 0 ${W} ${H}" role="img" aria-label="${esc(o.title || "chart")}">`;
    const yt = niceTicks(yMin, yMax, 5), xt = o.xTicks || niceTicks(xMin, xMax, 6);
    g += `<g class="grid">` + yt.map(v => `<line x1="${m.l}" x2="${W - m.r}" y1="${sy(v)}" y2="${sy(v)}"/>`).join("") + `</g>`;
    (o.bands || []).forEach(b => { g += `<rect x="${sx(b.x0)}" y="${m.t}" width="${Math.max(2, sx(b.x1) - sx(b.x0))}" height="${H - m.t - m.b}" fill="${b.color}"/>`; });
    g += `<g class="axis">` + yt.map(v => `<text x="${m.l - 8}" y="${sy(v) + 4}" text-anchor="end">${esc(yf(v))}</text>`).join("")
      + xt.map(v => `<text x="${sx(v)}" y="${H - m.b + 16}" text-anchor="middle">${esc(xf(v))}</text>`).join("");
    if (o.xLabel) g += `<text x="${(m.l + W - m.r) / 2}" y="${H - 6}" text-anchor="middle">${esc(o.xLabel)}</text>`;
    if (o.yLabel) g += `<text transform="translate(14 ${(m.t + H - m.b) / 2}) rotate(-90)" text-anchor="middle">${esc(o.yLabel)}</text>`;
    g += `</g><line x1="${m.l}" x2="${W - m.r}" y1="${H - m.b}" y2="${H - m.b}" stroke="#C9D9DC"/>`;
    (o.hlines || []).forEach(h => {
      if (h.y < yMin || h.y > yMax) return;
      g += `<line x1="${m.l}" x2="${W - m.r}" y1="${sy(h.y)}" y2="${sy(h.y)}" stroke="${h.color || "#D4574E"}" stroke-dasharray="6 4" stroke-width="1.5"/>`
        + (h.label ? `<text x="${W - m.r - 4}" y="${sy(h.y) - 5}" text-anchor="end" font-size="11" fill="${h.color || "#D4574E"}">${esc(h.label)}</text>` : "");
    });
    if (o.diagonal) g += `<line x1="${sx(xMin)}" y1="${sy(yMin)}" x2="${sx(xMax)}" y2="${sy(yMax)}" stroke="#A9BCC2" stroke-dasharray="4 4"/>`;
    if (o.strip) o.strip.x.forEach(x => { g += `<rect x="${sx(x) - 3}" y="${H - m.b - 7}" width="6" height="7" fill="${o.strip.color}" opacity=".85"/>`; });
    o.series.forEach((s, si) => {
      const d = s.x.map((x, i) => `${i ? "L" : "M"}${sx(x).toFixed(1)} ${sy(s.y[i]).toFixed(1)}`).join("");
      g += `<path d="${d}" fill="none" stroke="${s.color}" stroke-width="${s.width || 2}" ${s.dash ? `stroke-dasharray="${s.dash}"` : ""} stroke-linejoin="round" stroke-linecap="round"/>`;
      if (s.points) s.x.forEach((x, i) => {
        const sel = o.selected === i;
        g += `<circle class="pt" data-i="${i}" cx="${sx(x)}" cy="${sy(s.y[i])}" r="${sel ? 6 : 3.6}" fill="${sel ? "#fff" : s.color}" stroke="${s.color}" stroke-width="${sel ? 3 : 1}"><title>${esc((o.pointTitle ? o.pointTitle(i) : `${xf(x)}: ${yf(s.y[i])}`))}</title></circle>`;
      });
    });
    g += `</svg>`;
    let leg = "";
    if (o.legend !== false && o.series.length > 1 || o.legendExtra) {
      leg = `<div class="legend">` + o.series.map(s => `<span><i style="background:${s.color}"></i>${esc(s.name)}</span>`).join("") + (o.legendExtra || "") + `</div>`;
    }
    el.innerHTML = g + leg;
    if (o.onPoint) el.querySelectorAll(".pt").forEach(c => c.addEventListener("click", () => o.onPoint(+c.dataset.i)));
  }

  /* horizontal bars: items = [{label, value (0-100), text, top}] */
  function bars(el, items) {
    el.innerHTML = `<div class="bars">` + items.map(it =>
      `<div class="row ${it.top ? "top" : ""}"><span>${esc(it.label)}</span><div class="track"><div class="fill" style="width:${Math.max(0.5, it.value)}%"></div></div><span style="text-align:right">${esc(it.text)}</span></div>`).join("") + `</div>`;
  }

  return { line, bars, esc };
})();
