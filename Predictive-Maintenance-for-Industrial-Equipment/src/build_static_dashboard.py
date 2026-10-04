"""Build docs/index.html - a self-contained dashboard for GitHub Pages."""
import json

import pandas as pd

from .config import DATA_DIR, DOCS_DIR

TEMPLATE = r"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Predictive Maintenance Dashboard</title>
<style>
 :root{--bg:#f6f7f9;--card:#fff;--text:#1f2933;--muted:#6b7280;--line:#e5e7eb;--high:#c0392b;--med:#e67e22;--low:#27ae60}
 @media (prefers-color-scheme:dark){:root{--bg:#12161c;--card:#1b2129;--text:#e6e9ee;--muted:#9aa3af;--line:#2b3440}}
 body{margin:0;font-family:system-ui,Segoe UI,Arial,sans-serif;background:var(--bg);color:var(--text)}
 header{padding:20px 24px}h1{margin:0;font-size:22px}.sub{color:var(--muted);font-size:13px;margin-top:4px}
 main{padding:0 24px 32px;max-width:1100px;margin:auto}
 .cards{display:flex;gap:12px;flex-wrap:wrap;margin-bottom:16px}
 .card{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:14px 18px;min-width:130px}
 .card b{display:block;font-size:26px}.card span{color:var(--muted);font-size:12px}
 table{width:100%;border-collapse:collapse;background:var(--card);border:1px solid var(--line);border-radius:10px;overflow:hidden}
 th,td{padding:8px 10px;text-align:left;font-size:13px;border-bottom:1px solid var(--line)}
 th{color:var(--muted);font-weight:600}tbody tr{cursor:pointer}tbody tr:hover,tr.sel{background:rgba(120,140,170,.15)}
 .bar{height:8px;background:var(--line);border-radius:4px;width:110px;display:inline-block;vertical-align:middle}
 .bar i{display:block;height:100%;border-radius:4px}
 .pill{padding:2px 8px;border-radius:10px;color:#fff;font-size:12px}
 .High{background:var(--high)}.Medium{background:var(--med)}.Low{background:var(--low)}.Offline{background:#7f8c8d}
 .panel{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:14px;margin-top:16px}
 canvas{width:100%;height:260px}.scroll{overflow-x:auto}
</style></head><body>
<header><h1>Predictive Maintenance for Industrial Equipment</h1>
<div class="sub" id="sub"></div></header>
<main>
 <div class="cards" id="cards"></div>
 <div class="scroll"><table><thead><tr><th>Machine</th><th>Type</th><th>Risk</th><th>Level</th><th>Main driver</th>
 <th>Temp</th><th>Vib</th><th>Pressure</th><th>h since maint.</th></tr></thead><tbody id="rows"></tbody></table></div>
 <div class="panel"><b id="ctitle"></b><canvas id="chart" width="1000" height="260"></canvas>
 <div class="sub">Risk score over the last 14 days. Dashed line = alert threshold. Click a row above to switch machine.</div></div>
</main>
<script>
const D = __DATA__;
const col = {High:'#c0392b',Medium:'#e67e22',Low:'#27ae60',Offline:'#7f8c8d'};
document.getElementById('sub').textContent = `Model: ${D.meta.model_name} | predicts failure within ${D.meta.horizon_h} h | data as of ${D.meta.as_of}`;
const cnt = l => D.scores.filter(s => s.risk_level === l).length;
document.getElementById('cards').innerHTML =
 `<div class="card"><b>${D.scores.length}</b><span>machines</span></div>` +
 ['High','Medium','Low','Offline'].filter(l => l!=='Offline' || cnt(l)>0).map(l => `<div class="card"><b style="color:${col[l]}">${cnt(l)}</b><span>${l==='Offline'?'stopped (maintenance)':l+' risk'}</span></div>`).join('');
const tb = document.getElementById('rows');
D.scores.forEach((s, i) => {
  const tr = document.createElement('tr');
  tr.innerHTML = `<td><b>${s.machine_id}</b></td><td>${s.machine_type}</td>
   <td><span class="bar"><i style="width:${(s.risk_score*100).toFixed(0)}%;background:${col[s.risk_level]}"></i></span> ${s.risk_score.toFixed(2)}</td>
   <td><span class="pill ${s.risk_level}">${s.risk_level}</span></td><td>${s.main_driver}</td>
   <td>${s.temperature.toFixed(1)}</td><td>${s.vibration.toFixed(2)}</td><td>${s.pressure.toFixed(1)}</td><td>${s.hours_since_maintenance}</td>`;
  tr.onclick = () => select(s.machine_id);
  tr.dataset.id = s.machine_id;
  tb.appendChild(tr);
});
function select(id){
  document.querySelectorAll('tbody tr').forEach(r => r.classList.toggle('sel', r.dataset.id === id));
  document.getElementById('ctitle').textContent = 'Risk history - ' + id;
  draw(D.history[id] || []);
}
function draw(pts){
  const c = document.getElementById('chart'), g = c.getContext('2d'), W = c.width, H = c.height, P = 36;
  g.clearRect(0,0,W,H);
  const style = getComputedStyle(document.body);
  g.strokeStyle = style.getPropertyValue('--line'); g.fillStyle = style.getPropertyValue('--muted'); g.font = '12px sans-serif';
  [0,.25,.5,.75,1].forEach(v => { const y = H-P-(H-2*P)*v; g.beginPath(); g.moveTo(P,y); g.lineTo(W-10,y); g.stroke(); g.fillText(v.toFixed(2),2,y+4); });
  if(!pts.length) return;
  const x = i => P + (W-P-10)*i/(pts.length-1), y = v => H-P-(H-2*P)*v;
  g.setLineDash([6,4]); g.strokeStyle = '#888'; g.beginPath(); g.moveTo(P,y(D.meta.threshold)); g.lineTo(W-10,y(D.meta.threshold)); g.stroke(); g.setLineDash([]);
  g.strokeStyle = '#c0392b'; g.lineWidth = 2; g.beginPath();
  pts.forEach((p,i) => i ? g.lineTo(x(i),y(p[1])) : g.moveTo(x(i),y(p[1]))); g.stroke(); g.lineWidth = 1;
  g.fillText(pts[0][0].slice(0,10), P, H-10); g.fillText(pts[pts.length-1][0].slice(0,10), W-80, H-10);
}
select(D.scores[0].machine_id);
</script></body></html>
"""


def build():
    scores = pd.read_csv(DATA_DIR / "risk_scores.csv")
    hist = pd.read_csv(DATA_DIR / "risk_history.csv")
    meta = json.loads((DATA_DIR / "dashboard_meta.json").read_text())
    history = {m: [[t, float(r)] for t, r in zip(g["timestamp"], g["risk_score"])]
               for m, g in hist.groupby("machine_id")}
    data = {"meta": meta, "scores": json.loads(scores.to_json(orient="records")), "history": history}
    DOCS_DIR.mkdir(exist_ok=True)
    (DOCS_DIR / "index.html").write_text(TEMPLATE.replace("__DATA__", json.dumps(data)), encoding="utf-8")
    (DOCS_DIR / ".nojekyll").touch()
    print(f"[docs] wrote {DOCS_DIR / 'index.html'}")


if __name__ == "__main__":
    build()
