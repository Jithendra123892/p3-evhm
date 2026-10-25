
"use strict";
var slides = [].slice.call(document.querySelectorAll(".slide"));
var cur = 0, notesOn = false;
function show(i){
  cur = Math.max(0, Math.min(slides.length - 1, i));
  slides.forEach(function(s, j){ s.classList.toggle("on", j === cur); });
  document.getElementById("pg").textContent = (cur + 1) + " / " + slides.length;
  localStorage.setItem("p3slide", String(cur));
  if (cur === 1) anim2(); if (cur === 4) anim5();
}
document.addEventListener("keydown", function(e){
  if (e.key === "ArrowRight" || e.key === "PageDown"){ show(cur + 1); e.preventDefault(); }
  if (e.key === "ArrowLeft" || e.key === "PageUp"){ show(cur - 1); e.preventDefault(); }
  if (e.key === "s" || e.key === "S"){
    notesOn = !notesOn;
    [].forEach.call(document.querySelectorAll(".say"), function(el){ el.style.display = notesOn ? "" : "none"; });
  }
});
document.addEventListener("click", function(e){ if (e.clientX > innerWidth * 0.66) show(cur + 1); });
function fit(){
  var s = Math.min(innerWidth / 1920, innerHeight / 1080);
  document.getElementById("stage").style.transform = "translate(-50%,-50%) scale(" + s + ")";
}
addEventListener("resize", fit); fit();
var startSlide = parseInt((location.hash || "").replace("#", ""), 10);
if (!(startSlide >= 1 && startSlide <= slides.length))
  startSlide = parseInt(localStorage.getItem("p3slide") || "0", 10) + 1;
show(startSlide - 1);

/* ---- slide 2: invisible fade ---- */
function anim2(){
  var cv = document.getElementById("c2"), ctx = cv.getContext("2d");
  var W = cv.width, H = cv.height, prog = 0, raf = 0;
  function cap(t){ return 1.8565 - 0.0022 * t - 0.5 / (1 + Math.exp(-9 * (t / 118 - 1.15))) * 0.42; }
  function draw(){
    ctx.clearRect(0, 0, W, H);
    var x0 = 90, x1 = W - 40, y0 = 40, y1 = H - 70;
    var eolY = y1 - (1.4 - 1.1) / 0.9 * (y1 - y0);
    ctx.strokeStyle = "#ff6b6b"; ctx.lineWidth = 3; ctx.setLineDash([12, 10]);
    ctx.beginPath(); ctx.moveTo(x0, eolY); ctx.lineTo(x1, eolY); ctx.stroke(); ctx.setLineDash([]);
    ctx.fillStyle = "#ff6b6b"; ctx.font = "26px ui-monospace,Consolas,monospace";
    ctx.fillText("replacement threshold (70%)", x0 + 10, eolY - 14);
    ctx.strokeStyle = "#22314e"; ctx.lineWidth = 2;
    ctx.strokeRect(x0, y0, x1 - x0, y1 - y0);
    ctx.fillStyle = "#93a7c4"; ctx.font = "24px ui-monospace,Consolas,monospace";
    ctx.fillText("capacity (Ah)", x0 + 6, y0 + 34);
    ctx.fillText("cycle →", x1 - 130, y1 + 46);
    ctx.strokeStyle = "#3ddc97"; ctx.lineWidth = 5; ctx.beginPath();
    var steps = 300, upTo = Math.floor(steps * prog);
    var kneeX = null;
    for (var i = 0; i <= upTo; i++){
      var t = i / steps, c = cap(t * 150);
      var x = x0 + t * (x1 - x0), y = y1 - (c - 1.1) / 0.9 * (y1 - y0);
      if (c <= 1.4 && kneeX === null) kneeX = x;
      i ? ctx.lineTo(x, y) : ctx.moveTo(x, y);
    }
    ctx.stroke();
    if (kneeX !== null){
      ctx.fillStyle = "#f6c453"; ctx.font = "27px 'Segoe UI',sans-serif";
      ctx.fillText("the knee — invisible on a dashboard, obvious here", Math.max(x0 + 20, kneeX - 340), y0 + 90);
      ctx.strokeStyle = "#f6c453"; ctx.lineWidth = 2.5;
      ctx.beginPath(); ctx.moveTo(kneeX, y0 + 104); ctx.lineTo(kneeX, eolY + 22); ctx.stroke();
    }
    prog = Math.min(1, prog + 0.004);
    if (prog < 1 && cur === 1) raf = requestAnimationFrame(draw); else if (prog >= 1) setTimeout(function(){ if (cur === 1){ prog = 0; anim2(); } }, 2400);
  }
  cancelAnimationFrame(raf); draw();
}

/* ---- slide 3: live counters ---- */
setInterval(function(){
  if (cur !== 2) return;
  var s = Date.now() / 1000;
  document.getElementById("liveV").textContent = (4.2 - (s % 6) * 0.03).toFixed(2) + " V";
  document.getElementById("liveI").textContent = (-(2.0 + Math.sin(s) * 0.3)).toFixed(2) + " A";
  document.getElementById("liveT").textContent = (24 + Math.sin(s / 3) * 1.2 + (s % 6) * 0.4).toFixed(1) + " °C";
  document.getElementById("liveC").textContent = (2.0 - (s % 60) * 0.002).toFixed(3) + " Ah";
}, 120);

/* ---- slide 5: the money animation ---- */
var raf5 = 0;
function anim5(){
  var cv = document.getElementById("c5"), ctx = cv.getContext("2d");
  var W = cv.width, H = cv.height, x0 = 100, x1 = W - 420, y0 = 40, y1 = H - 80;
  var prog = 0, parts = [], t = 0, raf = 0;
  function cap(c){ return 1.8565 - 0.0022 * c - 0.5 / (1 + Math.exp(-9 * (c / 118 - 1.15))) * 0.42; }
  var eolC = 0; while (cap(eolC) > 1.4) eolC++;
  function X(c){ return x0 + c / 170 * (x1 - x0); }
  function Y(v){ return y1 - (v - 1.1) / 0.9 * (y1 - y0); }
  function one(){
    ctx.clearRect(0, 0, W, H);
    try {
    var eolY = Y(1.4);
    ctx.strokeStyle = "#ff6b6b"; ctx.lineWidth = 3; ctx.setLineDash([12, 10]);
    ctx.beginPath(); ctx.moveTo(x0, eolY); ctx.lineTo(W - 40, eolY); ctx.stroke(); ctx.setLineDash([]);
    ctx.fillStyle = "#ff6b6b"; ctx.font = "24px ui-monospace,Consolas,monospace";
    ctx.fillText("EOL 70% = 1.40 Ah", x0 + 8, eolY - 12);
    ctx.strokeStyle = "#22314e"; ctx.strokeRect(x0, y0, x1 - x0, y1 - y0);
    // measured capacity up to playhead
    var ph = Math.floor(prog * 100);
    ctx.strokeStyle = "#3ddc97"; ctx.lineWidth = 5; ctx.beginPath();
    for (var c = 0; c <= ph; c++){ c ? ctx.lineTo(X(c), Y(cap(c))) : ctx.moveTo(X(c), Y(cap(c))); }
    ctx.stroke();
    // PI forecast (dark line beyond playhead)
    ctx.strokeStyle = "#4cc3ff"; ctx.lineWidth = 3.5; ctx.beginPath();
    for (var k = 0; k <= 140 - ph; k++){
      var vv = cap(ph) - 0.0019 * k - 0.4 / (1 + Math.exp(-9 * ((ph + k) / 118 - 1.15))) * 0.42 + 0.4 / (1 + Math.exp(-9 * (ph / 118 - 1.15))) * 0.42;
      var xx = X(ph + k), yy = Y(Math.max(1.1, vv));
      k ? ctx.lineTo(xx, yy) : ctx.moveTo(xx, yy);
    }
    ctx.stroke(); ctx.setLineDash([]);
    ctx.fillStyle = "#4cc3ff"; ctx.font = "24px 'Segoe UI',sans-serif";
    ctx.fillText("physics-informed forecast", X(ph) + 14, Y(cap(ph)) - 18);
    // particle futures: each a small line from playhead to its own crossing
    ctx.lineWidth = 2;
    for (var i = 0; i < parts.length; i++){
      var p = parts[i];
      var crossX = X(p.cross), crossY = eolY;
      var slope = (eolY - Y(cap(ph))) / Math.max(p.cross - ph, 1);
      ctx.strokeStyle = "rgba(140,160,190,0.5)";
      ctx.beginPath(); ctx.moveTo(X(ph), Y(cap(ph))); ctx.lineTo(crossX, crossY); ctx.stroke();
      ctx.fillStyle = "rgba(76,195,255,0.85)";
      ctx.fillRect(crossX - 3, crossY - 7, 6, 7);
    }
    // RUL median + interval readout
    if (parts.length){
      var meds = parts.map(function(p){ return p.cross; }).sort(function(a, b){ return a - b; });
      var med = meds[Math.floor(meds.length / 2)], lo = meds[Math.floor(meds.length * 0.025)], hi = meds[Math.floor(meds.length * 0.975)];
      var rulM = med - ph, rulL = Math.max(0, lo - ph), rulH = hi - ph;
      ctx.fillStyle = "#eaf1fb"; ctx.font = "34px ui-monospace,Consolas,monospace";
      var bx = x1 + 30;
      ctx.fillText("RUL median", bx, y0 + 60); ctx.fillStyle = "#3ddc97";
      ctx.fillText(rulM + " cyc", bx, y0 + 110);
      ctx.fillStyle = "#eaf1fb"; ctx.fillText("95% interval", bx, y0 + 180);
      ctx.fillStyle = "#4cc3ff"; ctx.fillText("[" + rulL + ", " + rulH + "]", bx, y0 + 230);
      var alert = rulL <= 10 ? "ALERT" : rulL <= 25 ? "CAUTION" : "OK";
      ctx.fillStyle = alert === "OK" ? "#3ddc97" : alert === "CAUTION" ? "#f6c453" : "#ff6b6b";
      ctx.fillText("● " + alert, bx, y0 + 300);
      ctx.fillStyle = "#93a7c4"; ctx.font = "22px 'Segoe UI',sans-serif";
      ctx.fillText("maintenance rule:", bx, y0 + 360);
      ctx.fillText("lower bound → schedule", bx, y0 + 392);
    }
    // legend
    ctx.fillStyle = "#93a7c4"; ctx.font = "22px 'Segoe UI',sans-serif";
    ctx.fillText("measured capacity", x0 + 10, y1 + 44);
    ctx.fillStyle = "#3ddc97"; ctx.fillRect(x0 - 34, y1 + 30, 22, 5);
    // particle futures spawn on the first frame: panel + futures always visible,
    // even if the browser throttles the animation
    if (parts.length === 0){
      var s = (cycle0 * 7919) | 0;
      for (var j = 0; j < 90; j++){
        s = (s * 1103515245 + 12345) & 0x7fffffff;
        var jitter = ((s % 1000) / 1000 - 0.45) * 55;
        parts.push({ cross: Math.round(Math.min(168, Math.max(ph + 15, eolC + jitter))) });
      }
    }
    } catch (err) {
      ctx.fillStyle = "#ff6b6b"; ctx.font = "26px ui-monospace,Consolas,monospace";
      ctx.fillText("render error: " + err.message, x0, y0 + 40);
    }
    prog += 0.35;
    if (prog >= 100){ prog = 100; }
    if (cur === 4) raf = requestAnimationFrame(one);
  }
  cancelAnimationFrame(raf5); raf5 = requestAnimationFrame(one);
}
var cycle0 = 100;
