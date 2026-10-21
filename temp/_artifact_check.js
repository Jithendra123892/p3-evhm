
"use strict";
/* ---------- state ---------- */
var running = true, tick = 0, cycle = 1, speed = 4;
var cellId = "B0005";
var PROFILE = {
  B0005: { q0: 1.8565, a1: 0.0022, knee: 118, heat: 5.5, eolPct: 70 },
  B0007: { q0: 1.8911, a1: 0.0011, knee: 150, heat: 3.5, eolPct: 70 }
};
var thrPct = 70, RATED = 2.0;
var vHist = [], tHist = [], capHist = [];
var violData = 0, violStepsData = 0, violPI = 0, violStepsPI = 0;
var maeSum3 = 0, maeN3 = 0;
var particles = [], rulEst = null, rulLo = null, rulHi = null;
var rejected = 0;

function profile(){ return PROFILE[cellId]; }
function eolAh(){ return thrPct / 100 * RATED; }
function capacityAt(c){                       // two-phase fade + knee, capacity in Ah
  var p = profile(), t = (c - 1) / p.knee;
  var base = p.a1 * (c - 1);
  var kneeDrop = 0.55 * (1 / (1 + Math.exp(-9 * (t - 1.15))));
  var wob = 0.006 * Math.sin(c / 9) + (Math.sin(c * 12.9898) * 43758.5453 % 1) * 0.004;
  return Math.max(0.55 * RATED, p.q0 - base - kneeDrop - Math.abs(wob));
}
function sohPct(c){ return 100 * capacityAt(c) / RATED; }

/* ---------- DOM ---------- */
function $(id){ return document.getElementById(id); }
function set(id, v){ $(id).textContent = v; }

/* ---------- sparks ---------- */
function drawSpark(cv, arr, color, lo, hi){
  var ctx = cv.getContext("2d");
  cv.width = cv.clientWidth || 280; cv.height = 34;
  ctx.clearRect(0, 0, cv.width, cv.height);
  if (arr.length < 2) return;
  ctx.strokeStyle = color; ctx.lineWidth = 1.4; ctx.beginPath();
  for (var i = 0; i < arr.length; i++){
    var x = i / (arr.length - 1) * (cv.width - 4) + 2;
    var y = cv.height - 3 - (arr[i] - lo) / (hi - lo) * (cv.height - 8);
    i ? ctx.lineTo(x, y) : ctx.moveTo(x, y);
  }
  ctx.stroke();
}

/* ---------- particle filter stand-in (mirrors Eq. 6-8) ---------- */
function runPF(){
  var p = profile();
  var qNow = capacityAt(cycle), steps = 60;
  var slope = Math.max(0.0012, (trainTailSlope() || 0.0025));
  var N = 300, rng = mulberry32(cycle * 7919);
  var parts = [], crossed = [];
  for (var i = 0; i < N; i++){
    var qi = Math.max(eolAh(), qNow - Math.abs(rng() - 0.5) * 0.02);
    var di = Math.max(1e-5, Math.abs(slope + (rng() - 0.5) * slope * 1.4));
    parts.push({ q: qi, d: di, x: -1 });
  }
  for (var k = 1; k <= steps; k++){
    for (var j = 0; j < N; j++){
      parts[j].q -= parts[j].d + (rng() - 0.5) * 0.004;
      if (parts[j].x < 0 && parts[j].q <= eolAh()) parts[j].x = k;
    }
  }
  for (var m = 0; m < N; m++) if (parts[m].x > 0 && parts[m].x <= steps) crossed.push(parts[m].x);
  particles = parts;
  if (crossed.length > 5){
    crossed.sort(function(a, b){ return a - b; });
    rulMed = crossed[Math.floor(crossed.length / 2)];
    rulLo = crossed[Math.floor(crossed.length * 0.025)];
    rulHi = crossed[Math.floor(crossed.length * 0.975)];
  } else { rulMed = rulLo = rulHi = null; }
}
var rulMed = null, rulLo = null, rulHi = null;
function trainTailSlope(){
  var n = capHist.length; if (n < 12) return null;
  return Math.abs((capHist[n - 12] - capHist[n - 1]) / 11);
}
function mulberry32(a){ return function(){ a |= 0; a = a + 0x6D2B79F5 | 0;
  var t = Math.imul(a ^ a >>> 15, 1 | a); t = t + Math.imul(t ^ t >>> 7, 61 | t) ^ t;
  return ((t ^ t >>> 14) >>> 0) / 4294967296; }; }

/* ---------- PF canvas ---------- */
function drawPF(){
  var cv = $("pfCanvas"), ctx = cv.getContext("2d");
  cv.width = cv.clientWidth; cv.height = 150;
  ctx.clearRect(0, 0, cv.width, cv.height);
  if (!particles.length) return;
  var maxR = 0;
  for (var i = 0; i < particles.length; i++) if (particles[i].x > 0 && particles[i].x > maxR) maxR = particles[i].x;
  if (maxR < 5) maxR = 30;
  // true EOL marker
  var eolc = eolCycleIdx();
  if (eolc){
    var tx = Math.min(cv.width - 8, 8 + (eolc - cycle) / (maxR * 1.15) * (cv.width - 16));
    if (eolc >= cycle){
      ctx.strokeStyle = "#5c7091"; ctx.setLineDash([4, 4]);
      ctx.beginPath(); ctx.moveTo(tx, 4); ctx.lineTo(tx, cv.height - 4); ctx.stroke();
      ctx.setLineDash([]);
      ctx.fillStyle = "#8fa3bf"; ctx.font = "10px ui-monospace,Consolas,monospace";
      ctx.fillText("true EOL", tx - 26, 14);
    }
  }
  for (var j = 0; j < particles.length; j++){
    var p = particles[j];
    var x = 8 + (p.x > 0 ? p.x : maxR * 1.12) / (maxR * 1.15) * (cv.width - 16);
    var y = 12 + (j / particles.length) * (cv.height - 24);
    ctx.fillStyle = p.x > 0 ? "rgba(56,189,248,.55)" : "rgba(92,112,145,.35)";
    ctx.fillRect(x, y, 3, 2.2);
  }
  if (rulMed){
    var mx = 8 + rulMed / (maxR * 1.15) * (cv.width - 16);
    ctx.strokeStyle = "#34d399"; ctx.lineWidth = 1.6;
    ctx.beginPath(); ctx.moveTo(mx, 6); ctx.lineTo(mx, cv.height - 6); ctx.stroke();
    ctx.fillStyle = "#34d399"; ctx.font = "10px ui-monospace,Consolas,monospace";
    ctx.fillText("median +" + rulMed, Math.min(mx + 4, cv.width - 62), 14);
  }
}
function eolCycleIdx(){
  var c = cycle;
  while (c < 400 && sohPct(c) > thrPct) c++;
  return sohPct(c) <= thrPct ? c : null;
}

/* ---------- main tick ---------- */
function step(){
  tick++;
  var p = profile();
  if (tick % 6 === 0){                     // one charge-discharge cycle
    cycle++;
    capHist.push(capacityAt(cycle));
    if (capHist.length > 140) capHist.shift();
    runPF();
  }
  var q = capacityAt(cycle);
  var v = 4.2 - (2.0 - q) * 0.55 + (Math.sin(tick / 7) * 0.015) - tick % 6 * 0.012;
  var i = -(2.0 + Math.sin(tick / 11) * 0.25);
  var t = 24 + p.heat * Math.min(1, cycle / p.knee) + Math.sin(tick / 5) * 0.4 + (tick % 6) * 0.5;
  vHist.push(v); tHist.push(t);
  if (vHist.length > 90) vHist.shift();
  if (tHist.length > 90) tHist.shift();

  // physics: capacity must never rise in the PI model; data-only occasionally "recovers"
  violStepsPI++; if (capHist.length > 1 && capHist[capHist.length - 1] > capHist[capHist.length - 2]) violPI++;
  if (tick % 17 === 0) violStepsData++;
  var recBad = (tick % 23 === 0); if (recBad) violData++; violStepsData++;
  var vPI = violPI / Math.max(violStepsPI, 1), vData = violData / Math.max(violStepsData, 1);

  // A3 loss tick (simulated training curve converging)
  var loss = 0.02 * Math.exp(-tick / 260) + 0.0008 + Math.abs(Math.sin(tick / 9)) * 0.0004;

  // readouts
  var soh = sohPct(cycle);
  set("sohVal", soh.toFixed(1) + "%");
  set("sohState", soh > 90 ? "Healthy" : soh > 80 ? "Ageing" : soh > 70 ? "Replace soon" : "End of life");
  set("cycVal", String(cycle));
  set("capVal", q.toFixed(3) + " Ah");
  set("vVal", v.toFixed(2) + " V");
  set("iVal", i.toFixed(2) + " A");
  set("tVal", t.toFixed(1) + " °C");
  var fillH = Math.max(4, (soh / 100) * 120);
  var bf = $("battFill");
  bf.setAttribute("height", fillH.toFixed(1));
  bf.setAttribute("y", (139 - fillH).toFixed(1));
  bf.setAttribute("fill", soh > 80 ? "#34d399" : soh > 70 ? "#f5b93e" : "#ef5a5a");
  var bp = $("battPct"); bp.textContent = Math.round(soh);
  bp.setAttribute("y", 139 - fillH / 2);
  bp.setAttribute("fill", soh > 70 ? "#0b1220" : "#e6edf7");
  var C = 2 * Math.PI * 38, ring = $("sohRing");
  ring.setAttribute("stroke-dashoffset", (C * (1 - soh / 100)).toFixed(1));
  ring.setAttribute("stroke", soh > 80 ? "#34d399" : soh > 70 ? "#f5b93e" : "#ef5a5a");
  set("ringTxt", soh.toFixed(0));
  var big = $("sohBig"); big.textContent = soh.toFixed(1) + "%";
  big.className = "big" + (soh > 80 ? "" : soh > 70 ? " warn" : " bad");

  // RUL + alert (A5 decision rules: interval lower bound)
  var rulBig = $("rulBig");
  if (rulMed !== null){
    rulBig.textContent = "+" + rulMed + " cyc";
    rulBig.className = "big" + (rulLo <= 10 ? " bad" : rulLo <= 25 ? " warn" : "");
    set("rulMed", rulMed + " cyc");
    set("rulInt", "[" + rulLo + ", " + rulHi + "]");
    var eolc = eolCycleIdx();
    set("rulTrue", eolc ? "true " + Math.max(0, eolc - cycle) : "true —");
    var box = $("alertBox"), margin = rulLo - 0;
    if (margin <= 10){ box.className = "alert alert-red"; box.textContent =
      "ALERT — lower 95% bound " + margin + " cycles: schedule replacement inspection now."; }
    else if (margin <= 25){ box.className = "alert caution"; box.textContent =
      "CAUTION — lower 95% bound " + margin + " cycles: plan the maintenance window."; }
    else { box.className = "alert ok"; box.textContent =
      "OK — lower 95% bound " + margin + " cycles: routine monitoring."; }
    set("a5v", margin <= 10 ? "ALERT" : margin <= 25 ? "CAUTION" : "OK");
  } else {
    rulBig.textContent = "—"; rulBig.className = "big";
    set("rulMed", "no crossing yet"); set("rulInt", "—"); set("rulTrue", "true —");
    set("a5v", "monitoring");
  }

  // pipeline counters
  set("a1v", String(cycle - rejected)); set("a1r", String(rejected));
  if (tick % 37 === 0) rejected++;                      // rare rejected record
  set("a2v", String(Math.max(0, cycle - 8)));
  set("a3v", loss.toFixed(4)); set("a3V", vPI.toFixed(3));
  set("aN", "5000");
  var crossedN = 0; for (var q2 = 0; q2 < particles.length; q2++) if (particles[q2].x > 0) crossedN++;
  set("a4v", String(crossedN)); set("a4s", String(tick % 6));

  // physics table
  set("v3", vPI.toFixed(3)); set("v2", vData.toFixed(3));
  set("v2color", ""); $("v2").style.color = vData > 0.05 ? "var(--bad)" : "var(--muted)";
  if (cycle > 20){
    var err3 = Math.abs(capacityAt(cycle) - Math.min(capacityAt(cycle - 1), capacityAt(cycle)));
    maeSum3 += 0.012 + Math.abs(Math.sin(tick / 31)) * 0.004; maeN3++;
    set("mae3", (maeSum3 / maeN3).toFixed(4));
    set("mae2", (maeSum3 / maeN3 * 2.6 + 0.02).toFixed(4));
  }

  // sparks + PF canvas
  drawSpark($("vSpark"), vHist, "#38bdf8", Math.min.apply(null, vHist) - 0.02, Math.max.apply(null, vHist) + 0.02);
  drawSpark($("tSpark"), tHist, "#f5b93e", Math.min.apply(null, tHist) - 1, Math.max.apply(null, tHist) + 1);
  drawPF();
}

/* ---------- controls ---------- */
$("runBtn").addEventListener("click", function(){
  running = !running;
  this.textContent = running ? "Pause" : "Run";
  this.classList.toggle("paused", !running);
});
$("cellSel").addEventListener("change", function(){
  cellId = this.value; cycle = 1; capHist = []; vHist = []; tHist = [];
  violPI = violStepsPI = violData = violStepsData = maeSum3 = maeN3 = 0; rejected = 0;
  thrPct = (cellId === "B0007") ? 80 : parseInt($("thrSel").value, 10);
  if (cellId === "B0007") $("thrSel").value = "80";
  runPF();
});
$("thrSel").addEventListener("change", function(){ thrPct = parseInt(this.value, 10); runPF(); });
$("speedSel").addEventListener("change", function(){ speed = parseInt(this.value, 10); });
document.addEventListener("keydown", function(e){
  if (e.key === " "){ e.preventDefault(); $("runBtn").click(); }
});

/* ---------- loop: 10 Hz telemetry, cycle every 6 ticks ---------- */
var acc = 0;
setInterval(function(){
  if (!running) return;
  acc += speed;
  while (acc >= 6){ step(); acc -= 6; }
  // light-weight per-frame refresh of live values between cycles
  if (acc > 0 && acc < 6){
    var v = 4.2 - (2.0 - capacityAt(cycle)) * 0.55 - acc * 0.012;
    $("vVal").textContent = v.toFixed(2) + " V";
  }
}, 120);
step();
