
"use strict";
var W = 1920, H = 1080;
var cv = document.getElementById("cv"), ctx = cv.getContext("2d");
var STAGE = document.getElementById("stage");

/* ---------- timeline (all state is a pure function of T) ---------- */
var SCENES = [
  { t: 0,  cap: "An EV battery. Day after day." },
  { t: 13, cap: "Every cycle drains a little. Invisible." },
  { t: 31, cap: "Then the knee. <b>No warning.</b>" },
  { t: 43, cap: "<b>P3-EVHM</b> forecasts the curve — and warns early." },
  { t: 59, cap: "Breakdown vs planned maintenance." },
  { t: 71, cap: "See the cliff before it comes." },
];
var TOTAL = 80;
function sceneIdx(T){ var i = 0; for (var j = 0; j < SCENES.length; j++) if (T >= SCENES[j].t) i = j; return i; }

/* ---------- the cell model (same physics as the project) ---------- */
var RATED = 2.0, EOL_AH = 1.40;
function capAt(c){ return 1.8565 - 0.0022 * c - 0.42 / (1 + Math.exp(-9 * (c / 118 - 1.15))); }
var EOL_CYC = 0; while (capAt(EOL_CYC) > EOL_AH) EOL_CYC++;   // ~145
var DEATH_CYC = EOL_CYC + 3;                                    // car gives up just past EOL
function cycleAt(T){
  if (T < 13) return T;                                // s1: first cycles
  if (T < 31) return 12 + (T - 13) * 4.9;              // s2: time-lapse to ~100
  if (T < 43) return 110 + (T - 31) * 4.1;             // s3: through the knee
  if (T < 59){                                         // s4: rewind, then re-run WITH the model
    var w = Math.min(1, (T - 43) / 2.2);
    var rew = 110 + (DEATH_CYC - 110) * (1 - easeInOut(w)); // 158 -> 110
    return w < 1 ? 90 + (rew - 90) : 90 + (T - 45.2) * 4.4;
  }
  if (T < 71) return 150;                              // s5: split screens
  return 150;                                          // end card
}
function easeInOut(p){ return p < .5 ? 2 * p * p : 1 - Math.pow(-2 * p + 2, 2) / 2; }

/* ---------- palette ---------- */
var C = { bg:"#0b1220", sky:"#0d1626", road:"#141f33", line:"#22314e", ink:"#eaf1fb",
  muted:"#93a7c4", dim:"#42557a", acc:"#3ddc97", data:"#4cc3ff", warn:"#f6c453", bad:"#ff6b6b" };
var MONO = "24px ui-monospace,Consolas,monospace";
var MONO_S = "20px ui-monospace,Consolas,monospace";
var SANS = "30px 'Segoe UI',system-ui,sans-serif";

function rr(x, y, w, h, r){ ctx.beginPath(); ctx.roundRect(x, y, w, h, r); }
ctx.roundRect = function(x, y, w, h, r){
  ctx.beginPath(); ctx.moveTo(x + r, y);
  ctx.arcTo(x + w, y, x + w, y + h, r); ctx.arcTo(x + w, y + h, x, y + h, r);
  ctx.arcTo(x, y + h, x, y, r); ctx.arcTo(x, y, x + w, y, r); ctx.closePath();
};

/* ---------- world drawing ---------- */
function sky(T){
  ctx.fillStyle = C.sky; ctx.fillRect(0, 0, W, 600);
  var s = sceneIdx(T);
  var dayT = (T * 0.6) % 4;                      // drifting clouds
  ctx.fillStyle = "#131f36";
  for (var i = 0; i < 3; i++){
    var cx = ((i * 700 + dayT * 160) % 2400) - 300, cy = 90 + i * 40;
    ctx.beginPath(); ctx.ellipse(cx, cy, 150, 36, 0, 0, 7); ctx.fill();
  }
  if (s === 2 || s === 3){                       // trouble: dim sun, haze
    ctx.fillStyle = "rgba(255,107,107,0.06)"; ctx.fillRect(0, 0, W, 600);
  }
  ctx.fillStyle = "#26375a"; ctx.beginPath(); ctx.arc(1600, 110, 46, 0, 7); ctx.fill();
}
function road(T, speedMul, dimmed){
  ctx.fillStyle = dimmed ? "#101a2c" : C.road; ctx.fillRect(0, 560, W, 60);
  ctx.fillStyle = dimmed ? "#1a2740" : "#2c3f66";
  var off = (T * 260 * speedMul) % 160;
  for (var x = -160 + off; x < W; x += 160) ctx.fillRect(x, 588, 80, 5);
  ctx.strokeStyle = C.line; ctx.lineWidth = 2;
  ctx.beginPath(); ctx.moveTo(0, 558); ctx.lineTo(W, 558); ctx.moveTo(0, 622); ctx.lineTo(W, 622); ctx.stroke();
}
function car(x, y, sc, wheelRot, alive, dimmed, tilt){
  ctx.save(); ctx.translate(x, y); ctx.scale(sc, sc); ctx.rotate(tilt || 0);
  ctx.globalAlpha = dimmed ? 0.55 : 1;
  var body = alive ? "#dfe8f5" : "#8795ab";
  // headlight beam
  if (alive){
    ctx.fillStyle = "rgba(246,196,83,0.14)";
    ctx.beginPath(); ctx.moveTo(128, -18); ctx.lineTo(230, 4); ctx.lineTo(230, 30); ctx.lineTo(128, 4); ctx.fill();
  }
  // body
  ctx.fillStyle = body;
  ctx.beginPath();
  ctx.moveTo(-130, 0); ctx.lineTo(-128, -34);
  ctx.bezierCurveTo(-90, -38, -80, -40, -66, -58);
  ctx.bezierCurveTo(-40, -76, 30, -78, 52, -60);
  ctx.bezierCurveTo(64, -48, 74, -42, 96, -40);
  ctx.bezierCurveTo(120, -38, 128, -26, 128, 0);
  ctx.closePath(); ctx.fill();
  // windows
  ctx.fillStyle = alive ? "#122038" : "#0c1526";
  ctx.beginPath(); ctx.moveTo(-58, -56); ctx.lineTo(-16, -70); ctx.lineTo(-16, -42); ctx.lineTo(-58, -42); ctx.closePath(); ctx.fill();
  ctx.beginPath(); ctx.moveTo(-8, -70); ctx.lineTo(38, -64); ctx.bezierCurveTo(48, -60, 54, -54, 56, -42); ctx.lineTo(-8, -42); ctx.closePath(); ctx.fill();
  // lights
  ctx.fillStyle = alive ? "#ffe9a8" : "#5a4a3a";
  ctx.beginPath(); ctx.arc(122, -12, 7, 0, 7); ctx.fill();
  ctx.fillStyle = alive ? "#ff8f6b" : "#4a3535";
  ctx.beginPath(); ctx.arc(-124, -12, 6, 0, 7); ctx.fill();
  // battery pack under floor
  ctx.fillStyle = alive ? C.acc : "#31513f";
  ctx.fillRect(-70, 2, 140, 14);
  ctx.strokeStyle = "#0e1726"; ctx.lineWidth = 2; ctx.strokeRect(-70, 2, 140, 14);
  // wheels
  for (var i = 0; i < 2; i++){
    var wx = i ? 82 : -82;
    ctx.fillStyle = "#0d1526"; ctx.beginPath(); ctx.arc(wx, 14, 26, 0, 7); ctx.fill();
    ctx.strokeStyle = "#33456b"; ctx.lineWidth = 5; ctx.stroke();
    ctx.save(); ctx.translate(wx, 14); ctx.rotate(wheelRot);
    ctx.strokeStyle = "#4d6390"; ctx.lineWidth = 4;
    for (var k = 0; k < 4; k++){ ctx.rotate(Math.PI / 2); ctx.beginPath(); ctx.moveTo(0, 0); ctx.lineTo(0, -18); ctx.stroke(); }
    ctx.restore();
  }
  if (!alive){                                    // dead marks
    ctx.fillStyle = C.bad; ctx.font = "bold 44px 'Segoe UI',sans-serif"; ctx.textAlign = "center";
    ctx.fillText("×", -82, -60); ctx.fillText("×", 82, -60);
    ctx.fillStyle = C.bad; ctx.beginPath();
    ctx.moveTo(0, -96); ctx.lineTo(26, -62); ctx.lineTo(-26, -62); ctx.closePath(); ctx.fill();
    ctx.fillStyle = "#0b1220"; ctx.font = "bold 34px 'Segoe UI',sans-serif"; ctx.fillText("!", 0, -70);
  }
  ctx.restore(); ctx.globalAlpha = 1; ctx.textAlign = "left";
}

/* ---------- health monitor panel ---------- */
var PANEL = { x: 200, y: 700, w: 1520, h: 280 };
function panel(T, mode){
  var p = PANEL;
  ctx.fillStyle = "#0e1729"; ctx.strokeStyle = C.line; ctx.lineWidth = 2;
  ctx.roundRect(p.x - 20, p.y - 20, p.w + 40, p.h + 40, 16); ctx.fill(); ctx.stroke();
  ctx.fillStyle = C.dim; ctx.font = MONO_S;
  ctx.fillText("battery health — live", p.x - 4, p.y - 34);
  ctx.fillStyle = C.bad; ctx.textAlign = "right";
  ctx.fillText("end of life 70%", p.x + p.w + 4, p.y - 34); ctx.textAlign = "left";
  var X = function(c){ return p.x + c / 170 * p.w; };
  var Y = function(v){ return p.y + p.h - (v - 1.1) / 0.85 * p.h; };
  // grid + EOL line
  ctx.strokeStyle = "#1a2740"; ctx.lineWidth = 1;
  for (var g = 0; g <= 5; g++){ ctx.beginPath(); ctx.moveTo(p.x, p.y + g * p.h / 5); ctx.lineTo(p.x + p.w, p.y + g * p.h / 5); ctx.stroke(); }
  ctx.strokeStyle = C.bad; ctx.lineWidth = 3; ctx.setLineDash([14, 10]);
  ctx.beginPath(); ctx.moveTo(p.x, Y(EOL_AH)); ctx.lineTo(p.x + p.w, Y(EOL_AH)); ctx.stroke(); ctx.setLineDash([]);
  var cyc = cycleAt(T);
  // measured curve to playhead
  ctx.strokeStyle = C.acc; ctx.lineWidth = 5; ctx.beginPath();
  for (var c = 0; c <= cyc; c++){
    var x = X(c), y = Y(capAt(c));
    c ? ctx.lineTo(x, y) : ctx.moveTo(x, y);
  }
  ctx.stroke();
  if (mode === "fan" && cyc >= 90){                 // the model's forecast fan
    var s = (cyc * 7919) | 0, first = true;
    ctx.lineWidth = 1.6;
    for (var j = 0; j < 26; j++){
      s = (s * 1103515245 + 12345) & 0x7fffffff;
      var cross = Math.round(Math.min(170, Math.max(cyc + 10, EOL_CYC + ((s % 1000) / 1000 - 0.45) * 50)));
      var slope = (Y(EOL_AH) - Y(capAt(cyc))) / Math.max(cross - cyc, 1);
      ctx.strokeStyle = "rgba(76,195,255,0.30)";
      ctx.beginPath(); ctx.moveTo(X(cyc), Y(capAt(cyc))); ctx.lineTo(X(cross), Y(EOL_AH)); ctx.stroke();
      ctx.fillStyle = "rgba(76,195,255,0.85)"; ctx.fillRect(X(cross) - 3, Y(EOL_AH) - 7, 6, 7);
    }
    ctx.strokeStyle = C.data; ctx.lineWidth = 3;
    ctx.beginPath(); ctx.moveTo(X(cyc), Y(capAt(cyc))); ctx.lineTo(X(EOL_CYC), Y(EOL_AH)); ctx.stroke();
  }
  if (mode === "after"){                            // crossed: mark the failure
    ctx.fillStyle = C.bad; ctx.font = "bold 26px 'Segoe UI',sans-serif";
    ctx.fillText("battery failed", X(EOL_CYC) - 170, Y(EOL_AH) + 40);
  }
  // playhead marker + readouts
  var soh = 100 * capAt(cyc) / RATED;
  ctx.fillStyle = C.ink; ctx.beginPath(); ctx.arc(X(Math.min(cyc, 170)), Y(capAt(Math.min(cyc, 170))), 7, 0, 7); ctx.fill();
  ctx.font = MONO; ctx.fillStyle = C.muted;
  ctx.fillText("cycle " + Math.floor(cyc), p.x + p.w - 300, p.y + p.h + 64);
  ctx.fillStyle = soh > 75 ? C.acc : soh >= 70 ? C.warn : C.bad;
  ctx.fillText("SoH " + soh.toFixed(1) + "%", p.x + p.w - 140, p.y + p.h + 64);
  ctx.strokeStyle = C.line; ctx.strokeRect(p.x, p.y, p.w, p.h);
}

/* ---------- overlays ---------- */
function modelChip(T, on){
  if (!on) return;
  var cx = 960, cy = 250;
  ctx.strokeStyle = C.data; ctx.lineWidth = 3;
  ctx.roundRect(cx - 200, cy - 62, 400, 124, 18); ctx.stroke();
  ctx.fillStyle = "#101d33"; ctx.fill();
  ctx.fillStyle = C.data; ctx.font = "bold 30px 'Segoe UI',sans-serif"; ctx.textAlign = "center";
  ctx.fillText("P3-EVHM", cx, cy - 12);
  ctx.fillStyle = C.muted; ctx.font = "20px 'Segoe UI',sans-serif";
  ctx.fillText("watches the live stream · forecasts the curve", cx, cy + 22);
  ctx.textAlign = "left";
  // data pulses from car battery up to the chip
  for (var i = 0; i < 3; i++){
    var ph = ((T * 0.9 + i / 3) % 1);
    var px = cx + (ph - 0.5) * 0, py = 540 - ph * (540 - (cy + 62));
    var x0 = cx + (i - 1) * 40;
    ctx.fillStyle = "rgba(76,195,255," + (0.9 - ph * 0.5) + ")";
    ctx.beginPath();
    ctx.moveTo(x0 + ph * (cx - x0), 540 - ph * (540 - cy - 62));
    ctx.arc(x0 + ph * (cx - x0), 540 - ph * (540 - cy - 62), 6, 0, 7);
    ctx.fill();
  }
}
function badge(T, cyc){
  var rulL = Math.max(0, EOL_CYC - 26 - cyc);
  if (rulL > 40) return;
  var late = rulL <= 10;
  ctx.strokeStyle = late ? C.bad : C.warn; ctx.lineWidth = 3;
  ctx.roundRect(1210, 120, 500, 96, 14); ctx.stroke();
  ctx.fillStyle = late ? "rgba(255,107,107,0.08)" : "rgba(246,196,83,0.08)"; ctx.fill();
  ctx.fillStyle = late ? C.bad : C.warn; ctx.font = "bold 32px 'Segoe UI',sans-serif";
  ctx.fillText(late ? "SERVICE NOW" : "PLAN SERVICE", 1240, 158);
  ctx.fillStyle = C.ink; ctx.font = MONO;
  ctx.fillText("~" + Math.max(0, Math.round(EOL_CYC - cyc)) + " cycles left · interval [" + Math.round(rulL) + ", " + Math.round(EOL_CYC + 24 - cyc) + "]", 1240, 196);
}
function toast(txt, col, T, x, y){
  var a = Math.min(1, T); if (a <= 0) return;
  ctx.globalAlpha = a; ctx.strokeStyle = col; ctx.lineWidth = 3;
  ctx.roundRect(x, y, 560, 84, 14); ctx.stroke();
  ctx.fillStyle = "rgba(10,18,32,0.85)"; ctx.fill();
  ctx.fillStyle = col; ctx.font = "bold 26px 'Segoe UI',sans-serif"; ctx.fillText(txt, x + 26, y + 52);
  ctx.globalAlpha = 1;
}

/* ---------- scenes ---------- */
function sceneSplit(T){
  var p = Math.min(1, (T - 59) / 2.2), w = easeInOut(p) * 960;
  // LEFT: without us — dead
  ctx.save(); ctx.beginPath(); ctx.rect(0, 0, w, 1080); ctx.clip();
  ctx.fillStyle = "rgba(255,107,107,0.05)"; ctx.fillRect(0, 0, w, 1080);
  sky(T); road(T, 0, true);
  car(380, 520, 0.9, 0, false, false, -0.04);
  miniCurve(0, w, "after");
  ctx.fillStyle = C.bad; ctx.font = "bold 40px 'Segoe UI',sans-serif"; ctx.textAlign = "center";
  ctx.fillText("without P3-EVHM", w / 2, 100); ctx.textAlign = "left";
  toast("breakdown — unplanned", C.bad, 1, 140, 690);
  ctx.restore();
  // RIGHT: with us — alive, warned early
  ctx.save(); ctx.beginPath(); ctx.rect(w, 0, 1920 - w, 1080); ctx.clip();
  ctx.translate(w, 0);
  sky(T); road(T, 1, false);
  car(620, 520, 0.9, T * 5, true, false, 0);
  miniCurve(0, 960, "fan");
  ctx.fillStyle = C.acc; ctx.font = "bold 40px 'Segoe UI',sans-serif"; ctx.textAlign = "center";
  ctx.fillText("with P3-EVHM", 480, 100); ctx.textAlign = "left";
  toast("service scheduled — planned", C.acc, 1, 180, 690);
  ctx.restore();
}
function miniCurve(off, w, mode){
  var p = { x: off + 90, y: 740, w: w - 180, h: 220 };
  ctx.fillStyle = "#0e1729"; ctx.strokeStyle = C.line; ctx.lineWidth = 2;
  ctx.roundRect(p.x - 14, p.y - 14, p.w + 28, p.h + 28, 14); ctx.fill(); ctx.stroke();
  var X = function(c){ return p.x + c / 170 * p.w; };
  var Y = function(v){ return p.y + p.h - (v - 1.1) / 0.85 * p.h; };
  ctx.strokeStyle = C.bad; ctx.lineWidth = 2.5; ctx.setLineDash([10, 8]);
  ctx.beginPath(); ctx.moveTo(p.x, Y(EOL_AH)); ctx.lineTo(p.x + p.w, Y(EOL_AH)); ctx.stroke(); ctx.setLineDash([]);
  ctx.strokeStyle = mode === "fan" ? C.acc : C.bad; ctx.lineWidth = 4; ctx.beginPath();
  for (var c = 0; c <= (mode === "fan" ? 150 : 170); c++){
    var x = X(c), y = Y(capAt(c)); c ? ctx.lineTo(x, y) : ctx.moveTo(x, y);
  }
  ctx.stroke();
  if (mode === "fan"){
    ctx.strokeStyle = C.data; ctx.lineWidth = 2.5; ctx.setLineDash([6, 6]);
    ctx.beginPath(); ctx.moveTo(X(150), Y(capAt(150))); ctx.lineTo(X(EOL_CYC), Y(EOL_AH)); ctx.stroke(); ctx.setLineDash([]);
    ctx.fillStyle = C.data; ctx.beginPath(); ctx.arc(X(EOL_CYC), Y(EOL_AH), 6, 0, 7); ctx.fill();
    ctx.fillStyle = C.muted; ctx.font = "20px 'Segoe UI',sans-serif";
    ctx.fillText("warned ~35 cycles early", p.x + 40, p.y + 60);
  } else {
    ctx.fillStyle = C.bad; ctx.beginPath(); ctx.arc(X(EOL_CYC), Y(EOL_AH), 8, 0, 7); ctx.fill();
    ctx.fillStyle = C.bad; ctx.font = "20px 'Segoe UI',sans-serif";
    ctx.fillText("first sign: the failure", p.x + 40, p.y + 60);
  }
}
function sceneEnd(T){
  ctx.fillStyle = "#070d18"; ctx.fillRect(0, 0, W, H);
  var p = Math.min(1, (T - 71) / 2);
  ctx.globalAlpha = p;
  ctx.fillStyle = C.ink; ctx.font = "bold 74px 'Segoe UI',sans-serif"; ctx.textAlign = "center";
  ctx.fillText("See the cliff before it comes.", 960, 300);
  ctx.fillStyle = C.muted; ctx.font = "32px 'Segoe UI',sans-serif";
  ctx.fillText("P3-EVHM — physics-informed prognostics for EV battery health", 960, 372);
  var stats = [
    ["NASA PCoE", "real cell data, two byte-verified sources"],
    ["0.366 → 0.076", "implausible predictions eliminated by physics constraints"],
    ["16.5×", "GPU particle engine, measured at 500k particles"],
    ["20/20", "RUL estimates produced across both machines"],
  ];
  ctx.textAlign = "center";
  for (var i = 0; i < 4; i++){
    var x = 300 + i * 440;
    ctx.fillStyle = C.acc; ctx.font = "bold 52px " + MONO; ctx.fillText(stats[i][0], x, 560);
    ctx.fillStyle = C.muted; ctx.font = "24px 'Segoe UI',sans-serif";
    var words = stats[i][1].split(" "), lines = [words.slice(0, 3).join(" "), words.slice(3).join(" ")];
    ctx.fillText(lines[0], x, 614); ctx.fillText(lines[1], x, 650);
  }
  ctx.fillStyle = C.dim; ctx.font = "26px " + MONO;
  ctx.fillText("github.com/Jithendra123892/p3-evhm — MIT", 960, 800);
  ctx.globalAlpha = 1; ctx.textAlign = "left";
}

/* ---------- master render (pure function of T) ---------- */
function render(T){
  ctx.clearRect(0, 0, W, H);
  var s = sceneIdx(T);
  if (s === 5){ sceneEnd(T); }
  else if (s === 4){ sceneSplit(T); }
  else {
    sky(T);
    var cyc = cycleAt(T);
    var alive = (s === 3) ? true : cyc < DEATH_CYC;   // s4: car was serviced — stays alive
    var speed = alive ? 1 : 0;
    road(T, speed, !alive);
    car(760, 520, 1.15, T * 5 * speed, alive, false, alive ? 0 : 0.02);
    panel(T, s === 3 ? "fan" : s === 2 ? "after" : "plain");
    if (s === 3){
      modelChip(T, T > 47);
      badge(T, cyc);
      if (cyc > 118) toast("maintenance booked — battery replaced on schedule", C.acc, Math.min(1, (cyc - 118) / 6), 1210, 660);
    }
    if (s === 2){
      ctx.fillStyle = C.bad; ctx.font = "bold 42px 'Segoe UI',sans-serif"; ctx.textAlign = "center";
      ctx.globalAlpha = Math.min(1, (T - 38) / 3);
      ctx.fillText("the driver's first warning is the failure itself", 960, 160);
      ctx.globalAlpha = 1; ctx.textAlign = "left";
    }
  }
  // caption + hud
  var capEl = document.getElementById("cap");
  capEl.innerHTML = SCENES[s].cap;
  var soh = 100 * capAt(cycleAt(T)) / RATED;
  document.getElementById("meta").innerHTML =
    (s === 5 || s === 4) ? "" : "cycle " + Math.floor(cycleAt(T)) + " · <span class=\"soh\">SoH " + soh.toFixed(1) + "%</span>";
  var frac = T / TOTAL;
  document.getElementById("seekfill").style.width = (frac * 100) + "%";
  document.getElementById("seekdot").style.left = (frac * 100) + "%";
}

/* ---------- playback ---------- */
var T = 0, last = 0, playing = true, raf = 0, visible = false;
var m = (location.hash || "").match(/t=(\d+(\.\d+)?)/);
if (m) T = Math.min(parseFloat(m[1]), TOTAL - 0.01);   // deep-link: open at scene time
function tick(ts){
  var dt = last ? Math.min((ts - last) / 1000, 0.25) : 0;
  last = ts;
  if (playing) T += dt;
  if (T >= TOTAL) T = 0;                       // loop — presentation mode
  render(T);
  if (visible) raf = requestAnimationFrame(tick);
}
function fit(){
  var s = Math.min(innerWidth / W, innerHeight / H);
  STAGE.style.transform = "translate(-50%,-50%) scale(" + s + ")";
}
addEventListener("resize", fit); fit();
document.addEventListener("keydown", function(e){
  if (e.key === " "){ playing = !playing; e.preventDefault(); }
  if (e.key === "ArrowRight"){ T = SCENES[Math.min(sceneIdx(T) + 1, 5)].t + 0.01; }
  if (e.key === "ArrowLeft"){ T = SCENES[Math.max(sceneIdx(T) - 1, 0)].t + 0.01; }
  if (e.key === "r" || e.key === "R"){ T = 0; }
});
document.getElementById("seek").addEventListener("click", function(e){
  var r = this.getBoundingClientRect();
  var fx = (e.clientX - r.left) / r.width;
  var st = Math.max(innerWidth / W, innerHeight / H);
  T = fx * TOTAL;
});
new IntersectionObserver(function(e){ visible = e[e.length - 1].isIntersecting;
  if (visible){ last = 0; cancelAnimationFrame(raf); raf = requestAnimationFrame(tick); }
}, { rootMargin: "140px" }).observe(STAGE);
render(T);
visible = true; last = 0; raf = requestAnimationFrame(tick);
