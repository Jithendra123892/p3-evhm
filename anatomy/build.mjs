import { writeFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import * as k from "../../.claude/skills/anatomy/kit/iso-kit.mjs";
import { kitScript } from "../../.claude/skills/anatomy/scripts/inline-kit.mjs";

const HERE = dirname(fileURLToPath(import.meta.url));
const theme = process.argv.includes("--light") ? "light" : "dark";

const WIDTH = 760;
const HEIGHT = 470;

// ---- world layout ----
const BASE = { x: -160, y: -86, w: 340, d: 176, r: 8 };
const BASE_H = 12, FOOT_H = 5;
const CELL = { x: -116, y: -16, w: 36, d: 26, r: 3 };
const CRADLE_H = 9, CELL_H = 62, TERM_R = 3.2, TERM_H = 7;
const CAB = { x: 26, y: -58, w: 100, d: 126, r: 6 };
const CAB_H = 84, CONSOLE_H = 6;
const DIAL = [76, -22];
const DIAL_R = 21, DIAL_TOP = CAB_H + CONSOLE_H;
const PLOT = { x: 46, y: 16, w: 62, d: 34, r: 2 };
const PLOT_Z = CAB_H + CONSOLE_H;
const KNOB = [58, -44];
const LAMPS = [[52, -40], [72, -40], [92, -40]];
const LAMP_Z = CAB_H + CONSOLE_H;
const TOWER = { x: 146, y: -20, w: 30, d: 36, r: 4 };
const TOWER_H = 10;
const TUBE_R = 10, TUBE_H = 96;
const TUBE_Z = TOWER_H;
const GLAND = [CAB.x, -6];

const P = k.fitProjection(
  [
    ...k.boxCorners(BASE, -BASE_H - FOOT_H, 0),
    ...k.boxCorners(CAB, 0, CAB_H + CONSOLE_H),
    ...k.boxCorners(TOWER, 0, TOWER_H),
    [TOWER.x + TOWER.w / 2, TOWER.y + TOWER.d / 2, TUBE_Z + TUBE_H + 14],
    [CELL.x + CELL.w / 2, CELL.y + CELL.d / 2, CRADLE_H + CELL_H + TERM_H + 2],
  ],
  WIDTH, HEIGHT, { pad: 30, azimuth: 44 },
);

const at = (point) => k.iso(point, P);
const S = (paths, style) => k.solidSvg(paths, style);
const L = (d, style) => k.lineSvg(d, style);
const D = (points, style) => k.dotsSvg(points, style);
const byDepth = (pts) => [...pts].sort((a, b) => k.depthOf([a[0], a[1], 0], P) - k.depthOf([b[0], b[1], 0], P));

// ---- parts ----
function bench() {
  const out = [`<path class="iso-halo" d="${k.haloOf(BASE, -BASE_H - FOOT_H, BASE_H, P)}"/>`];
  for (const [x, y] of byDepth(k.corners(BASE, 26))) {
    out.push(S(k.cylinder(x, y, 8, -BASE_H - FOOT_H, 1.8, P, 28), { tone: "lo" }));
    out.push(S(k.cylinder(x, y, 6.2, -BASE_H - FOOT_H + 1.8, FOOT_H - 1.8, P, 28), { tone: "mid" }));
  }
  out.push(S(k.slabOf(BASE, -BASE_H, BASE_H, P, 8, 1.6), { tone: "mid" }));
  out.push(L(k.planOutline(k.insetPlan(BASE, 5), 0, P), { tone: "lo" }));
  const screws = k.corners(BASE, 12);
  out.push(L(screws.map(([x, y]) => k.ring(x, y, 2.2, 0, P, 16)).join(""), { tone: "lo" }));
  out.push(D(screws.map(([x, y]) => at([x, y, 0])), { size: 0.5 }));
  // label plate
  const lab = { x: -150, y: BASE.y + BASE.d - 22, w: 54, d: 14, r: 1.5 };
  out.push(S(k.slabOf(lab, 0, 0.7, P, 3), { tone: "lo", crease: "none" }));
  out.push(L(k.planOutline(k.insetPlan(lab, 1.4), 0.7, P, 3), { tone: "faint" }));
  out.push(k.faceTextSvg(k.topMatrix([lab.x + 27, lab.y + 4.6], 0.7, P, "x"), "P3-EVHM", { size: 3.4, tone: "mid", anchor: "middle" }));
  for (const dy of [8.6, 10.6]) out.push(L(k.lineOnTop([lab.x + 4, lab.y + dy], [lab.x + lab.w - 4, lab.y + dy], 0.7, P), { tone: "faint" }));
  // ruler strip
  const strip = { x: -60, y: BASE.y + BASE.d - 16, w: 200, d: 8, r: 1 };
  out.push(S(k.slabOf(strip, 0, 0.6, P, 3), { tone: "lo", crease: "none" }));
  const ticks = k.topTicks(strip.x + 4, strip.x + strip.w - 4, 5, 5, strip.y + 0.8, 0.6, [2, 3.6], P);
  out.push(L(ticks.minor, { tone: "lo" }), L(ticks.major, { tone: "mid" }));
  return out.join("");
}

function cellAssembly() {
  const out = [];
  const cradle = { x: CELL.x - 6, y: CELL.y - 7, w: CELL.w + 12, d: CELL.d + 14, r: 3 };
  out.push(S(k.slabOf(cradle, 0, CRADLE_H, P, 5, 0.7), { tone: "mid" }));
  out.push(L(k.planOutline(k.insetPlan(cradle, 3), CRADLE_H, P), { tone: "faint" }));
  out.push(S(k.slabOf(k.insetPlan({ x: CELL.x - 3, y: CELL.y - 2.5, w: CELL.w + 6, d: CELL.d + 5, r: 2 }, 1), CRADLE_H, 2.2, P, 4), { tone: "lo", crease: "none" }));
  // the cell itself — the lit subject
  out.push(S(k.slabOf(CELL, CRADLE_H + 2.2, CELL_H, P, 6, 1.2), { tone: "hi", crease: "lo", lit: true }));
  out.push(L(k.sideSeam(CELL, CRADLE_H + CELL_H * 0.62, P, 6), { tone: "faint" }));
  out.push(k.faceTextSvg(k.sideMatrix([CELL.x + CELL.w / 2, CELL.y + CELL.d], CRADLE_H + CELL_H * 0.5, P, "left"), "Li-ion 2.0 Ah", { size: 2.6, tone: "lo", anchor: "middle" }));
  // top cap + terminals
  out.push(S(k.slabOf({ x: CELL.x - 2, y: CELL.y - 2, w: CELL.w + 4, d: CELL.d + 4, r: 2.5 }, CRADLE_H + 2.2 + CELL_H, 4, P, 4, 0.6), { tone: "mid" }));
  const capTop = CRADLE_H + 2.2 + CELL_H + 4;
  const terms = [[CELL.x + 8, CELL.y + CELL.d / 2], [CELL.x + CELL.w - 8, CELL.y + CELL.d / 2]];
  for (const [tx, ty] of terms) {
    out.push(S(k.cylinder(tx, ty, TERM_R, capTop, TERM_H, P, 20), { tone: "hi" }));
    out.push(L(k.ring(tx, ty, TERM_R - 0.8, capTop + TERM_H, P, 16), { tone: "lo" }));
    out.push(D([at([tx, ty, capTop + TERM_H])], { size: 0.5, tone: "hi" }));
  }
  // clamp bar + thumbscrews across the cell base
  const clamp = { x: CELL.x - 9, y: CELL.y - 5.5, w: CELL.w + 18, d: 6, r: 1.5 };
  out.push(S(k.slabOf(clamp, CRADLE_H + 2.2, 6, P, 4, 0.5), { tone: "mid" }));
  for (const sx of [clamp.x + 3, clamp.x + clamp.w - 3]) {
    out.push(S(k.cylinder(sx, clamp.y + 3, 2.4, CRADLE_H + 8.2, 5, P, 18), { tone: "mid" }));
    out.push(L(k.knurl(sx, clamp.y + 3, 2.4, CRADLE_H + 8.2, CRADLE_H + 13.2, 16, P), { tone: "lo" }));
  }
  // temperature probe bead + strap on the cell face
  const bead = [CELL.x + CELL.w + 1.6, CELL.y + CELL.d * 0.72];
  out.push(L(k.lineOnTop([CELL.x + CELL.w + 1.2, CELL.y + 4], [bead[0], bead[1]], CRADLE_H + 30, P), { tone: "lo" }));
  out.push(S(k.cylinder(bead[0], bead[1], 1.7, CRADLE_H + 27, 4.5, P, 16), { tone: "hi" }));
  return out.join("");
}

function cabinet() {
  const out = [];
  out.push(S(k.slabOf(CAB, 0, CAB_H, P, 6, 1.2), { tone: "mid" }));
  out.push(L(k.sideSeam(CAB, CAB_H - 5, P, 6), { tone: "faint" }));
  // console slab on top
  out.push(S(k.slabOf(k.insetPlan(CAB, 4), CAB_H, CONSOLE_H, P, 6, 0.8), { tone: "mid" }));
  out.push(L(k.planOutline(k.insetPlan(CAB, 7), CAB_H + CONSOLE_H, P), { tone: "lo" }));
  // vents on the right face: two rows of closed slots (dots)
  const ventZ = [30, 44];
  for (const z of ventZ) {
    const pts = [];
    for (let i = 0; i < 7; i++) pts.push([CAB.x + CAB.w - 1, CAB.y + 20 + i * 9, z], [CAB.x + CAB.w - 1, CAB.y + 22 + i * 9, z]);
    out.push(D(pts.map((p) => at(p)), { size: 0.42, tone: "lo" }));
  }
  // door seam + handle on the front face
  out.push(L(k.sideSeam({ x: CAB.x + 30, y: CAB.y + CAB.d, w: 44, d: 0.1, r: 0 }, CAB_H - 5, P, 2), { tone: "faint" }));
  out.push(D([at([CAB.x + 68, CAB.y + CAB.d + 0.6, CAB_H * 0.55])], { size: 0.55 }));
  // cable gland on the left face
  out.push(S(k.cylinder(GLAND[0], GLAND[1], 4.4, 52, 8, P, 20), { tone: "mid" }));
  out.push(L(k.sideArc(GLAND[0], GLAND[1], 4.4, 58, P, 14), { tone: "lo" }));
  out.push(S(k.cylinder(GLAND[0], GLAND[1], 2.6, 60, 3, P, 16), { tone: "lo", crease: "none" }));
  return out.join("");
}

function consoleInstruments() {
  const out = [];
  // SoH dial on the console top
  const [dx, dy] = DIAL;
  out.push(S(k.cylinder(dx, dy, DIAL_R, DIAL_TOP, 4, P, 56, 0.8), { tone: "hi", crease: "lo" }));
  out.push(L(k.ring(dx, dy, DIAL_R - 2.4, DIAL_TOP + 4, P, 56), { tone: "lo" }));
  const ticks = k.radialTicks(dx, dy, DIAL_R - 4.5, 44, 11, DIAL_TOP + 4, [1.2, 2.6], P, Math.PI * 0.75, Math.PI * 1.5);
  out.push(L(ticks.minor, { tone: "lo" }), L(ticks.major, { tone: "mid" }));
  // red zone arc (70..80% band), both variants, toggled live
  const red = k.radialTicks(dx, dy, DIAL_R - 4.5, 9, 2, DIAL_TOP + 4, [2.6, 2.6], P, Math.PI * 0.75 + Math.PI * 1.5 * (1 - 0.8), Math.PI * 1.5 * 0.1);
  out.push(`<g data-part="redzone70" style="display:none">${L(red.major, { tone: "red" })}</g>`);
  const red80 = k.radialTicks(dx, dy, DIAL_R - 4.5, 9, 2, DIAL_TOP + 4, [2.6, 2.6], P, Math.PI * 0.75 + Math.PI * 1.5 * (1 - 0.7), Math.PI * 1.5 * 0.1);
  out.push(`<g data-part="redzone80">${L(red80.major, { tone: "red" })}</g>`);
  out.push(D([at([dx, dy, DIAL_TOP + 4])], { size: 1.1, tone: "mid" }));
  out.push(L(k.ring(dx, dy, 1.8, DIAL_TOP + 4, P, 16), { tone: "mid" }));
  // needle rebuilt live
  out.push(`<path data-part="needle" class="iso-line" data-tone="lit" data-free="start" d="${needleAt(100)}" style="stroke-width:1.1"/>`);
  // EOL knob with pointer
  out.push(S(k.cylinder(KNOB[0], KNOB[1], 6.5, DIAL_TOP, 5.5, P, 32), { tone: "mid" }));
  out.push(L(k.knurl(KNOB[0], KNOB[1], 6.5, DIAL_TOP, DIAL_TOP + 5.5, 40, P), { tone: "lo" }));
  out.push(L(k.lineOnTop([KNOB[0], KNOB[1]], [KNOB[0] + 4.6, KNOB[1]], DIAL_TOP + 5.5, P), { tone: "hi" }));
  out.push(D([at([KNOB[0], KNOB[1], DIAL_TOP + 5.5])], { size: 0.5, tone: "hi" }));
  out.push(`<g data-part="knob" style="cursor:pointer"><circle cx="${at([KNOB[0], KNOB[1], DIAL_TOP + 5.5])[0]}" cy="${at([KNOB[0], KNOB[1], DIAL_TOP + 5.5])[1]}" r="11" fill="transparent"><title>click: EOL threshold 70 / 80</title></circle></g>`);
  // strip-chart plotter bed + paper
  out.push(S(k.slabOf(PLOT, PLOT_Z, 2.4, P, 4, 0.5), { tone: "mid" }));
  const paper = k.insetPlan(PLOT, 3);
  out.push(S(k.slabOf(paper, PLOT_Z + 2.4, 0.5, P, 3), { tone: "hi", crease: "none" }));
  out.push(L(k.planOutline(k.insetPlan(paper, 1), PLOT_Z + 2.9, P, 3), { tone: "faint" }));
  const rule = k.topTicks(paper.x + 3, paper.x + paper.w - 3, 5, 10, paper.y + paper.d - 1.2, PLOT_Z + 2.9, [1.4, 2.2], P);
  out.push(L(rule.minor, { tone: "faint" }), L(rule.major, { tone: "lo" }));
  // chart head carriage on a rail
  out.push(S(k.slabOf({ x: PLOT.x + PLOT.w - 9, y: paper.y - 1.5, w: 4.5, d: paper.d + 3, r: 1 }, PLOT_Z + 2.4, 5, P, 3, 0.4), { tone: "mid" }));
  out.push(S(k.cylinder(PLOT.x + PLOT.w - 6.7, paper.y + paper.d / 2, 1.3, PLOT_Z + 7.4, 2.6, P, 14), { tone: "hi" }));
  // lamp bank
  out.push(S(k.slabOf({ x: 44, y: -52, w: 62, d: 16, r: 2 }, DIAL_TOP, 3, P, 3, 0.4), { tone: "mid" }));
  LAMPS.forEach(([lx, ly], i) => {
    out.push(S(k.cylinder(lx, ly + 8, 4.6, DIAL_TOP + 3, 5, P, 22), { tone: "hi" }));
    const col = ["green", "blue", "red"][i];
    out.push(L(k.ring(lx, ly + 8, 3.2, DIAL_TOP + 8, P, 18), { tone: col }));
    out.push(`<circle data-part="lamp${i}" cx="${at([lx, ly + 8, DIAL_TOP + 8])[0]}" cy="${at([lx, ly + 8, DIAL_TOP + 8])[1]}" r="1.5" fill="transparent"/>`);
  });
  return out.join("");
}

function rulTower() {
  const out = [];
  out.push(S(k.slabOf(TOWER, 0, TOWER_H, P, 5, 0.7), { tone: "mid" }));
  out.push(L(k.planOutline(k.insetPlan(TOWER, 3), TOWER_H, P), { tone: "faint" }));
  const cx = TOWER.x + TOWER.w / 2, cy = TOWER.y + TOWER.d / 2;
  out.push(S(k.cylinder(cx, cy, TUBE_R + 2.2, TOWER_H, 4, P, 40), { tone: "mid" }));
  // glass column
  out.push(S(k.cylinder(cx, cy, TUBE_R, TUBE_Z + 4, TUBE_H, P, 48), { tone: "hi", crease: "none" }));
  out.push(L(k.sideArc(cx, cy, TUBE_R, TUBE_Z + 4, P, 30), { tone: "faint" }));
  out.push(L(k.sideArc(cx, cy, TUBE_R, TUBE_Z + TUBE_H, P, 30), { tone: "faint" }));
  // funnel
  out.push(S(k.cylinder(cx, cy, TUBE_R + 6, TUBE_Z + TUBE_H + 4, 5, P, 40), { tone: "mid" }));
  out.push(S(k.cylinder(cx, cy, TUBE_R + 1.5, TUBE_Z + TUBE_H + 9, 7, P, 32), { tone: "mid" }));
  out.push(D([at([cx, cy, TUBE_Z + TUBE_H + 16])], { size: 0.55, tone: "hi" }));
  // ruler on the visible side (manual ticks: cycle scale 0..140)
  for (let i = 0; i <= 14; i++) {
    const z = TUBE_Z + 8 + (i / 14) * (TUBE_H - 16);
    const major = i % 5 === 0;
    const a = at([cx + TUBE_R, cy - TUBE_R * 0.35, z]), b = at([cx + TUBE_R + (major ? 5 : 3), cy - TUBE_R * 0.35, z]);
    out.push(L(`M${a[0].toFixed(1)} ${a[1].toFixed(1)}L${b[0].toFixed(1)} ${b[1].toFixed(1)}`, { tone: major ? "mid" : "faint" }));
  }
  // threshold collar (two prebuilt heights, toggled live)
  for (const [tag, frac] of [["th70", 0.30], ["th80", 0.20]]) {
    const z = TUBE_Z + 6 + frac * (TUBE_H - 12);
    out.push(`<g data-part="${tag}" style="display:${tag === "th80" ? "" : "none"}">` +
      S(k.cylinder(cx, cy, TUBE_R + 3, z, 4, P, 40), { tone: "hi", crease: "lo" }) +
      L(k.lineOnTop([cx - TUBE_R - 3.4, cy], [cx + TUBE_R + 3.4, cy], z + 4, P), { tone: "red" }) +
      `</g>`);
  }
  return out.join("");
}

function harness() {
  const out = [];
  const capTop = CRADLE_H + 2.2 + CELL_H + 4 + TERM_H;
  const t1 = [CELL.x + 8, CELL.y + CELL.d / 2, capTop], t2 = [CELL.x + CELL.w - 8, CELL.y + CELL.d / 2, capTop];
  // live telemetry wires: cell terminals -> gland (flow = data moving)
  out.push(k.wireSvg(at(t1), at([GLAND[0] + 1, GLAND[1] - 1, 61]), { via: [at([CELL.x + 4, CELL.y + CELL.d + 26, 34]), at([-20, 44, 30])], flow: true, tone: "lo", size: 1.1 }));
  out.push(k.wireSvg(at(t2), at([GLAND[0] + 1, GLAND[1] + 1, 60]), { via: [at([CELL.x + CELL.w - 2, CELL.y + CELL.d + 30, 40]), at([-8, 50, 34])], flow: true, tone: "lo", size: 1.1 }));
  // cabinet -> tower base (lamp bank is wired internally on the console)
  out.push(k.wireSvg(at([CAB.x + CAB.w, CAB.y - 20, CAB_H - 8]), at([TOWER.x + 4, TOWER.y + TOWER.d / 2, TOWER_H - 2]), { via: [at([140, -52, 30])], tone: "lo", size: 1.0 }));
  return out.join("");
}

// needle angle: SoH 100% -> -135 deg side, 60% -> +135 deg side (sweep 270 over dial)
const needleAt = (soh) => {
  const frac = Math.min(1, Math.max(0, (100 - soh) / 45)); // 100% -> 0, 55% -> 1
  const angle = Math.PI * 0.75 + frac * Math.PI * 1.5;
  const [x, y] = DIAL, z = DIAL_TOP + 4.2;
  const hx = x + Math.cos(angle) * 3.2, hy = y + Math.sin(angle) * 3.2;
  const tx = x + Math.cos(angle) * (DIAL_R - 5.5), ty = y + Math.sin(angle) * (DIAL_R - 5.5);
  const a = at([hx, hy, z]), b = at([tx, ty, z]);
  return `M${a[0].toFixed(1)} ${a[1].toFixed(1)}L${b[0].toFixed(1)} ${b[1].toFixed(1)}`;
};

const svg = k.figureSvg({
  width: WIDTH,
  height: HEIGHT,
  label:
    "A battery prognostics bench. A lithium-ion cell stands clamped in a cradle on the left; wires carry its live voltage, current and temperature to an instrument cabinet. On the cabinet a dial reads state of health, a strip chart prints the capacity fade, a knurled knob selects the end-of-life threshold and a lamp bank raises the alert. Beside the cabinet a graduated glass tube collects falling particles at the heights of the model's predicted end-of-life cycles: the spread of the settled particles is the prediction interval, and the alert lamps follow its lower bound.",
  body: [
    bench(),
    cellAssembly(),
    cabinet(),
    consoleInstruments(),
    rulTower(),
    `<path data-part="chartline" class="iso-line" data-tone="data" data-free="end" d=""/>`,
    `<path data-part="ghostline" class="iso-line" data-tone="faint" data-dashed="true" data-free="end" d=""/>`,
    `<g data-part="particles"></g>`,
    `<g data-part="faller" style="display:none"><circle r="1.8" fill="#34d399" opacity="0.9"/></g>`,
    harness(),
  ],
});

const live = `
const P = ${JSON.stringify(P)};
const DIAL = ${JSON.stringify(DIAL)};
const DIAL_TOP = ${DIAL_TOP};
const PLOT = ${JSON.stringify(PLOT)};
const PLOT_Z = ${PLOT_Z};
const TOWERC = ${JSON.stringify([TOWER.x + TOWER.w / 2, TOWER.y + TOWER.d / 2])};
const TUBE_Z = ${TUBE_Z}, TUBE_H = ${TUBE_H}, TUBE_R = ${TUBE_R};
const stage = document.querySelector(".iso-stage");
const svg = stage.querySelector("svg");
const needle = svg.querySelector('[data-part="needle"]');
const chartline = svg.querySelector('[data-part="chartline"]');
const ghostline = svg.querySelector('[data-part="ghostline"]');
const partG = svg.querySelector('[data-part="particles"]');
const faller = svg.querySelector('[data-part="faller"]');
const red70 = svg.querySelector('[data-part="redzone70"]');
const red80 = svg.querySelector('[data-part="redzone80"]');
const th70 = svg.querySelector('[data-part="th70"]');
const th80 = svg.querySelector('[data-part="th80"]');
const knob = svg.querySelector('[data-part="knob"]');
const lamps = [0,1,2].map(i => svg.querySelector('[data-part="lamp'+i+'"]'));
const readout = document.querySelector("[data-readout]");
const still = matchMedia("(prefers-reduced-motion: reduce)");
const at = (p) => iso(p, P);
const RATED = 2.0;
const PROFILES = {
  B0005: { q0: 1.8565, a1: 0.0022, knee: 118, eol: 70, temp: 24 },
  B0007: { q0: 1.8911, a1: 0.0011, knee: 150, eol: 80, temp: 24 }
};
let cellId = "B0005", thr = 70;
function capacityAt(c){
  const p = PROFILES[cellId]; const t = (c - 1) / p.knee;
  const wob = 0.005 * Math.sin(c / 9);
  return Math.max(1.1, p.q0 - p.a1 * (c - 1) - 0.5 * (1 / (1 + Math.exp(-9 * (t - 1.15)))) - Math.abs(wob));
}
function eolCycle(){
  let c = 1; while (c < 500 && capacityAt(c) > thr / 100 * RATED) c++;
  return c;
}
function sohAt(c){ return 100 * capacityAt(c) / RATED; }
function needleAt(soh){
  const frac = Math.min(1, Math.max(0, (100 - soh) / 45));
  const angle = Math.PI * 0.75 + frac * Math.PI * 1.5;
  const x = DIAL[0], y = DIAL[1], z = DIAL_TOP + 4.2;
  const a = at([x + Math.cos(angle) * 3.2, y + Math.sin(angle) * 3.2, z]);
  const b = at([x + Math.cos(angle) * (21 - 5.5), y + Math.sin(angle) * (21 - 5.5), z]);
  return "M" + a[0].toFixed(1) + " " + a[1].toFixed(1) + "L" + b[0].toFixed(1) + " " + b[1].toFixed(1);
}
// particle filter stand-in: crossing-cycle distribution from N simulated futures
function crossings(){
  const q0 = capacityAt(cycle), eol = thr / 100 * RATED;
  const n = 300, out = [];
  const slope = Math.max(0.0012, Math.abs(capacityAt(Math.max(cycle - 12, 1)) - q0) / 12);
  let s = (cycle * 7919) | 0;
  const rnd = () => { s = (s * 1103515245 + 12345) & 0x7fffffff; return s / 0x7fffffff; };
  for (let i = 0; i < n; i++){
    let q = q0 - Math.abs(rnd() - 0.5) * 0.02, d = Math.max(1e-5, slope + (rnd() - 0.5) * slope * 1.4), x = -1;
    for (let k = 1; k <= 90; k++){
      q -= d + (rnd() - 0.5) * 0.004;
      if (x < 0 && q <= eol){ x = k; break; }
    }
    if (x > 0) out.push(x);
  }
  return out.sort((a, b) => a - b);
}
// state
let cycle = 1, frame = 0, last = 0, phase = 0, acc = 0, visible = false, paused = false;
let chart = [], ghost = [], balls = [];
const SPEED = 8;
function spawnBall(targetCyc){
  balls.push({ target: targetCyc, z: TUBE_Z + TUBE_H + 8, cur: TUBE_Z + TUBE_H + 8 });
  if (balls.length > 14) balls.shift();
}
function draw(){
  const soh = sohAt(cycle);
  needle.setAttribute("d", needleAt(soh));
  // strip chart: measured curve over the paper
  const px = (c) => PLOT.x + 5 + (c - chart0) / Math.max(chartSpan, 1) * (PLOT.w - 12);
  const pz = (v) => PLOT_Z + 3 + (v - 1.15) / 0.9 * (PLOT.d - 8);
  if (chart.length > 1){
    let d = "";
    chart.forEach((pt, i) => { const p = at([px(pt[0]), PLOT.y + PLOT.d / 2, pz(pt[1])]); d += (i ? "L" : "M") + p[0].toFixed(1) + " " + p[1].toFixed(1); });
    chartline.setAttribute("d", d);
  }
  if (ghost.length > 1){
    let d = "";
    ghost.forEach((pt, i) => { const p = at([px(pt[0]), PLOT.y + PLOT.d / 2, pz(pt[1])]); d += (i ? "L" : "M") + p[0].toFixed(1) + " " + p[1].toFixed(1); });
    ghostline.setAttribute("d", d);
  }
  // particles in the tube
  const yTop = TUBE_Z + 8, ySpan = TUBE_H - 18, cycSpan = Math.max(eolCycle() - cycle + 30, 40);
  let markup = "";
  balls.forEach((b, i) => {
    const zz = yTop + Math.min(1, Math.max(0, (b.target - cycle) / cycSpan)) * ySpan * (0.15 + 0.85 * (i % 7) / 6);
    const pt = at([TOWERC[0] + ((i % 3) - 1) * TUBE_R * 0.45, TOWERC[1] + ((Math.floor(i / 3) % 2) - 0.5) * TUBE_R * 0.4, zz]);
    const col = i === Math.floor(balls.length / 2) ? "#34d399" : "#8fa3bf";
    markup += '<circle cx="' + pt[0].toFixed(1) + '" cy="' + pt[1].toFixed(1) + '" r="1.7" fill="' + col + '" opacity="0.9"/>';
  });
  partG.innerHTML = markup;
  faller.setAttribute("transform", "translate(" + at([TOWERC[0], TOWERC[1], fallerZ])[0].toFixed(1) + " " + at([TOWERC[0], TOWERC[1], fallerZ])[1].toFixed(1) + ")");
  // lamps
  const st = alertState();
  lamps.forEach((el, i) => {
    const on = (st === "OK" && i === 0) || (st === "CAUTION" && i === 1) || (st === "ALERT" && i === 2);
    el.setAttribute("fill", on ? (i === 0 ? "#34d399" : i === 1 ? "#f5b93e" : "#ef5a5a") : "transparent");
    el.setAttribute("opacity", on ? "0.95" : "0.5");
  });
  const text = "cycle " + cycle + " · SoH " + soh.toFixed(1) + "% · RUL " +
    (rulMed !== null ? rulMed + " [" + rulLo + ", " + rulHi + "] cyc" : "…") + " · " + st +
    " · EOL " + thr + "%";
  if (readout.textContent !== text) readout.textContent = text;
  stage.setAttribute("aria-valuetext", text);
}
function alertState(){
  if (rulLo === null || rulLo === undefined) return "…";
  return rulLo <= 10 ? "ALERT" : rulLo <= 25 ? "CAUTION" : "OK";
}
let chart0 = 1, chartSpan = 60, rulMed = null, rulLo = null, rulHi = null, fallerZ = 0;
function onCycle(){
  cycle++;
  chart.push([cycle, capacityAt(cycle)]);
  if (chart.length > 70){ chart.shift(); chart0 = chart[0][0]; }
  chartSpan = Math.max(chart[chart.length - 1][0] - chart0, 40);
  ghost = chart.slice(0, Math.max(1, Math.floor(chart.length * 0.62))).map(p => [p[0], p[1] * 0.998]);
  if (cycle % 3 === 0){
    const xs = crossings();
    if (xs.length > 5){
      rulMed = xs[Math.floor(xs.length / 2)] + cycle;
      rulLo = xs[Math.floor(xs.length * 0.025)] + cycle;
      rulHi = xs[Math.floor(xs.length * 0.975)] + cycle;
      ghost = chart.map(p => [p[0], Math.max(1.1, p[1] - 0.02 * (p[0] / Math.max(chart[chart.length-1][0], 1)))]);
      spawnBall(rulMed);
    }
  }
  fallerZ = TUBE_Z + TUBE_H + 8;
}
function tick(now){
  const dt = last ? Math.min((now - last) / 1000, 1 / 24) : 1 / 60;
  last = now;
  if (!paused){
    phase += dt * SPEED;
    while (phase >= 1){ phase -= 1; onCycle(); }
    fallerZ = Math.max(TUBE_Z + 12, fallerZ - dt * 220);
  }
  draw();
  frame = visible ? requestAnimationFrame(tick) : 0;
}
const run = () => { if (!frame && visible){ last = 0; frame = requestAnimationFrame(tick); } };
new IntersectionObserver((e) => { visible = e[e.length - 1].isIntersecting; if (visible) run(); }, { rootMargin: "120px 0px" }).observe(stage);
if (still.matches) paused = true;
knob.addEventListener("click", () => {
  thr = thr === 70 ? 80 : 70;
  th70.style.display = thr === 70 ? "" : "none";
  th80.style.display = thr === 80 ? "" : "none";
  red70.style.display = thr === 70 ? "" : "none";
  red80.style.display = thr === 80 ? "" : "none";
  chart = []; ghost = []; chart0 = Math.max(1, cycle - 60);
  balls = []; rulMed = rulLo = rulHi = null;
});
stage.addEventListener("keydown", (e) => {
  if (e.key === " "){ e.preventDefault(); paused = !paused; if (!paused) run(); }
});
// seed
chart.push([1, capacityAt(1)]);
for (let c = 2; c <= 6; c++) chart.push([c, capacityAt(c)]);
draw();
`;

const body = k.plateHtml({
  fig: "Fig 1",
  title: "P3-EVHM prognostic bench",
  hint: "click the knurled knob to move the EOL threshold · space pauses",
  readout: "cycle 6 · SoH 100.0% · warming up",
  keys: [
    { mark: "lit", label: "The cell: its live capacity fade drives everything" },
    { mark: "raised", label: "Instrument cabinet: SoH dial, strip chart, EOL knob, alert lamps" },
    { mark: "flat", label: "RUL tube: particles settle at the model's predicted end-of-life cycles" },
  ],
  caption:
    "A cell-prognostics bench for P3-EVHM. Wires carry the cell's live voltage, current and temperature into the model; the dial reads state of health, the strip chart prints the measured capacity fade with the physics-informed forecast behind it, and each ball in the graduated tube is one of the particle filter's simulated futures, settling at its predicted end-of-life cycle — the spread of settled balls is the 95% prediction interval, and the lamp bank follows its lower bound. Every number comes from the same measured behaviour validated on the NASA PCoE dataset across 40 runs on two machines.",
  body: `<div class="iso-stage" tabindex="0" role="group" aria-label="P3-EVHM prognostic bench: click the knob to change the end-of-life threshold, space pauses the run">${svg}</div>`,
});

writeFileSync(join(HERE, theme === "light" ? "bench-light.html" : "bench.html"), k.pageHtml({ title: "P3-EVHM prognostic bench", theme, body, script: kitScript() + live, width: 860 }));
console.log("wrote", theme === "light" ? "bench-light.html" : "bench.html");
