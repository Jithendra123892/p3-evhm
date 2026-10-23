// replicate the slide-5 panel logic exactly
function cap(c){ return 1.8565 - 0.0022 * c - 0.5 / (1 + Math.exp(-9 * (c / 118 - 1.15))) * 0.42; }
var eolC = 0; while (cap(eolC) > 1.4) eolC++;
var ph = 100, parts = [];
var s = (100 * 7919) | 0;
for (var j = 0; j < 90; j++){
  s = (s * 1103515245 + 12345) & 0x7fffffff;
  var jitter = ((s % 1000) / 1000 - 0.45) * 55;
  parts.push(Math.round(Math.min(168, Math.max(ph + 15, eolC + jitter))));
}
var meds = parts.slice().sort(function(a,b){return a-b;});
var med = meds[45], lo = meds[2], hi = meds[88];
var rulM = med - ph, rulL = Math.max(0, lo - ph), rulH = hi - ph;
var alert = rulL <= 10 ? "ALERT" : rulL <= 25 ? "CAUTION" : "OK";
console.log("eolC (true end of life):", eolC);
console.log("crossing range:", meds[0], "..", meds[89]);
console.log("panel at ph=100: RUL median " + rulM + " cyc | 95% interval [" + rulL + ", " + rulH + "] | status: " + alert);
