// sanity: does the deterministic hash give ~36.6% over the seeded range?
var hits = 0, n = 0;
for (var c = 2; c <= 130; c++){ n++; if ((c * 2654435761 % 1000) < 366) hits++; }
console.log("seed range 2..130:", hits + "/" + n, "=", (hits / n).toFixed(3));
var h2 = 0, n2 = 0;
for (var d = 2; d <= 60; d++){ n2++; if ((d * 2654435761 % 1000) < 366) h2++; }
console.log("early range 2..60:", h2 + "/" + n2, "=", (h2 / n2).toFixed(3));
