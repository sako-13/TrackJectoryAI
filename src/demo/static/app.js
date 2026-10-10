/* Vanilla-JS frontend for the Athlete Career Forecaster demo. */
(function () {
  "use strict";

  var LEVELS = { 1: "Major intl", 2: "Intl/national", 3: "National", 4: "Regional", 5: "Local" };
  var COLORS = { age_curve: "#1f77b4", gbm: "#ff7f0e", lstm: "#2ca02c", transformer: "#9467bd" };
  var state = { seasons: [], nextId: 0 };

  function el(id) { return document.getElementById(id); }

  function addSeasonRow(data) {
    var d = data || { age: "", season_best: "", n_races: 6, level: 3 };
    var id = state.nextId++;
    var row = document.createElement("div");
    row.className = "season-row";
    row.dataset.id = id;
    row.innerHTML =
      '<div><label>Age (yrs)</label><input type="number" step="0.1" min="10" max="50" ' +
      'placeholder="17.5" value="' + d.age + '" data-field="age"></div>' +
      '<div><label>Season best (s)</label><input type="number" step="0.01" min="9" max="17" ' +
      'placeholder="12.10" value="' + d.season_best + '" data-field="season_best"></div>' +
      '<div><label># races</label><input type="number" step="1" min="1" max="30" value="' +
      d.n_races + '" data-field="n_races"></div>' +
      '<div><label>Level</label><select data-field="level">' +
      Object.keys(LEVELS).map(function (k) {
        return '<option value="' + k + '"' + (String(d.level) === k ? " selected" : "") + ">" +
               k + " – " + LEVELS[k] + "</option>";
      }).join("") + "</select></div>" +
      '<div class="full"><button class="btn-ghost small remove" type="button">Remove season</button></div>';
    row.querySelector(".remove").addEventListener("click", function () { row.remove(); });
    el("seasons").appendChild(row);
  }

  function collectSeasons() {
    var rows = document.querySelectorAll("#seasons .season-row");
    var seasons = [];
    rows.forEach(function (row) {
      function val(f) { return row.querySelector('[data-field="' + f + '"]').value; }
      var age = parseFloat(val("age"));
      var best = parseFloat(val("season_best"));
      if (!isFinite(age) || !isFinite(best)) return;
      seasons.push({
        age: age,
        season_best: best,
        n_races: parseInt(val("n_races"), 10) || 5,
        level: parseInt(val("level"), 10) || 3,
      });
    });
    return seasons;
  }

  function showError(msg) {
    var e = el("error");
    e.textContent = msg;
    e.style.display = msg ? "block" : "none";
  }

  function fmt(v) { return (v == null || isNaN(v)) ? "—" : Number(v).toFixed(2); }

  function renderTable(model, rows) {
    var html = '<div class="model-block"><h3 style="color:' + (COLORS[model] || "#333") + '">' +
      model.toUpperCase() + '</h3><table><thead><tr><th>Horizon</th><th>Age</th>' +
      '<th>10%</th><th>Median</th><th>90%</th></tr></thead><tbody>';
    rows.forEach(function (r) {
      html += "<tr><td>+" + r.horizon + " season" + (r.horizon > 1 ? "s" : "") + "</td>" +
        "<td>" + r.age + "</td><td>" + fmt(r.q10) + "</td><td><strong>" + fmt(r.q50) +
        "</strong></td><td>" + fmt(r.q90) + "</td></tr>";
    });
    return html + "</tbody></table></div>";
  }

  function renderChart(history, forecasts) {
    var svg = el("chart");
    var W = 900, H = 380, P = { l: 56, r: 18, t: 16, b: 44 };
    svg.setAttribute("viewBox", "0 0 " + W + " " + H);
    svg.innerHTML = "";
    var allAges = history.map(function (s) { return s.age; });
    var allVals = history.map(function (s) { return s.season_best; });
    var modelNames = Object.keys(forecasts);
    modelNames.forEach(function (m) {
      forecasts[m].forEach(function (r) {
        allAges.push(r.age);
        allVals.push(r.q10, r.q90);
      });
    });
    var ageMin = Math.min.apply(null, allAges), ageMax = Math.max.apply(null, allAges);
    var vMin = Math.min.apply(null, allVals) - 0.1, vMax = Math.max.apply(null, allVals) + 0.1;
    function X(a) { return P.l + (a - ageMin) / (ageMax - ageMin || 1) * (W - P.l - P.r); }
    function Y(v) { return P.t + (v - vMin) / (vMax - vMin || 1) * (H - P.t - P.b); }

    function line(points, color, dash, width) {
      var p = document.createElementNS("http://www.w3.org/2000/svg", "polyline");
      p.setAttribute("points", points.map(function (q) { return X(q[0]) + "," + Y(q[1]); }).join(" "));
      p.setAttribute("fill", "none");
      p.setAttribute("stroke", color);
      p.setAttribute("stroke-width", width || 2);
      if (dash) p.setAttribute("stroke-dasharray", dash);
      svg.appendChild(p);
    }
    function dot(a, v, color) {
      var c = document.createElementNS("http://www.w3.org/2000/svg", "circle");
      c.setAttribute("cx", X(a)); c.setAttribute("cy", Y(v));
      c.setAttribute("r", 3.2); c.setAttribute("fill", color);
      svg.appendChild(c);
    }
    function text(x, y, str, anchor) {
      var t = document.createElementNS("http://www.w3.org/2000/svg", "text");
      t.setAttribute("x", x); t.setAttribute("y", y);
      t.setAttribute("font-size", "11"); t.setAttribute("fill", "#5b6b7b");
      if (anchor) t.setAttribute("text-anchor", anchor);
      t.textContent = str;
      svg.appendChild(t);
    }

    // axes
    for (var i = 0; i <= 4; i++) {
      var v = vMin + (vMax - vMin) * i / 4;
      line([[ageMin, v], [ageMax, v]], "#e3e9ef", null, 1);
      text(P.l - 8, Y(v) + 4, fmt(v), "end");
    }
    var ticks = 5;
    for (var j = 0; j <= ticks; j++) {
      var a = ageMin + (ageMax - ageMin) * j / ticks;
      text(X(a), H - P.b + 18, a.toFixed(0), "middle");
    }
    text(P.l - 40, P.t - 4, "Season best (s)");
    text(W - P.r, H - 8, "Age (years)", "end");

    // history
    line(history.map(function (s) { return [s.age, s.season_best]; }), "#333", null, 2.2);
    history.forEach(function (s) { dot(s.age, s.season_best, "#333"); });

    // forecasts: band (first model with q10/q90), then medians
    var bandDrawn = false;
    modelNames.forEach(function (m) {
      var rows = forecasts[m];
      if (!bandDrawn) {
        var top = rows.map(function (r) { return [r.age, r.q10]; });
        var bottom = rows.map(function (r) { return [r.age, r.q90]; }).reverse();
        var poly = document.createElementNS("http://www.w3.org/2000/svg", "polygon");
        poly.setAttribute("points", top.concat(bottom).map(function (q) { return X(q[0]) + "," + Y(q[1]); }).join(" "));
        poly.setAttribute("fill", COLORS[m] || "#2ca02c");
        poly.setAttribute("fill-opacity", "0.15");
        svg.appendChild(poly);
        bandDrawn = true;
      }
      line(rows.map(function (r) { return [r.age, r.q50]; }), COLORS[m] || "#2ca02c", "6 4", 2);
      rows.forEach(function (r) { dot(r.age, r.q50, COLORS[m] || "#2ca02c"); });
    });

    // legend
    var lx = P.l + 10, ly = P.t + 12;
    var items = [["observed", "#333"]].concat(modelNames.map(function (m) {
      return [m + " median", COLORS[m] || "#2ca02c"];
    }));
    items.forEach(function (it) {
      line([[lx, ly], [lx + 22, ly]], it[1], it[0].indexOf("median") >= 0 ? "6 4" : null, 2);
      text(lx + 28, ly + 4, it[0]);
      ly += 18;
    });
  }

  async function runForecast() {
    showError("");
    var seasons = collectSeasons();
    if (seasons.length < 3) {
      showError("Please enter at least 3 complete seasons (age and season best).");
      return;
    }
    var btn = el("forecast");
    btn.disabled = true;
    btn.textContent = "Forecasting…";
    try {
      var resp = await fetch("/predict", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ event: el("event").value, seasons: seasons }),
      });
      var data = await resp.json();
      if (!resp.ok) {
        showError(data.detail || "Prediction failed.");
        return;
      }
      el("results").style.display = "block";
      el("archetype-line").innerHTML =
        "Development archetype: <span class='badge'>" + data.archetype + "</span>" +
        " &nbsp;<span class='muted'>(clustered from your progression so far)</span>";
      var out = el("forecast-output");
      out.innerHTML = "";
      Object.keys(data.forecasts).forEach(function (m) {
        out.innerHTML += renderTable(m, data.forecasts[m]);
      });
      renderChart(seasons, data.forecasts);
      el("results").scrollIntoView({ behavior: "smooth", block: "start" });
    } catch (err) {
      showError("Could not reach the demo server: " + err);
    } finally {
      btn.disabled = false;
      btn.textContent = "Forecast my career";
    }
  }

  el("add-season").addEventListener("click", function () { addSeasonRow(); });
  el("clear").addEventListener("click", function () {
    el("seasons").innerHTML = "";
    el("results").style.display = "none";
    el("forecast-output").innerHTML = "";
  });
  el("load-example").addEventListener("click", function () {
    el("seasons").innerHTML = "";
    var example = [
      { age: 15.6, season_best: 12.85, n_races: 5, level: 5 },
      { age: 16.5, season_best: 12.20, n_races: 7, level: 4 },
      { age: 17.4, season_best: 11.75, n_races: 8, level: 3 },
      { age: 18.3, season_best: 11.42, n_races: 9, level: 3 },
      { age: 19.2, season_best: 11.21, n_races: 9, level: 2 },
      { age: 20.4, season_best: 11.09, n_races: 10, level: 2 },
    ];
    example.forEach(addSeasonRow);
  });
  el("forecast").addEventListener("click", runForecast);

  // start with three empty rows
  addSeasonRow(); addSeasonRow(); addSeasonRow();
})();
