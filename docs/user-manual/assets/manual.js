/* SENSI SpeechFeatures User Manual — behaviour. Plain JS, works from file:// (no fetch). */
(function () {
  "use strict";
  var ROOT = document.body.getAttribute("data-root") || "";

  /* outline: fold / unfold */
  document.querySelectorAll(".outline .toggle").forEach(function (b) {
    b.addEventListener("click", function () {
      var li = b.closest("li");
      li.classList.toggle("open");
      b.setAttribute("aria-expanded", li.classList.contains("open"));
    });
  });
  var cur = document.querySelector(".outline a.current");
  if (cur) cur.scrollIntoView({ block: "center" });

  /* narrow screens: menu button */
  var menu = document.querySelector(".menu-btn");
  if (menu) menu.addEventListener("click", function () { document.body.classList.toggle("menu-open"); });
  document.querySelector("main").addEventListener("click", function () { document.body.classList.remove("menu-open"); });

  /* tabs (MATLAB / output / variant switches): [data-tabs] groups */
  document.querySelectorAll("[data-tabs]").forEach(function (group) {
    var buttons = group.querySelectorAll(":scope > .tabs button");
    var panels = group.querySelectorAll(":scope > .tabpanel");
    buttons.forEach(function (btn, i) {
      btn.addEventListener("click", function () {
        buttons.forEach(function (b) { b.classList.remove("on"); b.setAttribute("aria-selected", "false"); });
        panels.forEach(function (p) { p.classList.remove("on"); });
        btn.classList.add("on"); btn.setAttribute("aria-selected", "true");
        if (panels[i]) panels[i].classList.add("on");
      });
    });
  });

  /* copy buttons on code blocks */
  document.querySelectorAll("pre[data-copy]").forEach(function (pre) {
    var b = document.createElement("button");
    b.className = "copy"; b.type = "button"; b.textContent = "Copy";
    b.addEventListener("click", function () {
      var text = pre.querySelector("code") ? pre.querySelector("code").innerText : pre.innerText;
      var done = function () { b.textContent = "Copied"; setTimeout(function () { b.textContent = "Copy"; }, 1400); };
      if (navigator.clipboard && window.isSecureContext) { navigator.clipboard.writeText(text).then(done); }
      else {
        var ta = document.createElement("textarea"); ta.value = text; document.body.appendChild(ta);
        ta.select(); try { document.execCommand("copy"); done(); } catch (e) {} ta.remove();
      }
    });
    pre.appendChild(b);
  });

  /* scroll-spy for "On this page" */
  var links = Array.prototype.slice.call(document.querySelectorAll(".right a[href^='#']"));
  var targets = links.map(function (a) { return document.getElementById(a.getAttribute("href").slice(1)); });
  function spy() {
    var y = window.scrollY + 120, idx = 0;
    targets.forEach(function (t, i) { if (t && t.offsetTop <= y) idx = i; });
    if (window.innerHeight + window.scrollY >= document.body.scrollHeight - 4) idx = targets.length - 1;
    links.forEach(function (a, i) { a.classList.toggle("active", i === idx); });
  }
  if (links.length) { window.addEventListener("scroll", spy, { passive: true }); spy(); }

  /* search over the prebuilt index (assets/search-index.js) */
  var input = document.querySelector(".search input");
  var box = document.querySelector(".search-results");
  var INDEX = window.MANUAL_INDEX || [];
  function norm(s) { return (s || "").toLowerCase(); }
  function score(e, terms) {
    var t = norm(e.t), k = norm(e.k), b = norm(e.b), s = 0;
    for (var i = 0; i < terms.length; i++) {
      var q = terms[i];
      if (t.indexOf(q) >= 0) s += t === q ? 40 : 12; else if (k.indexOf(q) >= 0) s += 10;
      else if (b.indexOf(q) >= 0) s += 2; else return 0;
    }
    return s + (e.w || 0);
  }
  function render(q) {
    var terms = norm(q).split(/\s+/).filter(Boolean);
    if (!terms.length) { box.hidden = true; return; }
    var hits = INDEX.map(function (e) { return [score(e, terms), e]; })
      .filter(function (h) { return h[0] > 0; })
      .sort(function (a, b) { return b[0] - a[0]; }).slice(0, 12);
    box.innerHTML = "";
    if (!hits.length) { box.innerHTML = '<div class="none">No matches. Try a column name such as Zipf_Freq, or a feature name.</div>'; }
    hits.forEach(function (h, i) {
      var a = document.createElement("a");
      a.href = ROOT + h[1].u; if (i === 0) a.className = "active";
      a.innerHTML = "";
      a.appendChild(document.createTextNode(h[1].t));
      var sm = document.createElement("small"); sm.textContent = h[1].p; a.appendChild(sm);
      box.appendChild(a);
    });
    box.hidden = false;
  }
  if (input && box) {
    box.hidden = true;
    input.addEventListener("input", function () { render(input.value); });
    input.addEventListener("keydown", function (ev) {
      var items = box.querySelectorAll("a"), act = box.querySelector("a.active"), i = Array.prototype.indexOf.call(items, act);
      if (ev.key === "Enter" && act) { window.location.href = act.href; }
      else if (ev.key === "Escape") { box.hidden = true; input.blur(); }
      else if ((ev.key === "ArrowDown" || ev.key === "ArrowUp") && items.length) {
        ev.preventDefault(); if (act) act.classList.remove("active");
        i = ev.key === "ArrowDown" ? Math.min(items.length - 1, i + 1) : Math.max(0, i - 1);
        items[i].classList.add("active"); items[i].scrollIntoView({ block: "nearest" });
      }
    });
    document.addEventListener("keydown", function (ev) {
      if (ev.key === "/" && document.activeElement !== input) { ev.preventDefault(); input.focus(); }
    });
    document.addEventListener("click", function (ev) { if (!ev.target.closest(".search")) box.hidden = true; });
  }
})();
