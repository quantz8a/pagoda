/* Pagoda 站点交互层：路由 / 全文检索 / 过滤 / 卡片 / 微交互 */
(function () {
  "use strict";

  var ROUTES = ["home", "why", "sglang", "laya", "arch", "graph", "docs"];
  var INDEX = window.DOC_INDEX || [];
  var LABELS = window.BUCKET_LABELS || {};
  var currentBucket = "all";
  var openDocId = null;
  var pendingOpenId = null;

  /* ---------------- 工具 ---------------- */
  function $(sel, root) { return (root || document).querySelector(sel); }
  function $$(sel, root) { return Array.prototype.slice.call((root || document).querySelectorAll(sel)); }
  function escapeHtml(s) {
    return String(s == null ? "" : s).replace(/[&<>"']/g, function (c) {
      return { "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c];
    });
  }
  function normalize(s) { return (s || "").toLowerCase().replace(/\s+/g, " ").trim(); }

  /* ---------------- 路由 ---------------- */
  function currentRoute() {
    var h = (location.hash || "#home").replace(/^#\/?/, "");
    return ROUTES.indexOf(h) >= 0 ? h : "home";
  }

  function applyRoute() {
    var r = currentRoute();
    $$(".page").forEach(function (p) { p.classList.toggle("active", p.id === "page-" + r); });
    $$(".nav-link").forEach(function (a) { a.classList.toggle("active", a.dataset.route === r); });
    $$(".topbar").forEach(function (b) { b.classList.toggle("docs-route", r === "docs"); });
    if (r === "docs") {
      renderIndex(currentBucket);
      if (pendingOpenId) { var pid = pendingOpenId; pendingOpenId = null; deployOpen(pid); }
    } else {
      if (openDocId) openDocId = null;
    }
    window.scrollTo({ top: 0, behavior: "auto" });
  }

  /* ---------------- 全文检索 ---------------- */
  function score(entry, q) {
    var t = normalize(entry.title), en = normalize(entry.en), kw = normalize((entry.keywords || []).join(" "));
    var bl = normalize(entry.blurb), body = normalize(entry.body);
    var s = 0;
    if (t.indexOf(q) === 0) s += 90; else if (t.indexOf(q) >= 0) s += 60;
    if (en.indexOf(q) >= 0) s += 40;
    if (kw.indexOf(q) >= 0) s += 30;
    if (bl.indexOf(q) === 0) s += 25; else if (bl.indexOf(q) >= 0) s += 15;
    if (body.indexOf(q) >= 0) s += 10;
    // 词组拆分（应对中文多词条查询）
    q.split(" ").filter(Boolean).forEach(function (w) {
      if (w.length < 2) return;
      if (normalize(entry.title).indexOf(w) >= 0) s += 12;
      else if (normalize(entry.blurb).indexOf(w) >= 0) s += 8;
      else if (normalize(entry.body).indexOf(w) >= 0) s += 5;
    });
    return s;
  }

  function search(q) {
    var n = normalize(q);
    if (!n) return [];
    return INDEX.map(function (e) { return { e: e, s: score(e, n) }; })
      .filter(function (x) { return x.s > 0; })
      .sort(function (a, b) { return b.s - a.s; })
      .slice(0, 24)
      .map(function (x) { return x.e; });
  }

  /* ---------------- 过滤桶 ---------------- */
  function renderChips() {
    var bar = $("#filter-bar");
    if (!bar) return;
    var order = ["all", "docs", "concept", "sglang", "laya", "model", "metrics", "why"];
    bar.innerHTML = order.map(function (k) {
      var lb = LABELS[k] || { zh: k, dot: "#fff" };
      return '<button class="chip' + (currentBucket === k ? " on" : "") + '" data-bucket="' + k + '">' +
        '<span class="chip-dot" style="background:' + lb.dot + '"></span>' + escapeHtml(lb.zh) + "</button>";
    }).join("");
    $$(".chip", bar).forEach(function (c) {
      c.addEventListener("click", function () {
        currentBucket = c.dataset.bucket;
        renderChips();
        renderIndex(currentBucket);
      });
    });
  }

  /* ---------------- 卡片渲染（文档页 = 可搜索索引） ---------------- */
  function renderIndex(bucket) {
    var grid = $("#docs-grid");
    if (!grid) return;
    var items = INDEX.filter(function (e) { return bucket === "all" || e.bucket === bucket; });
    if (currentBucket === "all") {
      // 按桶分组展示，形成「关系雷达」
      var order = ["docs", "concept", "sglang", "laya", "model", "metrics", "why"];
      var html = "";
      order.forEach(function (b) {
        var list = items.filter(function (e) { return e.bucket === b; });
        if (!list.length) return;
        var lb = LABELS[b] || { zh: b };
        html += '<div class="group" data-group="' + b + '">' +
          '<div class="group-head"><span class="group-dot" style="background:' + (LABELS[b] && LABELS[b].dot) + '"></span>' +
          '<h2 class="group-title">' + escapeHtml(lb.zh) + "</h2><span class='group-count'>" + list.length + "</span></div>" +
          '<div class="card-grid">' + list.map(cardHtml).join("") + "</div></div>";
      });
      grid.innerHTML = html;
    } else {
      grid.innerHTML = '<div class="card-grid">' + items.map(cardHtml).join("") + "</div>";
      if (!items.length) grid.innerHTML = '<div class="empty">这个分类下暂时没有条目。</div>';
    }
    bindCards();
  }

  function cardHtml(e) {
    var lb = LABELS[e.bucket] || { zh: "", dot: "#888" };
    var detail = "";
    if (e.points && e.points.length) {
      detail += '<ul class="points">' + e.points.map(function (p) { return "<li>" + escapeHtml(p) + "</li>"; }).join("") + "</ul>";
    }
    if (e.cmd) {
      detail += '<div class="codeblock"><div class="codehead"><span>terminal</span><button class="copy-btn" data-code="' + escapeHtml(e.cmd) + '">复制</button></div><pre><code>' + escapeHtml(e.cmd) + "</code></pre></div>";
    }
    if (e.note) {
      detail += '<p class="note"><span class="note-tag">要点</span>' + escapeHtml(e.note) + "</p>";
    }
    return '<article class="card" id="card-' + e.id + '" data-id="' + e.id + '">' +
      '<div class="card-chrome"><span class="card-dot" style="background:' + lb.dot + '"></span>' +
      '<span class="card-bucket">' + escapeHtml(lb.zh) + "</span>" +
      (e.minutes ? '<span class="card-min">' + escapeHtml(e.minutes) + "</span>" : "") + "</div>" +
      '<h3 class="card-title">' + escapeHtml(e.title) + "</h3>" +
      (e.en ? '<div class="card-en">' + escapeHtml(e.en) + "</div>" : "") +
      '<p class="card-blurb">' + escapeHtml(e.blurb) + "</p>" +
      (e.keywords && e.keywords.length ? '<div class="kw">' + e.keywords.slice(0, 5).map(function (k) { return '<span class="kw-i">#' + escapeHtml(k) + "</span>"; }).join("") + "</div>" : "") +
      (detail ? '<div class="card-detail">' + detail + "</div>" : "") +
      "</article>";
  }

  function bindCards() {
    $$("#docs-grid .card").forEach(function (card) {
      card.addEventListener("click", function (ev) {
        if (ev.target.closest(".copy-btn") || ev.target.closest("code")) return;
        card.classList.toggle("open");
      });
    });
    $$("#docs-grid .copy-btn").forEach(bindCopy);
  }

  function bindCopy(btn) {
    if (btn.dataset.bound) return; btn.dataset.bound = "1";
    btn.addEventListener("click", function (ev) {
      ev.stopPropagation();
      var txt = btn.dataset.code || "";
      function done() { btn.textContent = "已复制"; setTimeout(function () { btn.textContent = "复制"; }, 1400); }
      if (navigator.clipboard && navigator.clipboard.writeText) {
        navigator.clipboard.writeText(txt).then(done, function () { fallback(); });
      } else { fallback(); }
      function fallback() {
        var ta = document.createElement("textarea");
        ta.value = txt; document.body.appendChild(ta); ta.select();
        try { document.execCommand("copy"); } catch (e) {}
        document.body.removeChild(ta); done();
      }
    });
  }

  /* ---------------- 打开某条「文档」卡片 ---------------- */
  function deployOpen(id) {
    setTimeout(function () {
      var card = document.getElementById("card-" + id);
      if (card) {
        card.classList.add("open");
        card.scrollIntoView({ behavior: "smooth", block: "center" });
      }
    }, 60);
  }

  function openEntry(id) {
    closeSearch();
    currentBucket = "all";
    renderChips();
    if (location.hash !== "#docs") {
      pendingOpenId = id;
      location.hash = "docs";
    } else {
      renderIndex("all");
      deployOpen(id);
    }
  }

  /* ---------------- 搜索面板（命令面板） ---------------- */
  function openSearch() { $("#search-overlay").classList.add("open"); setTimeout(function () { $("#search-input").focus(); }, 40); }
  function closeSearch() { $("#search-overlay").classList.remove("open"); $("#search-input").value = ""; $("#search-results").innerHTML = ""; $("#search-count").textContent = ""; }

  function onSearchInput() {
    var q = $("#search-input").value;
    var res = search(q);
    var box = $("#search-results");
    $("#search-count").textContent = q ? (res.length ? "找到 " + res.length + " 条 · ↑↓ 选择 · Enter 打开 · Esc 关闭" : "没找到，换个词试试？比如 “checkpoint” 或 “前缀缓存”") : "输入关键词，检索 文档 / 机制 / 整合 / Laya / 指标";
    if (!q) { box.innerHTML = ""; return; }
    box.innerHTML = res.map(function (e, i) {
      var lb = LABELS[e.bucket] || { zh: "", dot: "#888" };
      return '<div class="result" data-id="' + e.id + '" data-i="' + i + '">' +
        '<span class="result-l" style="background:' + lb.dot + '"></span>' +
        '<div class="result-main"><div class="result-title">' + escapeHtml(e.title) + '</div>' +
        '<div class="result-blurb">' + escapeHtml(e.blurb) + '</div></div>' +
        '<span class="result-bucket">' + escapeHtml(lb.zh) + '</span></div>';
    }).join("");
    $$(".result", box).forEach(function (r) {
      r.addEventListener("click", function () { openEntry(r.dataset.id); });
    });
  }

  /* ---------------- 键盘快捷键 ---------------- */
  function onKey(e) {
    var tag = (document.activeElement && document.activeElement.tagName || "").toLowerCase();
    var typing = tag === "input" || tag === "textarea";
    var overlay = $("#search-overlay").classList.contains("open");
    if ((e.key === "/" && !typing) || ((e.metaKey || e.ctrlKey) && e.key.toLowerCase() === "k")) {
      e.preventDefault();
      if (overlay) closeSearch(); else openSearch();
      return;
    }
    if (e.key === "Escape" && overlay) { e.preventDefault(); closeSearch(); }
  }

  /* ---------------- 滚动进入动画 ---------------- */
  function initReveal() {
    if (!("IntersectionObserver" in window)) { $$(".reveal").forEach(function (el) { el.classList.add("in"); }); return; }
    var io = new IntersectionObserver(function (entries) {
      entries.forEach(function (en) { if (en.isIntersecting) { en.target.classList.add("in"); io.unobserve(en.target); } });
    }, { threshold: 0.12, rootMargin: "0px 0px -40px 0px" });
    $$(".reveal").forEach(function (el) { io.observe(el); });
  }

  /* ---------------- 首页 hero 光标辉光 ---------------- */
  function initGlow() {
    var hero = $("#hero");
    if (!hero || !window.matchMedia("(hover:hover)").matches) return;
    var glow = document.createElement("div"); glow.className = "hero-glow"; hero.appendChild(glow);
    hero.addEventListener("pointermove", function (e) {
      var r = hero.getBoundingClientRect();
      glow.style.transform = "translate(" + (e.clientX - r.left - 150) + "px," + (e.clientY - r.top - 150) + "px)";
    });
  }

  /* ---------------- tabs（架构页简单切换占位，如需可扩展） ---------------- */
  function initNav() {
    $$(".nav-link").forEach(function (a) {
      a.addEventListener("click", function () {
        if (a.dataset.route) { location.hash = a.dataset.route; }
      });
    });
    $("#search-btn").addEventListener("click", openSearch);
    $("#search-overlay").addEventListener("click", function (e) { if (e.target === this) closeSearch(); });
    $("#search-input").addEventListener("input", onSearchInput);
    $$("[data-open-search]").forEach(function (b) { b.addEventListener("click", openSearch); });
    $$("[data-open-entry]").forEach(function (el) {
      el.addEventListener("click", function () { openEntry(el.dataset.openEntry); });
    });
  }

  document.addEventListener("DOMContentLoaded", function () {
    initNav();
    renderChips();
    applyRoute();
    initReveal();
    initGlow();
    $$(".copy-btn").forEach(bindCopy);
    window.addEventListener("hashchange", applyRoute);
    document.addEventListener("keydown", onKey);
  });
})();