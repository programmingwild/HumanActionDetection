/* ==========================================================================
   ACTIONSCOPE — interface motion
   Preloader · custom cursor · scroll reveals · counters · magnetic buttons
   ========================================================================== */
(() => {
  "use strict";
  const reduce = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
  const fine = window.matchMedia("(pointer: fine)").matches;

  const when = (selector, cb, tries = 120) => {
    const el = document.querySelector(selector);
    if (el) return cb(el);
    if (tries > 0) setTimeout(() => when(selector, cb, tries - 1), 50);
  };

  /* ---------------- preloader ---------------- */
  function preloader() {
    when("#preloader", (pre) => {
      if (reduce) { pre.remove(); return; }
      const num = pre.querySelector(".pre-count");
      const bar = pre.querySelector(".pre-bar i");
      let n = 0;
      const tick = () => {
        n = Math.min(100, n + 18 + Math.floor(Math.random() * 22));
        if (num) num.innerHTML = String(n).padStart(3, "0") + "<small>%</small>";
        if (bar) bar.style.width = n + "%";
        if (n < 100) { setTimeout(tick, 35 + Math.random() * 35); return; }
        setTimeout(() => {
          pre.classList.add("done");
          setTimeout(() => pre.remove(), 650);
        }, 80);
      };
      setTimeout(tick, 120);
    });
  }

  /* ---------------- clock + nav ---------------- */
  function nav() {
    when(".nav", (navEl) => {
      const bar = navEl.querySelector(".nav-progress");
      const links = [...navEl.querySelectorAll(".nav-links a")];
      const ids = links.map((a) => a.getAttribute("href")).filter((h) => h && h.startsWith("#"));
      let ticking = false;
      const update = () => {
        const y = document.documentElement.scrollTop || window.scrollY || 0;
        navEl.classList.toggle("scrolled", y > 40);
        const max = document.documentElement.scrollHeight - window.innerHeight;
        if (bar) bar.style.width = (max > 0 ? (y / max) * 100 : 0) + "%";
        let current = ids[0] || null;
        for (const id of ids) {
          const el = document.querySelector(id);
          if (el && el.getBoundingClientRect().top < window.innerHeight * 0.45) current = id;
        }
        links.forEach((a) => a.classList.toggle("on", a.getAttribute("href") === current));
        ticking = false;
      };
      const onScroll = () => { if (!ticking) { ticking = true; requestAnimationFrame(update); } };
      window.addEventListener("scroll", onScroll, { passive: true });
      document.addEventListener("scroll", onScroll, { passive: true, capture: true });
      update();
    });

    const clock = document.querySelector(".clock");
    if (clock) {
      const tick = () => {
        const d = new Date();
        clock.textContent = [d.getHours(), d.getMinutes(), d.getSeconds()]
          .map((v) => String(v).padStart(2, "0")).join(":");
      };
      tick(); setInterval(tick, 1000);
    }

    // smooth anchor navigation with fixed-nav offset
    document.addEventListener("click", (e) => {
      const a = e.target.closest('a[href^="#"]');
      if (!a) return;
      const id = a.getAttribute("href");
      if (!id || id === "#") return;
      const el = document.querySelector(id);
      if (!el) return;
      e.preventDefault();
      const top = el.getBoundingClientRect().top + (document.documentElement.scrollTop || window.scrollY) - 78;
      document.documentElement.scrollTo({ top, behavior: reduce ? "auto" : "smooth" });
    });
  }

  /* ---------------- custom cursor ---------------- */
  function cursor() {
    if (!fine || reduce || document.querySelector(".cursor-dot")) return;
    const dot = document.createElement("div");
    const ring = document.createElement("div");
    dot.className = "cursor cursor-dot";
    ring.className = "cursor cursor-ring";
    document.body.append(dot, ring);
    document.body.classList.add("cursor-on");

    let mx = window.innerWidth / 2, my = window.innerHeight / 2;
    let rx = mx, ry = my;
    const move = (e) => {
      mx = e.clientX; my = e.clientY;
      dot.style.transform = `translate3d(${mx}px,${my}px,0)`;
      const hot = !!e.target.closest("a,button,input,textarea,select,label,.dropzone,.videozone,.verdict,.sample-btn");
      ring.classList.toggle("hot", hot);
      dot.classList.toggle("hot", hot);
    };
    window.addEventListener("pointermove", move, { passive: true });
    window.addEventListener("pointerdown", () => { ring.style.opacity = ".4"; });
    window.addEventListener("pointerup", () => { ring.style.opacity = "1"; });
    const raf = () => {
      rx += (mx - rx) * 0.16; ry += (my - ry) * 0.16;
      ring.style.transform = `translate3d(${rx}px,${ry}px,0)`;
      requestAnimationFrame(raf);
    };
    raf();
  }

  /* ---------------- reveal + counters ---------------- */
  function reveal() {
    const els = document.querySelectorAll(".reveal:not(.in), .stat b[data-count]:not(.counted)");
    if (!els.length) return;
    if (reduce) { els.forEach((el) => el.classList.add("in")); return; }
    const io = new IntersectionObserver((entries) => {
      entries.forEach((entry) => {
        if (!entry.isIntersecting) return;
        const el = entry.target;
        el.classList.add("in");
        if (el.dataset && el.dataset.count !== undefined) count(el);
        io.unobserve(el);
      });
    }, { threshold: 0.18, rootMargin: "0px 0px -6% 0px" });
    els.forEach((el) => io.observe(el));
  }

  function count(el) {
    if (el.classList.contains("counted")) return;
    el.classList.add("counted");
    const target = parseFloat(el.dataset.count);
    const decimals = parseInt(el.dataset.decimals || "0", 10);
    const suffix = el.dataset.suffix || "";
    const dur = 1300;
    const t0 = performance.now();
    const step = (t) => {
      const p = Math.min(1, (t - t0) / dur);
      const eased = 1 - Math.pow(1 - p, 3);
      const val = (target * eased).toFixed(decimals);
      el.innerHTML = `${val}<small>${suffix}</small>`;
      if (p < 1) requestAnimationFrame(step);
    };
    requestAnimationFrame(step);
  }

  /* ---------------- magnetic buttons + orb parallax ---------------- */
  function magnet() {
    if (reduce || !fine) return;
    document.querySelectorAll(".btn, button.run, button.sample-btn").forEach((el) => {
      if (el.dataset.mag) return;
      el.dataset.mag = "1";
      el.addEventListener("pointermove", (e) => {
        const r = el.getBoundingClientRect();
        const x = (e.clientX - r.left - r.width / 2) * 0.14;
        const y = (e.clientY - r.top - r.height / 2) * 0.18;
        el.style.transform = `translate(${x}px,${y}px)`;
      });
      el.addEventListener("pointerleave", () => { el.style.transform = ""; });
    });

    const orb = document.querySelector(".orb");
    if (orb && !orb.dataset.par) {
      orb.dataset.par = "1";
      window.addEventListener("pointermove", (e) => {
        const x = (e.clientX / window.innerWidth - 0.5) * 16;
        const y = (e.clientY / window.innerHeight - 0.5) * 16;
        orb.style.transform = `translate3d(${x}px,${y}px,0)`;
      }, { passive: true });
    }
  }

  /* ---------------- container gutter (Gradio scopes media queries) ------- */
  function fitContainer() {
    const c = document.querySelector("main.contain");
    if (!c) return;
    const w = window.innerWidth;
    const pad = w <= 760 ? "18px" : w <= 900 ? "24px" : w <= 1150 ? "32px" : "48px";
    c.style.setProperty("padding-left", pad, "important");
    c.style.setProperty("padding-right", pad, "important");
  }

  /* ---------------- image backdrops ---------------- */
  function imageBackdrops() {
    document.querySelectorAll(".dropzone .image-container img, .det-out .image-container img").forEach((img) => {
      const src = img.getAttribute("src");
      if (!src || src.startsWith("data:") || src.length < 5) return;
      const cont = img.closest(".image-container");
      if (!cont) return;
      let bg = cont.querySelector(".frame-blur");
      if (!bg) {
        bg = document.createElement("div");
        bg.className = "frame-blur";
        cont.insertBefore(bg, cont.firstChild);
      }
      if (bg.dataset.src !== src) {
        bg.dataset.src = src;
        bg.style.backgroundImage = `url("${src}")`;
        cont.classList.add("has-frame");
      }
    });
  }

  /* ---------------- screenshot / deep-link helper ---------------- */
  function deepLink() {
    const m = location.search.match(/[?&]chapter=(0[1-3])/);
    if (!m) return;
    const go = () => {
      const el = document.getElementById("chapter-" + m[1]);
      if (!el) return;
      const y = document.documentElement.scrollTop || window.scrollY;
      const top = el.getBoundingClientRect().top + y - 74;
      document.documentElement.scrollTo({ top, behavior: "instant" });
      if (Math.abs((document.documentElement.scrollTop || 0) - top) > 4) {
        document.documentElement.scrollBy({ top: top - (document.documentElement.scrollTop || 0), behavior: "instant" });
      }
    };
    const iv = setInterval(go, 400);
    setTimeout(() => clearInterval(iv), 9000);
    [400, 1200, 2400].forEach((t) => setTimeout(go, t));
  }

  /* ---------------- boot + re-arm on Gradio updates ---------------- */
  function boot() {
    fitContainer(); preloader(); nav(); cursor(); reveal(); magnet(); imageBackdrops(); deepLink();
  }
  window.addEventListener("resize", fitContainer, { passive: true });

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", boot);
  } else {
    boot();
  }

  // Gradio can mount/replace nodes after boot; re-run the cheap enhancers.
  let queued = false;
  const arm = () => {
    if (queued) return;
    queued = true;
    requestAnimationFrame(() => {
      queued = false;
      reveal(); magnet(); imageBackdrops(); fitContainer();
    });
  };
  new MutationObserver(arm).observe(document.documentElement, {
    childList: true, subtree: true, attributes: true, attributeFilter: ["src"],
  });
  window.addEventListener("load", () => { reveal(); magnet(); imageBackdrops(); });
})();
