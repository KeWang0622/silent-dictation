(() => {
  "use strict";
  const $ = (s, el = document) => el.querySelector(s);
  const $$ = (s, el = document) => Array.from(el.querySelectorAll(s));
  const send = (cmd, args = {}) => {
    try { window.webkit.messageHandlers.sd.postMessage(JSON.stringify({ cmd, ...args })); }
    catch (e) { console.log("send", cmd, args); }
  };
  window.addEventListener("error", (e) => send("log", { msg: "JS error: " + e.message + " @" + e.lineno + ":" + e.colno }));
  window.addEventListener("unhandledrejection", (e) => send("log", { msg: "JS rejection: " + e.reason }));
  const esc = (s) => String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
  const pct = (x) => (x == null ? "—" : Math.round(x * 100) + "%");
  const ago = (t) => {
    const s = Math.max(0, Date.now() / 1000 - t);
    if (s < 60) return "now";
    if (s < 3600) return Math.floor(s / 60) + "m ago";
    if (s < 86400) return Math.floor(s / 3600) + "h ago";
    return Math.floor(s / 86400) + "d ago";
  };
  const same = (a, b) => String(a).toLowerCase().replace(/[^a-z0-9' ]/g, "").trim() === String(b).toLowerCase().replace(/[^a-z0-9' ]/g, "").trim();
  const COPY_ICON = '<svg viewBox="0 0 24 24"><rect x="8" y="8" width="12" height="12" rx="2.5"/><path d="M16 8V6a2 2 0 0 0-2-2H6a2 2 0 0 0-2 2v8a2 2 0 0 0 2 2h2"/></svg>';

  let S = null;
  let page = null;
  let step = 0;
  let calCount = 60;
  let histFilter = "review";
  let keyTested = false;
  let showTrainView = false;
  let previewOn = false;
  const thumbs = {};
  const asked = new Set();
  const edits = {};
  const done = new Set();
  const sig = {};

  function toast(msg) {
    const t = $("#toast");
    t.textContent = msg;
    t.classList.add("on");
    clearTimeout(toast.timer);
    toast.timer = setTimeout(() => t.classList.remove("on"), 2200);
  }

  function changed(key, value) {
    const v = JSON.stringify(value);
    if (sig[key] === v) return false;
    sig[key] = v;
    return true;
  }

  // ------------------------------------------------------------------ navigation
  function go(p, anchor) {
    page = p;
    send("page", { page: p });
    $$(".page").forEach((el) => el.classList.toggle("active", el.id === "page-" + p));
    $$(".nav").forEach((el) => el.classList.toggle("active", el.dataset.page === p));
    if (p !== "calibrate" && S && !S.training.running) showTrainView = false;
    Object.keys(sig).forEach((k) => delete sig[k]);
    if (p === "setup") setStep(step, true);
    updatePreview();
    render();
    if (anchor) setTimeout(() => document.getElementById(anchor)?.scrollIntoView({ behavior: "smooth" }), 60);
    else $("#main").scrollTop = 0;
  }

  function setStep(n, quiet) {
    step = n;
    $$(".step").forEach((el) => el.classList.toggle("active", +el.dataset.step === n));
    $$("#steps i").forEach((el, i) => el.classList.toggle("done", i <= n));
    if (!quiet) { updatePreview(); render(); }
  }

  function wantPreview() {
    if (!S) return false;
    if (page === "setup") return step === 2;
    if (page === "calibrate") return !S.training.running && !showTrainView;
    if (page === "settings") return true;
    return false;
  }

  function updatePreview() {
    const want = wantPreview();
    if (want !== previewOn) { previewOn = want; send("preview", { on: want }); }
  }

  function setupIssues() {
    if (!S) return [];
    const out = [];
    if (S.permissions.camera !== "granted") out.push("camera access");
    if (!S.permissions.input || S.hotkeyError) out.push("Input Monitoring");
    if (!S.permissions.accessibility) out.push("Accessibility");
    return out;
  }

  // ------------------------------------------------------------------ render
  function render() {
    if (!S || !page) return;
    renderSidebar();
    if (page === "setup") renderSetup();
    else if (page === "home") renderHome();
    else if (page === "calibrate") renderCalibrate();
    else if (page === "history") renderHistory();
    else if (page === "settings") renderSettings();
  }

  function renderSidebar() {
    const st = $("#sideStatus");
    let cls = "ok", txt = "Ready";
    if (S.state === "loading") { cls = "busy"; txt = "Loading models…"; }
    else if (S.state === "error") { cls = "bad"; txt = "Something went wrong"; }
    else if (S.training.running) { cls = "busy"; txt = "Training…"; }
    else if (S.state === "listening") { cls = "rec"; txt = S.handsFree ? "Hands-free…" : "Reading lips…"; }
    else if (S.state === "decoding") { cls = "busy"; txt = "Transcribing…"; }
    else if (setupIssues().length) { cls = "bad"; txt = "Setup needed"; }
    st.className = "side-status " + cls;
    $(".txt", st).textContent = txt;
    st.dataset.page = setupIssues().length ? "setup" : "home";
    $("#sideHint").innerHTML = 'Hold <span class="kbd-inline">' + esc(S.settings.keyLabel) + "</span> to dictate · double-tap for hands-free";
    const review = S.recent.filter((r) => !r.reviewed && !done.has(r.id)).length;
    $("#navHist").textContent = review ? String(review) : "";
    $("#navCal").textContent = S.accuracyHistory.length ? "" : "New";
    $$("[data-keylabel]").forEach((el) => (el.textContent = S.settings.keyLabel));
    $$("[data-keycap]").forEach((el) => el.classList.toggle("down", !!S.keyDown));
  }

  // ---------------------------------------------------------------- setup
  const PERMS = [
    ["camera", '<svg viewBox="0 0 24 24"><rect x="3" y="6.5" width="12.5" height="11" rx="2.5"/><path d="M15.5 10.5 21 7.5v9l-5.5-3"/></svg>', "Camera", "Sees your lips. Video never leaves this Mac."],
    ["input", '<svg viewBox="0 0 24 24"><rect x="2.5" y="6" width="19" height="12" rx="2.5"/><path d="M6 9.5h.01M9.3 9.5h.01M12.6 9.5h.01M15.9 9.5h.01M18 9.5h.01M7.5 14.5h9"/></svg>', "Input Monitoring", "Notices when you hold the dictation key."],
    ["accessibility", '<svg viewBox="0 0 24 24"><path d="M9 4.5h6M9 19.5h6M12 4.5v15"/><path d="M4 9.5l3 2.5-3 2.5" class="mint"/></svg>', "Accessibility", "Types the text into the app you're using."],
  ];

  function permRows(el) {
    const rows = PERMS.map(([k, icon, title, desc]) => {
      const v = S.permissions[k];
      const ok = v === true || v === "granted";
      const action = ok ? '<span class="perm-state ok">Allowed</span>'
        : (k === "camera" && v === "denied")
          ? '<button class="btn" data-act="openSettings" data-kind="camera">Open Settings</button>'
          : '<button class="btn primary" data-act="permission" data-kind="' + k + '">Allow</button>';
      return '<div class="perm"><div class="perm-icon">' + icon + '</div><div class="perm-body"><b>' + title + "</b><span>" + desc + "</span></div>" + action + "</div>";
    }).join("");
    if (changed("perm-" + el.id, rows)) el.innerHTML = rows;
  }

  function renderSetup() {
    if (step === 1) {
      permRows($("#permList"));
      $("#restartNote").classList.toggle("hidden", !(S.permissions.input && S.hotkeyError));
      const missing = setupIssues().length;
      $("#permNext").textContent = missing ? "Continue anyway" : "Continue";
    }
    if (step === 2) {
      fillCameras($("#camSelectSetup"));
      renderChecks($("#camChecks"));
    }
    if (step === 3) {
      segState();
      $("#keyTest").textContent = S.keyDown ? "✓ Got it. Release to finish." : keyTested ? "✓ The key works." : "Press and hold the key to test it.";
      $("#keyTest").className = "hint" + (keyTested ? " ok" : "");
      const fn = S.settings.key === "fn";
      $("#globeNote").classList.toggle("hidden", !fn || S.globeOk);
      const conflict = fn && S.conflicts.length;
      $("#conflictNote").classList.toggle("hidden", !conflict);
      if (conflict) $("#conflictNote").textContent = S.conflicts.join(", ") + " also uses fn, so both would start together. Pick Right ⌥ here, or quit it while you're in public.";
    }
  }

  function renderChecks(el) {
    const p = S.cameraProblem || "";
    const items = [
      ["Camera access", S.permissions.camera === "granted"],
      ["Your face is visible", S.cameraOn && !/face|black|open/i.test(p) && S.permissions.camera === "granted"],
      ["Good light and distance", S.cameraOn && !p],
    ];
    const html = items.map(([t, ok]) => '<div class="check ' + (ok ? "ok" : "bad") + '"><i>' + (ok ? "✓" : "!") + "</i>" + t + "</div>").join("")
      + (p ? '<div class="hint">' + esc(p) + "</div>" : "");
    if (changed("checks-" + el.id, html)) el.innerHTML = html;
  }

  function fillCameras(sel) {
    const names = S.cameras.length ? S.cameras : ["No camera found"];
    const html = names.map((n) => "<option>" + esc(n) + "</option>").join("");
    if (changed("cams-" + sel.id, html)) sel.innerHTML = html;
    const cur = S.settings.camera_name || names[0];
    if (sel.value !== cur && document.activeElement !== sel) sel.value = cur;
  }

  function segState() {
    $$(".seg[data-setting]").forEach((seg) => {
      const v = S.settings[seg.dataset.setting];
      $$("button", seg).forEach((b) => b.classList.toggle("on", b.dataset.value === String(v)));
    });
    $$("input[data-setting]").forEach((inp) => { inp.checked = !!S.settings[inp.dataset.setting]; });
  }

  // ---------------------------------------------------------------- home
  function renderHome() {
    const k = S.settings.keyLabel;
    let title = "Hold " + k + " and mouth your words";
    let text = "Release to type into any app. Double-tap " + k + " for hands-free.";
    if (S.state === "loading") { title = "Getting ready…"; text = "Loading the lip-reading model. This takes a few seconds."; }
    else if (S.training.running) { title = "Training your personal model"; text = "Dictation resumes as soon as it finishes."; }
    else if (S.state === "listening") { title = S.handsFree ? "Hands-free — press " + k + " to finish" : "Reading your lips…"; text = "Mouth the words clearly, no sound needed."; }
    else if (S.state === "decoding") { title = "Transcribing…"; }
    else if (S.hotkeyError || !S.permissions.input) { title = "Allow Input Monitoring to use " + k; text = "Finish setup so the key can start dictation."; }
    $("#heroTitle").textContent = title;
    $("#heroText").textContent = text;
    const chips = [
      S.personal ? ["Personal model", "ok"] : ["Base model", ""],
      [S.cameraOn ? "Camera on" : "Camera off", S.cameraOn ? "accent" : ""],
      [{ fast: "Fast", balanced: "Balanced", accurate: "Accurate" }[S.settings.mode] || "Balanced", ""],
      S.llmActive ? ["AI clean-up on", "accent"] : null,
    ].filter(Boolean);
    const ch = chips.map(([t, c]) => '<span class="chip ' + c + '">' + esc(t) + "</span>").join("");
    if (changed("chips", ch)) $("#heroChips").innerHTML = ch;
    const issues = setupIssues();
    $("#setupBanner").classList.toggle("hidden", !issues.length);
    if (issues.length) $("#setupBannerText").textContent = "Still needed: " + issues.join(", ") + ".";
    $("#calBanner").classList.toggle("hidden", !!issues.length || S.accuracyHistory.length > 0);
    $("#statDict").textContent = S.stats.dictations.toLocaleString();
    $("#statWords").textContent = S.stats.words.toLocaleString();
    $("#statAcc").textContent = pct(S.stats.accuracy);
    $("#statAccFoot").textContent = S.stats.accuracy == null ? "Calibrate to measure" : "on sentences it never trained on";
    renderRecent();
  }

  function renderRecent() {
    const items = S.recent.slice(0, 6);
    if (!changed("recent", items.map((r) => r.id + r.text))) return;
    $("#recentList").innerHTML = items.length ? items.map((r) =>
      '<li><div class="r-text">' + esc(r.text) + (same(r.text, r.raw) ? "" : '<div class="r-raw">lip reading: ' + esc(r.raw.toLowerCase()) + "</div>") + '</div><span class="r-time">' + ago(r.time) +
      '</span><button class="icon-btn" title="Copy" data-act="copy" data-text="' + esc(r.text) + '">' + COPY_ICON + "</button></li>").join("")
      : '<li class="empty-line">Your dictations will appear here.</li>';
  }

  // ---------------------------------------------------------------- calibrate
  const STAGES = { baseline: "Measuring your starting accuracy…", adapt: "Adapting the model to your lips…",
    tune: "Tuning the decoder…", llm: "Checking AI clean-up…", done: "Done" };

  function renderCalibrate() {
    const cal = S.calibration, tr = S.training;
    const view = (tr.running || showTrainView) ? "calTraining" : cal.active ? "calSession" : "calIntro";
    $$(".cal-view").forEach((el) => el.classList.toggle("active", el.id === view));
    if (view === "calIntro") renderCalIntro();
    else if (view === "calSession") renderCalSession(cal);
    else renderTraining(tr);
    updatePreview();
  }

  function renderCalIntro() {
    $$("#calOptions .option").forEach((o) => o.classList.toggle("on", +o.dataset.count === calCount));
    renderChecks($("#calChecks"));
    $("#clipCount").textContent = S.clips ? S.clips + (S.clips === 1 ? " sentence" : " sentences") + " recorded" : "";
    $("#trainNowBtn").classList.toggle("hidden", S.clips < 20);
    renderChart();
  }

  function renderChart() {
    const h = S.accuracyHistory;
    if (!changed("chart", h)) return;
    const el = $("#accChart");
    if (!h.length) { el.innerHTML = '<div class="empty-chart">Your accuracy will appear here after your first calibration.</div>'; return; }
    const W = 800, H = 150, pad = 26;
    const n = Math.max(h.length, 2);
    const x = (i) => pad + (i * (W - 2 * pad)) / (n - 1 || 1);
    const y = (v) => H - pad - v * (H - 2 * pad);
    const line = (key) => h.map((p, i) => (i ? "L" : "M") + x(i).toFixed(1) + " " + y(p[key]).toFixed(1)).join(" ");
    const dots = h.map((p, i) => '<circle cx="' + x(i) + '" cy="' + y(p.personal) + '" r="4.5" fill="var(--accent)"/>' +
      '<text x="' + x(i) + '" y="' + (y(p.personal) - 11) + '" text-anchor="middle" font-size="12" font-weight="700" fill="var(--text)">' + pct(p.personal) + "</text>").join("");
    const grid = [0, 0.5, 1].map((v) => '<line x1="' + pad + '" x2="' + (W - pad) + '" y1="' + y(v) + '" y2="' + y(v) + '" stroke="var(--line)"/>' +
      '<text x="0" y="' + (y(v) + 4) + '" font-size="10.5" fill="var(--text3)">' + v * 100 + "%</text>").join("");
    el.innerHTML = '<svg viewBox="0 0 ' + W + " " + H + '" preserveAspectRatio="none">' + grid +
      '<path d="' + line("base") + '" fill="none" stroke="var(--text3)" stroke-width="2" stroke-dasharray="5 5"/>' +
      '<path d="' + line("personal") + '" fill="none" stroke="var(--accent)" stroke-width="3" stroke-linecap="round"/>' + dots + "</svg>";
  }

  function renderCalSession(cal) {
    const total = cal.total || 1;
    $("#calCounter").textContent = cal.done ? "All " + total + " sentences recorded" : "Sentence " + Math.min(cal.idx + 1, total) + " of " + total;
    $("#calAccuracy").textContent = cal.accuracy == null ? "" : "Accuracy so far: " + pct(cal.accuracy);
    $("#calBar").style.width = (100 * cal.idx / total) + "%";
    const prompt = cal.done ? "That's everything. Train your personal model 🎉" : cal.prompt;
    if (changed("prompt", prompt)) {
      const p = $("#calPrompt");
      p.textContent = prompt;
      p.style.animation = "none"; void p.offsetWidth; p.style.animation = "";
    }
    $("#calNext").textContent = cal.done ? "" : (cal.next || "");
    const st = $("#calStatus");
    let txt = "", cls = "";
    if (S.state === "loading") txt = "Loading models…";
    else if (cal.status === "recording" || S.state === "listening") { txt = "● Recording — release when you've finished the sentence"; cls = "rec"; }
    else if (cal.status === "saving") { txt = "Saving…"; }
    else if (cal.status === "saved") { txt = "✓ Saved. Next sentence."; cls = "ok"; }
    else if (cal.status === "retry") { txt = cal.message; cls = "warn"; }
    st.textContent = txt;
    st.className = "cal-status " + cls;
    const last = cal.last;
    const fb = last ? feedbackHtml(last) : '<div class="fb-label">What the model read</div><div class="muted">After each sentence you\'ll see what the current model read, word by word.</div>';
    if (changed("feedback", fb)) $("#calFeedback").innerHTML = fb;
    const fin = $("#calFinish");
    fin.disabled = S.clips < 20;
    fin.textContent = S.clips < 20 ? "Finish & train (" + S.clips + "/20)" : "Finish & train";
  }

  function feedbackHtml(last) {
    const words = last.align.map((o) => {
      if (o.op === "ok") return '<span class="w ok">' + esc(o.ref) + "</span>";
      if (o.op === "sub") return '<span class="w sub" title="you said “' + esc(o.ref) + '”"><s>' + esc(o.ref) + "</s>" + esc(o.hyp) + "</span>";
      if (o.op === "del") return '<span class="w del" title="missed">' + esc(o.ref) + "</span>";
      return '<span class="w ins" title="extra word">' + esc(o.hyp) + "</span>";
    }).join("");
    const right = last.align.filter((o) => o.op === "ok").length;
    return '<div class="fb-label"><span>What the model read</span><span>' + right + "/" + last.words + " words right</span></div><div class=\"words\">" + words + "</div>";
  }

  function renderTraining(tr) {
    const p = tr.running ? tr.progress : 1;
    $("#trainRing").style.strokeDashoffset = String(327 * (1 - p));
    $("#trainPct").textContent = Math.round(p * 100) + "%";
    $("#trainStage").textContent = tr.running ? (STAGES[tr.stage] || "Working…") : (tr.error ? tr.error : "");
    $("#trainHint").classList.toggle("hidden", !tr.running);
    const finished = !tr.running && (tr.result || tr.error);
    $("#trainTitle").textContent = tr.running ? "Training your personal model" : tr.error ? "Training didn't run" : "Training finished";
    $("#trainResult").classList.toggle("hidden", !tr.result || tr.running);
    $("#trainDone").classList.toggle("hidden", !finished);
    const note = $("#trainNote");
    note.classList.toggle("hidden", !tr.result || tr.running);
    if (tr.result) {
      $("#resBefore").textContent = pct(tr.result.base);
      $("#resAfter").textContent = pct(Math.max(tr.result.base, tr.result.personal));
      note.textContent = tr.result.active === "new" ? "Your personal model is now active."
        : tr.result.active === "current" ? "Your previous personal model is still the best, so it stays active."
        : "The base model is still better for now. Record more sentences and train again.";
    }
    const log = (tr.log || []).join("\n");
    if (changed("trainlog", log)) { const pre = $("#trainLog"); pre.textContent = log; pre.scrollTop = pre.scrollHeight; }
  }

  // ---------------------------------------------------------------- history
  function autosize(t) { t.style.height = "auto"; t.style.height = t.scrollHeight + 2 + "px"; }

  function renderHistory() {
    $$("#histFilter button").forEach((b) => b.classList.toggle("on", b.dataset.f === histFilter));
    const items = S.recent.filter((r) => histFilter === "all" || (!r.reviewed && !done.has(r.id)));
    const key = items.map((r) => r.id + (r.reviewed || done.has(r.id)));
    const tb = $("#histTrain");
    tb.disabled = S.clips < 20 || S.training.running;
    tb.textContent = S.training.running ? "Training…" : "Train personal model";
    tb.title = S.clips < 20 ? "Needs at least 20 saved sentences (" + S.clips + " so far)" : "";
    $("#histEmpty").classList.toggle("hidden", items.length > 0);
    if (!changed("hist", key)) return;
    $("#histList").innerHTML = items.map((r) => {
      const reviewed = r.reviewed || done.has(r.id);
      const img = thumbs[r.id] ? ' src="' + thumbs[r.id] + '"' : "";
      return '<li class="hitem' + (reviewed ? " reviewed" : "") + '" data-id="' + r.id + '"><img class="strip" data-thumb="' + r.id + '"' + img + ' alt="">' +
        '<div class="hbody"><textarea rows="1" spellcheck="false" data-id="' + r.id + '"' + (reviewed ? " disabled" : "") + ">" + esc(edits[r.id] ?? r.text) + "</textarea>" +
        '<span class="hmeta">' + (same(r.text, r.raw) ? "" : "lip reading: " + esc(r.raw.toLowerCase()) + " · ") + ago(r.time) + "</span></div>" +
        (reviewed ? '<span class="chip ok">Saved</span>' : '<div class="hactions"><button class="btn primary" data-act="save" data-id="' + r.id + '">Save</button><button class="btn ghost" data-act="discard" data-id="' + r.id + '">Discard</button></div>') + "</li>";
    }).join("");
    items.forEach((r) => { if (!thumbs[r.id] && !asked.has(r.id)) { asked.add(r.id); send("thumb", { id: r.id }); } });
    $$("#histList textarea").forEach(autosize);
  }

  // ---------------------------------------------------------------- settings
  function renderSettings() {
    segState();
    fillCameras($("#camSelect"));
    $("#setCamStatus").textContent = S.cameraProblem || (S.cameraOn ? "Working" : "Turns on when you dictate");
    const fn = S.settings.key === "fn";
    $("#setConflict").classList.toggle("hidden", !(fn && S.conflicts.length));
    $("#setConflictText").textContent = S.conflicts.join(", ") + " also uses fn. Choose another key or quit it while you're in public.";
    $("#modeHelp").textContent = { fast: "Quickest response, slightly less accurate", balanced: "Balanced speed and accuracy",
      accurate: "Most accurate, a little slower" }[S.settings.mode] || "";
    $("#llmSeg").classList.toggle("disabled", !S.llmAvailable);
    $$("#llmSeg button").forEach((b) => (b.disabled = !S.llmAvailable));
    $("#llmHelp").textContent = S.llmAvailable
      ? "A small on-device language model fixes look-alike words. Auto turns it on only if calibration shows it helps you."
      : "Optional 2 GB download: run ./setup.sh --with-llm in the source folder.";
    $("#personalHelp").textContent = S.personal ? "Active · " + pct(S.stats.accuracy) + " accuracy on held-out sentences" : "Using the base model. Calibrate to personalise it.";
    permRows($("#permSettings"));
    $("#version").textContent = S.version;
  }

  // ------------------------------------------------------------------ events
  document.addEventListener("click", (e) => {
    const link = e.target.closest("a[data-url]");
    if (link) { e.preventDefault(); send("openURL", { url: link.dataset.url }); return; }
    const nav = e.target.closest(".nav, .side-status");
    if (nav) { go(nav.dataset.page); return; }
    const opt = e.target.closest(".option");
    if (opt) { calCount = +opt.dataset.count; render(); return; }
    const filt = e.target.closest("#histFilter button");
    if (filt) { histFilter = filt.dataset.f; render(); return; }
    const segBtn = e.target.closest(".seg[data-setting] button");
    if (segBtn) {
      const key = segBtn.closest(".seg").dataset.setting;
      S.settings[key] = segBtn.dataset.value;
      if (key === "key") { keyTested = false; S.settings.keyLabel = segBtn.textContent; }
      send("set", { key, value: segBtn.dataset.value });
      render();
      return;
    }
    const el = e.target.closest("[data-act]");
    if (!el) return;
    const act = el.dataset.act;
    if (act === "step") setStep(+el.dataset.to);
    else if (act === "finishSetup") { send("set", { key: "onboarded", value: true }); go(el.dataset.then || "home"); }
    else if (act === "go") go(el.dataset.page);
    else if (act === "permission") send("permission", { kind: el.dataset.kind });
    else if (act === "openSettings") send("openSettings", { kind: el.dataset.kind });
    else if (act === "restart") send("restart");
    else if (act === "quit") send("quit");
    else if (act === "openData") send("openData");
    else if (act === "openURL") send("openURL", { url: el.dataset.url });
    else if (act === "calStart") { showTrainView = false; send("calibrate.start", { count: calCount }); }
    else if (act === "calMore") { showTrainView = false; send("calibrate.start", { count: 20 }); }
    else if (act === "calRedo") send("calibrate.redo");
    else if (act === "calSkip") send("calibrate.skip");
    else if (act === "calStop") send("calibrate.stop");
    else if (act === "train") { showTrainView = true; send("train"); if (page !== "calibrate") go("calibrate"); else render(); }
    else if (act === "copy") { send("copy", { text: el.dataset.text }); toast("Copied"); }
    else if (act === "save" || act === "discard") {
      const id = el.dataset.id;
      const input = $('textarea[data-id="' + id + '"]');
      if (act === "save") send("review.save", { id, text: input ? input.value : "" });
      else send("review.discard", { id });
      done.add(id);
      const li = el.closest(".hitem");
      if (li && histFilter === "review") { li.style.transition = "opacity .2s, transform .2s"; li.style.opacity = "0"; li.style.transform = "translateX(12px)"; }
      setTimeout(() => { delete sig.hist; render(); }, 200);
      if (act === "save") toast("Saved to your training data");
    }
  });

  document.addEventListener("change", (e) => {
    const el = e.target;
    if (el.matches("select[data-setting]")) { S.settings[el.dataset.setting] = el.value; send("set", { key: el.dataset.setting, value: el.value }); }
    if (el.matches("input[type=checkbox][data-setting]")) { S.settings[el.dataset.setting] = el.checked; send("set", { key: el.dataset.setting, value: el.checked }); }
  });
  document.addEventListener("input", (e) => {
    if (e.target.matches(".hbody textarea")) { edits[e.target.dataset.id] = e.target.value; autosize(e.target); }
  });
  document.addEventListener("keydown", (e) => {
    if (e.key === "Enter" && !e.shiftKey && e.target.matches(".hbody textarea")) {
      e.preventDefault();
      $('[data-act="save"][data-id="' + e.target.dataset.id + '"]')?.click();
    }
  });

  // ------------------------------------------------------------------ API used by the app
  window.SD = {
    init(o) { go(o.page || "home"); if (o.page === "setup") setStep(0); },
    go(o) { go(o.page, o.anchor); },
    state(s) {
      const first = !S;
      S = s;
      if (first && !page) go(S.settings.onboarded ? "home" : "setup");
      render();
    },
    key(down) {
      $$("[data-keycap]").forEach((el) => el.classList.toggle("down", down));
      if (S) S.keyDown = down;
      if (down && page === "setup" && step === 3) keyTested = true;
      if (page === "setup") renderSetup();
    },
    frame(f) {
      const scope = page === "setup" ? $('.step[data-step="2"]') : $("#page-" + page);
      if (!scope) return;
      $$("[data-cam]", scope).forEach((cam) => {
        if (cam.offsetParent === null) return;
        const img = $("img", cam);
        cam.classList.toggle("live", !!f.img);
        cam.classList.toggle("rec", !!f.recording);
        if (f.img) img.src = f.img;
        const box = $(".cam-box", cam);
        if (f.box && f.img) {
          box.style.display = "block";
          box.style.left = f.box[0] * 100 + "%"; box.style.top = f.box[1] * 100 + "%";
          box.style.width = f.box[2] * 100 + "%"; box.style.height = f.box[3] * 100 + "%";
        } else box.style.display = "none";
        const pr = $(".cam-problem", cam);
        pr.textContent = f.problem || "";
        pr.classList.toggle("on", !!f.problem);
        const empty = $(".cam-empty", cam);
        if (!f.on) empty.textContent = S && S.permissions.camera !== "granted" ? "Allow camera access first" : "Camera is off";
      });
      const mouth = $("[data-mouth]", scope);
      if (mouth && f.mouth) mouth.src = f.mouth;
      const meter = $("#calMeter");
      if (page === "calibrate" && meter && f.levels) {
        if (meter.children.length !== f.levels.length) meter.innerHTML = f.levels.map(() => "<i></i>").join("");
        f.levels.forEach((v, i) => (meter.children[i].style.height = Math.max(4, v * 34) + "px"));
      }
    },
    result(r) {
      if (r.delivered !== "app") return;
      const box = $("#tryBox");
      $(".try-empty", box)?.remove();
      const line = document.createElement("div");
      line.className = "try-line";
      line.innerHTML = esc(r.text) + "<small>" + (same(r.text, r.raw) ? "" : "lip reading: " + esc(r.raw) + " · ") + r.seconds + " s</small>";
      box.prepend(line);
      while (box.children.length > 4) box.lastChild.remove();
      if (page !== "home") toast(r.text);
    },
    thumb(t) {
      thumbs[t.id] = t.img;
      $$('img[data-thumb="' + t.id + '"]').forEach((img) => (img.src = t.img));
    },
  };

  send("ready");
})();
