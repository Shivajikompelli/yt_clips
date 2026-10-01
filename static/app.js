/* clipper Phase 3 UI — vanilla JS, no dependencies, works offline.
 *
 * Pages are hash-routed: #/ (home: create + list) and #/project/<id>.
 * All server data is inserted via textContent / createElement (never
 * innerHTML), so a hostile title can't inject markup.
 */
"use strict";

(() => {
  // ------------------------------------------------------------ constants
  const MAX_UPLOAD_MB = 1000; // server default (config.yaml max_upload_mb)
  const DEFAULT_CLIPS = 5;    // config.yaml clips_to_generate; used when num_clips is null
  const LIST_POLL_MS = 3000;  // home list refresh while anything is in flight
  const DETAIL_POLL_MS = 2000;// project page refresh until ready/failed
  const STAGES = ["queued", "downloading", "transcribing", "analyzing", "rendering", "ready"];
  const RUNNING_STATUSES = ["queued", "downloading", "transcribing", "analyzing", "rendering"];

  const $ = (sel, root) => (root || document).querySelector(sel);

  // ------------------------------------------------------------ helpers
  function el(tag, attrs, ...children) {
    const node = document.createElement(tag);
    if (attrs) {
      for (const [key, value] of Object.entries(attrs)) {
        if (value === null || value === undefined || value === false) continue;
        if (key === "class") node.className = value;
        else if (key === "text") node.textContent = value;
        else if (key.startsWith("on") && typeof value === "function") {
          node.addEventListener(key.slice(2), value);
        } else node.setAttribute(key, value);
        // NOTE: values land via setAttribute/textContent only — no HTML parsing
      }
    }
    for (const child of children) {
      if (child === null || child === undefined) continue;
      node.append(child);
    }
    return node;
  }

  function fmtTime(sec) {
    if (typeof sec !== "number" || !isFinite(sec) || sec < 0) return "—";
    const total = Math.round(sec);
    const m = Math.floor(total / 60);
    const s = total % 60;
    return `${m}:${String(s).padStart(2, "0")}`;
  }

  function fmtDate(iso) {
    if (!iso) return "";
    const d = new Date(iso);
    if (isNaN(d)) return iso;
    return d.toLocaleString(undefined, {
      year: "numeric", month: "short", day: "numeric",
      hour: "2-digit", minute: "2-digit",
    });
  }

  function statusKind(status) {
    if (status === "ready") return "ready";
    if (status === "failed") return "failed";
    return RUNNING_STATUSES.includes(status) ? "running" : "unknown";
  }

  async function api(path, options) {
    const resp = await fetch(path, options);
    let body = null;
    const text = await resp.text();
    if (text) {
      try { body = JSON.parse(text); } catch { body = text; }
    }
    if (!resp.ok) {
      const message = body && body.detail ? body.detail : `HTTP ${resp.status}`;
      const err = new Error(typeof message === "string" ? message : JSON.stringify(message));
      err.status = resp.status;
      throw err;
    }
    return body;
  }

  function toast(message, kind) {
    const region = $("#toast-region");
    if (!region) return;
    const node = el("div", { class: `toast${kind ? ` toast-${kind}` : ""}`, role: "status", text: message });
    region.append(node);
    setTimeout(() => node.remove(), 5000);
  }

  // Tracks per-render cancellation for polls/timeouts so a fast route
  // switch can't let a stale render clobber the new page.
  let renderToken = 0;

  // ------------------------------------------------------------ polling
  const Timers = {
    list: null,
    detail: null,
    health: null,
    clear(name) {
      if (this[name]) { clearInterval(this[name]); this[name] = null; }
    },
    clearAll() {
      this.clear("list"); this.clear("detail"); this.clear("health");
    },
  };

  function visibilityAware(fn) {
    // when the tab is hidden, skip the tick; it fires again on return
    return () => { if (!document.hidden) fn(); };
  }

  // ------------------------------------------------------------ health
  function pollHealth() {
    api("/health").then((h) => {
      const dot = $("#health-dot");
      const label = $("#health-label");
      if (!dot || !label) return;
      dot.className = `dot ${h.status === "ok" ? "dot-ok" : "dot-bad"}`;
      label.textContent = h.status === "ok" ? "api online" : "degraded";
    }).catch(() => {
      const dot = $("#health-dot", document);
      const label = $("#health-label");
      if (dot) dot.className = "dot dot-bad";
      if (label) label.textContent = "offline";
    });
  }

  // ------------------------------------------------------------ shared bits
  function skeletonList(n) {
    return el("div", { class: "proj-list" },
      ...Array.from({ length: n }, () => el("div", { class: "skeleton", "aria-hidden": "true" })));
  }

  function statusBadge(status) {
    const kind = statusKind(status);
    return el("span", { class: `badge badge-${kind}`, text: status });
  }

  function confirmDialog(title, bodyText) {
    // window.confirm keeps the UI dependency-free; message shows server detail
    return window.confirm(`${title}\n\n${bodyText}`);
  }

  // ------------------------------------------------------------ home page
  function renderHome() {
    Timers.clear("list");
    const app = $("#app");

    const urlInput = el("input", {
      id: "f-url", type: "text", name: "url",
      placeholder: "https://youtube.com/watch?v=… or C:\\videos\\talk.mp4 or file:///C:/…",
      autocomplete: "off", required: true,
    });
    const titleInput = el("input", {
      id: "f-title", type: "text", name: "title", placeholder: "auto from URL/file name",
      autocomplete: "off",
    });
    const clipsInput = el("input", {
      id: "f-clips", type: "number", name: "clips", min: "1", max: "20", value: String(DEFAULT_CLIPS),
    });
    const fileInput = el("input", {
      id: "f-file", type: "file", name: "file",
      accept: ".mp4,.mov,.mkv,.webm,video/mp4,video/quicktime,video/x-matroska,video/webm",
    });

    const errorBox = el("p", { class: "form-error hidden", id: "form-error", role: "alert" });
    const urlTab = el("button", { class: "tab active", type: "button", "data-tab": "url", text: "Link or path" });
    const fileTab = el("button", { class: "tab", type: "button", "data-tab": "file", text: "Upload file" });
    const submit = el("button", { class: "btn-primary", type: "submit", text: "Create clips" });

    const urlFields = el("div", null,
      el("div", { class: "field" },
        el("label", { for: "f-url", text: "Video URL or local path" }),
        urlInput,
        el("p", { class: "field-hint", text: "YouTube/HTTP link, a local file path, or a file:// URL." })),
      el("div", { class: "form-row" },
        el("div", { class: "field" },
          el("label", { for: "f-title", text: "Title (optional)" }),
          titleInput),
        el("div", { class: "field" },
          el("label", { for: "f-clips", text: "Clips" }),
          clipsInput,
          el("p", { class: "field-hint", text: "1–20" }))));

    // upload tab: the API's multipart branch reads only the 'file' field,
    // so no title/clips inputs here (title comes from the file name)
    const fileFields = el("div", { class: "hidden" },
      el("div", { class: "field" },
        el("label", { for: "f-file", text: "Video file" }),
        fileInput,
        el("p", { class: "field-hint", text: `.mp4 .mov .mkv .webm — up to ${MAX_UPLOAD_MB} MB · title comes from the file name` })));

    const uploadProgress = el("div", { class: "hidden", id: "upload-progress" },
      el("div", { class: "progress-track" }, el("div", { class: "progress-fill", id: "upload-fill" })),
      el("p", { class: "field-hint", id: "upload-label", text: "Uploading… 0%" }));

    const form = el("form", { id: "create-form", novalidate: true },
      el("div", { class: "tabs", role: "tablist" }, urlTab, fileTab),
      errorBox,
      urlFields,
      fileFields,
      uploadProgress,
      el("div", { class: "field" }, submit));

    let tab = "url";

    function showError(message) {
      errorBox.textContent = message;
      errorBox.classList.remove("hidden");
    }
    function clearError() { errorBox.classList.add("hidden"); }

    function setTab(next) {
      tab = next;
      urlTab.classList.toggle("active", next === "url");
      fileTab.classList.toggle("active", next === "file");
      urlFields.classList.toggle("hidden", next !== "url");
      fileFields.classList.toggle("hidden", next !== "file");
      clearError();
    }
    urlTab.addEventListener("click", () => setTab("url"));
    fileTab.addEventListener("click", () => setTab("file"));

    function validateClips(value) {
      const n = Number(value);
      if (!Number.isInteger(n) || n < 1 || n > 20) {
        return "Clips must be a whole number between 1 and 20.";
      }
      return null;
    }

    function submitJson(event) {
      event.preventDefault();
      clearError();
      const clipError = validateClips(clipsInput.value);
      if (clipError) return showError(clipError);
      const url = urlInput.value.trim();
      if (!url) return showError("Enter a video URL or local file path.");
      submit.disabled = true;
      api("/projects", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ url, title: titleInput.value.trim() || null, clips: Number(clipsInput.value) }),
      }).then((body) => { location.hash = `#/project/${body.id}`; })
        .catch((err) => { showError(err.message); submit.disabled = false; });
    }

    function submitUpload() {
      clearError();
      const file = fileInput.files && fileInput.files[0];
      if (!file) return showError("Choose a video file to upload.");

      const dot = file.name.lastIndexOf(".");
      const ext = dot >= 0 ? file.name.slice(dot).toLowerCase() : "";
      if (![".mp4", ".mov", ".mkv", ".webm"].includes(ext)) {
        return showError(`Unsupported file type '${ext || "(none)"}' (allowed: .mkv, .mov, .mp4, .webm)`);
      }
      if (file.size === 0) return showError("That file is empty.");
      if (file.size > MAX_UPLOAD_MB * 1024 * 1024) {
        return showError(`"${file.name}" is larger than the ${MAX_UPLOAD_MB} MB limit.`);
      }

      const data = new FormData();
      data.append("file", file, file.name);

      // XHR (not fetch) for upload progress events
      const xhr = new XMLHttpRequest();
      xhr.open("POST", "/projects");
      xhr.upload.addEventListener("progress", (e) => {
        if (!e.lengthComputable) return;
        const pct = Math.round((e.loaded / e.total) * 100);
        $("#upload-progress").classList.remove("hidden");
        $("#upload-fill").style.width = `${pct}%`;
        $("#upload-label").textContent = `Uploading… ${pct}%`;
      });
      xhr.addEventListener("load", () => {
        submit.disabled = false;
        uploadProgress.classList.add("hidden");
        let body = null;
        try { body = JSON.parse(xhr.responseText); } catch { /* non-JSON */ }
        if (xhr.status === 202 && body && body.id) {
          location.hash = `#/project/${body.id}`;
          return;
        }
        const detail = body && body.detail ? body.detail : `Upload failed (HTTP ${xhr.status})`;
        showError(typeof detail === "string" ? detail : JSON.stringify(detail));
      });
      xhr.addEventListener("error", () => {
        submit.disabled = false;
        uploadProgress.classList.add("hidden");
        showError("Network error during upload.");
      });
      submit.disabled = true;
      uploadProgress.classList.remove("hidden");
      $("#upload-fill").style.width = "0%";
      $("#upload-label").textContent = "Uploading… 0%";
      xhr.send(data);
    }

    form.addEventListener("submit", (event) => {
      event.preventDefault();
      if (tab === "url") submitJson(event); else submitUpload();
    });

    const listWrap = el("section", { id: "proj-list", "aria-busy": "true" }, skeletonList(3));

    app.replaceChildren(
      el("div", { class: "page-head" },
        el("div", null,
          el("h1", { class: "page-title", text: "Create clips" }),
          el("p", { class: "page-sub", text: "Paste a video link or local path, or upload a file. The pipeline downloads, transcribes, and picks highlight moments." }))),
      el("section", { class: "card", "aria-label": "Create a new project" }, form),
      el("div", { class: "clips-head" },
        el("h2", { text: "Projects" }),
        el("span", { class: "page-sub", id: "list-count", role: "status" })),
      listWrap);

    refreshList();
    Timers.clear("list");
    Timers.list = setInterval(visibilityAware(refreshList), LIST_POLL_MS);
  }

  function refreshList() {
    const wrap = $("#proj-list");
    if (!wrap) return;
    api("/projects").then((projects) => {
      if (!$("#proj-list")) return; // navigated away
      renderList(wrap, projects);
    }).catch((err) => {
      if (!$("#proj-list")) return;
      wrap.replaceChildren(el("p", { class: "form-error", text: `Could not load projects: ${err.message}` }));
    });
  }

  function renderList(wrap, projects) {
    wrap.setAttribute("aria-busy", "false");
    const count = $("#list-count");
    if (count) {
      const running = projects.filter((p) => statusKind(p.status) === "running").length;
      count.textContent = running > 0 ? `${projects.length} project(s), ${running} running` : `${projects.length} project(s)`;
    }
    if (!projects.length) {
      wrap.replaceChildren(el("div", { class: "empty" },
        el("h2", { text: "No projects yet" }),
        el("p", { text: "Create your first one with the form above — a YouTube link, a local file, or an upload." })));
      return;
    }
    const anyRunning = projects.some((p) => statusKind(p.status) === "running");
    if (!anyRunning) Timers.clear("list"); // nothing in flight: stop polling

    const cards = projects.map((p) => {
      const kind = statusKind(p.status);
      const running = kind === "running";
      const card = el("div", { class: "proj-card" },
        el("a", { class: "proj-row", href: `#/project/${p.id}`, "aria-label": `Open project ${p.title}` },
          el("div", { class: "proj-main" },
            el("p", { class: "proj-title", text: p.title }),
            el("p", { class: "proj-meta", text: `${p.source_type} · ${fmtDate(p.created_at)}` })),
          el("div", { class: "proj-status" },
            statusBadge(p.status),
            running ? el("div", { class: "mini-track" }, el("div", { class: "mini-fill", style: `width:${p.progress}%` })) : null)),
        kind === "failed" ? el("p", { class: "proj-error", text: p.error || "failed" }) : null,
        kind === "failed" || kind === "ready" ? el("div", { class: "proj-actions" },
          kind === "failed" ? el("button", {
            type: "button", text: "Retry",
            onclick: async () => {
              try {
                await api(`/projects/${p.id}/retry`, { method: "POST" });
                toast("Project requeued", "ok");
                refreshList();
              } catch (err) { toast(err.message, "error"); }
            },
          }) : null,
          el("button", {
            type: "button", class: "btn-danger", text: "Delete",
            onclick: async () => {
              if (!confirmDialog(`Delete "${p.title}"?`, "This permanently removes the project and its clips.")) return;
              try {
                await api(`/projects/${p.id}`, { method: "DELETE" });
                toast("Project deleted", "ok");
                refreshList();
              } catch (err) {
                // 409 shows the server's "project is currently rendering…" wording
                toast(err.message, "error");
                refreshList();
              }
            },
          })) : null);
      return card;
    });
    wrap.replaceChildren(...cards);
  }

  // ------------------------------------------------------------ project page
  function renderProject(id) {
    Timers.clear("detail");
    const app = $("#app");
    app.replaceChildren(el("div", { class: "skeleton", "aria-hidden": "true" }));

    const tick = () => {
      api(`/projects/${id}`).then((p) => {
        if (renderToken !== token) return; // superseded
        renderProjectView(p);
        const kind = statusKind(p.status);
        if (kind === "running") {
          if (!Timers.detail) {
            Timers.detail = setInterval(visibilityAware(tick), DETAIL_POLL_MS);
          }
        } else {
          Timers.clear("detail"); // terminal: stop polling
        }
      }).catch((err) => {
        if (renderToken !== token) return;
        if (err.status === 404) {
          $("#app").replaceChildren(el("div", { class: "empty" },
            el("h2", { text: "Project not found" }),
            el("p", { text: "It may have been deleted." }),
            el("p", null, el("a", { href: "#/", text: "← Back to projects" }))));
          Timers.clear("detail");
          return;
        }
        // transient error: keep the interval (if any) so a blip recovers
        toast(`Load failed: ${err.message}`, "error");
      });
    };

    const token = ++renderToken;
    tick();
  }

  function renderProjectView(p) {
    const app = $("#app");
    const kind = statusKind(p.status);
    const clips = Array.isArray(p.clips) ? p.clips : [];

    // stage tracker: queued → downloading → transcribing → analyzing → rendering → ready.
    // On failure the LAST stage the worker reported (p.stage) is the red one,
    // not "rendering"/"ready" — a download failure must read as such.
    const failedIdx = p.status === "failed" ? Math.max(0, STAGES.indexOf(p.stage)) : -1;
    const currentIdx = p.status === "failed" ? -1 : Math.max(0, STAGES.indexOf(p.status));
    const tracker = el("div", { class: "tracker", role: "list", "aria-label": "Pipeline stages" },
      ...STAGES.map((stage, i) => {
        let cls = "tracker-step";
        if (i === failedIdx) cls += " failed";
        else if (p.status === "ready" || i < currentIdx) cls += " done";
        else if (i === currentIdx) cls += " current";
        return el("div", { class: cls, role: "listitem" },
          el("span", { class: "tracker-dot", "aria-hidden": "true" }),
          el("span", { class: "tracker-label", text: stage }));
      }));

    const statusLine = el("p", {
      class: "page-sub", id: "proj-status", role: "status", "aria-live": "polite",
      text: p.status === "failed"
        ? `Failed during ${p.stage || "an early stage"} — see the error below.`
        : `${p.status}${p.stage && p.stage !== p.status ? ` · ${p.stage}` : ""} · ${p.progress}%`,
    });

    const progressLine = el("div", { class: "progress-line" },
      el("div", { class: "progress-track" }, el("div", { class: "progress-fill", style: `width:${p.progress}%` })),
      el("span", { text: `${p.progress}%` }));

    const card = el("section", { class: "card", "aria-label": "Project status" },
      el("div", { class: "page-head" },
        el("div", { style: "flex:1;min-width:0" },
          el("h1", { class: "page-title", text: p.title }),
          statusLine),
        statusBadge(p.status)),
      tracker,
      kind === "running" ? progressLine : null,
      p.error ? el("div", { class: "proj-error-box", role: "alert", text: p.error }) : null,
      el("dl", { class: "detail-grid" },
        el("div", null, el("dt", { text: "Source" }), el("dd", { text: p.source_value })),
        el("div", null, el("dt", { text: "Type" }), el("dd", { text: p.source_type })),
        el("div", null, el("dt", { text: "Clips" }),
          el("dd", { text: clips.length ? String(clips.length) : (p.num_clips == null ? String(DEFAULT_CLIPS) : `${p.num_clips} requested`) })),
        el("div", null, el("dt", { text: "Created" }), el("dd", { text: fmtDate(p.created_at) })),
        el("div", null, el("dt", { text: "Updated" }), el("dd", { text: fmtDate(p.updated_at) }))),
      kind === "failed" ? el("div", { class: "proj-actions", style: "justify-content:flex-start;margin-top:14px" },
        el("button", {
          type: "button", class: "btn-primary", text: "Retry project",
          onclick: async () => {
            try {
              await api(`/projects/${p.id}/retry`, { method: "POST" });
              toast("Project requeued", "ok");
              renderProject(p.id); // re-render immediately
            } catch (err) { toast(err.message, "error"); }
          },
        })) : null);

    let clipsSection = null;
    if (p.status === "ready" && clips.length) {
      clipsSection = [
        el("div", { class: "clips-head" },
          el("h2", { text: `Clips (${clips.length})` }),
          el("button", {
            type: "button", class: "btn-primary", text: "Download all",
            onclick: () => downloadAll(clips),
          })),
        el("div", { class: "clips-grid" },
          ...clips.map((clip) => renderClipCard(clip))),
      ];
    } else if (p.status === "ready" && !clips.length) {
      clipsSection = el("div", { class: "empty" },
        el("h2", { text: "No clips" }),
        el("p", { text: "The pipeline finished but produced no clips." }));
    }

    const page = [
      el("p", null, el("a", { class: "back-link", href: "#/", text: "← Back to projects" })),
      card,
      ...(clipsSection || []),
      el("div", { class: "proj-actions", style: "margin-top:26px" },
        el("button", {
          type: "button", class: "btn-danger", text: "Delete project",
          onclick: async () => {
            if (!confirmDialog(`Delete "${p.title}"?`, "This permanently removes the project and its clips.")) return;
            try {
              await api(`/projects/${p.id}`, { method: "DELETE" });
              toast("Project deleted", "ok");
              location.hash = "#/";
            } catch (err) {
              toast(err.message, "error"); // includes the 409 server wording
            }
          },
        })),
    ];

    // full re-render on each poll keeps progress/tracker fresh; the browser
    // only re-attaches videos when src changes, so playback is not interrupted
    app.replaceChildren(...page);
  }

  function renderClipCard(clip) {
    const range = el("p", { class: "clip-range" },
      `${fmtTime(clip.start_sec)} → ${fmtTime(clip.end_sec)} `,
      el("span", { class: "dur", text: `(${fmtTime((clip.end_sec || 0) - (clip.start_sec || 0))})` }));

    const reasonToggle = el("button", {
      class: "clip-reason-toggle", type: "button",
      "aria-expanded": "false", text: "Why this clip?",
    });
    const reason = clip.reason
      ? el("p", { class: "clip-reason hidden", text: clip.reason })
      : null;
    reasonToggle.addEventListener("click", () => {
      const expanded = reasonToggle.getAttribute("aria-expanded") === "true";
      reasonToggle.setAttribute("aria-expanded", String(!expanded));
      reasonToggle.textContent = expanded ? "Why this clip?" : "Hide explanation";
      if (reason) reason.classList.toggle("hidden");
    });

    return el("article", { class: "clip-card" },
      el("video", {
        class: "clip-video", controls: true, preload: "metadata",
        src: clip.stream_url, // 9:16 vertical preview with Range streaming
      }),
      el("div", { class: "clip-body" },
        el("div", { class: "clip-top" },
          el("h3", { class: "clip-title", text: clip.title || `Clip ${clip.idx + 1}` }),
          clip.score !== null && clip.score !== undefined
            ? el("span", { class: "score-badge", text: String(clip.score) }) : null),
        range,
        reasonToggle,
        reason,
        el("div", { class: "clip-actions" },
          el("a", { class: "btn-dl", href: clip.download_url, text: "Download" }))));
  }

  function downloadAll(clips) {
    // sequential via hidden iframe: no popup blockers, one file dialog each
    clips.forEach((clip, i) => {
      setTimeout(() => {
        const frame = document.createElement("iframe");
        frame.style.display = "none";
        frame.src = clip.download_url;
        document.body.append(frame);
        setTimeout(() => frame.remove(), 60000);
      }, i * 800);
    });
    toast(`Starting ${clips.length} download(s)…`, "ok");
  }

  // ------------------------------------------------------------ router
  function route() {
    Timers.clearAll();
    const hash = location.hash || "#/";
    const projectMatch = hash.match(/^#\/project\/([0-9a-fA-F]{32})$/);
    if (projectMatch) {
      renderProject(projectMatch[1]);
    } else if (hash === "#/" || hash === "#" || hash === "") {
      renderHome();
    } else {
      $("#app").replaceChildren(el("div", { class: "empty" },
        el("h2", { text: "Page not found" }),
        el("p", { text: "Unknown page." }),
        el("p", null, el("a", { href: "#/", text: "← Back to projects" }))));
    }
    window.scrollTo(0, 0);
  }

  window.addEventListener("hashchange", route);

  // ------------------------------------------------------------ boot
  route();
  pollHealth();
  Timers.health = setInterval(pollHealth, 30000);
  document.addEventListener("visibilitychange", () => {
    if (!document.hidden) {
      pollHealth();
      if (Timers.list) refreshList();
    }
  });
})();
