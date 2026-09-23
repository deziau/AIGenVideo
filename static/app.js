const $ = (sel, root = document) => root.querySelector(sel);
const $$ = (sel, root = document) => [...root.querySelectorAll(sel)];

let selectedStyle = "cartoon";
let busy = false;

// ---- Tabs -------------------------------------------------------------------
$$(".tab").forEach((tab) =>
  tab.addEventListener("click", () => {
    $$(".tab").forEach((t) => { t.classList.toggle("active", t === tab); t.setAttribute("aria-selected", t === tab); });
    $$(".panel").forEach((p) => p.classList.toggle("active", p.id === tab.dataset.tab));
  })
);

// ---- Drop zones with inline video preview -------------------------------------
$$(".drop").forEach((zone) => {
  const input = $("#" + zone.dataset.input);
  const preview = $(".drop-preview", zone);
  const hint = $(".drop-hint", zone);
  const show = () => {
    const file = input.files[0];
    if (!file) return;
    preview.src = URL.createObjectURL(file);
    preview.hidden = false;
    hint.hidden = true;
    preview.play().catch(() => {});
  };
  input.addEventListener("change", show);
  zone.addEventListener("dragover", (e) => { e.preventDefault(); zone.classList.add("drag"); });
  zone.addEventListener("dragleave", () => zone.classList.remove("drag"));
  zone.addEventListener("drop", (e) => {
    e.preventDefault();
    zone.classList.remove("drag");
    if (e.dataTransfer.files.length) { input.files = e.dataTransfer.files; show(); }
  });
});

// ---- Prompt idea chips ----------------------------------------------------------
$$("#prompt-ideas .chip").forEach((chip) =>
  chip.addEventListener("click", () => {
    const box = $("#prompt");
    box.value = box.value.trim() ? `${box.value.trim()} ${chip.textContent}.` : `${chip.textContent}.`;
    box.focus();
  })
);

// ---- Sliders -------------------------------------------------------------------
for (const id of ["strength", "smoothing"]) {
  $("#" + id).addEventListener("input", (e) => ($(`#${id}-out`).textContent = e.target.value));
}

// ---- Engine status + style list from the server -------------------------------------
async function loadConfig() {
  const cfg = await fetch("/api/config").then((r) => r.json());
  $("#engines").innerHTML = `
    <span class="pill ${cfg.claude ? "on" : "off"}">Claude director ${cfg.claude ? "on" : "off"}</span>
    <span class="pill ${cfg.runway ? "on" : "off"}">${cfg.runway ? "Runway video model on" : "Runway off: local preview"}</span>`;
  const box = $("#styles");
  box.innerHTML = "";
  for (const s of cfg.styles) {
    const [name, desc] = s.label.split(" - ");
    const btn = document.createElement("button");
    btn.type = "button";
    btn.className = "style-opt" + (s.id === selectedStyle ? " active" : "");
    btn.innerHTML = `${name}<small>${desc || ""}</small>`;
    btn.addEventListener("click", () => {
      selectedStyle = s.id;
      $$(".style-opt").forEach((b) => b.classList.toggle("active", b === btn));
    });
    box.appendChild(btn);
  }
}

// ---- Job handling ---------------------------------------------------------------
function setBusy(value) {
  busy = value;
  $$("form button").forEach((b) => (b.disabled = value));
}

function showJob() {
  $("#job").hidden = false;
  $("#result").hidden = true;
  $("#plan").hidden = true;
  $("#bar").style.width = "2%";
  $("#job-msg").className = "job-msg";
  $("#job-msg").textContent = "Uploading…";
  $("#job").scrollIntoView({ behavior: "smooth", block: "start" });
}

function showError(message) {
  $("#job-msg").className = "job-msg error";
  $("#job-msg").textContent = message;
}

const escapeHtml = (s) => String(s).replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));

function renderPlan(plan, engine) {
  const el = $("#plan");
  el.innerHTML = `
    <h3>${escapeHtml(plan.title)}</h3>
    <p><span class="label">Reference</span><br>${escapeHtml(plan.reference_summary)}</p>
    <p class="label">Prompt sent to the video model</p>
    <blockquote>${escapeHtml(plan.generation_prompt)}</blockquote>
    ${plan.notes ? `<p><span class="label">Director's notes</span><br>${escapeHtml(plan.notes)}</p>` : ""}
    ${engine === "local_preview" ? `<p class="label">Rendered as a local preview (${escapeHtml(plan.preview_look.replace("_", " "))}). Add RUNWAYML_API_SECRET to generate the full video.</p>` : ""}`;
  el.hidden = false;
}

async function pollJob(id) {
  while (true) {
    const job = await fetch(`/api/jobs/${id}`).then((r) => r.json());
    $("#bar").style.width = `${Math.max(2, Math.round(job.progress * 100))}%`;
    $("#job-msg").textContent = job.message;
    if (job.status === "done") return job.result;
    if (job.status === "error") throw new Error(job.error);
    await new Promise((r) => setTimeout(r, 1500));
  }
}

async function submitJob(url, formData) {
  if (busy) return;
  setBusy(true);
  showJob();
  try {
    const res = await fetch(url, { method: "POST", body: formData });
    const body = await res.json();
    if (!res.ok) throw new Error(body.detail || "Request failed");
    const result = await pollJob(body.id);
    if (result.plan) renderPlan(result.plan, result.engine);
    $("#result-video").src = result.video_url;
    $("#download").href = result.video_url;
    $("#result").hidden = false;
    $("#job-msg").textContent = "Done";
  } catch (err) {
    showError(err.message);
  } finally {
    setBusy(false);
  }
}

$("#create-form").addEventListener("submit", (e) => {
  e.preventDefault();
  const fd = new FormData();
  fd.append("video", $("#create-video").files[0]);
  fd.append("prompt", $("#prompt").value);
  submitJob("/api/generate", fd);
});

function stylizeForm() {
  const file = $("#stylize-video").files[0];
  if (!file) { alert("Choose a video first"); return null; }
  const fd = new FormData();
  fd.append("video", file);
  fd.append("style", selectedStyle);
  fd.append("strength", $("#strength").value);
  return fd;
}

$("#stylize-form").addEventListener("submit", (e) => {
  e.preventDefault();
  const fd = stylizeForm();
  if (!fd) return;
  fd.append("smoothing", $("#smoothing").value);
  submitJob("/api/stylize", fd);
});

$("#preview-btn").addEventListener("click", async () => {
  const fd = stylizeForm();
  if (!fd || busy) return;
  const btn = $("#preview-btn");
  btn.disabled = true;
  btn.textContent = "Rendering…";
  try {
    const res = await fetch("/api/stylize/preview", { method: "POST", body: fd });
    if (!res.ok) throw new Error((await res.json()).detail || "Preview failed");
    const img = $("#frame-preview");
    img.src = URL.createObjectURL(await res.blob());
    img.hidden = false;
  } catch (err) {
    alert(err.message);
  } finally {
    btn.disabled = false;
    btn.textContent = "Preview a frame";
  }
});

loadConfig();
