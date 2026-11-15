// A colour for the CHARACTER socket, so a character wire is distinguishable from an image wire.
import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

app.registerExtension({
  name: "omnichar.character",
  init() {
    const types = app.canvas?.default_connection_color_byType;
    const links = LiteGraph.LINK_COLORS ?? {};
    const colour = "#c77dff";
    if (types) types["CHARACTER"] = colour;
    links["CHARACTER"] = colour;
  },
});

// Licensed characters: a status line and two buttons on Load Character. Nothing here is saved
// with the workflow, and the secret only ever travels to this ComfyUI's own server.
const LOADER = "OmnicharLoadCharacter";
const STATUS = "license_status";
const live = new Set();

function widget(node, name) {
  return node.widgets?.find((w) => w.name === name);
}

function statusText(license) {
  if (!license.stored) return "Licensed · Enter secret";
  if (license.last_error?.message) return `Licensed · ${license.last_error.message}`;
  if (license.verified_until) {
    const when = new Date(license.verified_until);
    if (!Number.isNaN(when.getTime())) {
      return `Licensed · Verified until ${when.toLocaleString()}`;
    }
  }
  return "Licensed · Secret saved";
}

function unadorn(node) {
  for (const name of [STATUS, "Enter secret", "Forget"]) {
    const index = node.widgets?.findIndex((w) => w.name === name && w.omnicharLicense) ?? -1;
    if (index >= 0) node.widgets.splice(index, 1);
  }
  node.omnicharLicense = null;
  node.setSize([node.size[0], node.computeSize()[1]]);
  node.setDirtyCanvas(true, true);
}

function button(node, name, action) {
  const added = node.addWidget("button", name, null, action, { serialize: false });
  added.serialize = false;
  added.omnicharLicense = true;
  added.computeSize = (width) => [width, 24];
  return added;
}

// A status widget rather than canvas drawing, so the Vue node renderer shows it too.
function showStatus(node, text) {
  let status = node.widgets?.find((w) => w.name === STATUS && w.omnicharLicense);
  if (!status) status = button(node, STATUS, () => refresh(node));
  status.label = text;
}

async function refresh(node) {
  // Only the newest lookup may draw; an older reply for a previous character is dropped.
  const ticket = (node.omnicharRefresh = (node.omnicharRefresh ?? 0) + 1);
  const char = widget(node, "char")?.value ?? "";
  const charPath = widget(node, "char_path")?.value ?? "";
  const query = new URLSearchParams({ char, char_path: charPath });
  let found = { licensed: false };
  try {
    const response = await api.fetchApi(`/omnichar/character-license?${query}`);
    if (response.ok) found = await response.json();
  } catch {
    found = { licensed: false };
  }
  if (ticket !== node.omnicharRefresh) return;
  if (!found.licensed || !found.license_id) {
    if (node.omnicharLicense) unadorn(node);
    return;
  }
  let license = { stored: false };
  try {
    const response = await api.fetchApi(
      `/omnichar/license/${encodeURIComponent(found.license_id)}`,
    );
    if (response.ok) license = await response.json();
  } catch {
    license = { stored: false };
  }
  if (ticket !== node.omnicharRefresh) return;
  const fresh = !node.omnicharLicense;
  node.omnicharLicense = { id: found.license_id };
  showStatus(node, statusText(license));
  if (fresh) {
    button(node, "Enter secret", () => askForSecret(node));
    button(node, "Forget", () => forget(node));
  }
  node.setSize([node.size[0], Math.max(node.size[1], node.computeSize()[1])]);
  node.setDirtyCanvas(true, true);
}

async function forget(node) {
  const id = node.omnicharLicense?.id;
  if (!id) return;
  if (!confirm("Forget the license secret for this character?")) return;
  try {
    await api.fetchApi(`/omnichar/license/${encodeURIComponent(id)}`, { method: "DELETE" });
  } catch {
    // The status line below shows whether the secret is still stored.
  }
  await refresh(node);
}

function askForSecret(node) {
  const id = node.omnicharLicense?.id;
  if (!id) return;

  const backdrop = document.createElement("div");
  Object.assign(backdrop.style, {
    position: "fixed",
    inset: "0",
    background: "rgba(0, 0, 0, 0.5)",
    display: "flex",
    alignItems: "center",
    justifyContent: "center",
    zIndex: "10000",
  });
  const box = document.createElement("form");
  Object.assign(box.style, {
    background: "var(--comfy-menu-bg)",
    color: "var(--fg-color)",
    border: "1px solid var(--border-color)",
    borderRadius: "8px",
    padding: "16px",
    width: "360px",
    display: "flex",
    flexDirection: "column",
    gap: "10px",
    fontSize: "14px",
  });
  const label = document.createElement("label");
  label.textContent = "License secret from the share email";
  const input = document.createElement("input");
  input.type = "password";
  input.autocomplete = "off";
  input.spellcheck = false;
  input.placeholder = "lsk_live_...";
  Object.assign(input.style, {
    background: "var(--comfy-input-bg)",
    color: "var(--input-text)",
    border: "1px solid var(--border-color)",
    borderRadius: "4px",
    padding: "6px 8px",
  });
  const note = document.createElement("div");
  note.style.minHeight = "18px";
  note.style.color = "var(--descrip-text)";
  const actions = document.createElement("div");
  Object.assign(actions.style, { display: "flex", justifyContent: "flex-end", gap: "8px" });
  const cancel = document.createElement("button");
  cancel.type = "button";
  cancel.textContent = "Cancel";
  const save = document.createElement("button");
  save.type = "submit";
  save.textContent = "Save";
  actions.append(cancel, save);
  box.append(label, input, note, actions);
  backdrop.append(box);

  const close = () => {
    input.value = "";
    backdrop.remove();
    refresh(node);
  };
  cancel.addEventListener("click", close);
  backdrop.addEventListener("keydown", (event) => {
    if (event.key === "Escape") close();
  });
  box.addEventListener("submit", async (event) => {
    event.preventDefault();
    const secret = input.value.trim();
    input.value = "";
    if (!secret) return;
    save.disabled = true;
    try {
      const response = await api.fetchApi("/omnichar/license", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ license_id: id, secret }),
      });
      if (response.ok) {
        note.textContent = "Saved.";
        setTimeout(close, 600);
        return;
      }
      const answer = await response.json().catch(() => ({}));
      note.textContent = typeof answer.error === "string" ? answer.error : "Not saved.";
    } catch {
      note.textContent = "Not saved.";
    }
    save.disabled = false;
  });

  document.body.append(backdrop);
  input.focus();
}

app.registerExtension({
  name: "omnichar.license",
  async beforeRegisterNodeDef(nodeType, nodeData) {
    if (nodeData?.name !== LOADER) return;

    const created = nodeType.prototype.onNodeCreated;
    nodeType.prototype.onNodeCreated = function () {
      const out = created?.apply(this, arguments);
      live.add(this);
      let typing = null;
      for (const name of ["char", "char_path"]) {
        const watched = widget(this, name);
        if (!watched) continue;
        const callback = watched.callback;
        watched.callback = (...args) => {
          const result = callback?.apply(watched, args);
          if (name === "char") {
            refresh(this);
          } else {
            // A path is typed a character at a time; look it up once typing pauses.
            clearTimeout(typing);
            typing = setTimeout(() => refresh(this), 400);
          }
          return result;
        };
      }
      // After configure() has put a saved workflow's values in place.
      setTimeout(() => refresh(this), 0);
      return out;
    };

    const removed = nodeType.prototype.onRemoved;
    nodeType.prototype.onRemoved = function () {
      live.delete(this);
      return removed?.apply(this, arguments);
    };
  },
  setup() {
    const again = (event) => {
      const id = event?.detail?.node_id ?? event?.detail?.node;
      for (const node of live) {
        if (!node.omnicharLicense) continue;
        if (id == null || String(node.id) === String(id)) refresh(node);
      }
    };
    api.addEventListener("executed", again);
    api.addEventListener("execution_error", again);
    api.addEventListener("execution_success", () => {
      for (const node of live) if (node.omnicharLicense) refresh(node);
    });
  },
});
