/* helpers.js — shared UI utilities for TheNoise.
   Plain global script, loaded BEFORE app.js (no modules). */

const $ = id => document.getElementById(id);

// Reads that owe an instant answer pass this; /upscale and /controlmap hold one open for a render.
const PATIENCE = 10000;

/* ---------- persisted preferences (light/dark, sidebar left/right) ---------- */
function initToggle({ key, btnId, initial, current, apply, next }) {
  apply(initial());
  const btn = $(btnId);
  if (btn) btn.addEventListener('click', () => {
    const value = next(current());
    localStorage.setItem(key, value);
    apply(value);
  });
}

/* ---------- theme (light / dark) ---------- */
const THEME_UI = {
  light: { icon: '\u2600', label: 'Light' }, 
  dark: { icon: '\u263E', label: 'Dark' },
};

function applyTheme(theme) {
  const light = theme === 'light';
  document.body.classList.toggle('light', light);
  const meta = document.querySelector('meta[name="theme-color"]');
  if (meta) meta.setAttribute('content', getComputedStyle(document.body).getPropertyValue('--bg').trim());
  const ui = THEME_UI[theme];
  const icon = $('theme_icon');
  if (icon) icon.textContent = ui.icon;
  const label = $('theme_label');
  if (label) label.textContent = ui.label;
}

initToggle({
  key: 'thenoise-theme', btnId: 'theme_toggle',
  initial: () => localStorage.getItem('thenoise-theme') ||
    (window.matchMedia('(prefers-color-scheme: light)').matches ? 'light' : 'dark'),
  current: () => document.body.classList.contains('light') ? 'light' : 'dark',
  apply: applyTheme, next: v => v === 'light' ? 'dark' : 'light',
});

/* ---------- sidebar side (left / right) ---------- */
const SIDEBAR_UI = {
  left: { icon: '\u21E5' },
  right: { icon: '\u21E4' },
};

function applySidebar(side) {
  const right = side === 'right';
  document.body.classList.toggle('sidebar-right', right);
  const label = $('sidebar_label');
  if (label) label.textContent = right ? 'Right' : 'Left';
  const icon = $('sidebar_icon');
  if (icon) icon.textContent = right ? SIDEBAR_UI.right.icon : SIDEBAR_UI.left.icon;
}

initToggle({
  key: 'thenoise-sidebar', btnId: 'sidebar_toggle',
  initial: () => localStorage.getItem('thenoise-sidebar') === 'right' ? 'right' : 'left',
  current: () => document.body.classList.contains('sidebar-right') ? 'right' : 'left',
  apply: applySidebar, next: v => v === 'right' ? 'left' : 'right',
});

/* ---------- post-decode knobs (declared once server-side: thenoise/postprocess) ----------

The sidebar's postprocess controls are built from ``GET /postprocess``, so a new knob
appears in the form, the request body, the settings restore and the info panel without
a line of UI code. Ids follow the panes' convention: ``<prefix><name>``, where the
prefix is '' on Generate and 'edit_' on Edit.
*/
let postprocessOptions = [];

function hintSpan(text) {
  const span = document.createElement('span');
  span.className = 'hint';
  span.textContent = ' (' + text + ')';
  return span;
}

function checkControl(prefix, option) {
  const label = document.createElement('label');
  label.className = 'check';
  const input = document.createElement('input');
  input.type = 'checkbox';
  input.id = prefix + option.name;
  input.checked = option.default === true;
  input.addEventListener('change', refreshPostprocessNotes);
  label.append(input, document.createTextNode(' ' + option.label));
  if (option.hint) label.append(hintSpan(option.hint));
  return label;
}

function rangeControl(prefix, option) {
  const id = prefix + option.name;
  const box = document.createElement('div');
  box.className = 'range';

  const head = document.createElement('div');
  head.className = 'range-head';
  const label = document.createElement('label');
  label.className = 'label';
  label.htmlFor = id;
  label.textContent = option.label;
  const val = document.createElement('span');
  val.className = 'val';
  val.id = id + '_val';
  head.append(label, val);

  const input = document.createElement('input');
  input.type = 'range';
  input.id = id;
  // The knob's own range is what the server accepts; ui_min/ui_max is what is worth
  // offering on a slider (film grain is accepted to 10 but sane to 1).
  for (const [key, attr] of [['min', 'min'], ['max', 'max'], ['step', 'step']]) {
    if (option['ui_' + key] !== undefined) input[attr] = option['ui_' + key];
    else if (option[key] !== undefined) input[attr] = option[key];
  }
  input.value = option.default;
  input.addEventListener('input', () => {
    showRangeValue(prefix, option.name);
    refreshPostprocessNotes();
  });

  box.append(head, input);
  return box;
}

function buildPostprocessControls(prefix) {
  const box = $(prefix + 'postprocess_controls');
  if (!box) return;
  box.innerHTML = '';
  for (const option of postprocessOptions) {
    box.append(option.type === 'boolean'
      ? checkControl(prefix, option) : rangeControl(prefix, option));
  }
}

// "Saturation 0.30 · Film grain 0.50" — what the collapsed section would otherwise hide.
function postprocessNote(prefix) {
  const active = [];
  for (const option of postprocessOptions) {
    const input = $(prefix + option.name);
    if (!input) continue;
    if (option.type === 'boolean') {
      if (input.checked !== option.default) active.push(option.label);
      continue;
    }
    const value = input.valueAsNumber;
    if (Number.isFinite(value) && value !== option.default) {
      active.push(option.label + ' ' + value.toFixed(2));
    }
  }
  return active.join(' · ');
}

// Both panes carry the controls, so both headers are kept truthful.
function refreshPostprocessNotes() {
  for (const p of ['', 'edit_']) {
    const note = $(p + 'postprocess_note');
    if (note) note.textContent = postprocessNote(p);
  }
}

function showRangeValue(prefix, name) {
  const val = $(prefix + name + '_val');
  if (val) val.textContent = parseFloat($(prefix + name).value).toFixed(2);
}

// The knobs as the request body wants them.
function collectPostprocess(prefix) {
  const body = {};
  for (const option of postprocessOptions) {
    const input = $(prefix + option.name);
    if (!input) continue;
    body[option.name] = option.type === 'boolean' ? input.checked : input.valueAsNumber;
  }
  return body;
}

function restorePostprocess(prefix, meta) {
  for (const option of postprocessOptions) {
    const input = $(prefix + option.name);
    const value = meta[option.name];
    if (!input || value == null) continue;
    if (option.type === 'boolean') input.checked = !!value;
    else {
      input.value = value;
      showRangeValue(prefix, option.name);
    }
  }
  refreshPostprocessNotes();
}

// One row per knob the image was actually made with; a default is not worth showing.
function postprocessInfoRows(meta) {
  const rows = [];
  for (const option of postprocessOptions) {
    if (!(option.name in meta) || meta[option.name] === option.default) continue;
    rows.push([option.label, meta[option.name], '']);
  }
  return rows;
}

async function loadPostprocess() {
  const data = await fetchJSON('/postprocess');
  postprocessOptions = (data && data.options) || [];
  for (const p of ['', 'edit_']) {
    buildPostprocessControls(p);
  }
  refreshPostprocessNotes();
}

/* ---------- LoRA specs ---------- */
let loras = [];
let loraRoles = []; // the loaded model's LoRA-able component names

const LORA_PLACEHOLDER = 'one per line, e.g. style:0.8\ndit_low: style:0.8 (low expert)';

async function loadLoras() {
  const data = await fetchJSON('/lora');
  loras = data ? (data.loras || []).sort() : [];
  loraRoles = data ? (data.roles || []) : [];
}

const lorasLoaded = loadLoras();

// A "role: spec" line targets that component; any other line is a spec for the default role.
function parseLora(value) {
  const lines = value.trim() ? value.split('\n').map(l => l.trim()).filter(Boolean) : [];
  if (!lines.length) return null;
  if (loraRoles.length <= 1) return lines;
  const byRole = {};
  for (const line of lines) {
    const named = line.match(/^([a-z_]+):\s*(.+)$/);
    const role = named && loraRoles.includes(named[1]) ? named[1] : null;
    (byRole[role || 'dit'] = byRole[role || 'dit'] || []).push(role ? named[2] : line);
  }
  const roles = Object.keys(byRole);
  return roles.length === 1 && roles[0] === 'dit' ? byRole.dit : byRole;
}

function formatLora(specs) {
  if (Array.isArray(specs)) return specs.join(', ');
  if (specs && typeof specs === 'object') {
    return Object.entries(specs)
      .map(([role, list]) => list.join(', ') + (role === 'dit' ? '' : `  [${role}]`))
      .join('\n');
  }
  return String(specs);
}

function initLora(prefix) {
  const specs = () => $(prefix + 'lora_specs');
  const ac = () => $(prefix + 'lora_ac');

  lorasLoaded.then(() => {
    if (loraRoles.length > 1) specs().placeholder = LORA_PLACEHOLDER;
  });

  function openAc() { ac().classList.add('open'); }
  function closeAc() { ac().classList.remove('open'); ac().innerHTML = ''; }

  function tokenAtCursor(ta) {
    const s = ta.selectionStart;
    const before = ta.value.slice(0, s);
    const nl = before.lastIndexOf('\n');
    const lineStart = nl === -1 ? 0 : nl + 1;
    const sp = before.lastIndexOf(' ');
    const tokenStart = sp > lineStart ? sp + 1 : lineStart;
    return { token: ta.value.slice(tokenStart, s).trim(), tokenStart };
  }

  function insertLora(name, tokenStart) {
    const s = specs().selectionStart;
    specs().value = specs().value.slice(0, tokenStart) + name + specs().value.slice(s);
    const pos = tokenStart + name.length;
    specs().setSelectionRange(pos, pos);
    specs().focus();
  }

  function renderAc(items, tokenStart) {
    ac().innerHTML = '';
    if (!items.length) {
      const d = document.createElement('div');
      d.className = 'empty';
      d.textContent = 'No matching LoRAs';
      ac().appendChild(d);
      openAc();
      return;
    }
    items.forEach(name => {
      const d = document.createElement('div');
      d.className = 'item';
      d.dataset.name = name;
      d.dataset.tokenStart = tokenStart;
      d.textContent = name;
      ac().appendChild(d);
    });
    openAc();
  }

  specs().addEventListener('input', () => {
    const { token, tokenStart } = tokenAtCursor(specs());
    if (token.length < 2) { closeAc(); return; }
    const q = token.toLowerCase();
    renderAc(loras.filter(n => n.toLowerCase().includes(q)), tokenStart);
  });

  ac().addEventListener('mousedown', e => {
    const item = e.target.closest('.item');
    if (!item) return;
    e.preventDefault();
    insertLora(item.dataset.name, parseInt(item.dataset.tokenStart, 10));
    closeAc();
  });

  specs().addEventListener('keydown', e => {
    if (!ac().classList.contains('open')) return;
    if (e.key === 'Escape') { closeAc(); return; }
    const items = ac().querySelectorAll('.item');
    if (!items.length) return;
    if (e.key === 'ArrowDown' || e.key === 'ArrowUp') {
      e.preventDefault();
      let idx = -1;
      items.forEach((it, i) => { if (it.classList.contains('sel')) idx = i; });
      idx = e.key === 'ArrowDown'
        ? (idx + 1) % items.length
        : (idx === -1 ? items.length - 1 : (idx - 1 + items.length) % items.length);
      items.forEach((it, i) => it.classList.toggle('sel', i === idx));
    } else if (e.key === 'Enter' || e.key === 'Tab') {
      const sel = ac().querySelector('.item.sel');
      if (sel) {
        e.preventDefault();
        insertLora(sel.dataset.name, parseInt(sel.dataset.tokenStart, 10));
        closeAc();
      }
    }
  });
}

/* ---------- collapsible sidebar sections ----------

Sections sharing a ``data-pref`` open and close together, and remember it: both
panes carry the same controls, and a user who opened one means the other too.
*/
function collapsibles(pref) {
  return Array.from(document.querySelectorAll(`details.collapse[data-pref="${pref}"]`));
}

function initCollapses() {
  const prefKeys = new Set(
    Array.from(document.querySelectorAll('details.collapse[data-pref]'))
      .map(box => box.dataset.pref)
  );
  for (const pref of prefKeys) {
    const boxes = collapsibles(pref);
    const open = localStorage.getItem(pref) === '1';
    for (const box of boxes) {
      box.open = open;
      box.addEventListener('toggle', () => {
        localStorage.setItem(pref, box.open ? '1' : '0');
        for (const other of collapsibles(pref)) {
          if (other !== box) other.open = box.open;
        }
      });
    }
  }
}

initCollapses();

function bindRange(sliderId, valId, digits = 2) {
  $(sliderId).addEventListener('input', e =>
    $(valId).textContent = parseFloat(e.target.value).toFixed(digits));
}

function setRange(sliderId, valId, value, digits = 2) {
  $(sliderId).value = value;
  $(valId).textContent = value.toFixed(digits);
}

function rangeLabel(sliderId, valId, digits = 2) {
  $(valId).textContent = parseFloat($(sliderId).value).toFixed(digits);
}

function clampSlider(slider, max) {
  slider.max = max;
  if (parseFloat(slider.value) > max) slider.value = max;
}

const LATENT_SCALE = 2; // latent refiner multiplier

function renderRes(resEl, w, h, factor) {
  resEl.textContent = (w && h)
    ? ` ${Math.round(w * factor)} \u00d7 ${Math.round(h * factor)} px`
    : '';
}

function setTimer(elId, textId, state, text) {
  $(elId).className = 'timer ' + state;
  $(textId).textContent = text;
}

function readPngText(bytes) {
  const dv = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  const texts = {};
  if (dv.getUint32(0) !== 0x89504e47) return texts; // not a PNG
  let off = 8; // skip signature
  while (off + 8 <= bytes.byteLength) {
    const len = dv.getUint32(off);
    const type = String.fromCharCode(
      dv.getUint8(off + 4), dv.getUint8(off + 5),
      dv.getUint8(off + 6), dv.getUint8(off + 7)
    );
    const start = off + 8;
    if (type === 'tEXt') {
      // keyword: null-terminated Latin-1; text: rest of chunk
      const chunk = bytes.subarray(start, start + len);
      let sep = -1;
      for (let i = 0; i < chunk.length; i++) if (chunk[i] === 0) { sep = i; break; }
      if (sep >= 0) {
        const keyword = decodeLatin1(chunk.subarray(0, sep));
        const value = decodeLatin1(chunk.subarray(sep + 1));
        texts[keyword] = value;
      }
    }
    off = start + len + 4; // skip data + crc
  }
  return texts;
}

function decodeLatin1(bytes) {
  let s = '';
  for (const b of bytes) s += String.fromCharCode(b);
  return s;
}

async function decodeMeta(blob, keyword) {
  const buf = await blob.arrayBuffer();
  const texts = readPngText(new Uint8Array(buf));
  let meta = null;
  try { meta = texts[keyword] ? JSON.parse(texts[keyword]) : null; }
  catch { meta = null; }
  return meta;
}

const FIELD_LABELS = {
  model: 'Model', prompt: 'Prompt', negative_prompt: 'Negative prompt',
  width: 'Width', height: 'Height', steps: 'Steps',
  guidance_scale: 'CFG scale', seed: 'Seed', upscale: 'Upscale',
  upscale_factor: 'Upscale factor', upscale_type: 'Upscale type',
  sampler: 'Sampler', lora_specs: 'LoRA',
  mode: 'Mode', frame_count: 'Frames', fps: 'FPS', duration: 'Duration',
  references: 'References', attention: 'Attention',
  flow_shift: 'Flow shift', output_format: 'Output',
  latents: 'Detail level', octree_resolution: 'Voxel resolution',
  bounds: 'Bounding volume', mc_level: 'Surface offset', scale: 'Scale',
  picture: 'Reference image',
};

function addRow(grid, label, value, cls) {
  const dt = document.createElement('dt'); dt.textContent = label;
  const dd = document.createElement('dd'); if (cls) dd.className = cls;
  if (Array.isArray(value)) dd.textContent = value.join(', ') || '—';
  else dd.textContent = value ?? '—';
  grid.append(dt, dd);
}

function renderInfoRows(rows) {
  const grid = $('info_modal_grid');
  grid.innerHTML = '';
  for (const [label, value, cls] of rows) addRow(grid, label, value, cls);
}

function renderInfo(meta) {
  if (!meta || typeof meta !== 'object') {
    const grid = $('info_modal_grid');
    grid.innerHTML = '';
    const dd = document.createElement('dd');
    dd.className = 'none';
    dd.textContent = 'No metadata found in this image.';
    grid.appendChild(dd);
    return;
  }
  const rows = [];
  for (const key of ['model','prompt','negative_prompt','width','height','steps','sampler','guidance_scale','seed']) {
    if (key in meta) rows.push([FIELD_LABELS[key], meta[key], key === 'prompt' || key === 'negative_prompt' ? 'prompt' : '']);
  }
  if ('upscale' in meta) rows.push(['Upscale', meta.upscale, '']);
  if ('upscale_factor' in meta) rows.push(['Upscale factor', meta.upscale_factor, '']);
  if ('upscale_type' in meta) rows.push(['Upscale type', meta.upscale_type, '']);
  if ('pixel_upscaler' in meta && meta.pixel_upscaler) rows.push(['Pixel upscaler', meta.pixel_upscaler, '']);
  rows.push(...postprocessInfoRows(meta));
  if ('lora_specs' in meta && meta.lora_specs) rows.push([FIELD_LABELS.lora_specs, formatLora(meta.lora_specs), '']);
  renderInfoRows(rows);
}

function makeHistory({ containerId, rootId, alt, onSelect, video: isVideo = false }) {
  const items = []; // newest first, capped at 10: {url, meta, time}
  function select(i) {
    const item = items[i];
    if (!item) return;
    onSelect(item);
    $(containerId).querySelectorAll('.hist-item')
      .forEach((el, idx) => el.classList.toggle('current', idx === i));
  }
  function add(url, meta) {
    items.unshift({ url, meta, time: Date.now() });
    if (items.length > 10) {
      const removed = items.pop();
      if (removed.url) URL.revokeObjectURL(removed.url);
    }
    render();
  }
  function render() {
    const container = $(containerId);
    container.innerHTML = '';
    items.forEach((item, i) => {
      const div = document.createElement('div');
      div.className = 'hist-item' + (i === 0 ? ' current' : '');
      const el = document.createElement(isVideo ? 'video' : 'img');
      el.className = 'thumb-img';
      if (isVideo) {
        // A first frame is a thumbnail: metadata only, muted, never autoplaying.
        el.muted = true;
        el.playsInline = true;
        el.preload = 'metadata';
      } else {
        el.alt = alt + ' ' + (i + 1);
      }
      el.src = item.url;
      div.appendChild(el);
      const seed = document.createElement('span');
      seed.className = 'hist-seed';
      seed.textContent = item.meta && item.meta.seed != null ? item.meta.seed : '';
      div.appendChild(seed);
      div.addEventListener('click', () => select(i));
      container.appendChild(div);
    });
    $(rootId).classList.toggle('hidden', items.length === 0);
  }
  return { items, add };
}

function download(url, filename) {
  if (!url) return;
  const a = document.createElement('a');
  a.href = url;
  a.download = filename;
  document.body.appendChild(a);
  a.click();
  a.remove();
}

function setStageActions(dlBtn, infoBtn, enabled) {
  dlBtn.disabled = !enabled;
  infoBtn.disabled = !enabled;
}

function showResult(imgId, placeholderId, url) {
  $(imgId).src = url;
  $(imgId).style.display = 'block';
  $(placeholderId).style.display = 'none';
}

function hideResult(imgId, placeholderId) {
  $(imgId).style.display = 'none';
  $(placeholderId).style.display = 'block';
}

// What a browser says a file is. The server asks the bytes the same question again,
// because a name and a MIME type are both things a client chose.
function mediaKind(file) {
  if (file.type.startsWith('image/')) return 'picture';
  if (file.type.startsWith('video/')) return 'video';
  if (file.type.startsWith('audio/')) return 'audio';
  return 'unknown';
}

// A dropped file of any kind, as base64. A picture also comes with a preview and its
// size; anything else arrives as bytes, and the tile that stands in for a preview says
// what extension it came with.
function readMediaFile(file) {
  return new Promise(done => {
    if (!file) { done(null); return; }
    const reader = new FileReader();
    reader.onload = () => {
      const dataUrl = reader.result;
      const media = { b64: dataUrl.split(',')[1], dataUrl: null, kind: mediaKind(file) };
      if (media.kind !== 'picture') { done(media); return; }
      const img = new Image();
      img.onload = () => done({
        ...media, dataUrl,
        dims: { w: img.naturalWidth, h: img.naturalHeight },
      });
      img.onerror = () => done(null);   // a file that says image/ and is not one
      img.src = dataUrl;
    };
    reader.readAsDataURL(file);
  });
}

// A dropped image as every pane wants it: data URL, bare base64, size; null if it is not one.
function readImageFile(file) {
  return readMediaFile(file).then(media =>
    (media && media.dataUrl) ? media : null);
}

// Show a freshly loaded input image in a single/result split stage.
function showInputPreview(dataUrl, file, ids) {
  $(ids.singleImg).src = dataUrl;
  $(ids.single).classList.remove('hidden');
  $(ids.result).classList.add('hidden');
  $(ids.placeholder).style.display = 'none';
  $(ids.dzName).textContent = file.name;
  $(ids.dzName).style.display = 'block';
  $(ids.dzHint).style.display = 'none';
}

function addThumb(box, { src, alt, title, cls, label, onRemove }) {
  const div = document.createElement('div');
  div.className = 'ref-thumb' + (cls ? ' ' + cls : '');
  div.title = title;
  if (src) {
    const img = document.createElement('img');
    img.className = 'thumb-img';
    img.src = src;
    img.alt = alt;
    div.appendChild(img);
  } else {
    // No preview to show for a clip or a sound: the extension is as close as a name gets.
    const tag = document.createElement('div');
    tag.className = 'thumb-file';
    tag.textContent = label || '';
    div.appendChild(tag);
  }
  const rm = document.createElement('button');
  rm.className = 'ref-remove';
  rm.textContent = '\u00d7';
  rm.title = 'Remove';
  rm.addEventListener('click', onRemove);
  div.appendChild(rm);
  box.appendChild(div);
  return div;
}

// Swap a split stage from its input preview to its result image.
function showOutputPreview(singleId, resultId, placeholderId, resultImgId, url) {
  $(singleId).classList.add('hidden');
  $(resultId).classList.remove('hidden');
  $(placeholderId).style.display = 'none';
  $(resultImgId).src = url;
}

function seedName(meta, fallback) {
  return meta && meta.seed != null ? meta.seed : fallback;
}

function fillSelect(sel, names, label) {
  sel.innerHTML = '';
  for (const name of names) {
    const opt = document.createElement('option');
    opt.value = name;
    // ``label`` turns a value into the words shown to the user (defaults to the value).
    opt.textContent = label ? label(name) : name;
    sel.appendChild(opt);
  }
}

async function fetchJSON(url) {
  try {
    const res = await fetchOrGone(url, undefined, PATIENCE);
    if (res.ok) return await res.json();
  } catch (e) { /* fall through to null */ }
  return null;
}

// A refusal's reason: `error` from the job routes, FastAPI's `detail`, plain text otherwise.
async function errorMessage(res) {
  const text = await res.text().catch(() => '');
  if (!text) return res.statusText || `HTTP ${res.status}`;
  try {
    const refused = JSON.parse(text);
    return refused.error || refused.detail || text;
  } catch (e) {
    return text;
  }
}

// Every pane's fetch: a connection the server refused means it is not running, not merely slow.
async function fetchOrGone(url, init, patience = 0) {
  const ctl = patience ? new AbortController() : null;
  const deadline = ctl && setTimeout(() => ctl.abort(), patience);
  try {
    // The wait for an answer only: a downloaded artifact may still take its time.
    return await fetch(url, { ...init, signal: ctl ? ctl.signal : undefined });
  } catch (e) {
    throw new Error('the server is not answering — is it still running?', { cause: e });
  } finally {
    if (deadline) clearTimeout(deadline);
  }
}

// /health is answered even mid-render, so a failed read whose /health also fails is a gone server.
async function serverGone() {
  try {
    await fetchOrGone('/health', undefined, PATIENCE);
    return false;
  } catch (e) {
    return true;
  }
}

async function postJSON(url, body) {
  const res = await fetchOrGone(url, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  });
  if (!res.ok) throw new Error(await errorMessage(res));
  return await res.blob();
}

async function postJSONData(url, body) {
  const res = await fetchOrGone(url, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  }, PATIENCE);
  if (!res.ok) throw new Error(await errorMessage(res));
  return await res.json();
}

/* ---------- queued jobs: submit, watch, fetch ---------- */
const JOB_POLL_MS = 500;
const JOB_POLL_RETRIES = 5;

// Stage names are their own labels except this one.
const JOB_STAGE_LABELS = { postprocessing: 'post-processing' };

const sleep = (ms) => new Promise(done => setTimeout(done, ms));

// A hidden tab is watched by nobody, and browsers throttle its timers anyway.
async function waitVisible() {
  while (document.hidden) {
    await new Promise(done =>
      document.addEventListener('visibilitychange', done, { once: true }));
  }
}

// Null when the server no longer knows the id: a restarted server dropped its queue.
async function fetchJob(id) {
  const res = await fetchOrGone('/jobs/' + id, undefined, PATIENCE);
  if (res.status === 404) return null;
  if (!res.ok) throw new Error(await errorMessage(res));
  return await res.json();
}

// Forgives JOB_POLL_RETRIES consecutive failures while /health answers.
async function pollJob(id, onProgress) {
  let misses = 0;
  for (;;) {
    await waitVisible();
    await sleep(JOB_POLL_MS);
    let status;
    try {
      status = await fetchJob(id);
    } catch (e) {
      // One look at /health settles it: gone means there is nothing left to wait for.
      if (await serverGone()) throw e;
      if (++misses >= JOB_POLL_RETRIES) throw e;
      continue;
    }
    if (status === null) throw new Error('the server no longer knows this job');
    misses = 0;
    onProgress(status.progress || {});
    if (status.status !== 'queued' && status.status !== 'running') return status;
  }
}

// els: {root, label, bar} — the ids of one pane's progress block.
function setProgress(els, p) {
  $(els.root).classList.remove('hidden');
  const stage = JOB_STAGE_LABELS[p.stage] || p.stage || 'waiting';
  const bar = $(els.bar);
  if (p.of > 0) {
    $(els.label).textContent = `${stage} ${p.step}/${p.of}`;
    bar.classList.remove('indeterminate');
    bar.style.width = (100 * p.step / p.of).toFixed(1) + '%';
  } else {
    $(els.label).textContent = stage;
    bar.classList.add('indeterminate');
  }
}

function clearProgress(els) {
  $(els.root).classList.add('hidden');
  $(els.bar).classList.remove('indeterminate');
  $(els.bar).style.width = '0';
}

async function withBusy({ btn, overlay, timerEl, timerTextEl }, work) {
  const set = (state, text) => setTimer(timerEl, timerTextEl, state, text);
  btn.disabled = true;
  overlay.classList.remove('hidden');
  const start = Date.now();
  const elapsed = () => ((Date.now() - start) / 1000).toFixed(1) + 's';
  set('running', '0.0s');
  const clock = setInterval(() => set('running', elapsed()), 100);
  try {
    const end = await work(elapsed);
    set(end.state, end.text);
  } catch (e) {
    set('error', 'error: ' + e.message);
  } finally {
    clearInterval(clock);
    overlay.classList.add('hidden');
    btn.disabled = false;
  }
}

// `job` is the caller's box, so its Cancel button knows what to name.
async function runQueuedJob({ job, submit, body, btn, cancelBtn, overlay,
                              timerEl, timerTextEl, progress, onArtifact }) {
  try {
    await withBusy({ btn, overlay, timerEl, timerTextEl }, async (elapsed) => {
      job.id = (await postJSONData(submit, body)).id;
      cancelBtn.disabled = false;
      setProgress(progress, { stage: 'queued' });
      const status = await pollJob(job.id, (p) => setProgress(progress, p));
      if (status.status === 'canceled') return { state: 'error', text: 'canceled' };
      if (status.status !== 'completed') throw new Error(status.error || status.status);
      // Serving an artifact unlinks it: this read is the only one there will be.
      const res = await fetchOrGone('/jobs/' + job.id + '/content', undefined, PATIENCE);
      if (!res.ok) throw new Error('downloading the result failed: ' + await errorMessage(res));
      await onArtifact(await res.blob(), status);
      return { state: 'done', text: elapsed() };
    });
  } finally {
    job.id = null;
    cancelBtn.disabled = true;
    clearProgress(progress);
  }
}

// The poll reports the cancellation: the record stays running until a step boundary.
async function cancelJob(job, btn) {
  if (!job.id) return;
  btn.disabled = true;
  try {
    await postJSONData('/jobs/' + job.id + '/cancel');
  } catch (e) { /* it may have settled between the click and this request */ }
}

async function runWithBusy({ btn, overlay, timerEl, timerTextEl, request, onSuccess }) {
  await withBusy({ btn, overlay, timerEl, timerTextEl }, async (elapsed) => {
    await onSuccess(await request());
    return { state: 'done', text: elapsed() };
  });
}

function validateDims(prefix) {
  for (const f of ['width', 'height']) {
    const v = $(prefix + f).value === '' ? null : parseInt($(prefix + f).value, 10);
    // Empty means "auto" (the model default); 0 is just an invalid size.
    if (v !== null && !(v >= 1)) {
      alert(`error: ${f} must be at least 1 (got ${v}).`);
      return false;
    }
  }
  return true;
}

// An empty box is the model's default, so it leaves the body rather than going out as 0 or NaN.
function addNumbers(body, prefix, ints, floats) {
  for (const f of ints) {
    const v = $(prefix + f).value;
    if (v !== '') body[f] = parseInt(v, 10);
  }
  for (const f of floats) {
    const v = $(prefix + f).value;
    if (v !== '') body[f] = parseFloat(v);
  }
  return body;
}

function collectSettings(prefix, extra) {
  const body = {
    prompt: $(prefix + 'prompt').value,
    negative_prompt: $(prefix + 'negative_prompt').value,
    upscale_factor: parseFloat($(prefix + 'upscale_factor').value),
    upscale_type: $(prefix + 'upscale_type').value,
    pixel_upscaler: $(prefix + 'pixel_upscaler').value || null,
    ...collectPostprocess(prefix),
    lora_specs: parseLora($(prefix + 'lora_specs').value),
  };
  addNumbers(body, prefix, ['width', 'height', 'steps', 'seed'], ['guidance_scale']);
  const samplerVal = $(prefix + 'sampler').value;
  if (samplerVal) body.sampler = samplerVal;
  // Every medium is asked the same question, even one that answers it one way only.
  const attention = $(prefix + 'attention');
  if (attention && attention.value) body.attention = attention.value;
  Object.assign(body, extra);
  return body;
}

function parseTriState(value) {
  if (value === '' || value == null) return null;
  return value === 'true';
}

function bindSwap(swapId, wId, hId, after) {
  $(swapId).addEventListener('click', () => {
    const w = $(wId), h = $(hId);
    const tmp = w.value; w.value = h.value; h.value = tmp;
    if (after) after();
  });
}

function bindDropzone(dzId, inputId, onFile) {
  const dz = $(dzId);
  const input = $(inputId);
  dz.addEventListener('click', () => input.click());
  input.addEventListener('change', e => onFile(e.target.files));
  dz.addEventListener('dragover', e => { e.preventDefault(); dz.classList.add('over'); });
  dz.addEventListener('dragleave', () => dz.classList.remove('over'));
  dz.addEventListener('drop', e => {
    e.preventDefault(); dz.classList.remove('over');
    onFile(e.dataTransfer.files);
  });
}

function openInfo(title, render) {
  render();
  $('info_modal_title').textContent = title;
  $('info_modal').classList.remove('hidden');
}
function closeInfo() {
  $('info_modal').classList.add('hidden');
}
