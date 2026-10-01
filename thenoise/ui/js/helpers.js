/* helpers.js — shared UI utilities for TheNoise.
   Plain global script, loaded BEFORE app.js (no modules). */

const $ = id => document.getElementById(id);

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
  sampler: 'Sampler', qwen_vae_enhance: 'Reduce grid pattern',
  film_grain: 'Film grain', sharpening: 'Sharpening', lora_specs: 'LoRA',
};

function addRow(grid, label, value, cls) {
  const dt = document.createElement('dt'); dt.textContent = label;
  const dd = document.createElement('dd'); if (cls) dd.className = cls;
  if (Array.isArray(value)) dd.textContent = value.join(', ') || '—';
  else dd.textContent = value ?? '—';
  grid.append(dt, dd);
}

function renderInfo(meta) {
  const grid = $('info_modal_grid');
  grid.innerHTML = '';
  if (!meta || typeof meta !== 'object') {
    const dd = document.createElement('dd');
    dd.className = 'none';
    dd.textContent = 'No metadata found in this image.';
    grid.appendChild(dd);
    return;
  }
  for (const key of ['model','prompt','negative_prompt','width','height','steps','sampler','guidance_scale','seed']) {
    if (key in meta) addRow(grid, FIELD_LABELS[key], meta[key], key === 'prompt' || key === 'negative_prompt' ? 'prompt' : '');
  }
  if ('upscale' in meta) addRow(grid, 'Upscale', meta.upscale);
  if ('upscale_factor' in meta) addRow(grid, 'Upscale factor', meta.upscale_factor);
  if ('upscale_type' in meta) addRow(grid, 'Upscale type', meta.upscale_type);
  if ('pixel_upscaler' in meta && meta.pixel_upscaler) addRow(grid, 'Pixel upscaler', meta.pixel_upscaler);
  if ('qwen_vae_enhance' in meta) addRow(grid, 'Reduce grid pattern', meta.qwen_vae_enhance);
  for (const key of ['film_grain','sharpening','lora_specs']) {
    if (key in meta && (key !== 'film_grain' || meta.film_grain) && (key !== 'sharpening' || meta.sharpening)) {
      addRow(grid, FIELD_LABELS[key], meta[key], '');
    }
  }
}

function makeHistory({ containerId, rootId, alt, onSelect }) {
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
      const img = document.createElement('img');
      img.className = 'thumb-img';
      img.src = item.url;
      img.alt = alt + ' ' + (i + 1);
      div.appendChild(img);
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

function seedName(meta, fallback) {
  return meta && meta.seed != null ? meta.seed : fallback;
}

function fillSelect(sel, names) {
  sel.innerHTML = '';
  for (const name of names) {
    const opt = document.createElement('option');
    opt.value = name;
    opt.textContent = name;
    sel.appendChild(opt);
  }
}

async function fetchJSON(url) {
  try {
    const res = await fetch(url);
    if (res.ok) return await res.json();
  } catch (e) { /* fall through to null */ }
  return null;
}

async function postJSON(url, body) {
  const res = await fetch(url, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  });
  if (!res.ok) throw new Error(await res.text() || res.statusText);
  return await res.blob();
}

async function runWithBusy({ btn, overlay, timerEl, timerTextEl, request, onSuccess }) {
  const set = (state, text) => setTimer(timerEl, timerTextEl, state, text);
  btn.disabled = true;
  overlay.classList.remove('hidden');
  const start = Date.now();
  const elapsed = () => ((Date.now() - start) / 1000).toFixed(1) + 's';
  set('running', '0.0s');
  const timer = setInterval(() => set('running', elapsed()), 100);
  try {
    const result = await request();
    await onSuccess(result);
    set('done', elapsed());
  } catch (e) {
    set('error', 'error: ' + e.message);
  } finally {
    clearInterval(timer);
    overlay.classList.add('hidden');
    btn.disabled = false;
  }
}

function validateDims(prefix) {
  const MAX_DIM = 4096;
  for (const f of ['width', 'height']) {
    const v = $(prefix + f).value === '' ? null : parseInt($(prefix + f).value, 10);
    // Empty means "auto" (the model default); 0 is just an invalid size.
    if (v !== null && (v < 1 || v > MAX_DIM)) {
      alert(`error: ${f} must be between 1 and ${MAX_DIM} (got ${v}).`);
      return false;
    }
  }
  return true;
}

const SIGMA_SCHEDULES = {
  qwen_image21: {
    'Pruna Image 2.1 8 steps': [
      1, 0.933333333, 0.857142857, 0.769230769, 0.666666667, 0.545454545, 0.4,
      0.222222222,
    ],
    'Pruna Image 2.1 5 steps': [1, 0.94, 0.857142857, 0.666666667, 0.4],
    'Viggle Turbo': [1.0, 0.9375, 0.875, 0.75, 0.5, 0.25],
  },
};

let currentModel = null;

function schedulesFor(model) {
  return (model && SIGMA_SCHEDULES[model]) || {};
}

function selectedSigmas(prefix) {
  const sel = $(prefix + 'schedule');
  return (sel && sel.value && schedulesFor(currentModel)[sel.value]) || null;
}

function scheduleNameFor(sigmas) {
  if (!Array.isArray(sigmas) || !sigmas.length) return '';
  const grid = sigmas[sigmas.length - 1] === 0 ? sigmas.slice(0, -1) : sigmas;
  const presets = schedulesFor(currentModel);
  for (const name of Object.keys(presets)) {
    const values = presets[name];
    if (values.length === grid.length && values.every((v, i) => v === grid[i])) {
      return name;
    }
  }
  return '';
}

function syncSchedule(prefix) {
  const sel = $(prefix + 'schedule');
  const steps = $(prefix + 'steps');
  if (!sel || !steps) return;
  const grid = selectedSigmas(prefix);
  $(prefix + 'schedule_hint').textContent = grid ? grid.length + ' steps' : '';
  if (grid) {
    if (!steps.disabled) steps.dataset.auto = steps.value;
    steps.value = grid.length;
    steps.disabled = true;
  } else if (steps.disabled) {
    steps.disabled = false;
    steps.value = steps.dataset.auto || '';
  }
}

function collectSettings(prefix, extra) {
  const sigmas = selectedSigmas(prefix);
  const body = {
    prompt: $(prefix + 'prompt').value,
    negative_prompt: $(prefix + 'negative_prompt').value,
    upscale_factor: parseFloat($(prefix + 'upscale_factor').value),
    upscale_type: $(prefix + 'upscale_type').value,
    pixel_upscaler: $(prefix + 'pixel_upscaler').value || null,
    qwen_vae_enhance: $(prefix + 'qwen_vae_enhance').checked,
    film_grain: parseFloat($(prefix + 'film_grain').value),
    sharpening: parseFloat($(prefix + 'sharpening').value),
    lora_specs: parseLora($(prefix + 'lora_specs').value),
  };
  for (const f of ['width', 'height', 'steps', 'seed']) {
    if (f === 'steps' && sigmas) continue; // the grid owns the step count
    const v = $(prefix + f).value;
    if (v !== '') body[f] = parseInt(v, 10);
  }
  if (sigmas) body.sigmas = sigmas;
  const g = $(prefix + 'guidance_scale').value;
  if (g !== '') body.guidance_scale = parseFloat(g);
  const samplerVal = $(prefix + 'sampler').value;
  if (samplerVal) body.sampler = samplerVal;
  Object.assign(body, extra);
  return body;
}

function parseTriState(value) {
  if (value === '' || value == null) return null;
  return value === 'true';
}

function parseLora(value) {
  return value.trim()
    ? value.split('\n').map(l => l.trim()).filter(Boolean)
    : null;
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
