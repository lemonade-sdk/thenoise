/* app.js — view-specific logic for TheNoise (Generate / Edit / Upscale).
   Plain global script, loaded AFTER helpers.js. */

let currentUrl = null;
let currentMeta = null;

bindRange('film_grain', 'film_grain_val');
bindRange('sharpening', 'sharpening_val');
bindRange('edit_film_grain', 'edit_film_grain_val');
bindRange('edit_sharpening', 'edit_sharpening_val');

function updateFinalRes(prefix) {
  if (prefix === 'edit_') { updateEditFinalRes(); return; }
  const w = parseInt($('width').value, 10);
  const h = parseInt($('height').value, 10);
  const factor = parseFloat($('upscale_factor').value) || 1;
  renderRes($('final_res'), w, h, factor);
}

// Max slider value follows the selected pixel upscaler's native scale.
function updateUpscaleMax(prefix) {
  const slider = $(prefix + 'upscale_factor');
  const type = $(prefix + 'upscale_type').value;
  const name = $(prefix + 'pixel_upscaler').value;
  const scale = name ? (upscalerScales[name] || 2) : 0;
  const max = type === 'no-refiner'
    ? (scale || 1)
    : (scale ? LATENT_SCALE * scale : LATENT_SCALE);
  clampSlider(slider, max);
  rangeLabel(prefix + 'upscale_factor', prefix + 'upscale_factor_val');
  updateFinalRes(prefix);
}

['', 'edit_'].forEach(p => {
  $(p + 'upscale_type').addEventListener('change', () => updateUpscaleMax(p));
  $(p + 'pixel_upscaler').addEventListener('change', () => updateUpscaleMax(p));
  $(p + 'upscale_factor').addEventListener('input', () => {
    rangeLabel(p + 'upscale_factor', p + 'upscale_factor_val');
    updateFinalRes(p);
  });
});
$('width').addEventListener('input', () => updateFinalRes(''));
$('height').addEventListener('input', () => updateFinalRes(''));

let editRefs = []; // {dataUrl, b64, dims:{w,h}, name}

const EDIT_DEFAULT_SIZE = 1024; // largest side when no width/height given

function editTargetDims() {
  if (!editRefs.length) return null;
  const { w: iw, h: ih } = editRefs[0].dims;
  const ew = parseInt($('edit_width').value, 10);
  const eh = parseInt($('edit_height').value, 10);
  if (ew && eh) return { w: ew, h: eh };
  if (ew) return { w: ew, h: ih };
  if (eh) return { w: iw, h: eh };
  // No width/height: resize the first reference to 1024, aspect preserved.
  if (iw >= ih) return { w: EDIT_DEFAULT_SIZE, h: Math.round(ih * EDIT_DEFAULT_SIZE / iw) };
  return { w: Math.round(iw * EDIT_DEFAULT_SIZE / ih), h: EDIT_DEFAULT_SIZE };
}

function updateEditFinalRes() {
  const d = editTargetDims();
  const factor = parseFloat($('edit_upscale_factor').value) || 1;
  renderRes($('edit_final_res'), d && d.w, d && d.h, factor);
}
$('edit_width').addEventListener('input', updateEditFinalRes);
$('edit_height').addEventListener('input', updateEditFinalRes);

const genHist = makeHistory({
  containerId: 'history_items', rootId: 'history', alt: 'generation',
  onSelect: (item) => {
    currentUrl = item.url;
    currentMeta = item.meta;
    showResult('image', 'placeholder', item.url);
    renderInfo(item.meta);
    applySettings('', item.meta);
    setStageActions($('download'), $('info_btn'), true);
  },
});

function applySettings(prefix, meta) {
  if (!meta) return;
  const p = prefix ? prefix + '_' : '';
  $(p + 'prompt').value = meta.prompt ?? '';
  $(p + 'negative_prompt').value = meta.negative_prompt ?? '';
  if (meta.width != null) $(p + 'width').value = meta.width;
  if (meta.height != null) $(p + 'height').value = meta.height;
  if (meta.steps != null) $(p + 'steps').value = meta.steps;
  if (meta.guidance_scale != null) $(p + 'guidance_scale').value = meta.guidance_scale;
  if (meta.seed != null) $(p + 'seed').value = meta.seed;
  if (meta.sampler) $(p + 'sampler').value = meta.sampler;
  if (meta.upscale_factor != null) {
    setRange(p + 'upscale_factor', p + 'upscale_factor_val', meta.upscale_factor);
  } else if (meta.upscale === true) {
    // legacy metadata: 'upscale: true' == 2x refined
    setRange(p + 'upscale_factor', p + 'upscale_factor_val', 2);
    $(p + 'upscale_type').value = 'refined';
  }
  if (meta.upscale_type) $(p + 'upscale_type').value = meta.upscale_type;
  if (meta.qwen_vae_enhance != null) $(p + 'qwen_vae_enhance').checked = meta.qwen_vae_enhance;
  if (meta.film_grain != null) setRange(p + 'film_grain', p + 'film_grain_val', meta.film_grain);
  if (meta.sharpening != null) setRange(p + 'sharpening', p + 'sharpening_val', meta.sharpening);
  if (Array.isArray(meta.lora_specs)) {
    $(p + 'lora_specs').value = meta.lora_specs.join('\n');
  }
  if (meta.pixel_upscaler) {
    $(p + 'pixel_upscaler').value = meta.pixel_upscaler;
  }
  updateUpscaleMax(prefix);
}

bindSwap('swap', 'width', 'height');
bindSwap('eswap', 'edit_width', 'edit_height', updateEditFinalRes);

$('download').addEventListener('click', () =>
  download(currentUrl, `thenoise_${seedName(currentMeta, 'x')}.png`));

let loras = [];

async function loadLoras() {
  const data = await fetchJSON('/lora');
  loras = data ? (data.loras || []).sort() : [];
}

let upscalerScales = {};  // name -> detected native scale (2/4)

async function loadUpscalers() {
  let names = [];
  const data = await fetchJSON('/upscalers');
  if (data) {
    names = (data.upscalers || []).sort();
    upscalerScales = data.scales || {};
  }
  // Fill the generate + edit pixel-upscaler dropdowns (keep the 'none' option).
  for (const p of ['', 'edit_']) {
    const sel = $(p + 'pixel_upscaler');
    const none = sel.firstElementChild;
    sel.innerHTML = '';
    sel.appendChild(none);
    fillSelect(sel, names);
  }
  fillSelect($('upscaler_model'), names);
  applyUpscalerDefaults();
  // No upscaler models found: block the Upscale tab and hide the no-refiner option.
  const noUpscalers = names.length === 0;
  $('no_upscaler').classList.toggle('hidden', !noUpscalers);
  for (const p of ['', 'edit_']) {
    $(p + 'pixel_upscaler_field').classList.toggle('hidden', noUpscalers);
    const noRefiner = $(p + 'no_refiner_opt');
    if (noUpscalers) {
      if (noRefiner.parentElement) noRefiner.remove();
      if ($(p + 'upscale_type').value === 'no-refiner') $(p + 'upscale_type').value = 'refined';
    } else if (!$(p + 'upscale_type').contains(noRefiner)) {
      $(p + 'upscale_type').appendChild(noRefiner);
    }
  }
  updateUpscaleMax('');
  updateUpscaleMax('edit_');
}

function initLora(prefix) {
  const specs = () => $(prefix + 'lora_specs');
  const ac = () => $(prefix + 'lora_ac');

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

initLora('');
initLora('edit_');

// A single click closes any open lora autocomplete.
document.addEventListener('click', e => {
  if (!e.target.closest('.lora-field')) {
    document.querySelectorAll('.lora-ac.open').forEach(a => a.classList.remove('open'));
  }
});

loadLoras();
loadUpscalers();
updateUpscaleMax('');
updateUpscaleMax('edit_');

// Hide the generate/edit tabs when no model is loaded, and disable the
// edit tab when the loaded model doesn't support image editing.
async function applyModelState() {
  let hasModel = true;
  let supportsEdit = true;
  try {
    const res = await fetch('/health');
    if (res.ok) {
      const data = await res.json();
      hasModel = (data.models || []).length > 0;
      supportsEdit = !!(data.capabilities && data.capabilities.supports_edit);
    }
  } catch (e) { /* assume a model is present on network errors */ }
  $('no_model').classList.toggle('hidden', hasModel);
  $('edit_no_model').classList.toggle('hidden', hasModel);
  // Edit is usable only when a model is loaded AND it supports editing.
  const editAvailable = hasModel && supportsEdit;
  $('edit_no_support').classList.toggle('hidden', !hasModel || supportsEdit);
}
applyModelState();

$('generate').addEventListener('click', () => {
  if (!validateDims('')) return;
  runWithBusy({
    btn: $('generate'),
    overlay: $('overlay'),
    timerEl: 'timer',
    timerTextEl: 'timer_text',
    request: () => postJSON('/text2image', collectSettings('')),
    onSuccess: async (blob) => {
      currentUrl = URL.createObjectURL(blob);
      setStageActions($('download'), $('info_btn'), true);
      showResult('image', 'placeholder', currentUrl);
      const meta = await decodeMeta(blob, 'generation_data');
      currentMeta = meta;
      renderInfo(meta);
      genHist.add(currentUrl, meta);
    },
  });
});

document.querySelectorAll('.tab').forEach(t => t.addEventListener('click', () => {
  document.querySelectorAll('.view').forEach(v => v.classList.toggle('hidden', v.id !== 'view-' + t.dataset.tab));
  document.querySelectorAll('.tab').forEach(x => x.classList.toggle('active', x === t));
}));

let uInputB64 = null;      // base64 (no prefix) sent to /upscale
let uInputDims = null;     // {w, h} of the input
let uOutUrl = null;        // object URL of the upscaled result
let uOutMeta = null;       // upscale metadata

const dzName = $('dz_name');
function loadUpscaleFile(file) {
  if (!file || !file.type.startsWith('image/')) return;
  const reader = new FileReader();
  reader.onload = () => {
    const dataUrl = reader.result;
    const img = new Image();
    img.onload = () => {
      uInputB64 = dataUrl.split(',')[1];
      uInputDims = { w: img.naturalWidth, h: img.naturalHeight };
      updateUpscaleTabFactor();
      if (uOutUrl) { URL.revokeObjectURL(uOutUrl); uOutUrl = null; }
      uOutMeta = null;
      setStageActions($('udownload'), $('uinfo_btn'), false);
      $('usingle_img').src = dataUrl;
      $('usingle').classList.remove('hidden');
      $('uresult').classList.add('hidden');
      $('uplaceholder').style.display = 'none';
      dzName.textContent = file.name;
      dzName.style.display = 'block';
      $('dz_hint').style.display = 'none';
      $('upscale').disabled = false;
      applyUpscalerDefaults();
    };
    img.src = dataUrl;
  };
  reader.readAsDataURL(file);
}

bindDropzone('dropzone', 'file', files => loadUpscaleFile(files[0]));

function updateUpscaleTabFactor() {
  const factor = parseFloat($('u_factor').value) || 1;
  $('u_factor_val').textContent = factor.toFixed(2);
  renderRes($('u_final_res'),
    uInputDims && uInputDims.w, uInputDims && uInputDims.h, factor);
}
function updateUpscaleTabMax() {
  const model = $('upscaler_model').value;
  const max = model ? (upscalerScales[model] || 2) : 1;
  clampSlider($('u_factor'), max);
  updateUpscaleTabFactor();
}

function applyUpscalerDefaults() {
  const model = $('upscaler_model').value;
  const scale = upscalerScales[model];
  if (scale) $('u_factor').value = scale;
  updateUpscaleTabMax();
}
$('u_factor').addEventListener('input', updateUpscaleTabFactor);
$('upscaler_model').addEventListener('change', applyUpscalerDefaults);

function renderUpscaleInfo(meta) {
  const grid = $('info_modal_grid');
  grid.innerHTML = '';
  addRow(grid, 'Input resolution', uInputDims ? uInputDims.w + ' × ' + uInputDims.h : '—');
  const up = $('u_img');
  if (up.naturalWidth) addRow(grid, 'Output resolution', up.naturalWidth + ' × ' + up.naturalHeight);
  if (meta) {
    addRow(grid, 'Upscaler model', meta.upscaler_model);
    addRow(grid, 'Upscale factor', meta.upscale_factor);
  }
}

$('upscale').addEventListener('click', () => {
  const model = $('upscaler_model').value;
  const factor = Math.max(1, parseFloat($('u_factor').value) || 1);
  if (!model || !uInputB64) {
    setTimer('utimer', 'utimer_text', 'error',
      !model ? 'error: select an upscaler model' : 'error: load an input image');
    return;
  }

  runWithBusy({
    btn: $('upscale'),
    overlay: $('uoverlay'),
    timerEl: 'utimer',
    timerTextEl: 'utimer_text',
    request: async () => {
      return await postJSON('/upscale', {
        image_b64: uInputB64,
        upscale_factor: factor,
        pixel_upscaler: model,
      });
    },
    onSuccess: async (blob) => {
      if (uOutUrl) URL.revokeObjectURL(uOutUrl);
      uOutUrl = URL.createObjectURL(blob);
      const up = $('u_img');
      up.src = uOutUrl;
      $('usingle').classList.add('hidden');
      $('uplaceholder').style.display = 'none';
      $('uresult').classList.remove('hidden');
      setStageActions($('udownload'), $('uinfo_btn'), true);
      uOutMeta = await decodeMeta(blob, 'upscale_data');
      renderUpscaleInfo(uOutMeta);
    },
  });
});

$('udownload').addEventListener('click', () => {
  download(uOutUrl, 'upscaled.png');
});

let eOutUrl = null;      // object URL of the edited result
let eOutMeta = null;     // edit metadata

function addEditRefs(files) {
  [...files].filter(f => f.type.startsWith('image/')).forEach(file => {
    const reader = new FileReader();
    reader.onload = () => {
      const dataUrl = reader.result;
      const img = new Image();
      img.onload = () => {
        editRefs.push({
          dataUrl, b64: dataUrl.split(',')[1],
          dims: { w: img.naturalWidth, h: img.naturalHeight }, name: file.name,
        });
        renderEditRefs();
        resetEditResult();
        updateEditFinalRes();
      };
      img.src = dataUrl;
    };
    reader.readAsDataURL(file);
  });
}

function renderEditRefs() {
  const box = $('edit_refs');
  box.innerHTML = '';
  editRefs.forEach((r, i) => {
    const div = document.createElement('div');
    div.className = 'ref-thumb' + (i === 0 ? ' first' : '');
    div.title = r.name + (i === 0 ? ' (sets output size)' : '');
    const img = document.createElement('img');
    img.className = 'thumb-img';
    img.src = r.dataUrl;
    img.alt = 'reference ' + (i + 1);
    div.appendChild(img);
    const rm = document.createElement('button');
    rm.className = 'ref-remove';
    rm.textContent = '\u00d7';
    rm.title = 'Remove';
    rm.addEventListener('click', () => {
      editRefs.splice(i, 1);
      renderEditRefs();
      resetEditResult();
      updateEditFinalRes();
    });
    div.appendChild(rm);
    if (i === 0) {
      const badge = document.createElement('span');
      badge.className = 'first-badge';
      badge.textContent = '1st';
      div.appendChild(badge);
    }
    box.appendChild(div);
  });
}

function resetEditResult() {
  // Do NOT revoke eOutUrl: it may still be referenced by an entry in history.
  // History entries are revoked only when they are evicted (>10) or replaced.
  eOutUrl = null;
  eOutMeta = null;
  setStageActions($('edownload'), $('einfo_btn'), false);
  $('e_img').style.display = 'none';
  $('eplaceholder').style.display = 'block';
  $('edit_btn').disabled = editRefs.length === 0;
}

bindDropzone('edit_dropzone', 'edit_file', addEditRefs);

const editHist = makeHistory({
  containerId: 'ehistory_items', rootId: 'ehistory', alt: 'edit',
  onSelect: (item) => {
    // Don't revoke the previous eOutUrl: it lives on as an entry in history.
    eOutUrl = item.url;
    eOutMeta = item.meta;
    showResult('e_img', 'eplaceholder', item.url);
    renderInfo(item.meta);
    applySettings('edit', item.meta);
    setStageActions($('edownload'), $('einfo_btn'), true);
  },
});

$('edit_btn').addEventListener('click', () => {
  if (editRefs.length === 0) return;
  if (!validateDims('edit_')) return;

  runWithBusy({
    btn: $('edit_btn'),
    overlay: $('eoverlay'),
    timerEl: 'etimer',
    timerTextEl: 'etimer_text',
    request: () => postJSON('/edit', collectSettings('edit_', {
      // OpenAI-style: one image -> a string, many -> an array.
      image: editRefs.length === 1 ? editRefs[0].b64 : editRefs.map(r => r.b64),
    })),
    onSuccess: async (blob) => {
      // Don't revoke the previous eOutUrl: it is kept as an entry in history.
      eOutUrl = URL.createObjectURL(blob);
      showResult('e_img', 'eplaceholder', eOutUrl);
      setStageActions($('edownload'), $('einfo_btn'), true);
      eOutMeta = await decodeMeta(blob, 'generation_data');
      renderInfo(eOutMeta);
      editHist.add(eOutUrl, eOutMeta);
    },
  });
});

$('edownload').addEventListener('click', () =>
  download(eOutUrl, `thenoise_edit_${seedName(eOutMeta, 'x')}.png`));

$('info_btn').addEventListener('click', () => openInfo('Image info', () => renderInfo(currentMeta)));
$('uinfo_btn').addEventListener('click', () => openInfo('Upscale info', () => renderUpscaleInfo(uOutMeta)));
$('einfo_btn').addEventListener('click', () => openInfo('Edit info', () => renderInfo(eOutMeta)));
$('info_modal_close').addEventListener('click', closeInfo);
$('info_backdrop').addEventListener('click', closeInfo);
document.addEventListener('keydown', e => { if (e.key === 'Escape') closeInfo(); });
