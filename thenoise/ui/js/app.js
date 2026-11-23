/* app.js — view-specific logic for TheNoise (Generate / Edit / Upscale).
   Plain global script, loaded AFTER helpers.js. */

let currentUrl = null;
let currentMeta = null;

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

// A size left out is the model's to choose, and only it knows how: named sizes are all we state.
function editTargetDims() {
  const ew = parseInt($('edit_width').value, 10);
  const eh = parseInt($('edit_height').value, 10);
  return ew && eh ? { w: ew, h: eh } : null;
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
  restorePostprocess(p, meta);
  if (Array.isArray(meta.lora_specs)) {
    $(p + 'lora_specs').value = meta.lora_specs.join('\n');
  }
  if (meta.pixel_upscaler) {
    $(p + 'pixel_upscaler').value = meta.pixel_upscaler;
  }
  updateUpscaleMax(p);
}

bindSwap('swap', 'width', 'height');
bindSwap('eswap', 'edit_width', 'edit_height', updateEditFinalRes);

$('download').addEventListener('click', () =>
  download(currentUrl, `thenoise_${seedName(currentMeta, 'x')}.png`));

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

initLora('');
initLora('edit_');

// A single click closes any open lora autocomplete.
document.addEventListener('click', e => {
  if (!e.target.closest('.lora-field')) {
    document.querySelectorAll('.lora-ac.open').forEach(a => a.classList.remove('open'));
  }
});

loadUpscalers();
loadPostprocess();
updateUpscaleMax('');
updateUpscaleMax('edit_');

/* ---------- Generate is the medium this server renders ---------- */
// The checkpoint loaded at startup decides it; the other medium's routes are never installed.
// Upscale and Control Maps need no model, so whoever is on one when /health answers stays.
const MODEL_VIEWS = ['generate', 'video', 'mesh', 'edit'];

let mediumView = 'generate';   // what the loaded model serves
let shownView = 'generate';

function showView(view) {
  shownView = view;
  document.querySelectorAll('.view')
    .forEach(v => v.classList.toggle('hidden', v.id !== 'view-' + view));
  document.querySelectorAll('.tab[data-tab]')
    .forEach(x => x.classList.toggle('active', x.dataset.tab === view));
  $('generate_tab').classList.toggle('active', view === mediumView);
}

$('generate_tab').addEventListener('click', () => showView(mediumView));
document.querySelectorAll('.tab[data-tab]').forEach(t =>
  t.addEventListener('click', () => showView(t.dataset.tab)));

// Solver names are /health's to give, so a menu here holds no list of its own.
function fillSamplers(names) {
  if (!names || !names.length) return; // unreachable server: only `auto` is offered
  for (const id of ['sampler', 'edit_sampler', 'video_sampler', 'mesh_sampler']) {
    const sel = $(id);
    const keep = sel.value;
    fillSelect(sel, ['', ...names], name => name || 'auto');
    sel.value = keep === '' || names.includes(keep) ? keep : '';
  }
}

// Gate the UI on the loaded model's capabilities, which the pipeline enforces too.
async function applyModelState() {
  let models = null; // null = /health unreachable: assume a model that can do it all
  let caps = {};
  let samplers = null;
  let rgba = false; // true when the loaded model's VAE emits an alpha channel
  try {
    const res = await fetchOrGone('/health', undefined, PATIENCE);
    if (res.ok) {
      const data = await res.json();
      models = data.models || [];
      caps = data.capabilities || {};
      samplers = data.samplers || null;
      rgba = (data.pixel_channels || 3) > 3;
    }
  } catch (e) { /* assume a model is present on network errors */ }
  const hasModel = models !== null && models.length > 0;
  const noModel = models !== null && !hasModel;
  const capable = (name) => models === null || !!caps[name];
  // Nested under the medium's name, and null on a model of the other one.
  const video = caps.video || null;
  const mesh = caps.mesh || null;
  // One bit now: a model either takes a reference or does not. How it uses one
  // (``caps.ref_methods``) is a request field, not something the tabs split on.
  const takesImage = models === null || !!caps.image?.i2i;

  // An RGBA model can return transparent pixels; without something behind them
  // they would read as black (or as the panel colour) rather than as transparency.
  document.body.classList.toggle('rgba_out', rgba);

  mediumView = hasModel ? (video ? 'video' : mesh ? 'mesh' : 'generate') : 'generate';
  if (MODEL_VIEWS.includes(shownView)) showView(mediumView);
  // A video server installs no edit routes at all, so the tab would only open a refusal.
  $('edit_tab').classList.toggle('hidden', mediumView !== 'generate');

  $('no_model').classList.toggle('hidden', !noModel);
  $('edit_no_model').classList.toggle('hidden', !noModel);
  // Edit is usable only when an image model is loaded AND it supports editing.
  $('edit_no_support').classList.toggle('hidden', !hasModel || takesImage);
  // KV cache: off the menu (literally) when the model never wired the cache in.
  const kv = $('edit_kv_cache');
  const hasKvCache = capable('kv_cache');
  kv.disabled = !hasKvCache;
  if (!hasKvCache) kv.value = '';
  kv.title = hasKvCache ? '' : 'the loaded model has no reference-latent KV cache';
  // The ways a model can use a reference are the model's own: one way and there is
  // nothing to choose, and offering a method it lacks would only earn a refusal.
  const method = $('edit_ref_method');
  const methods = caps.ref_methods || [];
  const picked = method.value;
  method.replaceChildren(new Option('auto', ''), ...methods.map((m) => new Option(m, m)));
  method.value = methods.includes(picked) ? picked : '';
  method.disabled = methods.length < 2;
  method.title = method.disabled
    ? 'the loaded model has only one way with a reference' : '';
  $('video_i2v_field').classList.toggle('hidden', !video?.i2v);
  $('video_flf2v_field').classList.toggle('hidden', !video?.flf2v);
  const views = (mesh && mesh.mv2d) || 1;
  $('mesh_file').multiple = views > 1;
  $('mesh_views_hint').classList.toggle('hidden', views < 2);
  fillSamplers(samplers);
}
applyModelState();

// Each screen keeps the job it has in flight, for its Cancel button to name.
const genJob = { id: null };
const GEN_PROGRESS = { root: 'genprogress', label: 'genprogress_label', bar: 'genbar' };

$('generate').addEventListener('click', () => {
  if (!validateDims('')) return;
  runQueuedJob({
    job: genJob,
    submit: '/jobs/text2image',
    body: collectSettings(''),
    btn: $('generate'),
    cancelBtn: $('gencancel'),
    overlay: $('overlay'),
    timerEl: 'timer',
    timerTextEl: 'timer_text',
    progress: GEN_PROGRESS,
    onArtifact: async (blob) => {
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

$('gencancel').addEventListener('click', () => cancelJob(genJob, $('gencancel')));

let uInputB64 = null;      // base64 (no prefix) sent to /upscale
let uInputDims = null;     // {w, h} of the input
let uOutUrl = null;        // object URL of the upscaled result
let uOutMeta = null;       // upscale metadata

async function loadUpscaleFile(file) {
  const image = await readImageFile(file);
  if (!image) return;
  uInputB64 = image.b64;
  uInputDims = image.dims;
  updateUpscaleTabFactor();
  if (uOutUrl) { URL.revokeObjectURL(uOutUrl); uOutUrl = null; }
  uOutMeta = null;
  setStageActions($('udownload'), $('uinfo_btn'), false);
  showInputPreview(image.dataUrl, file, {
    singleImg: 'usingle_img', single: 'usingle', result: 'uresult',
    placeholder: 'uplaceholder', dzName: 'dz_name', dzHint: 'dz_hint',
  });
  $('upscale').disabled = false;
  applyUpscalerDefaults();
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
  const rows = [['Input resolution', uInputDims ? uInputDims.w + ' × ' + uInputDims.h : '—', '']];
  const up = $('u_img');
  if (up.naturalWidth) rows.push(['Output resolution', up.naturalWidth + ' × ' + up.naturalHeight, '']);
  if (meta) {
    rows.push(['Upscaler model', meta.upscaler_model, '']);
    rows.push(['Upscale factor', meta.upscale_factor, '']);
  }
  renderInfoRows(rows);
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
      showOutputPreview('usingle', 'uresult', 'uplaceholder', 'u_img', uOutUrl);
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
  [...files].forEach(async file => {
    const image = await readImageFile(file);
    if (!image) return;
    editRefs.push({ ...image, name: file.name });
    renderEditRefs();
    resetEditResult();
    updateEditFinalRes();
  });
}

function renderEditRefs() {
  const box = $('edit_refs');
  box.innerHTML = '';
  editRefs.forEach((r, i) => {
    const div = addThumb(box, {
      src: r.dataUrl, alt: 'reference ' + (i + 1), cls: i === 0 ? 'first' : '',
      title: r.name + (i === 0 ? ' (sets output size)' : ''),
      onRemove: () => {
        editRefs.splice(i, 1);
        renderEditRefs();
        resetEditResult();
        updateEditFinalRes();
      },
    });
    if (i === 0) {
      const badge = document.createElement('span');
      badge.className = 'first-badge';
      badge.textContent = '1st';
      div.appendChild(badge);
    }
  });
}

function resetEditResult() {
  // Do NOT revoke eOutUrl: it may still be referenced by an entry in history.
  // History entries are revoked only when they are evicted (>10) or replaced.
  eOutUrl = null;
  eOutMeta = null;
  setStageActions($('edownload'), $('einfo_btn'), false);
  hideResult('e_img', 'eplaceholder');
  $('edit_btn').disabled = editRefs.length === 0;
}

bindDropzone('edit_dropzone', 'edit_file', addEditRefs);

document.addEventListener('paste', e => {
  if ($('view-edit').classList.contains('hidden')) return;
  const images = [...(e.clipboardData?.items || [])]
    .filter(item => item.kind === 'file' && item.type.startsWith('image/'))
    .map(item => item.getAsFile())
    .filter(Boolean);
  if (!images.length) return;
  e.preventDefault();
  addEditRefs(images);
});

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

function editExtras() {
  const extra = {
    // OpenAI-style: one image -> a string, many -> an array.
    image: editRefs.length === 1 ? editRefs[0].b64 : editRefs.map(r => r.b64),
  };
  // A disabled control keeps its value, so also check it is actually enabled.
  const kvSelect = $('edit_kv_cache');
  const kv = kvSelect.disabled ? null : parseTriState(kvSelect.value);
  if (kv !== null) extra.kv_cache = kv;
  const method = $('edit_ref_method').value;
  if (method) extra.ref_method = method;
  return extra;
}

const editJob = { id: null };
const EDIT_PROGRESS = { root: 'editprogress', label: 'editprogress_label', bar: 'editbar' };

$('edit_btn').addEventListener('click', () => {
  if (editRefs.length === 0) return;
  if (!validateDims('edit_')) return;

  runQueuedJob({
    job: editJob,
    submit: '/jobs/edit',
    body: collectSettings('edit_', editExtras()),
    btn: $('edit_btn'),
    cancelBtn: $('editcancel'),
    overlay: $('eoverlay'),
    timerEl: 'etimer',
    timerTextEl: 'etimer_text',
    progress: EDIT_PROGRESS,
    onArtifact: async (blob) => {
      // Don't revoke the previous eOutUrl: it is kept as an entry in history.
      eOutUrl = URL.createObjectURL(blob);
      showResult('e_img', 'eplaceholder', eOutUrl);
      setStageActions($('edownload'), $('einfo_btn'), true);
      eOutMeta = await decodeMeta(blob, 'edit_data');
      renderInfo(eOutMeta);
      editHist.add(eOutUrl, eOutMeta);
    },
  });
});

$('editcancel').addEventListener('click', () => cancelJob(editJob, $('editcancel')));

$('edownload').addEventListener('click', () =>
  download(eOutUrl, `thenoise_edit_${seedName(eOutMeta, 'x')}.png`));

$('info_btn').addEventListener('click', () => openInfo('Image info', () => renderInfo(currentMeta)));
$('uinfo_btn').addEventListener('click', () => openInfo('Upscale info', () => renderUpscaleInfo(uOutMeta)));
$('einfo_btn').addEventListener('click', () => openInfo('Edit info', () => renderInfo(eOutMeta)));
$('info_modal_close').addEventListener('click', closeInfo);
$('info_backdrop').addEventListener('click', closeInfo);
document.addEventListener('keydown', e => { if (e.key === 'Escape') closeInfo(); });
