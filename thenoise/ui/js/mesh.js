
let meshOutUrl = null;
let meshOutMeta = null;
let meshShown = 0;   // a shape still being read when the next one arrives is stale

const meshJob = { id: null };
const MESH_PROGRESS = { root: 'mesh_progress', label: 'mesh_progress_label', bar: 'mesh_bar' };
const MESH_MODES = {i2d: 'image-to-shape', mv2d: 'views-to-shape', t2d: 'text-to-shape'};
const MESH_FORMATS = { glb: 'model/gltf-binary', obj: 'text/plain', stl: 'model/stl' };
const MESH_MAX_VIEWS = 4;
const meshViews = [];   // {b64, dataUrl, name}, in view order

function meshCeiling() {
  return $('mesh_file').multiple ? MESH_MAX_VIEWS : 1;
}

function meshViewLabel(index) {
  return meshCeiling() > 1 ? 'view ' + (index + 1) : 'image';
}

function meshFail(message) {
  setTimer('mesh_timer', 'mesh_timer_text', 'error', 'error: ' + message);
  return false;
}

function renderMeshViews() {
  const box = $('mesh_refs');
  box.innerHTML = '';
  meshViews.forEach((view, i) => addThumb(box, {
    src: view.dataUrl, alt: meshViewLabel(i), cls: i === 0 ? 'first' : '',
    title: meshViewLabel(i) + (view.name ? ': ' + view.name : ''),
    onRemove: () => {
      meshViews.splice(i, 1);
      renderMeshViews();
      resetMeshResult();
    },
  }));
  $('mesh_dz_hint').style.display = meshViews.length ? 'none' : '';
  const nameEl = $('mesh_dz_name');
  const names = meshViews.map(view => view.name).filter(Boolean).join(', ');
  nameEl.textContent = names
    ? (meshCeiling() > 1 && meshViews.length > 1
       ? meshViews.length + ' views: ' + names : names)
    : '';
  nameEl.style.display = nameEl.textContent ? 'block' : 'none';
}

async function loadMeshPicture(file) {
  const ceiling = meshCeiling();
  if (meshViews.length >= ceiling) {
    return meshFail(ceiling > 1 ? 'a shape takes at most ' + ceiling + ' views'
                                : 'this model takes one image');
  }
  const image = await readImageFile(file);
  if (!image) return;
  meshViews.push({ dataUrl: image.dataUrl, b64: image.b64, name: file.name });
  renderMeshViews();
  resetMeshResult();
}

bindDropzone('mesh_dropzone', 'mesh_file', async files => {
  for (const file of files) await loadMeshPicture(file);
});
initLora('mesh_');

function collectMeshSettings() {
  const body = {};
  addNumbers(body, 'mesh_', ['latents', 'steps', 'seed', 'octree_resolution'],
                              ['guidance_scale', 'bounds', 'mc_level', 'scale']);
  for (const [id, key] of [['mesh_sampler', 'sampler'],
                           ['mesh_output_format', 'output_format']]) {
    if ($(id).value) body[key] = $(id).value;
  }
  const lora_specs = parseLora($('mesh_lora_specs').value);
  if (lora_specs) body.lora_specs = lora_specs;
  body.image = meshViews.length === 1 ? meshViews[0].b64
                                      : meshViews.map(view => view.b64);
  return body;
}

function validateMeshInputs() {
  if (!meshViews.length) return meshFail('a shape needs an image');
  const check = (fields, ok, message) => {
    for (const f of fields) {
      const v = $('mesh_' + f).value;
      if (v !== '' && !ok(Number(v))) return meshFail(message);
    }
    return true;
  };
  const whole = (n) => Number.isInteger(n) && n >= 1;
  return check(['steps'], whole, 'steps must be a positive whole number')
    && check(['seed'], (n) => Number.isInteger(n) && n >= 0,
             'seed must be a nonnegative whole number (leave empty for random)')
    && check(['guidance_scale'], (n) => Number.isFinite(n) && n >= 0,
             'guidance must be a nonnegative number');
}

function meshName(meta) {
  return 'thenoise_mesh_' + seedName(meta, 'x') + '.' + ((meta && meta.output_format) || 'glb');
}

function resetMeshResult() {
  meshOutMeta = null;
  meshShown += 1;
  clearMesh();
  $('mesh_wire').disabled = true;
  setStageActions($('mesh_download'), $('mesh_info_btn'), false);
  $('mesh_out').style.display = 'none';
  $('meshplaceholder').style.display = '';
}

function showMeshResult(blob) {
  $('meshplaceholder').style.display = 'none';
  $('mesh_out').style.display = 'flex';
  $('mesh_out_name').textContent = meshName(meshOutMeta);
  setStageActions($('mesh_download'), $('mesh_info_btn'), true);
  const note = $('mesh_out_note');
  if (!meshViewerReady()) {
    note.textContent = 'no webgl2 in this browser — download it to look at the shape';
    return;
  }
  note.textContent = 'reading the shape\u2026';
  const shown = meshShown;
  showMeshBlob(blob, meshOutMeta.output_format).then(
    (shape) => {
      if (shown !== meshShown) return;
      $('mesh_wire').disabled = false;
      note.textContent = shape.triangles.toLocaleString() + ' triangles, '
        + shape.vertices.toLocaleString() + ' vertices \u2014 drag to turn, wheel to zoom';
    },
    (error) => {
      if (shown === meshShown) note.textContent = 'this viewer cannot read it: ' + error.message;
    },
  );
}

$('mesh_generate').addEventListener('click', () => {
  if (!validateMeshInputs()) return;
  const body = collectMeshSettings();
  resetMeshResult();
  runQueuedJob({
    job: meshJob,
    submit: '/jobs/mesh',
    body,
    btn: $('mesh_generate'),
    cancelBtn: $('mesh_cancel'),
    overlay: $('mesh_overlay'),
    timerEl: 'mesh_timer',
    timerTextEl: 'mesh_timer_text',
    progress: MESH_PROGRESS,
    onArtifact: async (blob, status) => {
      if (meshOutUrl) URL.revokeObjectURL(meshOutUrl);
      const { image, ...sent } = body;
      meshOutMeta = { ...sent, ...status.resolved, model: status.model,
                         picture: meshViews.map(view => view.name).join(', ') };
      meshOutUrl = URL.createObjectURL(new Blob([blob], {
        type: MESH_FORMATS[meshOutMeta.output_format] || 'application/octet-stream',
      }));
      showMeshResult(blob);
    },
  });
});

$('mesh_cancel').addEventListener('click', () => cancelJob(meshJob, $('mesh_cancel')));

$('mesh_download').addEventListener('click', () => download(meshOutUrl, meshName(meshOutMeta)));

$('mesh_info_btn').addEventListener('click', () => openInfo('Shape info', renderMeshInfo));

function renderMeshInfo() {
  const m = meshOutMeta || {};
  const rows = [];
  if (m.model) rows.push(['Model', m.model, '']);
  if (m.mode) rows.push([FIELD_LABELS.mode, MESH_MODES[m.mode] || m.mode, '']);
  for (const key of ['latents', 'steps', 'guidance_scale', 'sampler', 'octree_resolution',
                     'bounds', 'mc_level', 'scale', 'seed', 'output_format']) {
    if (m[key] != null && m[key] !== '') {
      rows.push([FIELD_LABELS[key], knobLabel(key, m[key]), '']);
    }
  }
  if (m.picture) rows.push([FIELD_LABELS.picture, m.picture, '']);
  const lora = m.lora_specs;
  const hasLora = Array.isArray(lora) ? lora.length : (lora ? Object.keys(lora).length : 0);
  if (hasLora) rows.push(['LoRA', formatLora(lora), 'prompt']);
  renderInfoRows(rows);
}
