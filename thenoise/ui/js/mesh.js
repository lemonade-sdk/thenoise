/* mesh.js — one picture in, POST /jobs/mesh, a mesh file out. Loaded after helpers.js. */

let meshOutUrl = null;
let meshOutMeta = null;

const meshJob = { id: null };
const MESH_PROGRESS = { root: 'mesh_progress', label: 'mesh_progress_label', bar: 'mesh_bar' };
const MESH_MODES = {i2d: 'image-to-shape', t2d: 'text-to-shape'};
const MESH_FORMATS = { glb: 'model/gltf-binary', obj: 'text/plain', stl: 'model/stl' };
const meshPicture = { b64: null, dataUrl: null, name: null };

function renderMeshPicture() {
  const box = $('mesh_refs');
  box.innerHTML = '';
  $('mesh_dz_hint').style.display = meshPicture.b64 ? 'none' : '';
  const nameEl = $('mesh_dz_name');
  nameEl.textContent = meshPicture.name || '';
  nameEl.style.display = meshPicture.name ? 'block' : 'none';
  if (!meshPicture.b64) return;
  addThumb(box, {
    src: meshPicture.dataUrl, alt: 'the picture', cls: 'first',
    title: 'the picture the shape is reconstructed from',
    onRemove: () => {
      Object.assign(meshPicture, { b64: null, dataUrl: null, name: null });
      renderMeshPicture();
      resetMeshResult();
    },
  });
}

async function loadMeshPicture(file) {
  const image = await readImageFile(file);
  if (!image) return;
  Object.assign(meshPicture, { dataUrl: image.dataUrl, b64: image.b64, name: file.name });
  renderMeshPicture();
  resetMeshResult();
}

bindDropzone('mesh_dropzone', 'mesh_file', files => loadMeshPicture(files[0]));
initLora('mesh_');

function collectMeshSettings() {
  const body = {};
  addNumbers(body, 'mesh_', ['latents', 'steps', 'seed', 'octree_resolution'],
                              ['guidance_scale', 'bounds', 'mc_level']);
  for (const [id, key] of [['mesh_sampler', 'sampler'],
                           ['mesh_output_format', 'output_format']]) {
    if ($(id).value) body[key] = $(id).value;
  }
  const lora_specs = parseLora($('mesh_lora_specs').value);
  if (lora_specs) body.lora_specs = lora_specs;
  body.image = meshPicture.b64;
  return body;
}

function validateMeshInputs() {
  const fail = (message) => {
    setTimer('mesh_timer', 'mesh_timer_text', 'error', 'error: ' + message);
    return false;
  };
  if (!meshPicture.b64) return fail('a shape needs a picture');
  const check = (fields, ok, message) => {
    for (const f of fields) {
      const v = $('mesh_' + f).value;
      if (v !== '' && !ok(Number(v))) return fail(message);
    }
    return true;
  };
  const whole = (n) => Number.isInteger(n) && n >= 1;
  const real = (n) => Number.isFinite(n);
  return check(['latents', 'steps', 'octree_resolution'], whole,
               'tokens, steps and octree must be positive whole numbers')
    && check(['seed'], (n) => Number.isInteger(n) && n >= 0,
             'seed must be a nonnegative whole number (leave empty for random)')
    && check(['guidance_scale'], (n) => real(n) && n >= 0,
             'guidance must be a nonnegative number')
    && check(['bounds'], (n) => real(n) && n > 0, 'bounds must be positive')
    && check(['mc_level'], real, 'the cut level must be a number');
}

function meshName(meta) {
  return 'thenoise_mesh_' + seedName(meta, 'x') + '.' + ((meta && meta.output_format) || 'glb');
}

function resetMeshResult() {
  meshOutMeta = null;
  setStageActions($('mesh_download'), $('mesh_info_btn'), false);
  $('mesh_out').style.display = 'none';
  $('meshplaceholder').style.display = '';
}

function showMeshResult() {
  $('meshplaceholder').style.display = 'none';
  $('mesh_out').style.display = 'flex';
  $('mesh_out_name').textContent = meshName(meshOutMeta);
  $('mesh_out_note').textContent = 'no viewer here — download it to look at the shape';
  setStageActions($('mesh_download'), $('mesh_info_btn'), true);
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
                         picture: meshPicture.name };
      meshOutUrl = URL.createObjectURL(new Blob([blob], {
        type: MESH_FORMATS[meshOutMeta.output_format] || 'application/octet-stream',
      }));
      showMeshResult();
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
                     'bounds', 'mc_level', 'seed', 'output_format']) {
    if (m[key] != null && m[key] !== '') rows.push([FIELD_LABELS[key], m[key], '']);
  }
  if (m.picture) rows.push(['Picture', m.picture, '']);
  const lora = m.lora_specs;
  const hasLora = Array.isArray(lora) ? lora.length : (lora ? Object.keys(lora).length : 0);
  if (hasLora) rows.push(['LoRA', formatLora(lora), 'prompt']);
  renderInfoRows(rows);
}
