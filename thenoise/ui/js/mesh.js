
const MESH_GLYPH = 'data:image/svg+xml;utf8,' + encodeURIComponent(
  '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 64 64">'
  + '<g fill="none" stroke="#8b8fa3" stroke-width="3" stroke-linejoin="round">'
  + '<path d="M32 6 58 20v24L32 58 6 44V20z"/><path d="M6 20l26 14 26-14M32 34v24"/></g></svg>');

let meshSource = null;
let meshGone = null;        
let meshKept = new Set();
let meshOutUrl = null;
let meshOutMeta = null;
let meshShown = 0;   

const meshJob = { id: null };
const MESH_PROGRESS = { root: 'mesh_progress', label: 'mesh_progress_label', bar: 'mesh_bar',
  labels: { conditioning: 'reading the image', sampling: 'shaping' } };
const MESH_MODES = {i2d: 'image-to-shape', mv2d: 'views-to-shape', t2d: 'text-to-shape',
                    shape: 'cut again from a kept shape'};
const MESH_SOLVED = ['latents', 'steps', 'seed', 'guidance_scale', 'sampler'];
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
  meshSource = meshGone = null;   // an image is a shape being made, not one being cut again
  meshViews.push({ dataUrl: image.dataUrl, b64: image.b64, name: file.name, file });
  renderMeshViews();
  resetMeshResult();
}

bindDropzone('mesh_dropzone', 'mesh_file', async files => {
  for (const file of files) await loadMeshPicture(file);
});
initLora('mesh_');

const meshHist = makeHistory({
  containerId: 'mesh_history_items', rootId: 'mesh_history', alt: 'shape',
  onSelect: showMeshShape,
});

async function showMeshShape(item) {
  meshOutMeta = item.meta;
  meshOutUrl = item.url;
  await loadMeshShapes();   // the answer the verdict below is made from
  // The browser still holds the shape to look at; only the server's latents can be cut again.
  const kept = meshKept.has(item.meta.job);
  meshSource = kept ? item.meta.job : null;
  meshGone = kept || !item.meta.job ? null : item.meta.job;
  if (!kept) await adoptMeshImages(item.meta);
  describeMeshSource();
  applyMeshSettings(item.meta);
  try {
    const answer = await fetch(item.url);
    if (!answer.ok) throw new Error('the browser has let it go');
    showMeshResult(await answer.blob());
  } catch (error) {
    meshFail('cannot show that shape again: ' + error.message);
  }
}

// A shape whose latents are gone can still be sampled again from the images it was cut from.
// The File handles are kept rather than their bytes: ten entries of up to four multi-megabyte
// pictures each is a hundred megabytes held for a shape that has already been downloaded.
async function adoptMeshImages(meta) {
  meshViews.length = 0;
  for (const file of meta.files || []) {
    const image = await readImageFile(file);
    if (image) meshViews.push({ dataUrl: image.dataUrl, b64: image.b64, name: file.name, file });
  }
  renderMeshViews();
}

function applyMeshSettings(meta) {
  for (const key of ['latents', 'steps', 'guidance_scale', 'sampler', 'seed',
                     'octree_resolution', 'bounds', 'mc_level', 'scale', 'output_format']) {
    const el = $('mesh_' + key);
    if (el && meta[key] != null && meta[key] !== '') el.value = meta[key];
  }
}

function describeMeshSource() {
  const note = $('mesh_shape_note');
  note.textContent = meshSource
    ? 'this cuts a shape you already made \u2014 nothing new is drawn'
    : meshGone
      ? (meshViews.length
         ? 'that shape is gone from the server \u2014 its pictures are back, so this draws a new one'
         : 'that shape is gone from the server \u2014 add a picture to draw a new one')
      : '';
  note.classList.toggle('hidden', !note.textContent);
  // The only way back to a new shape while a kept one is chosen, because the box takes none.
  $('mesh_shape_clear').classList.toggle('hidden', !meshSource);
  $('mesh_dropzone').inert = !!meshSource;   // inert covers the hidden input inside it

  const shaping = $('mesh_shape_knobs');
  shaping.inert = !!meshSource;
  shaping.classList.toggle('muted', !!meshSource);
}

async function loadMeshShapes() {
  const listing = await fetchJSON('/shapes');
  
  if (listing && Array.isArray(listing.shapes)) meshKept = new Set(listing.shapes);
}

registerKnobs('finish', '/finish');
loadKnobs('finish', ['mesh_']);

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
  if (meshSource) {
    body.from_shape = meshSource;
    
    for (const key of MESH_SOLVED) delete body[key];
  } else {
    body.image = meshViews.length === 1 ? meshViews[0].b64
                                        : meshViews.map(view => view.b64);
  }
  return { ...body, ...collectKnobs('finish', 'mesh_') };
}

function validateMeshInputs() {
  if (!meshViews.length && !meshSource) {
    if (meshGone) {
      return meshFail('that shape is no longer on the server \u2014 add an image for a new one');
    }
    return meshFail(meshHist.items.length
      ? 'a shape needs an image \u2014 or pick one from History to cut it again'
      : 'a shape needs an image');
  }
  
  if (meshSource) return true;
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
  meshGone = null;      // once a request is running, what an old entry lost is no longer the story
  describeMeshSource();
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
    note.textContent = 'this browser cannot show 3D — download it to look at the shape';
    return Promise.resolve(null);
  }
  note.textContent = 'putting it on screen\u2026';
  const shown = meshShown;
  return showMeshBlob(blob, meshOutMeta.output_format).then(
    (shape) => {
      if (shown !== meshShown) return null;
      $('mesh_wire').disabled = false;
      note.textContent = shape.triangles.toLocaleString() + ' triangles, '
        + shape.vertices.toLocaleString() + ' vertices'
        + (meshOutMeta.summary ? ' \u00b7 ' + meshOutMeta.summary : '')
        + ' \u2014 drag to turn, wheel to zoom';
      return meshSnapshot();
    },
    (error) => {
      if (shown === meshShown) note.textContent = 'could not open it: ' + error.message;
      return null;
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
      meshOutMeta = { ...status.resolved, model: status.model, job: status.id,
                      summary: status.summary || '',
                      files: meshViews.map(view => view.file),
                      picture: meshViews.map(view => view.name).join(', ') };
      const url = URL.createObjectURL(new Blob([blob], {
        type: MESH_FORMATS[meshOutMeta.output_format] || 'application/octet-stream',
      }));
      meshOutUrl = url;
      const meta = meshOutMeta;   // a newer shape may take the stage before this one is drawn
      showMeshResult(blob).then((still) => {
        meshHist.add(url, meta, still || MESH_GLYPH);
        loadMeshShapes();
      });
    },
  });
});

$('mesh_shape_clear').addEventListener('click', () => {
  meshSource = meshGone = null;
  describeMeshSource();
});
describeMeshSource();
loadMeshShapes();

$('mesh_cancel').addEventListener('click', () => cancelJob(meshJob, $('mesh_cancel')));

$('mesh_download').addEventListener('click', () => download(meshOutUrl, meshName(meshOutMeta)));

$('mesh_info_btn').addEventListener('click', () => openInfo('Shape info', renderMeshInfo));

function renderMeshInfo() {
  const m = meshOutMeta || {};
  const rows = [];
  if (m.model) rows.push(['Model', m.model, '']);
  if (m.mode) rows.push([FIELD_LABELS.mode, MESH_MODES[m.mode] || m.mode, '']);
  
  const ignored = m.mode === 'shape' ? MESH_SOLVED : [];
  for (const key of ['latents', 'steps', 'guidance_scale', 'sampler', 'octree_resolution',
                     'bounds', 'mc_level', 'scale', 'seed', 'output_format']) {
    if (ignored.includes(key)) continue;
    if (m[key] != null && m[key] !== '') {
      rows.push([FIELD_LABELS[key], knobLabel(key, m[key]), '']);
    }
  }
  if (m.picture) rows.push([FIELD_LABELS.picture, m.picture, '']);
  if (m.summary) rows.push(['Made', m.summary, '']);
  const lora = m.lora_specs;
  const hasLora = Array.isArray(lora) ? lora.length : (lora ? Object.keys(lora).length : 0);
  if (hasLora) rows.push(['LoRA', formatLora(lora), 'prompt']);
  rows.push(...knobInfoRows('finish', m.finish || {}));
  renderInfoRows(rows);
}
