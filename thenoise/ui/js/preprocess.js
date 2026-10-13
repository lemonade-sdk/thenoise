/* preprocess.js — the Preprocess tab: image -> extracted map (plain global script).
   The wire keeps the technical names; the words shown to the user are the plain ones. */

let preAvailable = {};    // control-map kind -> the methods implementing it
let preInputB64 = null;   // base64 (no data: prefix) sent to /preprocess
let preOutUrl = null;
let preOutMeta = null;

const KIND_LABELS = {
  canny: 'Edges', hed: 'Soft edges', softedge: 'Soft edges', lineart: 'Line drawing',
  sketch: 'Sketch lines', depth: 'Depth', normal: 'Normals', pose: 'Pose',
  binary: 'Silhouette',
};
const OPTION_LABELS = {
  low: 'Edge cutoff', high: 'Strong edge cutoff', thickness: 'Line width',
  invert: 'Invert depth',
};

const kindLabel = kind => KIND_LABELS[kind]
  ?? (kind ? kind.charAt(0).toUpperCase() + kind.slice(1) : kind);

function fillPreModels() {
  fillSelect($('p_model'), preAvailable[$('p_type').value] || []);
}

function showPreOptions() {
  const type = $('p_type').value;
  const methods = preAvailable[type] || [];
  // A kind with one implementation needs no method picker.
  $('p_model_field').classList.toggle('hidden', methods.length <= 1);
  $('p_canny_opts').classList.toggle('hidden', type !== 'canny');
  $('p_depth_opts').classList.toggle('hidden', type !== 'depth');
}

async function loadPreprocessors() {
  const data = await fetchJSON('/preprocessors');
  preAvailable = (data && data.preprocessors) || {};
  fillSelect($('p_type'), Object.keys(preAvailable), kindLabel);
  fillPreModels();
  showPreOptions();
  $('p_run').disabled = preInputB64 === null || !Object.keys(preAvailable).length;
}

function loadPreFile(file) {
  if (!file || !file.type.startsWith('image/')) return;
  const reader = new FileReader();
  reader.onload = () => {
    preInputB64 = reader.result.split(',')[1];
    if (preOutUrl) { URL.revokeObjectURL(preOutUrl); preOutUrl = null; }
    preOutMeta = null;
    setStageActions($('p_download'), $('p_info_btn'), false);
    $('p_single_img').src = reader.result;
    $('p_single').classList.remove('hidden');
    $('p_result').classList.add('hidden');
    $('p_placeholder').style.display = 'none';
    $('p_dz_name').textContent = file.name;
    $('p_dz_name').style.display = 'block';
    $('p_dz_hint').style.display = 'none';
    $('p_run').disabled = !Object.keys(preAvailable).length;
  };
  reader.readAsDataURL(file);
}

function renderPreInfo(meta) {
  const grid = $('info_modal_grid');
  grid.innerHTML = '';
  if (!meta || typeof meta !== 'object') {
    addRow(grid, 'Metadata', null, 'none');
    return;
  }
  addRow(grid, 'Detect', kindLabel(meta.type));
  addRow(grid, 'Method', meta.model);
  for (const [key, value] of Object.entries(meta.options || {})) {
    addRow(grid, OPTION_LABELS[key] || key, value);
  }
}

function preRequest() {
  const type = $('p_type').value;
  const body = {
    image_b64: preInputB64,
    type: type,
    // One method means there is nothing to choose: let the kind's default apply.
    model: $('p_model_field').classList.contains('hidden') ? null : ($('p_model').value || null),
  };
  if (type === 'canny') {
    body.canny = {
      low: parseFloat($('p_low').value) || 0,
      high: parseFloat($('p_high').value) || 0,
      thickness: parseInt($('p_thickness').value, 10) || 1,
    };
  }
  if (type === 'depth') {
    body.depth = { invert: $('p_invert').checked, colormap: 'gray' };
  }
  return body;
}

bindDropzone('p_dropzone', 'p_file', files => loadPreFile(files[0]));
$('p_type').addEventListener('change', () => { fillPreModels(); showPreOptions(); });

$('p_run').addEventListener('click', () => {
  runWithBusy({
    btn: $('p_run'),
    overlay: $('p_overlay'),
    timerEl: 'p_timer',
    timerTextEl: 'p_timer_text',
    request: () => postJSON('/preprocess', preRequest()),
    onSuccess: async (blob) => {
      if (preOutUrl) URL.revokeObjectURL(preOutUrl);
      preOutUrl = URL.createObjectURL(blob);
      $('p_img').src = preOutUrl;
      $('p_single').classList.add('hidden');
      $('p_result').classList.remove('hidden');
      $('p_placeholder').style.display = 'none';
      setStageActions($('p_download'), $('p_info_btn'), true);
      preOutMeta = await decodeMeta(blob, 'preprocess_data');
    },
  });
});

$('p_download').addEventListener('click', () =>
  download(preOutUrl, ($('p_type').value || 'map') + '.png'));
$('p_info_btn').addEventListener('click', () =>
  openInfo('Extraction info', () => renderPreInfo(preOutMeta)));

loadPreprocessors();
