/* preprocess.js — the Preprocess tab: image -> extracted map (plain global script).
   The wire keeps the technical names; the words shown to the user are the plain ones. */

let preAvailable = {};    // control-map kind -> the methods implementing it
let preDetectors = [];    // person detectors on disk, for the pose kind to crop with
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
  detector: 'Person finder', threshold: 'Person cutoff', kp_threshold: 'Joint cutoff',
  line_width: 'Line width', radius: 'Joint size', background: 'Canvas',
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
  $('p_pose_opts').classList.toggle('hidden', type !== 'pose');
  // With one detector there is nothing to choose, and the default is that one.
  $('p_detector_field').classList.toggle('hidden', preDetectors.length <= 1);
}

async function loadPreprocessors() {
  const data = await fetchJSON('/preprocessors');
  preAvailable = (data && data.preprocessors) || {};
  preDetectors = (data && data.detectors) || [];
  fillSelect($('p_type'), Object.keys(preAvailable), kindLabel);
  fillSelect($('p_detector'), preDetectors);
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
    showInputPreview(reader.result, file, {
      singleImg: 'p_single_img', single: 'p_single', result: 'p_result',
      placeholder: 'p_placeholder', dzName: 'p_dz_name', dzHint: 'p_dz_hint',
    });
    $('p_run').disabled = !Object.keys(preAvailable).length;
  };
  reader.readAsDataURL(file);
}

function renderPreInfo(meta) {
  const rows = [];
  if (!meta || typeof meta !== 'object') {
    renderInfoRows([['Metadata', null, 'none']]);
    return;
  }
  rows.push(['Detect', kindLabel(meta.type), '']);
  rows.push(['Method', meta.model, '']);
  for (const [key, value] of Object.entries(meta.options || {})) {
    rows.push([OPTION_LABELS[key] || key, value, '']);
  }
  renderInfoRows(rows);
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
  if (type === 'pose') {
    body.pose = {
      // No picker means one detector: let the pool's default apply.
      detector: $('p_detector_field').classList.contains('hidden')
        ? '' : $('p_detector').value,
      threshold: parseFloat($('p_person').value) || 0,
      kp_threshold: parseFloat($('p_joint').value) || 0,
      line_width: parseInt($('p_line').value, 10) || 1,
      radius: parseInt($('p_radius').value, 10) || 1,
      background: $('p_canvas').value,
    };
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
      showOutputPreview('p_single', 'p_result', 'p_placeholder', 'p_img', preOutUrl);
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
