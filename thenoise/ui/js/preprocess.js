/* preprocess.js — the Preprocess tab: image -> extracted map (plain global script).
   The wire keeps the technical names; the words shown to the user are the plain ones. */

let preAvailable = {};    // control-map kind -> the methods implementing it
let preOptions = {};      // control-map kind -> the option fields it takes
let preDetectors = [];    // person detectors on disk, for the pose kind to crop with
let preInputB64 = null;   // base64 (no data: prefix) sent to /preprocess
let preOutUrl = null;
let preOutMeta = null;

const KIND_LABELS = {
  canny: 'Edges', hed: 'Soft edges', softedge: 'Soft edges', lineart: 'Line drawing',
  sketch: 'Sketch lines', depth: 'Depth', normal: 'Normals', pose: 'Pose',
  binary: 'Silhouette',
};

const kindLabel = kind => KIND_LABELS[kind]
  ?? (kind ? kind.charAt(0).toUpperCase() + kind.slice(1) : kind);

// A choice's value is a wire name; the user sees the word with its first letter up.
const valueLabel = value => value ? value.charAt(0).toUpperCase() + value.slice(1) : value;

const isNumberField = option => option.type === 'number' || option.type === 'integer';

const optionFields = kind => preOptions[kind] || [];

function optionLabel(name) {
  for (const fields of Object.values(preOptions)) {
    const field = fields.find(entry => entry.name === name);
    if (field) return field.label;
  }
  return name;
}

function preCheckbox(option) {
  const label = document.createElement('label');
  label.className = 'check';
  const input = document.createElement('input');
  input.type = 'checkbox';
  input.id = 'p_opt_' + option.name;
  input.checked = option.default === true;
  label.append(input, document.createTextNode(' ' + option.label));
  return label;
}

function preField(option) {
  const id = 'p_opt_' + option.name;
  const field = document.createElement('div');
  field.className = 'field';
  const label = document.createElement('label');
  label.className = 'label';
  label.htmlFor = id;
  label.textContent = option.label;

  // A `pool` field offers what the server has on disk, named as it is there.
  const choices = option.pool ? preDetectors : option.choices;
  let control;
  if (choices) {
    control = document.createElement('select');
    fillSelect(control, choices, option.pool ? undefined : valueLabel);
    // With one thing to choose there is nothing to choose, and the default is that one.
    if (option.pool && choices.length <= 1) field.classList.add('hidden');
  } else {
    control = document.createElement('input');
    control.type = option.type === 'string' ? 'text' : 'number';
    if (option.min !== undefined) control.min = option.min;
    if (option.max !== undefined) control.max = option.max;
    if (option.step !== undefined) control.step = option.step;
    control.value = option.default;
  }
  control.id = id;
  field.append(label, control);
  return field;
}

function buildPreOptions(kind) {
  const box = $('p_options');
  box.innerHTML = '';
  let pending = null;   // a numeric field waiting for the one it shares a row with
  for (const option of optionFields(kind)) {
    if (!isNumberField(option)) {
      if (pending) { box.append(pending); pending = null; }
      box.append(option.type === 'boolean' ? preCheckbox(option) : preField(option));
      continue;
    }
    if (!pending) { pending = preField(option); continue; }
    const row = document.createElement('div');
    row.className = 'row';
    row.append(pending, preField(option));
    box.append(row);
    pending = null;
  }
  // A field left over at the end takes the full width rather than half a row.
  if (pending) box.append(pending);
}

// The chosen kind's options as the wire wants them: only what the form actually holds.
function preOptionValues(kind) {
  const values = {};
  for (const option of optionFields(kind)) {
    const input = $('p_opt_' + option.name);
    if (!input || input.closest('.hidden')) continue;
    if (option.type === 'boolean') values[option.name] = input.checked;
    else if (isNumberField(option)) {
      // Clearing a number field asks for the default; it is not the same as a zero.
      if (input.value !== '') values[option.name] = input.valueAsNumber;
    } else values[option.name] = input.value;
  }
  return values;
}

function fillPreModels() {
  fillSelect($('p_model'), preAvailable[$('p_type').value] || []);
}

function showPreOptions() {
  const type = $('p_type').value;
  const methods = preAvailable[type] || [];
  // A kind with one implementation needs no method picker.
  $('p_model_field').classList.toggle('hidden', methods.length <= 1);
  buildPreOptions(type);
}

async function loadPreprocessors() {
  const data = await fetchJSON('/preprocessors');
  preAvailable = (data && data.preprocessors) || {};
  preOptions = (data && data.options) || {};
  preDetectors = (data && data.detectors) || [];
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
    rows.push([optionLabel(key), value, '']);
  }
  renderInfoRows(rows);
}

function preRequest() {
  const type = $('p_type').value;
  return {
    image_b64: preInputB64,
    type: type,
    // One method means there is nothing to choose: let the kind's default apply.
    model: $('p_model_field').classList.contains('hidden') ? null : ($('p_model').value || null),
    [type]: preOptionValues(type),
  };
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
