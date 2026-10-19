/* controlmap.js — the Control Maps tab: image -> extracted map (plain global script).
   The wire keeps the technical names; the words shown to the user are the plain ones. */

let cmAvailable = {};    // control-map kind -> the methods implementing it
let cmOptions = {};      // control-map kind -> the option fields it takes
let cmDetectors = [];    // person detectors on disk, for the pose kind to crop with
let cmInputB64 = null;   // base64 (no data: prefix) sent to /controlmap
let cmOutUrl = null;
let cmOutMeta = null;

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

const optionFields = kind => cmOptions[kind] || [];

function optionLabel(name) {
  for (const fields of Object.values(cmOptions)) {
    const field = fields.find(entry => entry.name === name);
    if (field) return field.label;
  }
  return name;
}

function cmCheckbox(option) {
  const label = document.createElement('label');
  label.className = 'check';
  const input = document.createElement('input');
  input.type = 'checkbox';
  input.id = 'cm_opt_' + option.name;
  input.checked = option.default === true;
  label.append(input, document.createTextNode(' ' + option.label));
  return label;
}

function cmField(option) {
  const id = 'cm_opt_' + option.name;
  const field = document.createElement('div');
  field.className = 'field';
  const label = document.createElement('label');
  label.className = 'label';
  label.htmlFor = id;
  label.textContent = option.label;

  // A `pool` field offers what the server has on disk, named as it is there.
  const choices = option.pool ? cmDetectors : option.choices;
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

function buildCmOptions(kind) {
  const box = $('cm_options');
  box.innerHTML = '';
  let pending = null;   // a numeric field waiting for the one it shares a row with
  for (const option of optionFields(kind)) {
    if (!isNumberField(option)) {
      if (pending) { box.append(pending); pending = null; }
      box.append(option.type === 'boolean' ? cmCheckbox(option) : cmField(option));
      continue;
    }
    if (!pending) { pending = cmField(option); continue; }
    const row = document.createElement('div');
    row.className = 'row';
    row.append(pending, cmField(option));
    box.append(row);
    pending = null;
  }
  // A field left over at the end takes the full width rather than half a row.
  if (pending) box.append(pending);
}

// The chosen kind's options as the wire wants them: only what the form actually holds.
function cmOptionValues(kind) {
  const values = {};
  for (const option of optionFields(kind)) {
    const input = $('cm_opt_' + option.name);
    if (!input || input.closest('.hidden')) continue;
    if (option.type === 'boolean') values[option.name] = input.checked;
    else if (isNumberField(option)) {
      // Clearing a number field asks for the default; it is not the same as a zero.
      if (input.value !== '') values[option.name] = input.valueAsNumber;
    } else values[option.name] = input.value;
  }
  return values;
}

function fillCmModels() {
  fillSelect($('cm_model'), cmAvailable[$('cm_type').value] || []);
}

function showCmOptions() {
  const type = $('cm_type').value;
  const methods = cmAvailable[type] || [];
  // A kind with one implementation needs no method picker.
  $('cm_model_field').classList.toggle('hidden', methods.length <= 1);
  buildCmOptions(type);
}

async function loadControlmaps() {
  const data = await fetchJSON('/controlmaps');
  cmAvailable = (data && data.controlmaps) || {};
  cmOptions = (data && data.options) || {};
  cmDetectors = (data && data.detectors) || [];
  fillSelect($('cm_type'), Object.keys(cmAvailable), kindLabel);
  fillCmModels();
  showCmOptions();
  $('cm_run').disabled = cmInputB64 === null || !Object.keys(cmAvailable).length;
}

function loadCmFile(file) {
  if (!file || !file.type.startsWith('image/')) return;
  const reader = new FileReader();
  reader.onload = () => {
    cmInputB64 = reader.result.split(',')[1];
    if (cmOutUrl) { URL.revokeObjectURL(cmOutUrl); cmOutUrl = null; }
    cmOutMeta = null;
    setStageActions($('cm_download'), $('cm_info_btn'), false);
    showInputPreview(reader.result, file, {
      singleImg: 'cm_single_img', single: 'cm_single', result: 'cm_result',
      placeholder: 'cm_placeholder', dzName: 'cm_dz_name', dzHint: 'cm_dz_hint',
    });
    $('cm_run').disabled = !Object.keys(cmAvailable).length;
  };
  reader.readAsDataURL(file);
}

function renderCmInfo(meta) {
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

function cmRequest() {
  const type = $('cm_type').value;
  return {
    image_b64: cmInputB64,
    type: type,
    // One method means there is nothing to choose: let the kind's default apply.
    model: $('cm_model_field').classList.contains('hidden') ? null : ($('cm_model').value || null),
    [type]: cmOptionValues(type),
  };
}

bindDropzone('cm_dropzone', 'cm_file', files => loadCmFile(files[0]));
$('cm_type').addEventListener('change', () => { fillCmModels(); showCmOptions(); });

$('cm_run').addEventListener('click', () => {
  runWithBusy({
    btn: $('cm_run'),
    overlay: $('cm_overlay'),
    timerEl: 'cm_timer',
    timerTextEl: 'cm_timer_text',
    request: () => postJSON('/controlmap', cmRequest()),
    onSuccess: async (blob) => {
      if (cmOutUrl) URL.revokeObjectURL(cmOutUrl);
      cmOutUrl = URL.createObjectURL(blob);
      showOutputPreview('cm_single', 'cm_result', 'cm_placeholder', 'cm_img', cmOutUrl);
      setStageActions($('cm_download'), $('cm_info_btn'), true);
      cmOutMeta = await decodeMeta(blob, 'controlmap_data');
    },
  });
});

$('cm_download').addEventListener('click', () =>
  download(cmOutUrl, ($('cm_type').value || 'map') + '.png'));
$('cm_info_btn').addEventListener('click', () =>
  openInfo('Extraction info', () => renderCmInfo(cmOutMeta)));

loadControlmaps();
