/* video.js — the layout Generate shows for a video model: T2V, I2V and FLF2V through
   POST /jobs/video and the shared job read. Plain global script, loaded AFTER helpers.js. */

let vOutUrl = null;       // object URL of the current result
let vOutMeta = null;      // request + server-resolved settings of the current result

const videoJob = { id: null }; // in flight, for the Cancel button to name
const VIDEO_PROGRESS = { root: 'v_progress', label: 'v_progress_label', bar: 'v_bar' };
const VIDEO_MODES = {t2v: 'text-to-video', i2v: 'image-to-video',
                     flf2v: 'first and last frame', ref2v: 'reference-to-video'};

/* ---------- the frames a clip is pinned by ---------- */

// One entry per request field: what is loaded is what gets sent.
const vFrames = {
  image: {b64: null, dataUrl: null, name: null, box: 'video_refs',
          hint: 'video_dz_hint', label: 'video_dz_name', alt: 'initial frame',
          title: 'initial frame (image-to-video)'},
  end_image: {b64: null, dataUrl: null, name: null, box: 'video_end_refs',
              hint: 'video_end_dz_hint', label: 'video_end_dz_name', alt: 'last frame',
              title: 'last frame (first and last frame)'},
};

function renderVideoFrame(field) {
  const frame = vFrames[field];
  const box = $(frame.box);
  box.innerHTML = '';
  $(frame.hint).style.display = frame.b64 ? 'none' : '';
  const nameEl = $(frame.label);
  nameEl.textContent = frame.name || '';
  nameEl.style.display = frame.name ? 'block' : 'none';
  if (!frame.b64) return;
  addThumb(box, {src: frame.dataUrl, alt: frame.alt, cls: 'first',
                 title: frame.title, onRemove: () => removeVideoFrame(field)});
}

async function loadVideoFrame(field, file) {
  const image = await readImageFile(file);
  if (!image) return;
  Object.assign(vFrames[field], {dataUrl: image.dataUrl, b64: image.b64, name: file.name});
  renderVideoFrame(field);
  resetVideoResult();
}

function removeVideoFrame(field) {
  // A last frame with nothing to start from is not a request, so this takes it along.
  for (const key of field === 'image' ? ['image', 'end_image'] : [field]) {
    Object.assign(vFrames[key], {dataUrl: null, b64: null, name: null});
    renderVideoFrame(key);
  }
  resetVideoResult();
}

function resetVideoResult() {
  // Do NOT revoke vOutUrl: it may still be referenced by an entry in history.
  vOutMeta = null;
  setStageActions($('vdownload'), $('vinfo_btn'), false);
  hideResult('v_video', 'vplaceholder');
}

/* ---------- the pictures a clip is asked to look like ---------- */

// In the order they were picked, which is the order a prompt numbers them in.
let vRefs = [];

const REF_WORD = { picture: 'Picture', video: 'Video', audio: 'Audio' };

function renderVideoRefs() {
  const box = $('video_ref_refs');
  box.innerHTML = '';
  $('video_ref_dz_hint').style.display = vRefs.length ? 'none' : '';
  // Ordinals count each kind on its own, which is how a prompt points at one: a clip and
  // its soundtrack are numbered apart from the pictures beside them.
  const said = {};
  vRefs.forEach((ref, i) => {
    const word = REF_WORD[ref.kind] || 'Reference';
    said[word] = (said[word] || 0) + 1;
    addThumb(box, {
      src: ref.dataUrl, label: fileTag(ref.name), alt: word.toLowerCase() + ' ' + said[word],
      title: ref.name + ' — <' + word + ' ' + said[word] + '>',
      onRemove: () => { vRefs.splice(i, 1); renderVideoRefs(); resetVideoResult(); },
    });
  });
}

function fileTag(name) {
  const ext = name.includes('.') ? name.split('.').pop() : '';
  return ext ? ext.slice(0, 4).toUpperCase() : 'FILE';
}

async function addVideoRefs(files) {
  const picked = [...files];
  $('video_ref_file').value = ''; // so the same files can be picked again
  for (const file of picked) {
    const media = await readMediaFile(file);
    if (media) vRefs.push({dataUrl: media.dataUrl, b64: media.b64,
                           kind: media.kind, name: file.name});
  }
  renderVideoRefs();
  resetVideoResult();
}

bindDropzone('video_dropzone', 'video_file', files => loadVideoFrame('image', files[0]));
bindDropzone('video_end_dropzone', 'video_end_file',
             files => loadVideoFrame('end_image', files[0]));
bindDropzone('video_ref_dropzone', 'video_ref_file', addVideoRefs);
bindSwap('video_swap', 'video_width', 'video_height');

/* ---------- request ---------- */

initLora('video_');

function collectVideoSettings() {
  const body = {
    prompt: $('video_prompt').value,
    negative_prompt: $('video_negative_prompt').value,
  };
  addNumbers(body, 'video_', ['width', 'height', 'frames', 'fps', 'steps', 'seed'],
                              ['duration', 'guidance_scale', 'flow_shift']);
  const sampler = $('video_sampler').value;
  if (sampler) body.sampler = sampler;
  const output_format = $('video_output_format').value;
  if (output_format) body.output_format = output_format;
  const lora_specs = parseLora($('video_lora_specs').value);
  if (lora_specs) body.lora_specs = lora_specs;
  if (vFrames.image.b64) body.image = vFrames.image.b64;
  if (vFrames.end_image.b64) body.end_image = vFrames.end_image.b64;
  if (vRefs.length) body.references = vRefs.map(r => r.b64);
  const attention = $('video_attention').value;
  if (attention) body.attention = attention;
  return body;
}

function validateVideoInputs() {
  const fail = (message) => {
    setTimer('vtimer', 'vtimer_text', 'error', 'error: ' + message);
    return false;
  };
  if (!$('video_prompt').value.trim()) return fail('enter a prompt');
  const check = (fields, ok, message) => {
    for (const f of fields) {
      const v = $('video_' + f).value;
      if (v !== '' && !ok(Number(v))) {
        return fail(typeof message === 'function' ? message(f) : message);
      }
    }
    return true;
  };
  const whole = (n) => Number.isInteger(n);
  const real = (n) => Number.isFinite(n);
  // An empty field is the model's default, not a value of zero.
  return check(['width', 'height', 'frames', 'fps', 'steps'], n => whole(n) && n >= 1,
               f => f + ' must be a positive whole number')
    && check(['seed'], n => whole(n) && n >= 0,
             'seed must be a nonnegative whole number (leave empty for random)')
    && check(['duration'], n => real(n) && n > 0,
             'duration must be positive (leave empty for the model default)')
    && check(['guidance_scale'], n => real(n) && n >= 0, 'guidance must be a nonnegative number')
    && check(['flow_shift'], n => real(n) && n > 0,
             'flow shift must be positive (leave empty for the model default)');
}

$('video_generate').addEventListener('click', () => {
  if (!validateVideoInputs()) return;
  const body = collectVideoSettings();
  const refNames = vRefs.map(r => r.name);
  resetVideoResult();
  runQueuedJob({
    job: videoJob,
    submit: '/jobs/video',
    body,
    btn: $('video_generate'),
    cancelBtn: $('vcancel'),
    overlay: $('voverlay'),
    timerEl: 'vtimer',
    timerTextEl: 'vtimer_text',
    progress: VIDEO_PROGRESS,
    onArtifact: async (blob, status) => {
      if (vOutUrl) URL.revokeObjectURL(vOutUrl);
      vOutUrl = URL.createObjectURL(blob);
      // `resolved` has the grid-snapped size and the seed drawn; the frames are dropped.
      const { image, end_image, references, ...sent } = body;
      vOutMeta = { ...sent, references: refNames, model: status.model,
                   ...status.resolved };
      showVideoResult(vOutUrl);
      vHist.add(vOutUrl, vOutMeta);
    },
  });
});

$('vcancel').addEventListener('click', () => cancelJob(videoJob, $('vcancel')));

/* ---------- result ---------- */

function showVideoResult(url) {
  showResult('v_video', 'vplaceholder', url);
  setStageActions($('vdownload'), $('vinfo_btn'), true);
  $('v_video').play().catch(() => {}); // browsers block autoplay with sound; this is a nudge
}

$('vdownload').addEventListener('click', () => {
  const fmt = vOutMeta && vOutMeta.output_format === 'mp4' ? 'mp4' : 'webm';
  download(vOutUrl, `thenoise_video_${seedName(vOutMeta, 'x')}.${fmt}`);
});

$('vinfo_btn').addEventListener('click', () => openInfo('Video info', renderVideoInfo));

function renderVideoInfo() {
  const m = vOutMeta || {};
  const rows = [];
  if (m.model) rows.push(['Model', m.model, '']);
  if (m.mode) rows.push([FIELD_LABELS.mode, VIDEO_MODES[m.mode] || m.mode, '']);
  if (m.prompt) rows.push([FIELD_LABELS.prompt, m.prompt, 'prompt']);
  if (m.negative_prompt) rows.push([FIELD_LABELS.negative_prompt, m.negative_prompt, 'prompt']);
  for (const key of ['width', 'height', 'frame_count', 'fps', 'steps', 'guidance_scale',
                     'flow_shift', 'seed', 'sampler', 'output_format']) {
    if (m[key] != null && m[key] !== '') rows.push([FIELD_LABELS[key], m[key], '']);
  }
  if (m.duration != null) rows.push([FIELD_LABELS.duration, Number(m.duration).toFixed(2) + ' s', '']);
  if (Array.isArray(m.references) && m.references.length) {
    rows.push([FIELD_LABELS.references, m.references.join(', '), '']);
  }
  if (m.attention) rows.push([FIELD_LABELS.attention, m.attention, '']);
  const lora = m.lora_specs;
  const hasLora = Array.isArray(lora) ? lora.length : (lora ? Object.keys(lora).length : 0);
  if (hasLora) rows.push(['LoRA', formatLora(lora), 'prompt']);
  renderInfoRows(rows);
}

const vHist = makeHistory({
  containerId: 'vhistory_items', rootId: 'vhistory', alt: 'video', video: true,
  onSelect: (item) => {
    // The blob lives on in the history entry; the stage just points at it.
    vOutUrl = item.url;
    vOutMeta = item.meta;
    showVideoResult(item.url);
  },
});
