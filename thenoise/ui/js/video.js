/* video.js — the layout Generate shows for a video model: T2V / I2V through
   POST /jobs/video and the shared job read. Plain global script, loaded AFTER helpers.js. */

let vInputB64 = null;     // base64 (no prefix) of the I2V initial frame
let vInputDataUrl = null; // the same frame, for the thumbnail
let vOutUrl = null;       // object URL of the current result
let vOutMeta = null;      // request + server-resolved settings of the current result

const videoJob = { id: null }; // in flight, for the Cancel button to name
const VIDEO_PROGRESS = { root: 'v_progress', label: 'v_progress_label', bar: 'v_bar' };
const VIDEO_POLL = { first: 2500, ceiling: 8000 }; // a clip renders in minutes

/* ---------- I2V initial frame ---------- */

async function loadVideoFrame(file) {
  const image = await readImageFile(file);
  if (!image) return;
  vInputDataUrl = image.dataUrl;
  vInputB64 = image.b64;
  renderVideoRef(file.name);
  resetVideoResult();
}

function removeVideoFrame() {
  vInputDataUrl = null;
  vInputB64 = null;
  renderVideoRef('');
  resetVideoResult();
}

function renderVideoRef(name) {
  const box = $('video_refs');
  box.innerHTML = '';
  const hasFrame = !!vInputB64;
  $('video_dz_hint').style.display = hasFrame ? 'none' : '';
  const nameEl = $('video_dz_name');
  nameEl.textContent = name || '';
  nameEl.style.display = name ? 'block' : 'none';
  if (!hasFrame) return;
  addThumb(box, {
    src: vInputDataUrl, alt: 'initial frame', cls: 'first',
    title: 'initial frame (image-to-video)', onRemove: removeVideoFrame,
  });
}

function resetVideoResult() {
  // Do NOT revoke vOutUrl: it may still be referenced by an entry in history.
  vOutMeta = null;
  setStageActions($('vdownload'), $('vinfo_btn'), false);
  hideResult('v_video', 'vplaceholder');
}

bindDropzone('video_dropzone', 'video_file', files => loadVideoFrame(files[0]));
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
  if (vInputB64) body.image = vInputB64;
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
    poll: VIDEO_POLL,
    onArtifact: async (blob, status) => {
      if (vOutUrl) URL.revokeObjectURL(vOutUrl);
      vOutUrl = URL.createObjectURL(blob);
      // `resolved` has the grid-snapped size and the seed drawn; the frame is dropped.
      const { image, ...sent } = body;
      vOutMeta = { ...sent, model: status.model, ...status.resolved };
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
  if (m.mode) rows.push([FIELD_LABELS.mode, m.mode === 'i2v' ? 'image-to-video' : 'text-to-video', '']);
  if (m.prompt) rows.push([FIELD_LABELS.prompt, m.prompt, 'prompt']);
  if (m.negative_prompt) rows.push([FIELD_LABELS.negative_prompt, m.negative_prompt, 'prompt']);
  for (const key of ['width', 'height', 'frame_count', 'fps', 'steps', 'guidance_scale',
                     'flow_shift', 'seed', 'sampler', 'output_format']) {
    if (m[key] != null && m[key] !== '') rows.push([FIELD_LABELS[key], m[key], '']);
  }
  if (m.duration != null) rows.push([FIELD_LABELS.duration, Number(m.duration).toFixed(2) + ' s', '']);
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
