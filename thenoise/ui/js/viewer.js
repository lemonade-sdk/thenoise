
const MESH_VIEWED_FORMATS = ['glb', 'obj', 'stl'];
const MESH_FOV = 0.8;
const MESH_REST = {yaw: 0.7, pitch: 0.28};   // a three-quarter view of whatever arrives
const MESH_TILT = 1.45;                      // short of straight down, where up stops existing

const MESH_VIEW = {
  gl: null, program: null, uniform: {}, probe: undefined,
  surface: null, wire: null, vbo: null, holdings: [],
  taken: null,                               // kept for the wireframe, which is built on demand
  triangles: 0, vertices: 0, indexed: false,
  center: [0, 0, 0], radius: 1, distance: 3,
  yaw: MESH_REST.yaw, pitch: MESH_REST.pitch,
  wireframe: false, frame: 0,
};

const MESH_VERTEX = `#version 300 es
layout(location = 0) in vec3 a_position;
uniform mat4 u_view;
uniform mat4 u_proj;
out vec3 v_view;
void main() {
  vec4 placed = u_view * vec4(a_position, 1.0);
  v_view = placed.xyz;
  gl_Position = u_proj * placed;
}`;

const MESH_FRAGMENT = `#version 300 es
precision highp float;
in vec3 v_view;
uniform vec3 u_tint;
out vec4 colour;
void main() {
  vec3 normal = normalize(cross(dFdx(v_view), dFdy(v_view)));
  normal *= gl_FrontFacing ? 1.0 : -1.0;
  float key = max(dot(normal, normalize(vec3(0.35, 0.65, 0.8))), 0.0);
  float fill = max(dot(normal, normalize(vec3(-0.6, -0.25, 0.4))), 0.0);
  float facing = max(dot(normal, normalize(-v_view)), 0.0);
  float rim = pow(1.0 - facing, 3.0);
  colour = vec4(u_tint * (0.16 + 0.7 * key + 0.22 * fill + 0.22 * rim), 1.0);
}`;

/* ---------- the three files the writer makes ---------- */

const GLB_MAGIC = 0x46546C67;
const GLB_JSON_CHUNK = 0x4E4F534A;
const GLB_BIN_CHUNK = 0x004E4942;
const GLB_COMPONENTS = {
  5120: Int8Array, 5121: Uint8Array, 5122: Int16Array,
  5123: Uint16Array, 5125: Uint32Array, 5126: Float32Array,
};
const GLB_ELEMENTS = {SCALAR: 1, VEC2: 2, VEC3: 3, VEC4: 4};
const GLB_INDICES = [5121, 5123, 5125];

function glbView(json, buffer, binStart, number) {
  const accessor = json.accessors[number];
  const kind = accessor && GLB_COMPONENTS[accessor.componentType];
  const fields = accessor && GLB_ELEMENTS[accessor.type];
  if (!kind || !fields) throw new Error('a glb attribute this viewer cannot read');
  const view = json.bufferViews[accessor.bufferView || 0] || {};
  if (view.byteStride) throw new Error('a glb that interleaves its attributes');
  return new kind(buffer, binStart + (view.byteOffset || 0) + (accessor.byteOffset || 0),
                  accessor.count * fields);
}

function parseGlb(buffer) {
  const head = new DataView(buffer);
  if (head.byteLength < 12 || head.getUint32(0, true) !== GLB_MAGIC) {
    throw new Error('not a glb file');
  }
  let json = null, binStart = 0, at = 12;
  while (at + 8 <= head.byteLength) {
    const length = head.getUint32(at, true);
    const chunk = head.getUint32(at + 4, true);
    const start = at + 8;
    if (chunk === GLB_JSON_CHUNK) {
      json = JSON.parse(new TextDecoder().decode(new Uint8Array(buffer, start, length)));
    } else if (chunk === GLB_BIN_CHUNK) {
      binStart = start;
    }
    at = start + length;
  }
  const mesh = json && (json.meshes || [])[0];
  if (!mesh) throw new Error('a glb with no mesh in it');
  const wanted = (mesh.primitives || []).filter(one => one.attributes && one.attributes.POSITION != null);
  if (!wanted.length) throw new Error('a glb primitive with no positions');

  const corners = wanted.reduce((sum, one) => sum + json.accessors[one.attributes.POSITION].count, 0);
  const faces = wanted.reduce((sum, one) => sum + (one.indices == null ? 0
                                                      : json.accessors[one.indices].count), 0);
  const positions = new Float32Array(corners * 3);
  const indices = faces ? new Uint32Array(faces) : null;
  let atCorner = 0, atFace = 0;
  for (const primitive of wanted) {
    const place = json.accessors[primitive.attributes.POSITION];
    if (place.type !== 'VEC3' || place.componentType !== 5126) {
      throw new Error('glb positions that are not three floats');
    }
    const raw = glbView(json, buffer, binStart, primitive.attributes.POSITION);
    positions.set(raw, atCorner);
    if (primitive.indices != null && indices) {
      const numbered = json.accessors[primitive.indices];
      if (numbered.type !== 'SCALAR' || !GLB_INDICES.includes(numbered.componentType)) {
        throw new Error('glb indices that are not unsigned integers');
      }
      const taken = glbView(json, buffer, binStart, primitive.indices);
      const base = atCorner / 3;
      const here = place.count;
      for (let i = 0; i < taken.length; i += 1) {
        if (taken[i] >= here) throw new Error('a glb index past the end of its vertices');
        indices[atFace + i] = base + taken[i];
      }
      atFace += taken.length;
    }
    atCorner += raw.length;
  }
  return meshShape(positions, indices);
}

const OBJ_VERTEX = /^v\s+(\S+)\s+(\S+)\s+(\S+)/;

function parseObj(text) {
  const lines = text.split('\n');
  const positions = new Float32Array(lines.length * 3);
  let taken = new Uint32Array(lines.length * 3);
  let at = 0, used = 0;
  for (const line of lines) {
    const kind = line[0];
    if (kind !== 'v' && kind !== 'f') continue;
    if (kind === 'v') {
      const corner = OBJ_VERTEX.exec(line);
      if (!corner) continue;
      positions[at] = Number(corner[1]);
      positions[at + 1] = Number(corner[2]);
      positions[at + 2] = Number(corner[3]);
      at += 3;
      continue;
    }
    let corners = 0, first = 0, previous = 0;
    for (const part of line.slice(2).split(' ')) {
      if (!part) continue;
      const written = Number(part.split('/')[0]);
      const corner = written < 0 ? at / 3 + written : written - 1;
      if (corner < 0 || corner >= at / 3) {
        throw new Error('an obj face points at a vertex that is not there');
      }
      if (corners === 0) first = corner;
      else if (corners >= 2) {
        if (used + 3 > taken.length) {
          const bigger = new Uint32Array(taken.length * 2);
          bigger.set(taken);
          taken = bigger;
        }
        taken[used] = first;
        taken[used + 1] = previous;
        taken[used + 2] = corner;
        used += 3;
      }
      previous = corner;
      corners += 1;
    }
  }
  return meshShape(positions.subarray(0, at), used ? taken.subarray(0, used) : null);
}

function parseStl(buffer) {
  const head = new TextDecoder().decode(new Uint8Array(buffer, 0, Math.min(512, buffer.byteLength)));
  if (/^\s*solid/.test(head) && head.includes('facet')) {
    const lines = new TextDecoder().decode(buffer).split('\n');
    const positions = new Float32Array(lines.length * 3);   // one vertex is a line's limit
    let at = 0;
    for (const line of lines) {
      const corner = line.indexOf('vertex ');
      if (corner < 0) continue;
      const parts = line.slice(corner + 7).trim().split(/\s+/);
      positions[at] = Number(parts[0]);
      positions[at + 1] = Number(parts[1]);
      positions[at + 2] = Number(parts[2]);
      at += 3;
    }
    return meshShape(positions.subarray(0, at), null);
  }
  const view = new DataView(buffer);
  const triangles = view.byteLength >= 84 ? view.getUint32(80, true) : 0;
  if (view.byteLength < 84 + triangles * 50) throw new Error('not an stl file');
  const positions = new Float32Array(triangles * 9);
  for (let t = 0; t < triangles; t += 1) {
    let at = 84 + t * 50 + 12;
    for (let c = 0; c < 9; c += 1, at += 4) positions[t * 9 + c] = view.getFloat32(at, true);
  }
  return meshShape(positions, null);
}

function meshShape(positions, indices) {
  if (!positions.length) throw new Error('there are no triangles to look at');
  const floor = [Infinity, Infinity, Infinity];
  const ceiling = [-Infinity, -Infinity, -Infinity];
  for (let i = 0; i + 2 < positions.length; i += 3) {
    for (let axis = 0; axis < 3; axis += 1) {
      const value = positions[i + axis];
      if (!Number.isFinite(value)) throw new Error('a vertex that is not a number');
      if (value < floor[axis]) floor[axis] = value;
      if (value > ceiling[axis]) ceiling[axis] = value;
    }
  }
  const center = floor.map((value, axis) => (value + ceiling[axis]) / 2);
  const radius = Math.max(ceiling[0] - floor[0], ceiling[1] - floor[1], ceiling[2] - floor[2]) / 2;
  return {positions, indices, vertices: positions.length / 3,
          triangles: (indices ? indices.length : positions.length / 3) / 3,
          center, radius: Number.isFinite(radius) && radius > 0 ? radius : 1};
}

function parseMesh(buffer, format) {
  if (format === 'glb') return parseGlb(buffer);
  if (format === 'obj') return parseObj(new TextDecoder().decode(buffer));
  if (format === 'stl') return parseStl(buffer);
  throw new Error('nothing here views a .' + format + ' file');
}

/* ---------- gl ---------- */

function meshViewerReady() {
  if (MESH_VIEW.probe === undefined) MESH_VIEW.probe = !!window.WebGL2RenderingContext;
  return MESH_VIEW.probe;
}

function meshCompiledShader(gl, type, source) {
  const shader = gl.createShader(type);
  gl.shaderSource(shader, source);
  gl.compileShader(shader);
  if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS)) {
    throw new Error('the viewer shader was refused: ' + gl.getShaderInfoLog(shader));
  }
  return shader;
}

function meshGL() {
  if (MESH_VIEW.gl) return MESH_VIEW.gl;
  const gl = $('mesh_view').getContext('webgl2', {alpha: true, antialias: true});
  if (!gl) throw new Error('this browser has no webgl2');
  const program = gl.createProgram();
  gl.attachShader(program, meshCompiledShader(gl, gl.VERTEX_SHADER, MESH_VERTEX));
  gl.attachShader(program, meshCompiledShader(gl, gl.FRAGMENT_SHADER, MESH_FRAGMENT));
  gl.linkProgram(program);
  if (!gl.getProgramParameter(program, gl.LINK_STATUS)) {
    throw new Error('the viewer shader was refused: ' + gl.getProgramInfoLog(program));
  }
  gl.enable(gl.DEPTH_TEST);
  Object.assign(MESH_VIEW, {gl, program, uniform: {
    view: gl.getUniformLocation(program, 'u_view'),
    proj: gl.getUniformLocation(program, 'u_proj'),
    tint: gl.getUniformLocation(program, 'u_tint'),
  }});
  return gl;
}

function meshBuffers(shape) {
  clearMesh();
  const gl = MESH_VIEW.gl;
  const vbo = gl.createBuffer();
  gl.bindBuffer(gl.ARRAY_BUFFER, vbo);
  gl.bufferData(gl.ARRAY_BUFFER, shape.positions, gl.STATIC_DRAW);
  MESH_VIEW.holdings.push(vbo);
  MESH_VIEW.vbo = vbo;
  MESH_VIEW.taken = shape.indices;
  MESH_VIEW.indexed = shape.indices !== null;

  const surface = gl.createVertexArray();
  gl.bindVertexArray(surface);
  gl.enableVertexAttribArray(0);
  gl.vertexAttribPointer(0, 3, gl.FLOAT, false, 0, 0);
  if (MESH_VIEW.indexed) {
    gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, meshIndexBuffer(gl, shape.indices));
  }
  gl.bindVertexArray(null);
  MESH_VIEW.surface = surface;
  MESH_VIEW.triangles = shape.triangles;
  MESH_VIEW.vertices = shape.vertices;
}

function meshIndexBuffer(gl, indices) {
  const buffer = gl.createBuffer();
  gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, buffer);
  gl.bufferData(gl.ELEMENT_ARRAY_BUFFER, indices, gl.STATIC_DRAW);
  MESH_VIEW.holdings.push(buffer);
  return buffer;
}

function meshWireFrame() {
  const gl = MESH_VIEW.gl;
  const taken = MESH_VIEW.taken;
  const lines = new Uint32Array(MESH_VIEW.triangles * 6);
  for (let t = 0, at = 0; t < MESH_VIEW.triangles; t += 1) {
    const corner = taken ? [taken[t * 3], taken[t * 3 + 1], taken[t * 3 + 2]]
                         : [t * 3, t * 3 + 1, t * 3 + 2];
    lines[at++] = corner[0]; lines[at++] = corner[1];
    lines[at++] = corner[1]; lines[at++] = corner[2];
    lines[at++] = corner[2]; lines[at++] = corner[0];
  }
  const vao = gl.createVertexArray();
  gl.bindVertexArray(vao);
  gl.enableVertexAttribArray(0);
  gl.vertexAttribPointer(0, 3, gl.FLOAT, false, 0, 0);
  gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, meshIndexBuffer(gl, lines));
  gl.bindVertexArray(null);
  MESH_VIEW.wire = vao;
  return vao;
}

function meshPerspective(aspect) {
  const near = Math.max(MESH_VIEW.distance - MESH_VIEW.radius * 1.5,
                        MESH_VIEW.distance * 0.01);
  const far = MESH_VIEW.distance + MESH_VIEW.radius * 4;
  const depth = 1 / Math.tan(MESH_FOV / 2);
  const out = new Float32Array(16);
  out[0] = depth / aspect;
  out[5] = depth;
  out[10] = (far + near) / (near - far);
  out[11] = -1;
  out[14] = 2 * far * near / (near - far);
  return out;
}

function meshLook(eye) {
  const [cx, cy, cz] = MESH_VIEW.center;
  let forward = [eye[0] - cx, eye[1] - cy, eye[2] - cz];
  const length = Math.hypot(forward[0], forward[1], forward[2]) || 1;
  forward = forward.map(n => n / length);
  let side = [forward[2], 0, -forward[0]];
  const flat = Math.hypot(side[0], side[2]) || 1;
  side = side.map(n => n / flat);
  const up = [forward[1] * side[2] - forward[2] * side[1],
              forward[2] * side[0] - forward[0] * side[2],
              forward[0] * side[1] - forward[1] * side[0]];
  return new Float32Array([
    side[0], up[0], forward[0], 0,
    side[1], up[1], forward[1], 0,
    side[2], up[2], forward[2], 0,
    -(side[0] * eye[0] + side[1] * eye[1] + side[2] * eye[2]),
    -(up[0] * eye[0] + up[1] * eye[1] + up[2] * eye[2]),
    -(forward[0] * eye[0] + forward[1] * eye[1] + forward[2] * eye[2]),
    1,
  ]);
}

function meshResize() {
  const canvas = $('mesh_view');
  const scale = Math.min(window.devicePixelRatio || 1, 2);
  const width = Math.max(1, Math.round(canvas.clientWidth * scale));
  const height = Math.max(1, Math.round(canvas.clientHeight * scale));
  if (canvas.width !== width || canvas.height !== height) {
    canvas.width = width;
    canvas.height = height;
  }
  return canvas;
}

function drawMesh() {
  MESH_VIEW.frame = 0;
  const gl = MESH_VIEW.gl;
  if (!gl || !MESH_VIEW.surface) return;
  const canvas = meshResize();
  gl.viewport(0, 0, canvas.width, canvas.height);
  gl.clearColor(0, 0, 0, 0);
  gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);
  const reach = MESH_VIEW.distance, pitch = MESH_VIEW.pitch, yaw = MESH_VIEW.yaw;
  const eye = [MESH_VIEW.center[0] + reach * Math.cos(pitch) * Math.sin(yaw),
               MESH_VIEW.center[1] + reach * Math.sin(pitch),
               MESH_VIEW.center[2] + reach * Math.cos(pitch) * Math.cos(yaw)];
  gl.useProgram(MESH_VIEW.program);
  gl.uniformMatrix4fv(MESH_VIEW.uniform.view, false, meshLook(eye));
  gl.uniformMatrix4fv(MESH_VIEW.uniform.proj, false, meshPerspective(canvas.width / canvas.height));
  gl.enable(gl.POLYGON_OFFSET_FILL);
  gl.polygonOffset(1, 1);
  gl.uniform3f(MESH_VIEW.uniform.tint, 0.72, 0.78, 0.88);
  gl.bindVertexArray(MESH_VIEW.surface);
  if (MESH_VIEW.indexed) {
    gl.drawElements(gl.TRIANGLES, MESH_VIEW.triangles * 3, gl.UNSIGNED_INT, 0);
  } else {
    gl.drawArrays(gl.TRIANGLES, 0, MESH_VIEW.vertices);
  }
  if (MESH_VIEW.wireframe) {
    gl.uniform3f(MESH_VIEW.uniform.tint, 0.89, 0.7, 0.26);
    gl.bindVertexArray(MESH_VIEW.wire || meshWireFrame());
    gl.drawElements(gl.LINES, MESH_VIEW.triangles * 6, gl.UNSIGNED_INT, 0);
  }
  gl.disable(gl.POLYGON_OFFSET_FILL);
  gl.bindVertexArray(null);
}

function requestMeshFrame() {
  if (!MESH_VIEW.gl || MESH_VIEW.frame) return;
  MESH_VIEW.frame = requestAnimationFrame(drawMesh);
}

function fitMesh() {
  MESH_VIEW.distance = MESH_VIEW.radius / Math.sin(MESH_FOV / 2) * 1.25;
}

function meshTilt(yaw, pitch) {
  MESH_VIEW.yaw = yaw;
  MESH_VIEW.pitch = Math.min(MESH_TILT, Math.max(-MESH_TILT, pitch));
}

function meshZoom(by) {
  MESH_VIEW.distance = Math.min(MESH_VIEW.radius * 40,
                                Math.max(MESH_VIEW.radius * 0.2, MESH_VIEW.distance * by));
}

/* ---------- stage ---------- */

async function showMeshBlob(blob, format) {
  const gl = meshGL();
  const parsed = parseMesh(await blob.arrayBuffer(), String(format || 'glb').toLowerCase());
  Object.assign(MESH_VIEW, {center: parsed.center, radius: parsed.radius}, MESH_REST);
  fitMesh();
  meshBuffers(parsed);
  requestMeshFrame();
  return parsed;
}

function clearMesh() {
  const gl = MESH_VIEW.gl;
  if (!gl) return;
  for (const buffer of MESH_VIEW.holdings) gl.deleteBuffer(buffer);
  for (const vao of [MESH_VIEW.surface, MESH_VIEW.wire]) {
    if (vao) gl.deleteVertexArray(vao);
  }
  Object.assign(MESH_VIEW, {holdings: [], surface: null, wire: null, vbo: null, taken: null,
                            triangles: 0});
}

initToggle({
  key: 'thenoise-mesh-wire', btnId: 'mesh_wire',
  initial: () => localStorage.getItem('thenoise-mesh-wire') === 'true',
  current: () => MESH_VIEW.wireframe,
  apply: (on) => {
    MESH_VIEW.wireframe = on;
    const btn = $('mesh_wire');
    if (btn) btn.setAttribute('aria-pressed', on ? 'true' : 'false');
    requestMeshFrame();
  },
  next: on => !on,
});

$('mesh_view').addEventListener('pointerdown', (event) => {
  const canvas = event.currentTarget;
  canvas.setPointerCapture(event.pointerId);
  const last = {x: event.clientX, y: event.clientY};
  const move = (dragged) => {
    meshTilt(MESH_VIEW.yaw - (dragged.clientX - last.x) * 0.01,
             MESH_VIEW.pitch + (dragged.clientY - last.y) * 0.01);
    last.x = dragged.clientX;
    last.y = dragged.clientY;
    requestMeshFrame();
  };
  const stop = () => {
    canvas.removeEventListener('pointermove', move);
    canvas.removeEventListener('pointerup', stop);
    canvas.removeEventListener('pointercancel', stop);
  };
  canvas.addEventListener('pointermove', move);
  canvas.addEventListener('pointerup', stop);
  canvas.addEventListener('pointercancel', stop);
});

$('mesh_view').addEventListener('wheel', (event) => {
  event.preventDefault();
  const notch = event.deltaMode === 1 ? 16 : event.deltaMode === 2 ? 100 : 1;
  meshZoom(Math.exp(event.deltaY * notch * 0.001));
  requestMeshFrame();
}, {passive: false});

$('mesh_view').addEventListener('keydown', (event) => {
  const turn = {ArrowLeft: [-0.12, 0], ArrowRight: [0.12, 0],
                ArrowUp: [0, 0.12], ArrowDown: [0, -0.12]}[event.key];
  if (turn) meshTilt(MESH_VIEW.yaw + turn[0], MESH_VIEW.pitch + turn[1]);
  else if (event.key === '+' || event.key === '=') meshZoom(0.8);
  else if (event.key === '-' || event.key === '_') meshZoom(1.25);
  else if (event.key === '0') fitMesh();
  else return;
  event.preventDefault();
  requestMeshFrame();
});

$('mesh_view').addEventListener('dblclick', () => {
  Object.assign(MESH_VIEW, MESH_REST);
  fitMesh();
  requestMeshFrame();
});

new ResizeObserver(requestMeshFrame).observe($('mesh_view'));
