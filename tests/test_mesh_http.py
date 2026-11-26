import base64
import io
import os

import torch
from fastapi import FastAPI
from fastapi.testclient import TestClient
from PIL import Image
from test_mesh_controller import DEFAULTS, FakeShapeModel

from thenoise.jobs import JobQueue
from thenoise.mesh.controller import MeshPipelineController
from thenoise.mesh.http import install_mesh_routes
from thenoise.mesh.service import mesh_planner
from thenoise.mesh.shapes import ShapeStore

MODEL = "hunyuan3d21"


def test_the_shape_server_declares_its_finish_chain(tmp_path):
    """The pane builds its Finish sliders from this, so the words and bounds live on the server."""
    queue = JobQueue(tmp_path)
    try:
        with _client(_controller(), queue) as client:
            described = client.get("/finish").json()["options"]
    finally:
        queue.close()
    assert [entry["name"] for entry in described] == ["floaters", "triangles", "smooth"]
    assert all(entry["type"] == "number" for entry in described)
    assert {entry["name"]: entry["default"] for entry in described} == {"floaters": 0.0,
                                                                       "triangles": 1.0,
                                                                       "smooth": 0.0}
    assert {entry["name"]: entry["max"] for entry in described} == {"floaters": 1.0,
                                                                   "triangles": 1.0,
                                                                   "smooth": 1.0}


def _cut_with_junk():
    from thenoise.mesh.extract import Mesh, extract_surface
    from thenoise.mesh.octree import sample_sdf

    shape = extract_surface(sample_sdf(lambda points: points.norm(dim=-1) - 0.4, 32,
                                       device="cpu", bounds=1.0, level=0.0, start=8))
    corners = torch.tensor([[1, 1, 1], [-1, -1, 1], [-1, 1, -1], [1, -1, -1]],
                           dtype=torch.float32) * 0.05
    corners += torch.tensor([3.0, 3.0, 3.0], dtype=torch.float32)
    piece = torch.tensor([[0, 1, 2], [0, 3, 1], [0, 2, 3], [1, 3, 2]])
    return Mesh(torch.cat((shape.vertices, corners)),
                torch.cat((shape.faces, piece + len(shape.vertices))))


_CUT = _cut_with_junk()


class _JunkFactory(FakeShapeModel):
    """The defaults and the latents of the fake model, with a cut that has junk beside it."""

    def decode_mesh(self, latents, params):
        return _CUT


def test_the_finish_knobs_a_pane_sends_reach_the_written_file(tmp_path):
    """One HTTP body to the bytes on disk: the wire field, the resolved spec, the chain."""
    queue = JobQueue(tmp_path)
    try:
        with _client(MeshPipelineController(_JunkFactory()), queue) as client:
            answer = client.post("/jobs/mesh", json={"image": _b64(), "output_format": "obj",
                                                     "floaters": 0.15, "triangles": 0.5})
            assert answer.status_code == 202
            ident = answer.json()["id"]
            until(lambda: client.get(f"/jobs/{ident}").json()["status"] == "completed")
            written = client.get(f"/jobs/{ident}/content").text.count("f ")
    finally:
        queue.close()
    assert written == max(4, round((len(_CUT.faces) - 4) * 0.5))


def test_the_wire_knows_the_finish_knobs_it_offers(tmp_path):
    """A knob the server advertises is a field the request accepts, or a slider returns 422."""
    queue = JobQueue(tmp_path)
    try:
        with _client(_controller(), queue) as client:
            for body in ({"floaters": 0.01}, {"triangles": 0.25}):
                answer = client.post("/jobs/mesh", json={"image": _b64(), **body})
                assert answer.status_code == 202
                ident = answer.json()["id"]
                until(lambda: client.get(f"/jobs/{ident}").json()["status"] == "completed")
    finally:
        queue.close()


def _b64(color="red") -> str:
    buffer = io.BytesIO()
    Image.new("RGB", (40, 30), color=color).save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("ascii")


class _Runtime:
    """What a mesh job asks of the runtime: defaults at resolve, a controller per run."""

    model_name = MODEL
    medium = "mesh"

    def __init__(self, pipeline):
        self.pipeline = pipeline

    def run(self, action):
        return action(self.pipeline)


def _client(pipeline, queue, shapes=None):
    app = FastAPI()
    from thenoise.jobs_http import install_job_routes

    install_job_routes(app, queue)
    store = shapes if shapes is not None else ShapeStore(queue.root)
    install_mesh_routes(app, queue, mesh_planner(_Runtime(pipeline), store), MODEL, store)
    return TestClient(app)


def _controller(body=b"glTF fake"):
    real = MeshPipelineController(FakeShapeModel())

    class Controller:
        # The real resolve, so what the controller refuses is what the route answers 400.
        resolve = real.resolve

        def render(self, spec, token, progress, output, shapes=None):
            progress("sampling", spec.steps, spec.steps)
            output.write_bytes(body)

    return Controller()


def _failing(failure):
    class Controller:
        def resolve(self, spec):
            raise failure

        def render(self, spec, token, progress, output, shapes=None):
            raise failure

    return Controller()


def until(predicate, timeout=2):
    import time
    stop = time.monotonic() + timeout
    while time.monotonic() < stop:
        if predicate():
            return
        time.sleep(0.005)
    raise AssertionError("mesh job did not settle")


def test_a_queued_shape_comes_back_as_a_file(tmp_path):
    queue = JobQueue(tmp_path)
    try:
        with _client(_controller(), queue) as client:
            answer = client.post("/jobs/mesh", json={"model": MODEL, "image": _b64()})
            assert answer.status_code == 202
            job = answer.json()
            assert job["kind"] == "mesh"
            ident = job["id"]
            until(lambda: client.get(f"/jobs/{ident}").json()["status"] == "completed")
            status = client.get(f"/jobs/{ident}").json()
            assert status["resolved"]["mode"] == "i2d"
            assert status["resolved"]["latents"] == DEFAULTS["num_latents"]
            assert status["resolved"]["octree_resolution"] == DEFAULTS["octree_resolution"]
            assert "image" not in status["resolved"]
            content = client.get(f"/jobs/{ident}/content")
            assert content.content == b"glTF fake"
            assert content.headers["content-type"].startswith("model/gltf-binary")
    finally:
        queue.close()


def test_the_object_container_is_offered_too(tmp_path):
    queue = JobQueue(tmp_path)
    try:
        with _client(_controller(b"v 0 0 0"), queue) as client:
            answer = client.post("/jobs/mesh",
                                 json={"model": MODEL, "image": _b64(),
                                       "output_format": "obj", "latents": 128,
                                       "octree_resolution": 32, "seed": 5})
            ident = answer.json()["id"]
            until(lambda: client.get(f"/jobs/{ident}").json()["status"] == "completed")
            status = client.get(f"/jobs/{ident}").json()
            assert status["resolved"]["latents"] == 128
            assert status["resolved"]["seed"] == 5
            assert status["resolved"]["output_format"] == "obj"
            assert client.get(f"/jobs/{ident}/content").content == b"v 0 0 0"
    finally:
        queue.close()


def test_the_shape_is_returned_directly_when_nobody_wants_a_job(tmp_path):
    queue = JobQueue(tmp_path)
    try:
        with _client(_controller(), queue) as client:
            answer = client.post("/mesh", json={"image": _b64(), "steps": 2})
            assert answer.status_code == 200
            assert answer.content == b"glTF fake"
            assert not list(tmp_path.glob("sync_*"))
    finally:
        queue.close()


def test_a_model_other_than_the_loaded_one_is_refused(tmp_path):
    queue = JobQueue(tmp_path)
    try:
        with _client(_controller(), queue) as client:
            answer = client.post("/mesh", json={"model": "other", "image": _b64()})
            assert answer.status_code == 400
            assert "not loaded" in answer.text
    finally:
        queue.close()


def test_a_picture_that_is_not_a_picture_is_refused_before_the_minutes_are_spent(tmp_path):
    queue = JobQueue(tmp_path)
    try:
        with _client(_controller(), queue) as client:
            answer = client.post("/mesh", json={"image": "not base64 at all!!"})
            assert answer.status_code == 400
            assert "base64" in answer.text or "valid image" in answer.text

            text = base64.b64encode(b"hello, not a png").decode("ascii")
            answer = client.post("/mesh", json={"image": text})
            assert answer.status_code == 400
            assert "valid image" in answer.text
    finally:
        queue.close()


def test_a_request_without_a_picture_cannot_ask_for_a_shape(tmp_path):
    queue = JobQueue(tmp_path)
    try:
        with _client(_controller(), queue) as client:
            answer = client.post("/mesh", json={})
            assert answer.status_code == 400
            assert "t2d" in answer.text
    finally:
        queue.close()


def test_a_saved_shape_can_stand_in_for_an_image(tmp_path):
    """A request bringing both an image and a shape is refused, not decided for."""
    queue = JobQueue(tmp_path)
    try:
        with _client(_controller(), queue) as client:
            answer = client.post("/jobs/mesh", json={"from_shape": "job_first"})
            assert answer.status_code == 202
            ident = answer.json()["id"]
            until(lambda: client.get(f"/jobs/{ident}").json()["status"] == "completed")
            resolved = client.get(f"/jobs/{ident}").json()["resolved"]
            assert resolved["mode"] == "shape"
            assert resolved["from_shape"] == "job_first"

            answer = client.post("/mesh", json={"image": _b64(), "from_shape": "job_first"})
            assert answer.status_code == 400
            assert "either image or from_shape" in answer.text
    finally:
        queue.close()


def test_the_shapes_a_request_can_name_are_listed_newest_first(tmp_path):
    """The list the refusals speak from: what the server holds now, and nothing after a restart."""
    queue = JobQueue(tmp_path)
    shapes = ShapeStore(tmp_path)
    try:
        with _client(_controller(), queue, shapes) as client:
            assert client.get("/shapes").json() == {"shapes": []}
            shapes.save("job_alpha", torch.zeros(1, 2, 2), MODEL)
            shapes.save("job_beta", torch.zeros(1, 2, 2), MODEL)
            os.utime(shapes.path("job_beta"), (1, 1))
            assert client.get("/shapes").json() == {"shapes": ["job_alpha", "job_beta"]}
    finally:
        queue.close()


def test_a_controller_refusal_is_a_bad_request_not_a_crash(tmp_path):
    queue = JobQueue(tmp_path)
    try:
        with _client(_failing(ValueError("this shape is impossible")), queue) as client:
            answer = client.post("/mesh", json={"image": _b64()})
            assert answer.status_code == 400
            assert "impossible" in answer.text
            answer = client.post("/jobs/mesh", json={"image": _b64()})
            assert answer.status_code == 400
    finally:
        queue.close()


def test_a_field_nobody_offers_is_refused_rather_than_ignored(tmp_path):
    queue = JobQueue(tmp_path)
    try:
        with _client(_controller(), queue) as client:
            answer = client.post("/mesh", json={"image": _b64(), "num_frames": 4})
            assert answer.status_code == 422
    finally:
        queue.close()


def test_several_pictures_are_queued_as_the_views_of_one_shape(tmp_path):
    queue = JobQueue(tmp_path)
    try:
        with _client(_controller(), queue) as client:
            answer = client.post("/jobs/mesh",
                                 json={"model": MODEL, "image": [_b64("red"), _b64("blue")]})
            assert answer.status_code == 202
            ident = answer.json()["id"]
            until(lambda: client.get(f"/jobs/{ident}").json()["status"] == "completed")
            assert client.get(f"/jobs/{ident}").json()["resolved"]["mode"] == "mv2d"
    finally:
        queue.close()


def test_views_past_the_ones_the_model_numbers_are_refused_before_the_job(tmp_path):
    queue = JobQueue(tmp_path)
    try:
        with _client(_controller(), queue) as client:
            answer = client.post("/jobs/mesh",
                                 json={"model": MODEL, "image": [_b64()] * 5})
            assert answer.status_code == 400
            assert "at most 4 views" in answer.text
    finally:
        queue.close()


def test_a_knob_no_shape_can_use_is_refused_before_the_job(tmp_path):
    """An unreachable knob is refused before the job, not after 80 GB of octree."""
    queue = JobQueue(tmp_path)
    try:
        with _client(_controller(), queue) as client:
            for knob, value in [("bounds", 700.0), ("mc_level", 3.0), ("latents", 100_000)]:
                answer = client.post("/jobs/mesh", json={"image": _b64(), knob: value})
                assert answer.status_code == 400, (knob, answer.text)
                assert knob in answer.text, (knob, answer.text)
    finally:
        queue.close()
