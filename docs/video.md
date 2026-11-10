# Video generation

Wan 2.2 TI2V-5B supports text-to-video and first-frame image-to-video via
asynchronous jobs. TheNoise runs the native 3D transformer and causal video VAE;
Diffusers supplies the pipeline orchestration and schedulers; the native
transformer and video VAE load and execute without Diffusers model weights.
The model loads from a local Wan 2.2 TI2V-5B checkpoint directory on first use.

## Setup

```sh
uv pip install -e '.[video]'
python -m thenoise serve --video-model-dir /path/to/Wan2.2-TI2V-5B-Diffusers \
    --video-offload --port 8000
```

`--video-offload` stages model components through CPU. Video and image model
checkpoints cannot be supplied in the same `serve` invocation.

## HTTP API

```sh
curl -X POST http://localhost:8000/v1/videos \
  -H 'Content-Type: application/json' \
  -d '{"prompt":"A fox walking through snow", "video_frames":33, "fps":24, "width":720, "height":480, "steps":8, "output_format":"mp4"}'
```

The response contains a job `id` and status `queued`. Poll
`GET /v1/videos/{id}` for progress and the final status. Retrieve the video with
`GET /v1/videos/{id}/content`, or cancel using
`POST /v1/videos/{id}/cancel`.

For image-to-video add `"image":"<base64-encoded PNG or JPEG>"`. Optional
parameters include `negative_prompt`, `cfg_scale`, `seed`, `duration`,
`sampling_method` (`unipc` or `euler`) and `flow_shift`. MP4/H.264 and WebM/VP9
are supported. Frames are rounded to Wan's `4n+1` temporal grid.

Jobs run one at a time, with up to four waiting in the queue. Clips are stored
under `~/.cache/thenoise/videos` (configurable with `--video-cache-dir`). Jobs
are not persisted across restarts. Cancellation is cooperative and cannot stop
a running GPU kernel immediately. Decoded frames are currently buffered before
container encoding; spatial VAE tiling is not implemented.
