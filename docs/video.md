# Video generation

TheNoise supports native Wan 2.2 TI2V-5B text-to-video and image-to-video generation.
Use the [Comfy-Org/Wan-2.2](https://huggingface.co/Comfy-Org/Wan-2.2)
checkpoints under one directory:

```
models/wan22/split_files/
  diffusion_models/wan2.2_ti2v_5B_fp16.safetensors
  text_encoders/umt5_xxl_fp16.safetensors
  vae/wan2.2_vae.safetensors
```

Install TheNoise with the downloader extra, fetch the Wan model, and start the server:

```sh
uv pip install -e ".[scripts]"
python scripts/download.py --model wan22
python -m thenoise serve \
  --dit ./models/wan22/split_files/diffusion_models/wan2.2_ti2v_5B_fp16.safetensors \
  --text-encoder ./models/wan22/split_files/text_encoders/umt5_xxl_fp16.safetensors \
  --vae ./models/wan22/split_files/vae/wan2.2_vae.safetensors
```

```sh
curl -X POST http://localhost:8000/v1/videos \
  -H 'Content-Type: application/json' \
  -d '{"prompt":"A fox walking through snow", "frames":33, "fps":24, "width":720, "height":480, "steps":8, "output_format":"mp4"}'
```

Poll `GET /v1/videos/{id}` and download `GET /v1/videos/{id}/content`.
Cancel with `POST /v1/videos/{id}/cancel`.
For I2V include `"image":"<base64-encoded JPEG or PNG>"`.
Output is MP4/H.264 or WebM/VP9, and frames follow Wan's `4n+1` grid.
Video and image models use the same model-path flags and memory manager. CPU
offloading is the default; use `--offload-device cuda` to keep weights resident
when memory allows.
The Wan 2.2 flow shift is fixed by the model; it is not an HTTP option.
The FP8 text encoder is not yet supported; use the FP16 checkpoint.
