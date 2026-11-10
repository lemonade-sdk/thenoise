"""Wan 2.2 TI2V-5B video generation."""
from __future__ import annotations

import gc
import io
import logging
import time
from contextlib import ExitStack
from pathlib import Path

from .container import encode_video
from .jobs import Cancellation
from .spec import ResolvedVideo


log = logging.getLogger("uvicorn.error")


class WanVideo:
    name = "Wan2.2-TI2V-5B"

    def __init__(self, model_dir: str, *, device: str = "cuda", offload: bool = False):
        self.model_dir = str(Path(model_dir).expanduser())
        self.device = device
        self.offload = offload
        self._pipe = None
        self._mode = None
        self._placed = False
        self._base_scheduler_config = None

    def _get_pipe(self, mode: str, cancel: Cancellation):
        # Lazy load: a queued cancellation must not even start allocating weights.
        cancel.check()
        import torch
        from diffusers import WanPipeline, WanImageToVideoPipeline

        if self._pipe is None:
            if not Path(self.model_dir, "model_index.json").is_file():
                raise ValueError("--video-model-dir must contain a local Diffusers model_index.json")
            import json
            from transformers import T5TokenizerFast, UMT5EncoderModel
            from diffusers import UniPCMultistepScheduler
            from .native_checkpoint import load_wan_native_transformer
            from .native_vae import load_wan_native_video_vae

            index = json.loads(Path(self.model_dir, "model_index.json").read_text())
            if (index.get("_class_name") != "WanPipeline" or
                    index.get("expand_timesteps") is not True or
                    index.get("transformer_2") != [None, None] or
                    index.get("scheduler") != ["diffusers", "UniPCMultistepScheduler"]):
                raise ValueError("native Wan loader requires TI2V-5B model_index.json")

            vae = load_wan_native_video_vae(self.model_dir, check_cancel=cancel.check)
            transformer = load_wan_native_transformer(
                self.model_dir, dtype=torch.bfloat16, check_cancel=cancel.check
            )
            cancel.check()
            tokenizer = T5TokenizerFast.from_pretrained(
                str(Path(self.model_dir, "tokenizer")), local_files_only=True
            )
            text_encoder = UMT5EncoderModel.from_pretrained(
                str(Path(self.model_dir, "text_encoder")), dtype=torch.bfloat16,
                local_files_only=True,
            )
            scheduler = UniPCMultistepScheduler.from_pretrained(
                self.model_dir, subfolder="scheduler", local_files_only=True
            )
            candidate = WanPipeline(
                tokenizer=tokenizer, text_encoder=text_encoder, vae=vae,
                scheduler=scheduler, transformer=transformer,
                transformer_2=None, boundary_ratio=None, expand_timesteps=True,
            )
            if getattr(candidate, "transformer_2", None) is not None:
                raise ValueError("only Wan 2.2 TI2V-5B single-DiT checkpoints are supported")
            if (getattr(getattr(candidate.transformer, "config", None), "in_channels", None) != 48 or
                    getattr(getattr(candidate.vae, "config", None), "z_dim", None) != 48 or
                    candidate.config.get("expand_timesteps") is not True):
                raise ValueError("checkpoint is not the 48-channel Wan 2.2 TI2V-5B family")
            # Preserve checkpoint solver settings across T2V/I2V mode switches.
            # Rebuilding UniPC from a previous Euler config would silently lose
            # `use_flow_sigmas` and `prediction_type=flow_prediction`.
            base_scheduler_config = dict(candidate.scheduler.config)
            if (not base_scheduler_config.get("use_flow_sigmas") or
                    base_scheduler_config.get("prediction_type") != "flow_prediction"):
                raise ValueError("Wan TI2V-5B requires the checkpoint's flow UniPC scheduler")
            log.info("Wan native TI2V-5B transformer and FP32 causal video VAE loaded")
            # Publish the instance only once validated: a rejected checkpoint
            # must not leave a half-initialized adapter for the next request.
            self._pipe = candidate
            self._base_scheduler_config = base_scheduler_config
            self._mode = "t2v"
            self._placed = False
        cancel.check()
        if self._mode != mode:
            # Switch pipeline wrappers without recasting BF16 DiT or FP32 VAE weights.
            if self.offload and self._placed:
                self._pipe.remove_all_hooks()
            old = self._pipe
            cls = WanImageToVideoPipeline if mode == "i2v" else WanPipeline
            self._pipe = cls(
                tokenizer=old.tokenizer,
                text_encoder=old.text_encoder,
                vae=old.vae,
                scheduler=old.scheduler,
                transformer=old.transformer,
                transformer_2=getattr(old, "transformer_2", None),
                boundary_ratio=old.config.get("boundary_ratio"),
                expand_timesteps=old.config.get("expand_timesteps", False),
            )
            self._mode = mode
            self._placed = False
        if not self._placed:
            first_weight = next(self._pipe.vae.parameters(), None)
            if first_weight is None or first_weight.dtype != torch.float32:
                raise RuntimeError("Wan 2.2 video VAE must remain in FP32")
            if self.offload:
                # CPU <-> GPU staging. Hooks are installed once per wrapper.
                self._pipe.enable_model_cpu_offload(device=self.device)
            else:
                self._pipe.to(self.device)
            self._placed = True
        cancel.check()
        return self._pipe

    def generate(self, spec: ResolvedVideo, cancel: Cancellation, progress, output: Path) -> None:
        import torch
        from PIL import Image, ImageOps
        from thenoise.inference import inference, inference_lock

        with inference_lock, inference():
            progress("loading", 0, 0)
            pipe = self._get_pipe(spec.mode, cancel)
            cancel.check()
            from diffusers import FlowMatchEulerDiscreteScheduler
            from diffusers.schedulers.scheduling_unipc_multistep import UniPCMultistepScheduler
            if spec.sampling_method == "unipc":
                pipe.scheduler = UniPCMultistepScheduler.from_config(
                    self._base_scheduler_config, flow_shift=spec.flow_shift
                )
            else:
                pipe.scheduler = FlowMatchEulerDiscreteScheduler(
                    num_train_timesteps=self._base_scheduler_config["num_train_timesteps"],
                    shift=spec.flow_shift,
                )
            params = dict(
                prompt=spec.prompt, negative_prompt=spec.negative_prompt,
                height=spec.height, width=spec.width,
                num_frames=spec.frame_count, num_inference_steps=spec.steps,
                guidance_scale=spec.guidance_scale,
                generator=torch.Generator(device="cpu").manual_seed(spec.seed),
                output_type="pil",
            )
            if spec.image is not None:
                with Image.open(io.BytesIO(spec.image)) as img:
                    if img.width * img.height > 25_000_000:
                        raise ValueError("I2V input image exceeds 25 megapixels")
                    # Fit the source frame to the model canvas.
                    params["image"] = ImageOps.fit(
                        img.convert("RGB"), (spec.width, spec.height),
                        method=Image.Resampling.LANCZOS,
                    )

            latent_chunks = (spec.frame_count - 1) // 4 + 1
            sampling_start = time.monotonic()

            def step_end(_pipe, step, _timestep, kwargs):
                progress("sampling", step + 1, spec.steps)
                cancel.check()
                if step + 1 == spec.steps:
                    log.info("Wan sampling finished in %.1fs; preparing FP32 VAE decode "
                             "(offload=%s, device=%s)",
                             time.monotonic() - sampling_start, self.offload, self.device)
                    # CPU offload may take significant time before decoder.forward
                    # is entered. Do not leave the job showing sampling 100%.
                    progress("decoding", 0, latent_chunks)
                return kwargs

            # Decoder hooks make causal-frame progress and cancellation visible.
            decoding_started = False
            decoded_chunks = 0

            def before_decode(_module, _inputs):
                nonlocal decoding_started
                if not decoding_started:
                    decoding_started = True
                    weight = next(_module.parameters(), None)
                    decoder_device = weight.device if weight is not None else "unknown"
                    log.info("Wan VAE decoder entered on %s (%d latent chunks)",
                             decoder_device, latent_chunks)
                cancel.check()

            def after_decode(_module, _inputs, _output):
                nonlocal decoded_chunks
                decoded_chunks += 1
                progress("decoding", decoded_chunks, latent_chunks)
                if decoded_chunks == latent_chunks:
                    log.info("Wan VAE decoder completed; finalizing frames")

            cancel.check()
            params["callback_on_step_end"] = step_end
            with ExitStack() as hooks:
                for module in (pipe.text_encoder, pipe.transformer, pipe.vae.encoder):
                    hooks.callback(module.register_forward_pre_hook(
                        lambda *_: cancel.check()).remove)
                hooks.callback(pipe.vae.decoder.register_forward_pre_hook(
                    before_decode).remove)
                hooks.callback(pipe.vae.decoder.register_forward_hook(
                    after_decode).remove)
                try:
                    result = pipe(**params)
                finally:
                    # Cancellation can skip normal VAE cache cleanup.
                    pipe.vae.clear_cache()
            cancel.check()
            progress("encoding", 0, spec.frame_count)
            log.info("Wan pipeline returned; encoding %d frames as %s",
                     spec.frame_count, spec.output_format)

            frames = result.frames[0]
            del result
        # PIL frames are CPU-side. Release the GPU inference lock before the
        # CPU video encoder runs; other model-free TheNoise ops can use the GPU.
        cancel.check()
        encode_video(frames, spec, cancel, output, progress)
        log.info("Wan video saved: %s", output)

    def unload(self) -> None:
        self._pipe = None
        self._mode = None
        self._placed = False
        self._base_scheduler_config = None
        gc.collect()
        import torch
        if self.device.startswith("cuda") and torch.cuda.is_available():
            torch.cuda.empty_cache()
