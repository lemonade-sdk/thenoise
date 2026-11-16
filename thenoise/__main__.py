"""Entrypoint: parse CLI, load a single model, then serve via uvicorn."""
from __future__ import annotations

import logging
from pathlib import Path
from .cli import build_parser


class _QuietReads(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        line = record.getMessage()
        return not ('"GET /jobs' in line and '" 200' in line)


def _serve(args) -> None:
    from .runtime import Settings, ModelPaths, Runtime
    settings = Settings(
        device=args.device, host=args.host, port=args.port,
        upscaler_dir=args.upscaler_dir, offload_device=args.offload_device,
        controlmap_dir=args.controlmap_dir,
    )

    runtime = Runtime(settings)

    # --dit alone identifies a model. Auxiliary checkpoints require --dit.
    if args.dit:
        runtime.load(
            ModelPaths(
                dit_path=args.dit,
                vae_path=args.vae,
                text_encoder_path=args.text_encoder,
                lora_dir=args.lora_dir,
                dit_low_path=args.dit_low,
                ltx_audio_vae_path=args.ltx_audio_vae,
                ltx_upscaler_path=args.ltx_upscaler,
                ltx_text_projection_path=args.ltx_text_projection,
            ),
        )
    elif any((
        args.vae,
        args.text_encoder,
        args.dit_low,
        args.ltx_audio_vae,
        args.ltx_upscaler,
        args.ltx_text_projection,
    )):
        raise SystemExit("serve: auxiliary model checkpoints require --dit")

    # The queue exists whether or not there is a model to serve: a submit that finds
    # nothing loaded is refused, and an unused queue never creates its directory.
    from .jobs import JobQueue
    jobs = JobQueue(Path(args.cache_dir).expanduser() / "jobs")

    from .api import create_app
    import uvicorn

    app = create_app(runtime, jobs=jobs)
    medium = runtime.medium
    if medium == "image":
        print(f"thenoise serving model '{runtime.model_name}' on {settings.device}")
    elif medium:
        print(f"thenoise serving {medium} model '{runtime.model_name}' "
              f"on {settings.device}")
    else:
        print(f"thenoise serving WITHOUT a model on {settings.device} "
              f"(only upscaling and control-map extraction are available)")
    logging.getLogger("uvicorn.access").addFilter(_QuietReads())
    uvicorn.run(app, host=settings.host, port=settings.port)


def main(argv=None) -> None:
    args = build_parser().parse_args(argv)
    if args.command == "serve":
        _serve(args)
    elif args.command == "generate":
        from .generate import run_generate
        run_generate(args)
    elif args.command == "edit":
        from .generate import run_edit
        run_edit(args)
    elif args.command == "video":
        from .video_cli import run_video
        run_video(args)
    elif args.command == "upscale":
        from .upscale_cli import run_upscale
        run_upscale(args)
    elif args.command == "controlmap":
        from .controlmap_cli import run_controlmap
        run_controlmap(args)
    else:  # pragma: no cover - argparse requires a subcommand
        raise SystemExit(
            "choose a subcommand: serve | generate | edit | video | upscale | controlmap"
        )


if __name__ == "__main__":
    main()
