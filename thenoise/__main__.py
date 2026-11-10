"""Entrypoint: parse CLI, load a single model, then serve via uvicorn."""
from __future__ import annotations

from pathlib import Path
from .cli import build_parser


def _serve(args) -> None:
    from .runtime import Settings, ModelPaths, Runtime
    settings = Settings(
        device=args.device, host=args.host, port=args.port,
        upscaler_dir=args.upscaler_dir, offload_device=args.offload_device,
        controlmap_dir=args.controlmap_dir,
    )

    runtime = Runtime(settings)

    # Load the model only when all three checkpoints are supplied; otherwise
    # serve model-free (upscale only).
    model_paths = [args.dit, args.vae, args.text_encoder]
    if args.video_model_dir and any(model_paths):
        raise SystemExit("serve: --video-model-dir is video-only; do not combine it with image model paths")

    if any(model_paths):
        if not all(model_paths):
            raise SystemExit(
                "serve: to load a model you must supply --dit, --vae and "
                "--text-encoder together; omit all three to run without a model"
            )
        runtime.load(
            ModelPaths(
                dit_path=args.dit,
                vae_path=args.vae,
                text_encoder_path=args.text_encoder,
                lora_dir=args.lora_dir,
            ),
        )

    video_jobs = None
    video_model = ""
    if args.video_model_dir:
        from .video.service import create_video_service
        video_jobs, video_model = create_video_service(
            args.video_model_dir, device=settings.device, offload=args.video_offload,
            cache_dir=str(Path(args.video_cache_dir).expanduser()),
        )

    from .api import create_app
    import uvicorn

    app = create_app(runtime, video_jobs=video_jobs, video_model=video_model)
    if video_jobs is not None:
        print(f"thenoise serving video model '{video_model}' on {settings.device}")
    elif runtime.available():
        print(f"thenoise serving model '{runtime.model_name}' on {settings.device}")
    else:
        print(f"thenoise serving WITHOUT a model on {settings.device} "
              f"(only upscaling and control-map extraction are available)")
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
    elif args.command == "upscale":
        from .upscale_cli import run_upscale
        run_upscale(args)
    elif args.command == "controlmap":
        from .controlmap_cli import run_controlmap
        run_controlmap(args)
    else:  # pragma: no cover - argparse requires a subcommand
        raise SystemExit(
            "choose a subcommand: serve | generate | edit | upscale | controlmap"
        )


if __name__ == "__main__":
    main()
