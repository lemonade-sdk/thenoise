"""Entrypoint: parse CLI, load a single model, then serve via uvicorn."""
from __future__ import annotations

from .cli import build_parser


def _serve(args) -> None:
    from .cli import resolve_model_paths
    from .runtime import Settings, ModelPaths, Runtime
    settings = Settings(
        device=args.device, host=args.host, port=args.port,
        upscaler_dir=args.upscaler_dir,
    )

    runtime = Runtime(settings)

    # Model paths are optional. Load the model only when checkpoints are supplied
    # (either --checkpoint or the --dit/--vae/--text-encoder trio); otherwise
    # serve model-free (only upscale is available).
    if args.checkpoint or args.dit or args.vae or args.text_encoder:
        runtime.load(ModelPaths(**resolve_model_paths(args)))

    from .api import create_app
    import uvicorn

    app = create_app(runtime)
    if runtime.available():
        print(f"thenoise serving model '{runtime.model_name}' on {settings.device}")
    else:
        print(f"thenoise serving WITHOUT a model on {settings.device} "
              f"(only upscaling is available)")
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
    else:  # pragma: no cover - argparse requires a subcommand
        raise SystemExit("choose a subcommand: serve | generate | edit | upscale")


if __name__ == "__main__":
    main()
