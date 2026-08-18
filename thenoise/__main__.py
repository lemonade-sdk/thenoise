"""Entrypoint: parse CLI, load a single model, then serve via uvicorn."""
from __future__ import annotations

from .cli import build_parser


def _serve(args) -> None:
    import logging
    logging.basicConfig(level=logging.INFO)

    from .runtime import Settings, ModelPaths, Runtime
    settings = Settings(
        device=args.device, host=args.host, port=args.port,
        upscaler_dir=args.upscaler_dir,
        gallery_dir=args.gallery,
    )

    runtime = Runtime(settings)
    runtime.load(
        ModelPaths(
            dit_path=args.dit,
            vae_path=args.vae,
            text_encoder_path=args.text_encoder,
            lora_dir=args.lora_dir,
        ),
    )

    from .api import create_app
    from .api import Gallery
    import uvicorn

    app = create_app(runtime, gallery=Gallery(settings.gallery_dir))
    print(f"thenoise serving model '{runtime.model_name}' on {settings.device}")
    uvicorn.run(app, host=settings.host, port=settings.port)


def main(argv=None) -> None:
    args = build_parser().parse_args(argv)
    if args.command == "serve":
        _serve(args)
    elif args.command == "generate":
        from .generate import run_generate
        run_generate(args)
    else:  # pragma: no cover - argparse requires a subcommand
        raise SystemExit("choose a subcommand: serve | generate")


if __name__ == "__main__":
    main()
