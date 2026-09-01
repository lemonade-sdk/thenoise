# AGENTS.md

Guidance for AI coding agents working in this repository.

## Project overview

`thenoise` is a focused diffusion inference engine for ROCm. It loads **one model at
a time** and exposes a small, explicit surface. The main deployment target is
Strix Halo / gfx1151, RDNA 3.5, native BF16/FP16, 128GB unified RAM. Other tested
targets are gfx1150 and gfx1152.

## Critical constraints

1. **Never run `uv sync`.** It would replace/break the ROCm `torch` build that the
   maintainer installs directly into the venv. Use `uv pip install` instead
   (e.g. `uv pip install -e .`). `torch` is intentionally not listed in
   `pyproject.toml`.

2. **Never run the program yourself.** You are in a containerized environment that
   cannot run the project with real models — there is no GPU and not enough RAM.
   Do not attempt to start the server, run `generate`, or load a model.

3. **When unsure on the correct way to proceed, ALWAYS ask the user** rather than 
   guessing or overthinking.

4. **The user is happy to test.** Do not be too afraid of breaking stuff, just
   inform the user of potentially risky changes and ask them to test them out.

## Workflow

- Verify changes with the test suite: `.venv/bin/python -m pytest tests/ -q`
  (tests are designed to run without real weights).
- Do not run heavy/compute-heavy commands.
