# Repository Guidelines

## Project Structure & Module Organization
Keep the FastAPI entrypoint in `main.py`; place IO- or AI-heavy helpers beside it (_prefixed when internal) and keep the AnimateDiff backend inside `video_generation/` via the `AnimateDiffGenerator`. Templates for HTML live under `templates/`, while CSS/JS/assets reside in `static/`. Generated media, audio, or previews must stay under `media/` (gitignored) so it can be purged before commits. Mirror this layout in tests—e.g., `tests/test_main.py` for `main.py` or `tests/video_generation/test_backend.py` for backend helpers.

## Build, Test, and Development Commands
Create the virtualenv with `python -m venv .venv` followed by `.\\.venv\\Scripts\\activate` on Windows or `source .venv/bin/activate` on Unix. Install runtime deps via `pip install -r requirements.txt`; GPU users may override Torch by pointing pip at the cu118 wheel URL noted in the docs. Run the API locally with `uvicorn main:app --reload --host 127.0.0.1 --port 8000`, or execute `./run.ps1` / `./run.sh` for scripted env setup plus AnimateDiff bootstrapping.

## Coding Style & Naming Conventions
Follow Black defaults (4 spaces, 88-char lines). Use snake_case for functions, PascalCase for classes, and SCREAMING_SNAKE_CASE for constants such as `MEDIA_DIR`. Route handlers should remain thin; move heavy lifting to helpers like `_generate_visual_asset`. Add docstrings when functions have side effects and always log through the shared logger instead of `print`.

## Testing Guidelines
Use `pytest` with `httpx.AsyncClient` for FastAPI routes and `pytest-mock` for diffuser, TTS, or moviepy integrations. Name files `test_<feature>.py` and mirror the runtime modules. Run `pytest -q` for quick checks, and `pytest --cov=main --cov=video_generation` before pushing; deterministic helpers (e.g., `_generate_plan_and_contexts`) should sit above 80% coverage.

## Commit & Pull Request Guidelines
Write concise, imperative commit subjects under 72 chars ("Add AnimateDiff backend"). Pull requests must link an issue or state user impact, include manual verification notes ("Ran ./run.ps1 with VIDEO_GENERATOR=animatediff"), and attach UI or media diffs whenever visuals change. Confirm `media/` is clean before requesting review.

## Security & Configuration Tips
Store `OPENAI_API_KEY`, `VIDEO_GENERATOR`, `HUGGINGFACE_TOKEN`, and any `ANIMATEDIFF_*` overrides in `.env` and never commit them. Ensure feature flags degrade gracefully when credentials or GPUs are missing. Keep large model weights out of the repo, and scrub the `media/` directory before sharing artifacts.
