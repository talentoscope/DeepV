# Contributing

Thank you for helping improve DeepV. This file describes the minimal developer workflow and useful commands to get changes verified and submitted.

Branching
- Create a descriptive branch from `main` / `master`:

```bash
git checkout -b draft/<topic>-<short>
```

- Keep commits small and focused. Use a single topic per branch.

Tests & quick validation
- Run the environment validator before heavy installs:

```bash
python scripts/validate_env.py
```

- Install requirements and run tests locally (use a virtualenv or Docker):

```bash
pip install -r requirements.txt
pip install pytest
pytest -q
```

Local linting and formatting

- This repository does not use GitHub Actions or remote CI. Run all checks and tests locally before pushing.

- To lint/format locally:

```bash
pip install flake8 black
flake8 .
black .
```

Utilities and gotchas
- `util_files/file_utils.py` provides `require_empty()` — it was renamed from the old `util_files/os.py` to avoid shadowing the stdlib.
- Many scripts use absolute defaults for paths (`/code`, `/data`, `/logs`). Prefer overriding `argparse` flags rather than changing defaults in code.

Docker / Windows / WSL
- The provided `docker/Dockerfile` can be used to build a reproducible environment. Example run on Windows with WSL or Docker Desktop:

```powershell
docker build -t deepv:latest .
docker run --rm -it --shm-size 128G \
  --mount type=bind,source="C:/path/to/DeepV",target=/code \
  --mount type=bind,source="C:/path/to/data",target=/data \
  --mount type=bind,source="C:/path/to/logs",target=/logs \
  --name deepv-container deepv:latest /bin/bash
```

Pushing and PRs
- Push your branch and open a PR against `main` / `master`. Use a concise title and paste the contents of `PR_DESCRIPTION.md` as the PR body when applicable.

Maintenance notes
- Upgrading `torch` requires selecting a wheel matching your CUDA runtime. Use the official PyTorch install selector (https://pytorch.org/get-started/locally/).

Where to get help
- For questions about specific modules see:
  - `cleaning/` — cleaning & UNet training
  - `vectorization/` — model specs and training
  - `refinement/` and `merging/` — optimization + postprocessing
  - `util_files/` — shared utilities
