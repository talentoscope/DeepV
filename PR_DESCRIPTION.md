Summary of changes
------------------

This branch contains a set of developer-focused improvements to make the repository easier to inspect, test, and iterate on:

- GitHub Actions CI: add `.github/workflows/ci.yml` with linting, formatting checks, tests, and a lightweight smoke job.
- Tests: add `tests/test_smoke.py` and `tests/test_file_utils.py` to provide a minimal import and unit coverage for critical utilities.
- Linting/format: add `.flake8` and enable `black --check` in CI.
- Utilities: rename `util_files/os.py` → `util_files/file_utils.py` to avoid shadowing the stdlib `os`; update imports accordingly.
- Docs: update `README.md` and `.github/copilot-instructions.md` with Windows/WSL Docker notes and a note about the util rename.
- Dev tooling: add `scripts/validate_env.py` to quickly verify Python, key packages, and CUDA availability.
- Dependency guidance: add `requirements-updated.txt` with suggested package upgrades and install notes.

Why
---

These changes aim to provide a lightweight, reproducible developer experience (CI + smoke tests) and remove a common import foot-gun (`util_files/os.py`). The requirements draft and validator lower the barrier for testing upgrades.

How to test locally
-------------------

1. Validate environment:

```bash
python scripts/validate_env.py
```

2. Install dependencies and run tests:

```bash
pip install -r requirements.txt
pip install pytest
pytest -q
```

Notes and next steps
--------------------

- I did not push a remote branch — please review locally or let me push to the repository when you confirm.
- Consider iterating on `requirements-updated.txt` by testing `torch` wheel compatibility with your CUDA driver before upgrading.
- Next PR steps could include adding CI checks for type-checking (`mypy`) and incremental refactors of long functions in `refinement/`.
