# Dependency audit and remediation notes

GitHub reported dependency vulnerability findings for this repository when pushing recent changes. This file explains how to audit and triage dependency issues locally and via automated tools.

Recommended quick checks

- Use `pip-audit` to enumerate known issues from PyPI:

```bash
python -m pip install pip-audit
pip-audit
```

- Optionally use `safety` or `pipdeptree` for dependency trees:

```bash
pip install safety pipdeptree
safety check
pipdeptree
```

What Dependabot will do

- Dependabot (enabled) will open weekly PRs for `pip` and `github-actions` updates. Review CI runs for each PR before merging.

How to pick upgrade candidates

- Prefer patch/minor upgrades first; test locally and in CI. Use `pip install 'package>=x.y.z'` and run `pytest`.
- For critical vulnerabilities where an upgrade is not possible (compatibility), consider:
  - pinning to a safe version and adding a `dependabot` ignore rule, or
  - replacing the dependency with a maintained alternative.

Automated remediation steps (recommended workflow)

1. Create a new branch from `master`.
2. Run `pip-audit` and inspect findings.
3. Update `requirements.txt` (or `requirements-dev.txt`) to the minimal safe versions.
4. Run the test suite locally (`pytest -q`) and push the branch.
5. Open a PR and let the CI run on GitHub Actions. If CI is green, merge.

Notes about heavyweight packages

- Some packages (e.g., `torch`, `pycairo`, `chamferdist`) may have GPU or platform-specific constraints. Test upgrades in a staging environment or with the Docker image.

If you want, I can:
- open a scripted audit (add a `scripts/audit_deps.sh`) to run `pip-audit` and produce a report, or
- prepare an initial branch with safe upgrades for non-breaking dev dependencies (pytest, networkx, flake8).
