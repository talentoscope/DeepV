#!/usr/bin/env bash
set -euo pipefail

# Simple dependency audit helper. Creates `reports/` and writes JSON/text reports.
# Requires: pip-audit, safety, pipdeptree (install in a venv via requirements-dev.txt)

OUTDIR="reports"
mkdir -p "$OUTDIR"

echo "Running pip-audit..."
if command -v pip-audit >/dev/null 2>&1; then
  pip-audit --format json --output "$OUTDIR/pip_audit.json" || true
else
  echo "pip-audit not installed; skipping pip-audit" > "$OUTDIR/pip_audit.json"
fi

echo "Running safety..."
if command -v safety >/dev/null 2>&1; then
  safety check --json > "$OUTDIR/safety.json" || true
else
  echo "safety not installed; skipping safety" > "$OUTDIR/safety.json"
fi

echo "Writing dependency tree via pipdeptree..."
if command -v pipdeptree >/dev/null 2>&1; then
  pipdeptree --warn silence > "$OUTDIR/pipdeptree.txt" || true
else
  echo "pipdeptree not installed; skipping pipdeptree" > "$OUTDIR/pipdeptree.txt"
fi

echo "Audit complete. Reports written to $OUTDIR/"
