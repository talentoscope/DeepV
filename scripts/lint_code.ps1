<#
Run code formatters and linters locally.
Usage: run from repository root in PowerShell:
  .\scripts\lint_code.ps1
#>

if (-Not (Test-Path ".venv")) {
    Write-Host "Create a venv first: python -m venv .venv"; exit 1
}

& .\.venv\Scripts\Activate.ps1

python -m pip install --upgrade pip
python -m pip install -r requirements-dev.txt

Write-Host "Running black (check)..."
python -m black --check .

Write-Host "Running isort (check)..."
python -m isort --check-only .

Write-Host "Running flake8..."
python -m flake8 .
