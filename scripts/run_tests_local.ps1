<#
PowerShell helper to create a venv, install dev requirements, run the env validator, and run tests.
Run from repository root in PowerShell:
  .\scripts\run_tests_local.ps1
#>

$venvPath = ".venv"
if (-Not (Test-Path $venvPath)) {
    Write-Host "Creating virtual environment..."
    python -m venv $venvPath
}

Write-Host "Activating venv..."
& "$venvPath\Scripts\Activate.ps1"

Write-Host "Upgrading pip and installing dev requirements..."
python -m pip install --upgrade pip
python -m pip install -r requirements-dev.txt

Write-Host "Running environment validator..."
python scripts/validate_env.py

Write-Host "Running pytest..."
python -m pytest -q
