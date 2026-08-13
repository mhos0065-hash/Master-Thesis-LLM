# setup_and_run.ps1
# ===================
# Run this once from PowerShell, inside the folder containing all the
# project files, to set up the environment and fire off the smallest
# possible first test.
#
# Usage:
#   powershell -ExecutionPolicy Bypass -File setup_and_run.ps1
# (the -ExecutionPolicy Bypass flag here only affects this one script run,
#  it does not change any system-wide setting)

Write-Host "=== Step 1: Checking Python ===" -ForegroundColor Cyan
$pythonCmd = $null
foreach ($cmd in @("python", "py")) {
    if (Get-Command $cmd -ErrorAction SilentlyContinue) {
        $pythonCmd = $cmd
        break
    }
}
if (-not $pythonCmd) {
    Write-Host "Python not found on PATH. Install it from https://python.org (check 'Add python.exe to PATH' during install), then re-run this script." -ForegroundColor Red
    exit 1
}
& $pythonCmd --version

Write-Host "`n=== Step 2: Creating virtual environment (venv/) ===" -ForegroundColor Cyan
if (-not (Test-Path "venv")) {
    & $pythonCmd -m venv venv
} else {
    Write-Host "venv already exists, skipping."
}

Write-Host "`n=== Step 3: Activating virtual environment ===" -ForegroundColor Cyan
try {
    & .\venv\Scripts\Activate.ps1
} catch {
    Write-Host "Activation blocked by execution policy. Allowing it for this process only..." -ForegroundColor Yellow
    Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass -Force
    & .\venv\Scripts\Activate.ps1
}

Write-Host "`n=== Step 4: Installing dependencies ===" -ForegroundColor Cyan
pip install -r requirements.txt

Write-Host "`n=== Step 5: Checking for .env ===" -ForegroundColor Cyan
if (-not (Test-Path ".env")) {
    Copy-Item ".env.example" ".env"
    Write-Host "Created .env from .env.example." -ForegroundColor Yellow
    Write-Host "Opening it in Notepad -- paste your API key into LLM_API_KEY, save, and close Notepad to continue." -ForegroundColor Yellow
    Start-Process notepad.exe ".env" -Wait
} else {
    Write-Host ".env already exists, skipping."
}

# quick sanity check that a key was actually entered
$envContent = Get-Content ".env" -Raw
if ($envContent -match "LLM_API_KEY=\s*(gsk_your_key_here|sk-your_key_here|)\s*($|\r?\n)") {
    Write-Host "`nLLM_API_KEY still looks like a placeholder or is empty. Edit .env and set a real key before continuing." -ForegroundColor Red
    Write-Host "Re-run this script once it's set." -ForegroundColor Red
    exit 1
}

Write-Host "`n=== Step 6: Confirming test chunks exist ===" -ForegroundColor Cyan
if (-not (Test-Path "chunks_csv\manifest.csv")) {
    python generate_all_chunks.py
} else {
    Write-Host "chunks_csv\manifest.csv already exists, skipping regeneration."
}

Write-Host "`n=== Step 7: Running the smallest first test ===" -ForegroundColor Cyan
Write-Host "single_agent.py on one chunk (cyclehire_outlier_sample)...`n"
python single_agent.py --chunk cyclehire_outlier_sample

Write-Host "`n=== Done ===" -ForegroundColor Green
Write-Host "Check the runs\ folder for a new <timestamp>_single_..._cyclehire_outlier_sample folder with report.md inside."
Write-Host "If that worked, try:  python multi_agent.py --chunk cyclehire_outlier_sample"
Write-Host "Then the full sweep:  python run_experiment.py"
