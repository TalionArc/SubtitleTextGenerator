[CmdletBinding()]
param(
    [switch]$SkipTests
)

$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest
$projectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$venvPython = Join-Path $projectRoot '.venv\Scripts\python.exe'
$pyinstaller = Join-Path $projectRoot '.venv\Scripts\pyinstaller.exe'

Push-Location $projectRoot
try {
    if (-not (Test-Path -LiteralPath $venvPython)) {
        python -m venv .venv
    }
    & $venvPython -m pip install --upgrade pip
    & $venvPython -m pip install -r requirements-build.txt
    if (-not $SkipTests) {
        & $venvPython -m ruff check lecture_subtitle_batcher tests run.py
        & $venvPython -m pytest
    }
    & $pyinstaller `
        --noconfirm `
        --clean `
        --onefile `
        --windowed `
        --name LectureSubtitleBatcher `
        --distpath output `
        --workpath build `
        --version-file packaging\version_info.txt `
        run.py
    Write-Host "완료: $projectRoot\output\LectureSubtitleBatcher.exe"
}
finally {
    Pop-Location
}
