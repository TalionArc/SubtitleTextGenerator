[CmdletBinding()]
param(
    [switch]$SkipTests
)

$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest
$projectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$venvPython = Join-Path $projectRoot '.venv\Scripts\python.exe'
$pyinstaller = Join-Path $projectRoot '.venv\Scripts\pyinstaller.exe'

function Get-Sha256Hex {
    param([Parameter(Mandatory)][string]$FilePath)

    $algorithm = [System.Security.Cryptography.SHA256]::Create()
    $stream = [System.IO.File]::OpenRead($FilePath)
    try {
        $bytes = $algorithm.ComputeHash($stream)
        return ([System.BitConverter]::ToString($bytes)).Replace('-', '')
    }
    finally {
        $stream.Dispose()
        $algorithm.Dispose()
    }
}

function Backup-Executable {
    param([Parameter(Mandatory)][string]$ExecutablePath)

    if (-not (Test-Path -LiteralPath $ExecutablePath)) {
        return
    }
    $item = Get-Item -LiteralPath $ExecutablePath
    $version = ([string]$item.VersionInfo.FileVersion).Trim()
    if ([string]::IsNullOrWhiteSpace($version)) {
        $version = 'unknown'
    }
    $sourceHash = Get-Sha256Hex $item.FullName
    $backupDirectory = Join-Path $projectRoot (Join-Path 'backup' $version)
    New-Item -ItemType Directory -Force -Path $backupDirectory | Out-Null
    $target = Join-Path $backupDirectory $item.Name
    if (Test-Path -LiteralPath $target) {
        $targetHash = Get-Sha256Hex $target
        if ($targetHash -eq $sourceHash) {
            Write-Host "기존 빌드 보관 확인: $target"
            return
        }
        $target = Join-Path $backupDirectory (
            '{0}-{1}{2}' -f $item.BaseName, $sourceHash.Substring(0, 12), $item.Extension
        )
        if (Test-Path -LiteralPath $target) {
            $targetHash = Get-Sha256Hex $target
            if ($targetHash -eq $sourceHash) {
                Write-Host "기존 빌드 보관 확인: $target"
                return
            }
            throw "같은 이름의 다른 백업 파일이 이미 있습니다: $target"
        }
    }
    Copy-Item -LiteralPath $item.FullName -Destination $target
    Write-Host "기존 빌드 보관 완료: $target"
}

Push-Location $projectRoot
try {
    Backup-Executable (Join-Path $projectRoot 'dist\SubtitleTextGenerator.exe')
    Backup-Executable (Join-Path $projectRoot 'output\SubtitleTextGenerator.exe')
    if (-not (Test-Path -LiteralPath $venvPython)) {
        python -m venv .venv
    }
    & $venvPython -m pip install --upgrade pip
    & $venvPython -m pip install -r requirements-build.txt
    if (-not $SkipTests) {
        & $venvPython -m ruff check subtitle_text_generator tests run.py
        & $venvPython -m pytest
    }
    & $pyinstaller `
        --noconfirm `
        --clean `
        --onefile `
        --windowed `
        --name SubtitleTextGenerator `
        --distpath output `
        --workpath build `
        --version-file packaging\version_info.txt `
        run.py
    Backup-Executable (Join-Path $projectRoot 'output\SubtitleTextGenerator.exe')
    Write-Host "완료: $projectRoot\output\SubtitleTextGenerator.exe"
}
finally {
    Pop-Location
}
