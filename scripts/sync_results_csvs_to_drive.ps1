# Sync all results/*.csv to Google Drive, preserving directory structure.
#
# Prereq: Google Drive for Desktop installed and syncing.
# Target:  <Drive>/My Drive/vlm_research_results/results_csvs/
#
# Usage (from repo root):
#   powershell -File scripts/sync_results_csvs_to_drive.ps1
#   powershell -File scripts/sync_results_csvs_to_drive.ps1 -DriveRoot "G:\My Drive"

param(
    [string]$DriveRoot = "",
    [string]$RepoRoot = ""
)

if (-not $RepoRoot) {
    $RepoRoot = Split-Path -Parent $PSScriptRoot
}

$ResultsDir = Join-Path $RepoRoot "results"
if (-not (Test-Path $ResultsDir)) {
    Write-Error "results/ not found at $ResultsDir"
    exit 1
}

# Auto-detect Google Drive mount if not passed explicitly.
if (-not $DriveRoot) {
    $candidates = @(
        "G:\My Drive",
        "H:\My Drive",
        "I:\My Drive",
        (Join-Path $env:USERPROFILE "Google Drive"),
        (Join-Path $env:USERPROFILE "Google Drive\My Drive")
    )
    foreach ($c in $candidates) {
        if (Test-Path $c) {
            $DriveRoot = $c
            break
        }
    }
}

if (-not $DriveRoot -or -not (Test-Path $DriveRoot)) {
    Write-Host "Google Drive not found. Either:"
    Write-Host "  1. Install Google Drive for Desktop, then re-run this script"
    Write-Host "  2. Upload vlm_results_csvs.zip (repo root) via drive.google.com"
    Write-Host "  3. Pass -DriveRoot explicitly, e.g.:"
    Write-Host '     powershell -File scripts/sync_results_csvs_to_drive.ps1 -DriveRoot "G:\My Drive"'
    exit 1
}

$DestRoot = Join-Path $DriveRoot "vlm_research_results\results_csvs"
New-Item -ItemType Directory -Force -Path $DestRoot | Out-Null

$csvs = Get-ChildItem -Recurse -File $ResultsDir -Filter "*.csv"
$copied = 0
$skipped = 0
$bytes = 0

foreach ($csv in $csvs) {
    $rel = $csv.FullName.Substring($ResultsDir.Length).TrimStart('\', '/')
    $dest = Join-Path $DestRoot $rel
    $destDir = Split-Path $dest -Parent
    New-Item -ItemType Directory -Force -Path $destDir | Out-Null

    if ((Test-Path $dest) -and ((Get-Item $dest).Length -eq $csv.Length)) {
        $skipped++
        continue
    }
    Copy-Item $csv.FullName $dest -Force
    $copied++
    $bytes += $csv.Length
}

Write-Host ""
Write-Host "Drive sync complete"
Write-Host "  Source:  $ResultsDir"
Write-Host "  Dest:    $DestRoot"
Write-Host "  Copied:  $copied files ($([math]::Round($bytes/1MB,1)) MB)"
Write-Host "  Skipped: $skipped (already up to date)"
Write-Host "  Total:   $($csvs.Count) CSVs"
