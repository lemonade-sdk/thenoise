# ===========================================================================
# download_artifact.ps1 — Download a GitHub Actions artifact by name to a local
# file, with retries + resumable download. Windows counterpart of the inline
# curl download used in the Linux qualify job (actions/download-artifact@v4 is
# unreliable for multi-GB artifacts on self-hosted runners).
#
# Usage: download_artifact.ps1 -Name <artifact_name> -Out <output_file>
#   Token defaults to $env:GH_TOKEN (set by the workflow).
# ===========================================================================
param(
  [Parameter(Mandatory = $true)][string]$Name,
  [Parameter(Mandatory = $true)][string]$Out,
  [string]$Token = $env:GH_TOKEN
)

$ErrorActionPreference = "Stop"
if (-not $Token) { Write-Error "no GH_TOKEN available"; exit 1 }
if (-not $env:GITHUB_REPOSITORY -or -not $env:GITHUB_RUN_ID) {
  Write-Error "GITHUB_REPOSITORY / GITHUB_RUN_ID not set; run from a GitHub Actions job"
  exit 1
}

$Headers = @{ Authorization = "Bearer $Token"; Accept = "application/vnd.github+json" }
$Base = "https://api.github.com/repos/$env:GITHUB_REPOSITORY/actions"

# Locate the artifact. It may appear late on the runner, so poll briefly.
$Art = $null
for ($attempt = 1; $attempt -le 5; $attempt++) {
  try {
    $run = Invoke-RestMethod -Uri "$Base/runs/$env:GITHUB_RUN_ID/artifacts" -Headers $Headers
    $Art = $run.artifacts | Where-Object { $_.name -eq $Name } | Select-Object -First 1
    if ($Art) { break }
  } catch {}
  Write-Host "artifact '$Name' not found yet; retrying in 10s"
  Start-Sleep -Seconds 10
}
if (-not $Art) { Write-Error "artifact '$Name' not found"; exit 1 }

# Download with retries (curl.exe is resumable).
$ok = $false
for ($attempt = 1; $attempt -le 3; $attempt++) {
  curl.exe -fL --retry 5 --retry-delay 10 --retry-all-errors -C - `
    -H "Authorization: Bearer $Token" -H "Accept: application/vnd.github+json" `
    -o $Out "$Base/artifacts/$($Art.id)/zip"
  if ($LASTEXITCODE -eq 0) { $ok = $true; break }
  Write-Host "download attempt $attempt failed; retrying in 15s"
  Start-Sleep -Seconds 15
}
if (-not $ok) { Write-Error "artifact download failed for '$Name'"; exit 1 }
Write-Host "downloaded artifact '$Name': $((Get-Item $Out).Length) bytes"
