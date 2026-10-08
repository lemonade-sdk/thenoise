# ---------------------------------------------------------------------------
# thenoise.ps1 — Bootstrap the venv (if needed) and launch the project on
# Windows. All CLI arguments are forwarded to `python -m thenoise`.
#
# Invoke via the thin `thenoise.bat` wrapper (double-click friendly) or
# directly from a terminal:
#   powershell -NoProfile -ExecutionPolicy Bypass -File thenoise.ps1 [args...]
# ---------------------------------------------------------------------------
param([Parameter(ValueFromRemainingArguments = $true)]$ForwardArgs)

$ErrorActionPreference = "Stop"

$ProjectDir = $PSScriptRoot
if (-not $ProjectDir) { $ProjectDir = (Get-Location).Path }
$VenvDir = Join-Path $ProjectDir ".venv"

# ---------------------------------------------------------------------------
# AMD GPU -> ROCm gfx target (used by step 3)
#
# Order of resolution:
#   1. PCI device id (PNPDeviceID DEV_xxxx) - survives the generic Name
#      strings ("AMD Radeon(TM) Graphics") that the marketing-name table can
#      never match.
#   2. A gfx code baked into the driver-reported name.
#   3. Marketing name fragments, longest pattern first.
# ---------------------------------------------------------------------------

# PCI device id (lowercase hex, no "0x") -> 'gfx code | human label'. One id
# can cover several marketing names (binned/fused SKUs of the same die).
$PciDevToGfx = @{
    # RDNA4 - Navi44/Navi48
    '7590' = 'gfx1200 | Navi 44 (RX 9060 XT)'
    '7550' = 'gfx1201 | Navi 48 (RX 9070/9070 XT/9070 GRE)'
    '7551' = 'gfx1201 | Navi 48 (Radeon AI PRO R9700)'
    '7580' = 'gfx1201 | Navi 48 (RX 9070 XT)'
    '7581' = 'gfx1201 | Navi 48 (RX 9070)'
    '7591' = 'gfx1201 | Navi 44 (RX 9060 XT)'
    '75a1' = 'gfx1201 | Navi 48 (RX 9070 GRE)'
    '75b0' = 'gfx1201 | Navi 48 (RX 9070 XT)'

    # RDNA3.5 - Strix / Strix Halo / Krackan Point
    '150e' = 'gfx1150 | Strix (880M/890M)'
    '1586' = 'gfx1151 | Strix Halo (8050S/8060S)'
    '1114' = 'gfx1152 | Krackan Point (840M/860M)'
    '1590' = 'gfx1150 | Strix Point (880M)'
    '1591' = 'gfx1150 | Strix Point (890M)'
    '15d0' = 'gfx1152 | Krackan Point (860M)'

    # RDNA3 - Navi31/32/33, Phoenix
    '7448' = 'gfx1100 | Navi 31 (Pro W7900)'
    '7449' = 'gfx1100 | Navi 31 (Pro W7800 48GB)'
    '744a' = 'gfx1100 | Navi 31 (Pro W7900 Dual Slot)'
    '744b' = 'gfx1100 | Navi 31 (Pro W7900D)'
    '744c' = 'gfx1100 | Navi 31 (RX 7900 XT/XTX/GRE/7900M)'
    '745e' = 'gfx1100 | Navi 31 (Pro W7800)'
    '7460' = 'gfx1101 | Navi 32 (Pro V710)'
    '7461' = 'gfx1101 | Navi 32 (Pro V710)'
    '7470' = 'gfx1101 | Navi 32 (Pro W7700)'
    '747e' = 'gfx1101 | Navi 32 (RX 7700 XT/7800 XT)'
    '7480' = 'gfx1102 | Navi 33 (RX 7600 series/Pro W7600)'
    '7481' = 'gfx1102 | Navi 33'
    '7483' = 'gfx1102 | Navi 33 (RX 7600M/7600M XT)'
    '7487' = 'gfx1102 | Navi 33'
    '7489' = 'gfx1102 | Navi 33 (Pro W7500)'
    '748b' = 'gfx1102 | Navi 33'
    '7499' = 'gfx1102 | Navi 33 (RX 7400/7300/Pro W7400)'
    '749f' = 'gfx1102 | Navi 33 (RX 7500)'
    '15bf' = 'gfx1103 | Phoenix1 (780M/760M/740M)'
    '15c8' = 'gfx1103 | Phoenix2 (780M/760M/740M)'
    '164f' = 'gfx1103 | Phoenix (780M/760M/740M)'

    # RDNA2 - Navi21/22/23/24, Rembrandt, Mendocino
    '73a1' = 'gfx1030 | Navi 21 (Pro V620)'
    '73a2' = 'gfx1030 | Navi 21 (Pro W6900X)'
    '73a3' = 'gfx1030 | Navi 21 (Pro W6800)'
    '73a5' = 'gfx1030 | Navi 21 (RX 6950 XT)'
    '73ab' = 'gfx1030 | Navi 21 (Pro W6800X/W6800X Duo)'
    '73ae' = 'gfx1030 | Navi 21 (Pro V620 MxGPU)'
    '73af' = 'gfx1030 | Navi 21 (RX 6900 XT)'
    '73bf' = 'gfx1030 | Navi 21 (RX 6800/6800 XT/6900 XT)'
    '73c3' = 'gfx1031 | Navi 22'
    '73ce' = 'gfx1031 | Navi 22 (SRIOV MxGPU)'
    '73df' = 'gfx1031 | Navi 22 (RX 6700/6750 XT/6800M/6850M XT)'
    '73e0' = 'gfx1032 | Navi 23'
    '73e1' = 'gfx1032 | Navi 23 (Pro W6600M)'
    '73e3' = 'gfx1032 | Navi 23 (Pro W6600)'
    '73ef' = 'gfx1032 | Navi 23 (RX 6650 XT/6700S/6800S)'
    '73ff' = 'gfx1032 | Navi 23 (RX 6600/6600 XT/6600M)'
    '73e2' = 'gfx1032 | Navi 23 (RX 6600 OEM)'
    '73f0' = 'gfx1032 | Navi 23 (RX 6650 XT OEM)'
    '163f' = 'gfx1033 | Van Gogh'
    '7421' = 'gfx1034 | Navi 24 (Pro W6500M)'
    '7422' = 'gfx1034 | Navi 24 (Pro W6400)'
    '7423' = 'gfx1034 | Navi 24 (Pro W6300/W6300M)'
    '7424' = 'gfx1034 | Navi 24 (RX 6300)'
    '743f' = 'gfx1034 | Navi 24 (RX 6400/6500 XT/6500M)'
    '1681' = 'gfx1035 | Rembrandt (680M/660M)'
    '1506' = 'gfx1036 | Mendocino (610M)'

    # RDNA1 - Navi10/12/14
    '7310' = 'gfx1010 | Navi 10 (Pro W5700X)'
    '7312' = 'gfx1010 | Navi 10 (Pro W5700)'
    '7319' = 'gfx1010 | Navi 10 (Pro 5700 XT)'
    '731b' = 'gfx1010 | Navi 10 (Pro 5700)'
    '731f' = 'gfx1010 | Navi 10 (RX 5600/5700 series)'
    '7360' = 'gfx1011 | Navi 12 (Pro 5600M/V520/BC-160)'
    '7362' = 'gfx1011 | Navi 12 (Pro V520/V540)'
    '7340' = 'gfx1012 | Navi 14 (RX 5500/5500M/Pro 5300)'
    '7341' = 'gfx1012 | Navi 14 (Pro W5500)'
    '7347' = 'gfx1012 | Navi 14 (Pro W5500M)'
    '734f' = 'gfx1012 | Navi 14 (Pro W5300M)'

    # Data Center / Enterprise
    '74a0' = 'gfx942 | Aqua Vanjaram (Instinct MI300A)'
    '74a1' = 'gfx942 | Aqua Vanjaram (Instinct MI300X)'
    '74a2' = 'gfx942 | Aqua Vanjaram (Instinct MI308X)'
    '74a5' = 'gfx942 | Aqua Vanjaram (Instinct MI325X)'
    '74a9' = 'gfx942 | Aqua Vanjaram (Instinct MI300X HF)'
    '74b5' = 'gfx942 | Aqua Vanjaram (Instinct MI300X VF)'
    '74b9' = 'gfx942 | Aqua Vanjaram (Instinct MI325X VF)'
    '74bd' = 'gfx942 | Aqua Vanjaram (Instinct MI300X HF)'
    '75a0' = 'gfx950 | Aqua Vanjaram (Instinct MI350X)'
    '75a3' = 'gfx950 | Aqua Vanjaram (Instinct MI355X)'

    # GCN5 / Vega
    '6860' = 'gfx900 | Vega 10 (Instinct MI25/V340/V320)'
    '6861' = 'gfx900 | Vega 10 (Pro WX 9100)'
    '6862' = 'gfx900 | Vega 10 (Pro SSG)'
    '6863' = 'gfx900 | Vega 10 (Vega Frontier Edition)'
    '6864' = 'gfx900 | Vega 10 (Pro V340/Instinct MI25x2)'
    '6867' = 'gfx900 | Vega 10 (Pro Vega 56)'
    '6868' = 'gfx900 | Vega 10 (Pro WX 8100/8200)'
    '6869' = 'gfx900 | Vega 10 (Pro Vega 48)'
    '686b' = 'gfx900 | Vega 10 (Pro Vega 64X)'
    '686c' = 'gfx900 | Vega 10 (Instinct MI25 MxGPU)'
    '687f' = 'gfx900 | Vega 10 (RX Vega 56/64)'
    '66a0' = 'gfx906 | Vega 20 (Pro/Instinct)'
    '66a1' = 'gfx906 | Vega 20 (Pro VII/Instinct MI50)'
    '66a3' = 'gfx906 | Vega 20 (Pro Vega II/Vega II Duo)'
    '66a7' = 'gfx906 | Vega 20 (Pro Vega 20)'
    '66af' = 'gfx906 | Vega 20 (Radeon VII)'
    '738c' = 'gfx908 | Arcturus (Instinct MI100)'
    '738e' = 'gfx908 | Arcturus (Instinct MI100)'
    '7408' = 'gfx90a | Aldebaran (Instinct MI250X)'
    '740c' = 'gfx90a | Aldebaran (Instinct MI250X/MI250)'
    '740f' = 'gfx90a | Aldebaran (Instinct MI210)'
}

# Marketing-name fallback table, one row per gfx target, formatted as
#   'name fragment, fragment, ... | gfx code | architecture label'
# Fragments are matched case-insensitively as substrings of the adapter name.
$GfxNameToGfx = @(
  'rx 9060 | gfx1200 | RDNA 4'
  'rx 9070, r9700, r9600 | gfx1201 | RDNA 4'
  '890m, 880m | gfx1150 | Strix Point'
  '8060s, 8050s, 8040s | gfx1151 | Strix Halo'
  '860m, 840m, 820m | gfx1152 | Krackan Point'
  'rx 7900, w7900, w7800 | gfx1100 | RDNA 3'
  'rx 7800, rx 7700, w7700 | gfx1101 | RDNA 3'
  'rx 7700s, rx 7650, rx 7600, w7600, w7500, rx 7400, w7400 | gfx1102 | RDNA 3'
  '780m, 760m, 740m | gfx1103 | RDNA 3'
  'rx 6950, rx 6900, rx 6800, w6800, v620 | gfx1030 | RDNA 2'
  'rx 6750, rx 6700, rx 6800m, rx 6700m, rx 6800s, rx 6700s | gfx1031 | RDNA 2'
  'rx 6650, rx 6600, w6600, rx 6650m, rx 6600m, rx 6600s | gfx1032 | RDNA 2'
  'van gogh, amd custom apu 0405 | gfx1033 | RDNA 2'
  'rx 6550, rx 6500, rx 6450, rx 6400, w6500, w6400, rx 6300, w6300, rx 6500m, rx 6450m, rx 6300m, rx 6550m, rx 6550s | gfx1034 | RDNA 2'
  '680m, 660m | gfx1035 | RDNA 2'
  '610m | gfx1036 | RDNA 2'
  'rx 5700, rx 5600 | gfx1010 | RDNA 1'
  'radeon pro v520 | gfx1011 | RDNA 1 (Navi 12)'
  'rx 5500 | gfx1012 | RDNA 1 (Navi 14)'
  'radeon pro vii | gfx906 | Radeon Pro VII / Vega 20'
  'mi300a, mi300x, mi325x | gfx942 | MI300/MI325'
  'mi350x, mi355x | gfx950 | MI350/MI355'
  'rx vega, vega 64, vega 56, vega frontier | gfx900 | Vega 10 / GCN5'
  'radeon vii, vega 20 | gfx906 | Vega 20 / GCN5'
  'instinct mi100 | gfx908 | Arcturus / MI100'
  'instinct mi200, instinct mi210, instinct mi250 | gfx90a | Aldebaran / MI200'
)

# Flatten the name table into one pattern list, longest pattern first, so a
# more specific fragment always wins over a shorter one that also matches
# ("rx 6800m" -> gfx1031, not the "rx 6800" gfx1030 entry; "rx 7700s" ->
# gfx1102, not the "rx 7700" gfx1101 entry). Ties keep table order.
$_patternOrder = 0
$_gfxNamePatterns = foreach ($row in $GfxNameToGfx) {
  # 'frag, frag | gfx code | label'
  $cols = $row -split '\|'
  foreach ($frag in ($cols[0] -split ',')) {
    [pscustomobject]@{
      Pattern = $frag.Trim().ToLower()
      Gfx     = $cols[1].Trim()
      Arch    = $cols[2].Trim()
      Order   = $_patternOrder
    }
    $_patternOrder++
  }
}
$GfxNamePatterns = @($_gfxNamePatterns |
  Sort-Object @{ Expression = { $_.Pattern.Length }; Descending = $true },
              @{ Expression = { $_.Order }; Ascending = $true })

# gfx codes that actually have a win_amd64 ROCm torch/torchvision wheel on the
# AMD index used in step 3. Detected targets outside this list get a clear
# message instead of a confusing resolver error. (Checked against
# https://rc.repo.amd.com/rocm/whl-next/; gfx1250/gfx942/gfx950 are Linux-only
# and gfx900/gfx906 (Vega, Radeon VII) are not on the index at all.)
$GfxWindowsWheels = @(
  'gfx908', 'gfx90a',
  'gfx1010', 'gfx1011', 'gfx1012',
  'gfx1030', 'gfx1031', 'gfx1032', 'gfx1033', 'gfx1034', 'gfx1035', 'gfx1036',
  'gfx1100', 'gfx1101', 'gfx1102', 'gfx1103',
  'gfx1150', 'gfx1151', 'gfx1152', 'gfx1153',
  'gfx1200', 'gfx1201'
)

function Get-AmdGpu {
  # AMD display adapters as @(Name, PNPDeviceID, DevId) records. AMD's PCI
  # vendor id (VEN_1002) is the reliable test; the name checks are only a
  # fallback for the rare case where PNPDeviceID is missing.
  $gpus = @()
  try {
    $adapters = @(Get-CimInstance Win32_VideoController -ErrorAction Stop |
      Select-Object -Property Name, PNPDeviceID)
  } catch {
    Write-Host "  Win32_VideoController query failed: $($_.Exception.Message)" -ForegroundColor Yellow
    return $gpus
  }
  foreach ($a in $adapters) {
    $name = ([string]$a.Name).Trim()
    $pnp  = ([string]$a.PNPDeviceID).Trim()
    if (-not (($pnp -match '(?i)VEN_1002') -or ($name -match '(?i)radeon') -or
              ($name -match '(?i)(^|[^a-z])amd([^a-z]|$)'))) { continue }
    $devId = ''
    if ($pnp -match '(?i)DEV_([0-9a-f]{4})') { $devId = $Matches[1].ToLower() }
    $gpus += [pscustomobject]@{ Name = $name; Pnp = $pnp; DevId = $devId }
  }
  # Plain `return $gpus` (not `,$gpus`): callers wrap in @(), which turns "no
  # output" into a zero-length array, but a nested array into one bogus entry.
  return $gpus
}

function Find-GfxTarget {
  param([object[]]$Gpus)

  # Normalize (and drop empty entries) so the three passes below can assume
  # Name/DevId are strings.
  $list = @(foreach ($g in $Gpus) {
    if (-not $g) { continue }
    [pscustomobject]@{ Name = ([string]$g.Name).Trim(); DevId = ([string]$g.DevId).ToLower() }
  })
  if ($list.Count -eq 0) { return $null }

  # 1. PCI device id - checked for every GPU first, so a discrete card wins
  #    over an iGPU that only reports "AMD Radeon(TM) Graphics".
  foreach ($g in $list) {
    if ($g.DevId -and $PciDevToGfx.ContainsKey($g.DevId)) {
      $cols = $PciDevToGfx[$g.DevId] -split '\|', 2
      return [pscustomobject]@{ Gfx = $cols[0].Trim(); Arch = $cols[1].Trim(); Name = $g.Name;
                                How = "PCI device id $($g.DevId)" }
    }
  }

  # 2. A gfx code baked into the driver-reported name.
  foreach ($g in $list) {
    $m = [regex]::Match($g.Name, '(?i)gfx[0-9]{3}[0-9a-z]?')
    if ($m.Success) {
      return [pscustomobject]@{ Gfx = $m.Value.ToLower(); Arch = 'reported by driver';
                                Name = $g.Name; How = 'gfx code in device name' }
    }
  }

  # 3. Marketing name fragments (longest match first).
  foreach ($g in $list) {
    $lower = $g.Name.ToLower()
    foreach ($p in $GfxNamePatterns) {
      if ($lower.Contains($p.Pattern)) {
        return [pscustomobject]@{ Gfx = $p.Gfx; Arch = $p.Arch; Name = $g.Name;
                                  How = "name match '$($p.Pattern)'" }
      }
    }
  }

  return $null
}

function Get-GfxArch {
  # Return the gfx code to install torch for, or $null after printing the
  # reason (and the suggested fix); the caller then exits.

  # An explicit GFX_ARCH env var always wins over auto-detection.
  if ($env:GFX_ARCH) {
    $explicit = $env:GFX_ARCH.Trim().ToLower()   # wheel extras are lowercase
    Write-Host "GFX_ARCH=$explicit (from environment)"
    if ($GfxWindowsWheels -notcontains $explicit) {
      Write-Host "Warning: no Windows ROCm wheel for $explicit on the AMD index; the install below may fail." -ForegroundColor Yellow
    }
    return $explicit
  }

  $gpus = @(Get-AmdGpu)
  if ($gpus.Count -eq 0) {
    Write-Host ""
    Write-Host "Error: no AMD GPU found." -ForegroundColor Red
    Write-Host "Install (or repair) your AMD graphics driver, or pick the target by hand:"
    Write-Host '  $env:GFX_ARCH = "gfx1151"   # then re-run thenoise.bat'
    Write-Host "Supported targets: $($GfxWindowsWheels -join ', ')  (gfx1151 = Strix Halo)"
    return $null
  }

  Write-Host "Found $($gpus.Count) AMD GPU(s):"
  foreach ($g in $gpus) { Write-Host "  - $($g.Name)  [$($g.Pnp)]" }

  $hit = Find-GfxTarget -Gpus $gpus
  if (-not $hit) {
    Write-Host ""
    Write-Host "Error: could not map the GPU(s) above to a gfx target." -ForegroundColor Red
    Write-Host "Pick it by hand and re-run:"
    Write-Host '  $env:GFX_ARCH = "gfx1151"   # thenoise.bat'
    Write-Host "Supported targets: $($GfxWindowsWheels -join ', ')  (gfx1151 = Strix Halo)"
    return $null
  }

  if ($GfxWindowsWheels -notcontains $hit.Gfx) {
    Write-Host ""
    Write-Host "Error: detected $($hit.Name) -> $($hit.Arch) ($($hit.Gfx)), which has no" -ForegroundColor Red
    Write-Host "Windows ROCm wheel on the AMD index. thenoise supports:"
    Write-Host "  $($GfxWindowsWheels -join ', ')"
    return $null
  }

  Write-Host "Auto-detected GFX_ARCH=$($hit.Gfx) - $($hit.Arch), via $($hit.How) (set `$env:GFX_ARCH to override)"
  return $hit.Gfx
}

# ---- 1. Check that uv is available ----------------------------------------
if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
  Write-Host @"
Error: uv is not installed.

Install it with:
  powershell -ExecutionPolicy Bypass -Command "irm https://astral.sh/uv/install.ps1 | iex"

Then open a new terminal and try again.
"@
  exit 1
}

# ---- 2. Create the venv if it does not exist ------------------------------
if (-not (Test-Path $VenvDir)) {
  Write-Host "Creating virtual environment ($VenvDir) with Python 3.13 ..."
  # --managed-python forces uv to use its own standalone CPython build rather
  # than a system python3.13. (On Windows, Triton JIT is unavailable anyway;
  # this still gives a self-contained, sudo-free setup.)
  & uv venv $VenvDir --python 3.13 --managed-python
  if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
}

# ---- 3. Install torch (ROCm build) ----------------------------------------
$Py = Join-Path $VenvDir "Scripts\python.exe"
if (-not (Test-Path $Py)) {
  Write-Host "Error: venv python not found: $Py" -ForegroundColor Red
  exit 1
}

$torchInstalled = $false
try { & $Py -c "import torch" 2>$null; if ($LASTEXITCODE -eq 0) { $torchInstalled = $true } } catch {}

if (-not $torchInstalled) {
  Write-Host "Installing ROCm torch ..."

  $GFX = Get-GfxArch
  if (-not $GFX) { exit 1 }   # Get-GfxArch has already printed the reason and the fix

  & uv pip install `
    "torch[device-$GFX]==2.14.0+rocm10.1.0rc2" `
    "torchvision[device-$GFX]==0.29.0a0+rocm10.1.0rc2" `
    "triton-windows<3.9" `
    --extra-index-url https://pypi.org/simple/ `
    --index-url https://rc.repo.amd.com/rocm/whl-next/
  if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
}

# ---- 4. Install the project in editable mode ------------------------------
& uv pip install -e $ProjectDir
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

# ---- 5. Set ROCm-specific environment variables ---------------------------
$env:TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL = "1"
$env:MIOPEN_FIND_MODE = "FAST"
$env:TORCH_COMPILE_DISABLE = "1"
$env:TORCHDYNAMO_DISABLE = "1"

# ---- 6. Launch the project, forwarding all arguments ----------------------
& $Py -m thenoise @ForwardArgs
exit $LASTEXITCODE
