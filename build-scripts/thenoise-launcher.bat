@echo off
rem ===========================================================================
rem thenoise.bat - Portable launcher for the Windows thenoise bundle.
rem Runs the bundled standalone CPython against the bundled ROCm PyTorch with
rem no system dependency. Sets PATH to the bundled ROCm runtime libs so the
rem HIP DLLs resolve (Windows uses PATH, not LD_LIBRARY_PATH), and disables
rem torch.compile because Triton is Linux-only on ROCm for now.
rem
rem This file is copied to the bundle root as thenoise.bat by
rem build-scripts/build_portable.ps1.
rem ===========================================================================
setlocal
set "ROOT=%~dp0"
set "SP=%ROOT%Lib\site-packages"
rem With Windows long paths off, hipBLASLt/rocBLAS crash on kernel files whose
rem path exceeds 259 characters (ROCm/rocm-libraries#9962).
if defined THENOISE_ALLOW_DEEP_PATH goto :path_ok
set "_TN_TAIL=%ROOT:~88%"
if not defined _TN_TAIL goto :path_ok
set "_TN_LP="
for /f "tokens=3" %%v in ('reg query "HKLM\SYSTEM\CurrentControlSet\Control\FileSystem" /v LongPathsEnabled 2^>nul ^| find /i "LongPathsEnabled"') do set "_TN_LP=%%v"
if /i "%_TN_LP%"=="0x1" goto :path_ok
>&2 echo thenoise: the install folder is too long for the GPU libraries:
>&2 echo   "%ROOT%"
>&2 echo Move it to a shorter path such as "%LOCALAPPDATA%\Programs\thenoise",
>&2 echo or enable Windows long paths. Set THENOISE_ALLOW_DEEP_PATH=1 to skip
>&2 echo this check for CPU-only use.
exit /b 1
:path_ok
set "PATH=%SP%\_rocm_sdk_core\bin;%SP%\_rocm_sdk_core\lib;%SP%\_rocm_sdk_core\lib\llvm\lib;%SP%\_rocm_sdk_libraries\bin;%SP%\torch\lib;%SP%;%ROOT%;%PATH%"
set "TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1"
set "MIOPEN_FIND_MODE=FAST"
set "TORCH_COMPILE_DISABLE=1"
set "TORCHDYNAMO_DISABLE=1"
"%ROOT%python.exe" -s -m thenoise %*
exit /b %errorlevel%
