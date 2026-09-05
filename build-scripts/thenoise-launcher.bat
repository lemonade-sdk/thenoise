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
set "PATH=%SP%\_rocm_sdk_core\bin;%SP%\_rocm_sdk_core\lib;%SP%\_rocm_sdk_core\lib\llvm\lib;%SP%\_rocm_sdk_libraries\bin;%SP%\torch\lib;%SP%;%ROOT%;%PATH%"
set "TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1"
set "MIOPEN_FIND_MODE=FAST"
set "TORCH_BLAS_PREFER_HIPBLASLT=1"
set "TORCH_COMPILE_DISABLE=1"
set "TORCHDYNAMO_DISABLE=1"
"%ROOT%python.exe" -s -m thenoise %*
exit /b %errorlevel%
