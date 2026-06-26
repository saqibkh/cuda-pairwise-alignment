# =============================================================================
#  scripts/build_nvcc.ps1   (PowerShell)
# -----------------------------------------------------------------------------
#  The SIMPLEST possible build: just call nvcc directly. No CMake, no Visual
#  Studio project -- handy when you want to see, in one screen, exactly what
#  command turns these sources into runnable programs.
#
#  It builds TWO executables into the build/ folder:
#     build\demo.exe     -- the guided showcase (run this first)
#     build\verify.exe   -- the correctness test suite
#
#  USAGE (from the project root):
#     powershell -ExecutionPolicy Bypass -File scripts\build_nvcc.ps1
#  or, to target a different GPU architecture:
#     powershell -ExecutionPolicy Bypass -File scripts\build_nvcc.ps1 -Arch 86
#
#  PREREQUISITE: nvcc must be on your PATH (installing the CUDA Toolkit and
#  opening a fresh terminal normally does this). Check with:  nvcc --version
# =============================================================================

param(
    # Target GPU compute capability WITHOUT the "sm_" prefix. 75 = RTX 2080 etc.
    [string]$Arch = "75"
)

# Stop on the first error so a failed compile does not look like success.
$ErrorActionPreference = "Stop"

# Resolve paths relative to THIS script, so the build works no matter what the
# current directory is. $PSScriptRoot is the folder containing this .ps1.
$root    = Split-Path -Parent $PSScriptRoot      # project root (one level up)
$include = Join-Path $root "include"
$src     = Join-Path $root "src"
$build   = Join-Path $root "build"

# Make sure the output directory exists.
New-Item -ItemType Directory -Force -Path $build | Out-Null

# The shared core sources both executables need.
$core = @(
    (Join-Path $src "alignment_common.cpp"),
    (Join-Path $src "cpu_reference.cpp"),
    (Join-Path $src "gpu_wavefront.cu"),
    (Join-Path $src "gpu_batched.cu")
)

Write-Host "Building for GPU architecture sm_$Arch ..." -ForegroundColor Cyan

# --- Build the demo ----------------------------------------------------------
# Flag-by-flag:
#   -O2            optimise host code
#   -std=c++14     C++ standard for the host side
#   -arch=sm_$Arch generate device code for this compute capability
#   -I$include     where to find the project headers
#   -o ...         output file
Write-Host "  -> build\demo.exe"
& nvcc -O2 -std=c++14 -arch="sm_$Arch" -I"$include" `
    (Join-Path $root "demo\demo_main.cpp") @core `
    -o (Join-Path $build "demo.exe")

# --- Build the verifier ------------------------------------------------------
Write-Host "  -> build\verify.exe"
& nvcc -O2 -std=c++14 -arch="sm_$Arch" -I"$include" `
    (Join-Path $root "tests\verify_main.cpp") @core `
    -o (Join-Path $build "verify.exe")

Write-Host "Done. Run:" -ForegroundColor Green
Write-Host "    .\build\demo.exe"
Write-Host "    .\build\verify.exe"
