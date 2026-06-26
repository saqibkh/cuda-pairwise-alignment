# 03 — Build & Run (and troubleshooting)

There are **three** ways to build this project. They compile the identical
sources; pick whichever you like. This page covers each in detail plus the
problems you are most likely to hit.

> **Reference environment** (everything below was verified on it):
> Windows 11 · NVIDIA RTX 2080 SUPER (`sm_75`) · CUDA Toolkit 13.3 ·
> Visual Studio Community 2026 (toolset `v145`, MSVC 14.51) · CMake 4.3.

## 0. Prerequisites & finding your GPU architecture

1. Install the **CUDA Toolkit** and confirm `nvcc` works:
   ```
   nvcc --version
   ```
2. Confirm the driver sees your GPU:
   ```
   nvidia-smi
   ```
3. Determine your GPU's **compute capability** ("`sm_XX`"). Quick references:
   | GPU family | Example cards | Arch |
   |---|---|---|
   | Turing | RTX 2080/2070/2060, GTX 1660, T4 | `75` |
   | Ampere | RTX 3090/3080/3070/3060, A100 | `86` (A100: `80`) |
   | Ada Lovelace | RTX 4090/4080/4070 | `89` |
   | Hopper | H100 | `90` |
   The **demo prints it for you** under "GPU device". The default build targets
   `sm_75`; change it to match your card (instructions per‑path below).

---

## 1. Visual Studio 2026

1. Double‑click **`CudaPairwiseAlignment.sln`**.
2. Set the configuration dropdowns to **`Release`** and **`x64`**.
3. Build: **Build → Build Solution** (`Ctrl+Shift+B`).
4. Run `x64\Release\CudaPairwiseAlignment.exe` (or press `Ctrl+F5`).

**Changing the target GPU:** open `CudaPairwiseAlignment.vcxproj`, find both
`<CodeGeneration>compute_75,sm_75</CodeGeneration>` lines, change `75` to your
arch. (Project → Properties → CUDA C/C++ → Device → Code Generation does the same
in the GUI.)

**Using a different Visual Studio:**
- VS 2022 → `<PlatformToolset>v143</PlatformToolset>`
- VS 2019 → `v142`

**Using a different CUDA version:** the project imports
`CUDA 13.3.props` and `CUDA 13.3.targets`. Change `13.3` in **both** lines to your
installed version. Those files come from the Toolkit's *Visual Studio
integration*, which must be installed (it is, by default, when you install CUDA
with VS present). They live at:
```
<VS>\MSBuild\Microsoft\VC\v180\BuildCustomizations\CUDA <ver>.props/.targets
```

> The `.sln` builds only the **demo**. The **verify** test program has its own
> `main()` and is built via CMake or the nvcc scripts (two `main()`s can't live in
> one executable).

---

## 2. CMake

CMake can generate the VS solution *or* build directly (Ninja/Makefiles). It
treats CUDA as a first‑class language, so there are no hand‑written nvcc lines.

```bat
:: Configure (generate a VS 2026 solution under build\)
cmake -S . -B build -G "Visual Studio 18 2026" -A x64

:: Build both targets in Release
cmake --build build --config Release

:: Run
build\Release\demo.exe
build\Release\verify.exe
```

Faster single‑config build with **Ninja** (if installed):
```bat
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build build
build\demo.exe
```

**Changing the target GPU:**
```bat
cmake -S . -B build -DCMAKE_CUDA_ARCHITECTURES=86
:: multiple at once (a "fat binary"):
cmake -S . -B build -DCMAKE_CUDA_ARCHITECTURES="75;86;89"
```

---

## 3. Bare `nvcc` scripts

The most transparent option — one `nvcc` command per executable.

```powershell
:: PowerShell
powershell -ExecutionPolicy Bypass -File scripts\build_nvcc.ps1          # sm_75
powershell -ExecutionPolicy Bypass -File scripts\build_nvcc.ps1 -Arch 86 # other arch
```
```bat
:: cmd.exe
scripts\build_nvcc.bat            :: sm_75
scripts\build_nvcc.bat 86         :: other arch
```
Both produce `build\demo.exe` and `build\verify.exe`. The exact command they run,
for reference:
```
nvcc -O2 -std=c++14 -arch=sm_75 -Iinclude ^
     demo\demo_main.cpp src\alignment_common.cpp src\cpu_reference.cpp ^
     src\gpu_wavefront.cu src\gpu_batched.cu -o build\demo.exe
```

---

## What success looks like

`demo.exe` prints your GPU, two small alignments **with their DP matrices**, a
CPU‑vs‑GPU timing comparison, and a batched throughput test — each ending in
`[VERIFY] … OK`. `verify.exe` ends with:
```
=== RESULT: 980 passed, 0 failed ===
ALL TESTS PASSED.
```
Both return exit code `0` on success (so CI can check them).

---

## Troubleshooting

**`nvcc: command not found` / not recognised.**
The Toolkit's `bin` isn't on `PATH`. Open a fresh terminal after installing, or
add `…\CUDA\v13.3\bin` to `PATH`. In Visual Studio this never matters — the
project finds nvcc via the CUDA integration.

**MSBuild: `The imported project "…\BuildCustomizations\CUDA 13.3.props" was not
found.`**
The CUDA↔VS integration for your CUDA version isn't installed, or the version in
the two import lines doesn't match what you have. Re‑run the CUDA installer and
enable "Visual Studio Integration", or edit the `13.3` in the `.vcxproj` to your
version.

**Runtime: `no kernel image is available for execution on the device`.**
You built for the wrong architecture. Set the arch to match your GPU (Section 0)
and rebuild. This is the single most common CUDA beginner error.

**Runtime: `CUDA driver version is insufficient for CUDA runtime version`.**
Your GPU driver is older than the Toolkit expects. Update the NVIDIA driver.

**The GPU wavefront is *slower* than the CPU.**
Expected for a single moderate alignment — see the performance caveats in
[02 — CUDA parallelisation](02_CUDA_PARALLELIZATION.md). The batched test is where
the GPU pulls ahead.

**Mismatch / a `[FAIL]` line in `verify.exe`.**
That would indicate a real bug. The failing case prints its scoring scheme,
algorithm, lengths, and the exact sequences (the RNG is deterministic, so it
reproduces). Please open an issue with that line.

**Out of GPU memory on a huge alignment.**
The wavefront kernel materialises the full `(m+1)×(n+1)` `int` matrix —
~16 GB at 64k×64k. Use shorter sequences, or implement the linear‑space
(Hirschberg) extension noted in the code.
