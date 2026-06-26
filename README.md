# CUDA Pairwise Alignment — Smith‑Waterman & Needleman‑Wunsch

A **didactic, heavily‑commented CUDA C++ implementation** of the two classic
pairwise sequence‑alignment algorithms:

- **Needleman–Wunsch** (1970) — *global* alignment (align two sequences end‑to‑end)
- **Smith–Waterman** (1981) — *local* alignment (find the best shared sub‑region)

This repository is built to be **study material**. Every file is annotated to
the point of being a tutorial in disguise: what each function does, what every
variable is for, why the logic is shaped the way it is, and how the pieces fit
together. If you are learning CUDA, read the sources top‑to‑bottom — they are
meant to be read, not just run.

> Verified on an **NVIDIA RTX 2080 SUPER** (compute capability `sm_75`) with
> **CUDA 13.3** and **Visual Studio 2026** on Windows 11. All 980 correctness
> checks pass; the GPU output is bit‑for‑bit identical to the CPU reference.

---

## Why this project exists

Dynamic‑programming (DP) alignment is the *perfect* first non‑trivial CUDA
problem, because it forces you to confront the central question of GPU
programming head‑on:

> **The DP recurrence looks sequential — each cell depends on its neighbours.
> So how do you make it parallel?**

This project answers that question **twice**, using two completely different
parallelisation strategies, so you can compare them:

| Strategy | File | Idea | Good for |
|---|---|---|---|
| **Intra‑sequence (wavefront)** | [`src/gpu_wavefront.cu`](src/gpu_wavefront.cu) | One alignment, many threads sweeping the matrix along its **anti‑diagonals** | One or a few *large* alignments |
| **Inter‑sequence (batched)** | [`src/gpu_batched.cu`](src/gpu_batched.cu) | Many alignments, **one thread each**, all at once | Scanning a *database* of many pairs |

Seeing both side by side — and measuring where each one wins — is the whole
point.

---

## What you get

- A clean, serial **CPU reference** ([`src/cpu_reference.cpp`](src/cpu_reference.cpp))
  that is the "golden" correctness oracle and performance baseline.
- The **wavefront GPU kernel** (intra‑sequence parallelism) with full traceback.
- The **batched GPU kernel** (inter‑sequence parallelism) demonstrating
  rolling‑row memory and **memory coalescing**.
- A **demo** ([`demo/demo_main.cpp`](demo/demo_main.cpp)) that prints DP
  matrices, reconstructs alignments, verifies GPU == CPU, and benchmarks both.
- An **exhaustive test harness** ([`tests/verify_main.cpp`](tests/verify_main.cpp))
  with algebraic invariants, hand‑computed cases, and randomized fuzzing.
- **Three build paths**, all verified: a Visual Studio solution, a CMake build,
  and bare `nvcc` scripts.
- A set of **guides** in [`docs/`](docs/) covering the biology, the math, and
  the CUDA parallelisation in depth.

---

## Repository layout

```
SW_NW_PWA/
├── README.md                       ← you are here
├── LICENSE                         ← MIT
├── CMakeLists.txt                  ← build path #2 (CMake)
├── CudaPairwiseAlignment.sln       ← build path #1 (Visual Studio 2026)
├── CudaPairwiseAlignment.vcxproj   ← ...the project it references
├── include/                        ← public headers (interfaces + heavy docs)
│   ├── alignment_common.h          ← shared types: scoring, enums, results
│   ├── cpu_reference.h
│   ├── cuda_utils.cuh              ← CUDA error‑checking + GPU timer
│   ├── gpu_wavefront.cuh
│   └── gpu_batched.cuh
├── src/                            ← implementations
│   ├── alignment_common.cpp        ← formatting + reproducible RNG
│   ├── cpu_reference.cpp           ← serial NW + SW (the oracle)  ★ read first
│   ├── gpu_wavefront.cu            ← anti‑diagonal wavefront kernel ★ centerpiece
│   └── gpu_batched.cu              ← one‑thread‑per‑pair batched kernel
├── demo/
│   └── demo_main.cpp               ← the guided showcase (run this)
├── tests/
│   └── verify_main.cpp             ← the correctness test suite
├── scripts/                        ← build path #3 (bare nvcc)
│   ├── build_nvcc.ps1
│   └── build_nvcc.bat
└── docs/
    ├── 00_INTRODUCTION.md          ← the biology, gently
    ├── 01_ALGORITHMS.md            ← the math: recurrences + worked examples
    ├── 02_CUDA_PARALLELIZATION.md  ← wavefront vs batched, memory, performance
    ├── 03_BUILD_AND_RUN.md         ← detailed build/troubleshooting guide
    └── changelog/                  ← one "what was added" note per push
        └── 2026-06-25-initial-release.md
```

**Suggested reading order for learning:**
1. [`docs/00_INTRODUCTION.md`](docs/00_INTRODUCTION.md) and
   [`docs/01_ALGORITHMS.md`](docs/01_ALGORITHMS.md) — understand the problem.
2. [`include/alignment_common.h`](include/alignment_common.h) — the vocabulary.
3. [`src/cpu_reference.cpp`](src/cpu_reference.cpp) — the algorithm, sequentially.
4. [`docs/02_CUDA_PARALLELIZATION.md`](docs/02_CUDA_PARALLELIZATION.md) — the GPU strategy.
5. [`src/gpu_wavefront.cu`](src/gpu_wavefront.cu) then
   [`src/gpu_batched.cu`](src/gpu_batched.cu) — the two kernels.
6. [`demo/demo_main.cpp`](demo/demo_main.cpp) — see it all tie together.

---

## Quick start

### Prerequisites
- An NVIDIA GPU + the **CUDA Toolkit** (this repo was tested with 13.3; anything
  reasonably recent works). Check with `nvcc --version`.
- Find your GPU's **compute capability** (e.g. `7.5` for RTX 2080, `8.6` for
  RTX 30‑series, `8.9` for RTX 40‑series). The demo prints it; the default build
  targets `sm_75`. Change it if your GPU differs (see below).

### Option 1 — Visual Studio 2026
```
Open CudaPairwiseAlignment.sln, pick "Release | x64", press Ctrl+Shift+B.
Run x64\Release\CudaPairwiseAlignment.exe
```
For VS 2022 change `<PlatformToolset>v145</PlatformToolset>` to `v143` and the
two `CUDA 13.3.props/.targets` imports to your installed CUDA version.

### Option 2 — CMake (generates the VS solution for you, or builds with Ninja)
```bat
cmake -S . -B build -G "Visual Studio 18 2026" -A x64
cmake --build build --config Release
build\Release\demo.exe
build\Release\verify.exe
```
Target a different GPU: add `-DCMAKE_CUDA_ARCHITECTURES=86` (Ampere), `89` (Ada), etc.

### Option 3 — bare `nvcc`
```powershell
powershell -ExecutionPolicy Bypass -File scripts\build_nvcc.ps1        # sm_75
powershell -ExecutionPolicy Bypass -File scripts\build_nvcc.ps1 -Arch 86
.\build\demo.exe
.\build\verify.exe
```

See [`docs/03_BUILD_AND_RUN.md`](docs/03_BUILD_AND_RUN.md) for the full guide and
troubleshooting.

---

## Sample output (abridged)

```
--- NW: GCATGCU vs GATTACA ---
  DP score matrix (B across the top, A down the side):
        *   G   A   T   T   A   C   A
   *    0  -2  -4  -6  -8 -10 -12 -14
   G   -2   2   0  -2  -4  -6  -8 -10
   C   -4   0   1  -1  -3  -5  -4  -6
   A   -6  -2   2   0  -2  -1  -3  -2
   T   -8  -4   0   4   2   0  -2  -4
   G  -10  -6  -2   2   3   1  -1  -3
   C  -12  -8  -4   0   1   2   3   1
   U  -14 -10  -6  -2  -1   0   1   2
  Optimal score: 2
GCATGCU
|..|.|.
GATTACA
  [VERIFY] CPU and GPU agree exactly.  OK

--- SW: shared motif (local) ---
  Optimal score: 12
GTTGAC
||||||
GTTGAC
  [VERIFY] CPU and GPU agree exactly.  OK

  Scoring 20000 independent random pairs (one GPU thread per pair):
    CPU time           :   364.68 ms  (54842 alignments/sec)
    GPU time           :    18.12 ms  (1103455 alignments/sec, kernel only)
    correctness        : OK (all 20000 scores match)
    GPU speedup        : 20.1x over single-threaded CPU
```

Note the honest result: for **one** moderate alignment the wavefront GPU is
often *slower* than the CPU (per‑diagonal launch overhead). The GPU's advantage
shows up on the **batched** workload, where there is enough independent work to
fill it. This trade‑off is explained in detail in the code and docs.

---

## Scope, simplifications, and exercises

This is a teaching implementation. It deliberately favours clarity, and each
simplification is a natural next step to explore:

- **Linear gap penalty** (not affine). Real tools use the Gotoh affine‑gap model
  (open + extend) with three matrices — a great extension.
- **Full O(m·n) matrix** materialised for traceback. Hirschberg's algorithm does
  global traceback in O(min(m,n)) space.
- **One kernel launch per anti‑diagonal.** Tiling + a persistent kernel with
  cooperative‑groups grid synchronisation would slash launch overhead.
- **Uncoalesced matrix access** in the wavefront kernel. A diagonal‑major or
  tiled layout fixes it.
- **Match/mismatch scalar scoring.** Swap in a substitution matrix (BLOSUM62) for
  proteins by editing one function.

Each of these is called out at its relevant spot in the source comments.

---

## Documentation index

- [Introduction — the biology, gently](docs/00_INTRODUCTION.md)
- [Algorithms — recurrences and worked examples](docs/01_ALGORITHMS.md)
- [CUDA parallelisation — wavefront vs batched, memory, performance](docs/02_CUDA_PARALLELIZATION.md)
- [Build & run — every path, plus troubleshooting](docs/03_BUILD_AND_RUN.md)
- [Changelog](docs/changelog/) — a per‑push note describing what was added

---

## License

MIT — see [LICENSE](LICENSE).

## References

- S. B. Needleman, C. D. Wunsch (1970). *A general method applicable to the
  search for similarities in the amino acid sequence of two proteins.* JMB 48(3).
- T. F. Smith, M. S. Waterman (1981). *Identification of common molecular
  subsequences.* JMB 147(1).
- O. Gotoh (1982). *An improved algorithm for matching biological sequences.*
  JMB 162(3). (affine gaps)
- D. S. Hirschberg (1975). *A linear space algorithm for computing maximal common
  subsequences.* CACM 18(6). (linear‑space traceback)
