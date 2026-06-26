# Changelog — 2026‑06‑25 — Initial release

This is the first push: a complete, working, heavily‑commented CUDA C++
implementation of Smith‑Waterman and Needleman‑Wunsch pairwise alignment, with
three build systems, a demo, a test suite, and full documentation.

> **Per‑push changelog policy.** Every future push to this repository adds a new
> dated file in `docs/changelog/` describing exactly what changed and why. This
> file documents what the repository contained at creation.

---

## What was added

### Core library (`include/` + `src/`)
| File | What it provides |
|---|---|
| `include/alignment_common.h` | Shared vocabulary: `ScoringScheme`, `AlignmentKind` (Global/Local), `AlignmentResult`, the `cell_t` matrix type, the `ALN_HD` host/device macro, and the `substitution_score` helper. |
| `src/alignment_common.cpp` | `format_alignment` (the pretty 3‑line block) and `random_sequence` (a deterministic xorshift DNA generator for reproducible tests). |
| `include/cpu_reference.h` / `src/cpu_reference.cpp` | The **serial golden model**: `cpu_align` (full matrix + traceback), `cpu_score_only` (rolling‑rows), the shared `traceback_from_matrix`, and `cpu_compute_matrix` (for visualisation). |
| `include/cuda_utils.cuh` | `CUDA_CHECK` / `CUDA_CHECK_KERNEL` error‑checking macros and the `GpuTimer` (CUDA‑event stopwatch). |
| `include/gpu_wavefront.cuh` / `src/gpu_wavefront.cu` | **Strategy A — intra‑sequence wavefront.** One alignment parallelised across its anti‑diagonals; one kernel launch per diagonal; host‑side traceback reused from the CPU. Plus `gpu_print_device_info`. |
| `include/gpu_batched.cuh` / `src/gpu_batched.cu` | **Strategy B — inter‑sequence batched.** Thousands of pairs, one thread each, rolling rows in a **coalesced** global‑memory layout. |

### Demo & tests
- `demo/demo_main.cpp` — the guided showcase: device info, two small alignments
  with printed DP matrices, CPU‑vs‑GPU verification, single‑alignment timing, and
  a 20,000‑pair batched throughput benchmark.
- `tests/verify_main.cpp` — a four‑layer correctness suite (algebraic
  invariants, hand‑computed cases, randomized CPU‑vs‑GPU wavefront fuzzing across
  several scoring schemes, and batched‑vs‑CPU scoring). **980 checks, all pass.**

### Build systems (all three verified to build *and run*)
- `CudaPairwiseAlignment.sln` + `.vcxproj` + `.vcxproj.filters` — Visual Studio
  2026 (toolset `v145`) with the CUDA 13.3 MSBuild integration. Builds the demo.
- `CMakeLists.txt` — portable CMake build with two targets (`demo`, `verify`);
  can generate the VS solution or build with Ninja.
- `scripts/build_nvcc.ps1` / `scripts/build_nvcc.bat` — the bare‑`nvcc` path.

### Documentation (`docs/` + `README.md`)
- `README.md` — overview, repo map, suggested reading order, quick‑start for all
  three builds, sample output, and the list of simplifications/exercises.
- `docs/00_INTRODUCTION.md` — the biology of sequence alignment, gently.
- `docs/01_ALGORITHMS.md` — the recurrences, initialisation, traceback, and two
  fully worked matrix examples (matching the demo's output).
- `docs/02_CUDA_PARALLELIZATION.md` — the wavefront and batched strategies,
  memory coalescing, and honest performance analysis.
- `docs/03_BUILD_AND_RUN.md` — detailed build instructions and a troubleshooting
  section.
- `LICENSE` (MIT), `.gitignore` (excludes all build artifacts).

---

## Verification performed before this push

- Built with **nvcc** (CLI), **MSBuild** (the `.sln`), and **CMake** (VS 2026
  generator) — all three succeed.
- Ran `verify.exe`: **980 / 980 checks pass**, including 320 randomized
  CPU‑vs‑GPU wavefront cases (full alignment strings compared, not just scores)
  and 10,000 batched scores across NW and SW.
- Ran `demo.exe` on an RTX 2080 SUPER: DP matrices correct, the Smith‑Waterman
  example recovers the shared `GTTGAC` motif (score 12), CPU and GPU agree
  exactly, and the batched workload reaches **16–22× speedup** over one CPU core.
- Ran an **adversarial multi-agent code review** across five dimensions (wavefront
  CUDA correctness, batched CUDA correctness, algorithm math, didactic accuracy of
  the comments/docs, and build portability), with independent verification of every
  high-severity finding. It confirmed **no logic bugs** in the kernels, the
  algorithm, or the traceback (the last brute-forced over 200k pairs). The few real
  findings were documentation/config issues, all fixed before this push: a couple of
  numeric/labelling slips in the worked examples, a complexity-bound wording
  inconsistency (the score-only path now genuinely achieves `O(min(m,n))`), a
  defensive guard added to the traceback's LEFT branch, a documented precondition on
  the batched kernel's 32-bit offsets, and a corrected `.gitignore` (its directory
  patterns had been silently disabled by trailing inline comments, which the
  gitignore format does not support).

## Known simplifications (intentional; documented in‑code as exercises)
- Linear (not affine/Gotoh) gap model.
- Full `O(m·n)` matrix materialised for wavefront traceback (no Hirschberg).
- One kernel launch per anti‑diagonal (no tiling / persistent kernel).
- Uncoalesced matrix access in the wavefront kernel (the batched kernel *is*
  coalesced, by contrast — deliberately, to teach both).
- Scalar match/mismatch scoring (no BLOSUM/PAM substitution matrix).
