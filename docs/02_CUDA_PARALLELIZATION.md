# 02 — CUDA Parallelisation: two strategies for the same algorithm

This is the heart of the project. The serial algorithm
([01 — Algorithms](01_ALGORITHMS.md)) fills the DP matrix one cell at a time, in
row order. A GPU has thousands of threads; how do we use them on a recurrence
where every cell depends on its neighbours? There are two good answers, and we
implement both.

---

## Strategy A — Intra‑sequence: the anti‑diagonal **wavefront**

**File:** [`src/gpu_wavefront.cu`](../src/gpu_wavefront.cu) ·
parallelises **one** alignment across many threads.

### The dependency, and the insight

Cell `(i, j)` reads `(i-1, j-1)`, `(i-1, j)`, and `(i, j-1)`. Group cells by their
**anti‑diagonal index** `d = i + j`. Then:

```
(i-1, j-1) is on diagonal d-2
(i-1, j  ) is on diagonal d-1
(i,   j-1) is on diagonal d-1
```

Every dependency of a cell on diagonal `d` lives on diagonal `d-1` or `d-2` —
**never on `d` itself**. Therefore **all cells on the same anti‑diagonal are
independent and can be computed simultaneously.** Picture a diagonal wave
sweeping from the top‑left to the bottom‑right corner:

```
 j →   0    1    2    3    4
i 0    .   d1   d2   d3   d4
  1   d1   d2   d3   d4   d5
  2   d2   d3   d4   d5   d6
  3   d3   d4   d5   d6   d7
```

All cells labelled `d3` can be filled in parallel, once `d1` and `d2` are done.
This is the classic **wavefront** (a.k.a. *systolic*) parallelisation of a 2‑D
dynamic program.

### Mapping to CUDA

- The matrix lives in **global memory**, flat and row‑major:
  `H[i*(n+1) + j]`.
- We launch **one kernel per anti‑diagonal**, for `d = 2 … m+n`.
- **Each kernel launch is a global barrier:** when the launch for diagonal `d`
  returns, all of `d` is finished before `d+1` begins. We lean on the runtime's
  between‑launch ordering to enforce the cross‑diagonal dependency — the simplest
  correct approach.
- Within a launch, **thread `t` owns one cell**: row `i = i_lo + t`, column
  `j = d - i`. Cells on a diagonal are independent, so **no intra‑launch
  synchronisation is needed at all**.

The valid rows on diagonal `d` are `i ∈ [max(1, d-n), min(m, d-1)]`; that range
length is the number of threads to launch.

### Why traceback stays on the host

We fill the matrix on the GPU, copy it back, and run the **same serial
`traceback_from_matrix`** the CPU uses. Traceback is inherently sequential (each
step depends on where the last one landed) and cheap — `O(m+n)`. Parallelising
the expensive `O(m·n)` fill while leaving the cheap walk serial is the right call,
and reusing one traceback function guarantees CPU and GPU return identical
alignments.

### Honest performance caveats (this is a teaching kernel)

The demo shows the wavefront kernel **losing to the CPU on a single 2000×2000
alignment**. That is expected, not a bug. Three reasons, each a great exercise:

1. **Launch overhead.** `m+n` tiny launches, each a few microseconds. For short
   diagonals the launch cost dwarfs the arithmetic. *Fix:* tile the matrix and
   launch far fewer times, or use a single **persistent kernel** with
   cooperative‑groups **grid synchronisation**.
2. **Uncoalesced memory.** Threads on a diagonal access `H[i*(n+1) + (d-i)]` for
   consecutive `i` — a stride of `(n+1) - 1`, the opposite of contiguous. Each
   warp's loads scatter across memory. *Fix:* store the matrix **diagonal‑major**
   or in tiles so a warp's accesses are contiguous.
3. **Full `O(m·n)` matrix + full copy back.** *Fix:* store only **2‑bit traceback
   directions** instead of full scores, or use **Hirschberg's** linear‑space
   method.

The kernel is written for *clarity first*. Knowing what "good" looks like is part
of the lesson.

---

## Strategy B — Inter‑sequence: **batched**, one thread per pair

**File:** [`src/gpu_batched.cu`](../src/gpu_batched.cu) ·
parallelises **many** alignments, one thread each.

### Why this is the "easy" (and often faster) parallelism

When you have *thousands of independent pairs* to align — e.g. a query against a
database — there is **no dependency between pairs at all**. Give each thread its
own pair and let them all run. No anti‑diagonals, no synchronisation, no clever
ordering: **embarrassingly parallel**. This is how GPUs are typically used for
sequence‑database scanning, and it is where the GPU's throughput truly shows
(the demo measures **17–20× over a single CPU core**).

The trade‑off: each thread computes **score only** (no traceback), using the
rolling‑two‑rows trick so its memory is `O(n)`, not `O(m·n)`. (Score everything
fast; re‑align just the top hits in full later with Strategy A.)

### Two memory lessons baked into this kernel

**(1) Rolling rows.** A full matrix per thread × thousands of threads would
exhaust GPU memory. Each thread keeps just `prev` and `curr` rows and rolls
forward — exactly like `cpu_score_only`.

**(2) Coalesced layout (the big one).** The rolling rows for *all* threads share
two big buffers. The crucial choice is the index order. We store the cell at
column `col` for thread `k` at:

```
buffer[ col * numPairs + k ]      ← pair index k is the FAST/contiguous dimension
```

Why? Threads `k, k+1, k+2, …` in a warp process the **same column at the same
time**, so their addresses are `…+k, …+k+1, …+k+2` — 32 consecutive 4‑byte words.
The hardware fuses them into **one wide "coalesced" memory transaction** instead
of 32 scattered ones. The naive alternative — `buffer[k * width + col]`, each
thread's row contiguous — makes warp‑neighbours stride by `width` and shreds
bandwidth. **Memory coalescing is one of the single biggest performance levers in
all of CUDA**, which is why this kernel goes out of its way to demonstrate it.

> Note: when sequence lengths vary within a warp, threads finish their rows at
> different times and the perfect coalescing degrades somewhat. This is realistic;
> production tools sometimes sort the batch by length to keep warps uniform.

### Flattening ragged data for the GPU

A kernel can't take a `vector<string>`. On the host we concatenate all A‑sequences
into one `char` buffer plus per‑pair **offset** and **length** arrays (and the
same for B). Thread `k` reads its A at `seqA[offA[k] … offA[k]+lenA[k])`. This
**concatenate + offsets** idiom appears whenever you ship variable‑length data to
a GPU.

---

## Side‑by‑side

| | Wavefront (A) | Batched (B) |
|---|---|---|
| Parallelism | within **one** alignment | across **many** alignments |
| Thread does | one matrix **cell** per launch | one whole **pair** |
| Dependency | anti‑diagonal ordering (hard) | none (embarrassingly parallel) |
| Synchronisation | one barrier **per diagonal** (kernel launches) | none |
| Output | full alignment (with traceback) | score only |
| Memory | full `O(m·n)` matrix | two rolling rows `O(n)` per thread |
| Wins when | few, **large** alignments | many, **small** alignments |
| Key lesson | turning a recurrence parallel via wavefront | **memory coalescing** + throughput |

---

## CUDA mechanics this project also demonstrates

- **Error checking that actually catches things** — `CUDA_CHECK` (wrap every
  runtime call) and `CUDA_CHECK_KERNEL` (check the launch *and* synchronise to
  surface asynchronous in‑kernel errors). See
  [`include/cuda_utils.cuh`](../include/cuda_utils.cuh).
- **Event‑based timing** — `GpuTimer` uses `cudaEvent` to measure real device
  time, because kernel launches are asynchronous and host clocks would lie.
- **`__host__ __device__` code sharing** — `substitution_score` is compiled for
  both CPU and GPU from one source line via the `ALN_HD` macro, guaranteeing
  identical scoring.
- **Passing small structs by value** — `ScoringScheme` is handed to kernels by
  value; it lands in fast parameter/constant memory, no global indirection.
- **Block/grid sizing** — `gridDim = ceil(work / blockDim)` with a guard
  `if (t >= count) return;` for the surplus threads in the last block.

Next: [03 — Build & run](03_BUILD_AND_RUN.md).
