// ============================================================================
//  gpu_wavefront.cu
// ----------------------------------------------------------------------------
//  GPU implementation of Needleman-Wunsch / Smith-Waterman that parallelises a
//  SINGLE alignment by sweeping the score matrix along its ANTI-DIAGONALS.
//
//  ========================================================================
//  THE CORE PROBLEM: THE DP RECURRENCE HAS A DEPENDENCY
//  ========================================================================
//  Recall the recurrence (from cpu_reference.cpp):
//
//      H[i][j] = max( H[i-1][j-1] + sub,  H[i-1][j] + gap,  H[i][j-1] + gap )
//
//  Cell (i, j) reads its three neighbours: up-left, up, and left. This means we
//  CANNOT just throw one thread at every cell and run them all at once -- a cell
//  must wait for its neighbours to be finished first. Naive full parallelism
//  would read garbage. So how do we extract parallelism from a recurrence that
//  is, on its face, sequential?
//
//  ========================================================================
//  THE KEY OBSERVATION: ANTI-DIAGONALS ARE INDEPENDENT
//  ========================================================================
//  Group the cells by the value d = i + j (their "anti-diagonal index"). Look at
//  where cell (i, j)'s dependencies live:
//
//      (i-1, j-1)  -> on anti-diagonal d-2
//      (i-1, j  )  -> on anti-diagonal d-1
//      (i,   j-1)  -> on anti-diagonal d-1
//
//  EVERY dependency of a cell on diagonal d lies on diagonal d-1 or d-2 -- NEVER
//  on diagonal d itself. Therefore all cells sharing the same d are mutually
//  independent and can be computed simultaneously! Picture a diagonal "wave"
//  sweeping from the top-left corner to the bottom-right:
//
//        j ->   0   1   2   3   4
//        i  0   .   d1  d2  d3  d4
//        |  1   d1  d2  d3  d4  d5
//        v  2   d2  d3  d4  d5  d6
//           3   d3  d4  d5  d6  d7
//
//  All cells labelled "d3" can be filled in parallel once "d1" and "d2" are done.
//  This is the classic WAVEFRONT (a.k.a. "systolic") parallelisation of a 2-D
//  dynamic program. It is the GPU-friendly reordering of the exact same work the
//  CPU did row by row.
//
//  ========================================================================
//  HOW WE MAP IT ONTO CUDA
//  ========================================================================
//  * The matrix lives in GPU global memory as a flat (m+1)*(n+1) array of
//    cell_t, row-major: cell (i, j) is at H[i*(n+1) + j].
//  * We launch ONE kernel per anti-diagonal, for d = 2, 3, ..., m+n.
//  * Each kernel launch is a GLOBAL BARRIER: when launch for diagonal d returns,
//    every cell of diagonal d is guaranteed finished before diagonal d+1 starts.
//    This is the simplest correct way to enforce the cross-diagonal ordering --
//    we lean on the runtime's between-launch synchronisation instead of trying
//    to coordinate blocks within one launch.
//  * Within a launch, thread t handles one cell on that diagonal. Because the
//    cells are independent, no intra-launch synchronisation is needed at all.
//
//  ========================================================================
//  HONEST PERFORMANCE CAVEATS (this is a TEACHING kernel)
//  ========================================================================
//  This implementation is optimised for CLARITY, not raw speed. Three known
//  inefficiencies, each a great "next exercise":
//    1. LAUNCH OVERHEAD: we do m+n tiny launches. Each launch costs a few
//       microseconds. For short sequences that overhead dwarfs the actual math,
//       so the GPU can be SLOWER than the CPU here -- and that's an honest,
//       instructive result, not a bug. Production kernels tile the matrix and
//       use far fewer launches, or a single persistent kernel with cooperative-
//       groups grid synchronisation.
//    2. UNCOALESCED MEMORY: threads on an anti-diagonal touch cells (i, d-i) for
//       consecutive i, i.e. addresses i*(n+1)+(d-i) that differ by (n+1)-1 -- a
//       large stride. Those reads/writes do not coalesce well. Real kernels
//       store the matrix in a diagonal-major or tiled layout to fix this.
//    3. FULL O(m*n) MATRIX in memory + a full copy back for traceback. Linear-
//       space methods (Hirschberg) or storing 2-bit traceback directions would
//       shrink this dramatically.
//  We call these out loudly so you know what "good" would look like.
// ============================================================================

#include "gpu_wavefront.cuh"
#include "cpu_reference.h"   // we REUSE traceback_from_matrix() for the walk-back
#include "cuda_utils.cuh"    // CUDA_CHECK, CUDA_CHECK_KERNEL, GpuTimer

#include <vector>
#include <algorithm>         // std::max
#include <cstdint>

// ----------------------------------------------------------------------------
//  device_max3() -- tiny device helper returning the max of three cells.
//  Written as a __device__ function purely for readability inside the kernel.
// ----------------------------------------------------------------------------
__device__ inline cell_t device_max3(cell_t x, cell_t y, cell_t z) {
    cell_t m = x;
    if (y > m) m = y;
    if (z > m) m = z;
    return m;
}

// ----------------------------------------------------------------------------
//  gpu_print_device_info()  (host function -- declared in gpu_wavefront.cuh)
//  ---------------------------------------------------------------------------
//  Queries CUDA device 0 and prints the facts a student most wants to see: the
//  card's name, its COMPUTE CAPABILITY (the "sm_XX" number that decides which
//  features and which -arch flag apply), total memory, and the number of
//  streaming multiprocessors (SMs -- the GPU's independent execution units).
// ----------------------------------------------------------------------------
void gpu_print_device_info() {
    int dev_count = 0;
    CUDA_CHECK(cudaGetDeviceCount(&dev_count));
    if (dev_count == 0) {
        std::printf("  No CUDA-capable device found.\n");
        return;
    }

    int dev = 0;
    CUDA_CHECK(cudaGetDevice(&dev));

    cudaDeviceProp p;
    CUDA_CHECK(cudaGetDeviceProperties(&p, dev));

    std::printf("  Device %d: %s\n", dev, p.name);
    std::printf("  Compute capability : sm_%d%d\n", p.major, p.minor);
    std::printf("  Global memory      : %.2f GB\n",
                static_cast<double>(p.totalGlobalMem) / (1024.0 * 1024.0 * 1024.0));
    std::printf("  Multiprocessors    : %d SMs\n", p.multiProcessorCount);
    std::printf("  Max threads/block  : %d\n", p.maxThreadsPerBlock);
    std::printf("  Warp size          : %d\n", p.warpSize);
}

// ----------------------------------------------------------------------------
//  KERNEL 1: init_borders_kernel
//  -----------------------------
//  Fills row 0 and column 0 of the matrix (the "empty prefix" border) before the
//  wavefront begins. Runs over indices 0..max(m,n); each thread sets at most one
//  cell of the top row and one of the left column.
//
//    H        : the device matrix.
//    m, n     : sequence lengths (matrix is (m+1) x (n+1)).
//    gap      : the (negative) gap penalty.
//    is_local : 1 for Smith-Waterman (borders are all 0), 0 for Needleman-
//               Wunsch (borders are cumulative gap penalties).
//
//  Doing this on the GPU (rather than memset-ing on the host and copying) keeps
//  the data on the device and demonstrates a second, trivially-parallel kernel.
// ----------------------------------------------------------------------------
__global__ void init_borders_kernel(cell_t* H, int m, int n, cell_t gap, int is_local) {
    const int t = blockIdx.x * blockDim.x + threadIdx.x;
    const int W = n + 1;   // row stride

    // Top row: cells (0, j) for j = 0..n.
    if (t <= n) {
        H[0 * W + t] = is_local ? 0 : (cell_t)t * gap;
    }
    // Left column: cells (i, 0) for i = 0..m. (Cell (0,0) is written by both the
    // t<=n and t<=m branches with the same value 0, which is harmless.)
    if (t <= m) {
        H[t * W + 0] = is_local ? 0 : (cell_t)t * gap;
    }
}

// ----------------------------------------------------------------------------
//  KERNEL 2: wavefront_diag_kernel
//  -------------------------------
//  Fills every interior cell on ONE anti-diagonal d in parallel. Launched once
//  per diagonal by the host loop below.
//
//  Parameters
//    H        : device matrix (read neighbours, write this diagonal).
//    A, B     : device copies of the two sequences (raw char arrays).
//    m, n     : sequence lengths.
//    d        : the anti-diagonal index being filled (d = i + j).
//    i_lo     : the smallest row index i present on this diagonal (precomputed
//               on the host -- see the loop for the formula).
//    count    : how many cells this diagonal has (number of threads needed).
//    scheme   : scoring rulebook (passed BY VALUE -- it is a tiny POD struct, so
//               copying it into the kernel's parameter space is cheaper than a
//               global-memory indirection, and it lands in fast constant/param
//               memory automatically).
//    is_local : 1 -> clamp negatives to 0 (Smith-Waterman); 0 -> Needleman-Wunsch.
//
//  THREAD -> CELL MAPPING: thread t owns row i = i_lo + t, and since i + j = d,
//  its column is j = d - i. Each valid t therefore addresses a distinct cell on
//  the diagonal, and threads with t >= count simply return.
// ----------------------------------------------------------------------------
__global__ void wavefront_diag_kernel(cell_t* H, const char* A, const char* B,
                                      int m, int n, int d, int i_lo, int count,
                                      ScoringScheme scheme, int is_local) {
    const int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= count) return;              // surplus threads in the last block

    const int i = i_lo + t;              // this thread's row
    const int j = d - i;                 // ... and column (because i + j = d)
    const int W = n + 1;                 // row stride for flat indexing

    // Fetch the two letters being compared at this cell. Remember the "-1":
    // matrix cell (i, j) compares the i-th letter of A (A[i-1]) with the j-th of
    // B (B[j-1]).
    const char ai = A[i - 1];
    const char bj = B[j - 1];

    // The three candidate moves. These three reads are the ONLY global-memory
    // loads in the hot path; all three land on diagonals d-1 / d-2, already
    // finalised by previous launches.
    const cell_t diag = H[(i - 1) * W + (j - 1)] + substitution_score(ai, bj, scheme);
    const cell_t up   = H[(i - 1) * W + (j    )] + scheme.gap;  // gap in B
    const cell_t left = H[(i    ) * W + (j - 1)] + scheme.gap;  // gap in A

    cell_t best = device_max3(diag, up, left);

    // Smith-Waterman's defining clamp: a local alignment may restart from zero.
    if (is_local && best < 0) best = 0;

    H[i * W + j] = best;                 // the single global-memory store
}

// ----------------------------------------------------------------------------
//  gpu_wavefront_align()  (host orchestration -- declared in the .cuh)
//  ---------------------------------------------------------------------------
//  Ties everything together: allocate, upload, init, sweep the diagonals, copy
//  back, and trace back on the host. See the header comment for the high-level
//  pipeline.
// ----------------------------------------------------------------------------
AlignmentResult gpu_wavefront_align(const std::string& a,
                                    const std::string& b,
                                    const ScoringScheme& scheme,
                                    AlignmentKind kind,
                                    float* out_kernel_ms) {
    const int m = static_cast<int>(a.size());
    const int n = static_cast<int>(b.size());
    const int W = n + 1;                            // matrix row width
    const int is_local = (kind == AlignmentKind::Local) ? 1 : 0;

    // --- Handle the degenerate empty-input case up front -------------------
    // If either sequence is empty there are no interior cells to fill; the
    // answer is entirely determined by the border, and traceback is trivial. We
    // simply fall back to the CPU path, which already handles this cleanly.
    if (m == 0 || n == 0) {
        if (out_kernel_ms) *out_kernel_ms = 0.0f;
        return cpu_align(a, b, scheme, kind);
    }

    // --- 1. Allocate device buffers ---------------------------------------
    const size_t matrix_cells = static_cast<size_t>(m + 1) * static_cast<size_t>(n + 1);
    const size_t matrix_bytes = matrix_cells * sizeof(cell_t);

    cell_t* dH = nullptr;     // the DP matrix on the device
    char*   dA = nullptr;     // device copy of sequence A
    char*   dB = nullptr;     // device copy of sequence B
    CUDA_CHECK(cudaMalloc(&dH, matrix_bytes));
    CUDA_CHECK(cudaMalloc(&dA, static_cast<size_t>(m)));
    CUDA_CHECK(cudaMalloc(&dB, static_cast<size_t>(n)));

    // --- 2. Upload the sequences (host -> device) -------------------------
    // a.data() is a contiguous const char*; we copy exactly m / n bytes. The
    // matrix dH does NOT need uploading -- it is produced entirely on the GPU.
    CUDA_CHECK(cudaMemcpy(dA, a.data(), static_cast<size_t>(m), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dB, b.data(), static_cast<size_t>(n), cudaMemcpyHostToDevice));

    // A standard block size. 256 threads/block is a safe, generally-good default
    // on every CUDA architecture; the grid is sized per-launch to cover the work.
    const int BLOCK = 256;

    // GpuTimer measures ONLY the kernel work (init + all diagonal launches), so
    // the reported time is a fair "device compute" figure excluding transfers.
    GpuTimer timer;
    timer.start();

    // --- 3. Initialise the borders on the device --------------------------
    {
        const int border_threads = std::max(m, n) + 1;
        const int grid = (border_threads + BLOCK - 1) / BLOCK;
        init_borders_kernel<<<grid, BLOCK>>>(dH, m, n, scheme.gap, is_local);
        CUDA_CHECK_KERNEL();
    }

    // --- 4. Sweep the anti-diagonals --------------------------------------
    // Interior cells have i in [1, m] and j in [1, n], so d = i + j ranges from
    // 1+1 = 2 up to m+n. For each d we compute which rows are present:
    //
    //   * j = d - i must satisfy 1 <= j <= n   ->  d-n <= i <= d-1
    //   * i must also satisfy     1 <= i <= m
    //   combining:  i_lo = max(1, d - n),  i_hi = min(m, d - 1)
    //
    // The number of cells on the diagonal is (i_hi - i_lo + 1).
    for (int d = 2; d <= m + n; ++d) {
        const int i_lo = std::max(1, d - n);
        const int i_hi = std::min(m, d - 1);
        const int count = i_hi - i_lo + 1;
        if (count <= 0) continue;                 // (cannot happen for d in range,
                                                  //  but cheap insurance)

        const int grid = (count + BLOCK - 1) / BLOCK;
        wavefront_diag_kernel<<<grid, BLOCK>>>(dH, dA, dB, m, n, d, i_lo, count,
                                               scheme, is_local);
        // NOTE: we deliberately do NOT call CUDA_CHECK_KERNEL() (which
        // synchronises) inside this hot loop -- synchronising m+n times would
        // wreck the timing and serialise launch overhead. The kernels run
        // back-to-back in the default stream, which already orders them
        // correctly. We check for errors ONCE after the loop instead.
    }
    // One synchronise + error check covering the whole diagonal sweep.
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    timer.stop();
    if (out_kernel_ms) *out_kernel_ms = timer.elapsed_ms();

    // --- 5. Copy the finished matrix back (device -> host) ----------------
    // We need the whole matrix on the host to run the serial traceback. For very
    // large alignments this O(m*n) transfer is the dominant cost -- another
    // reason production tools avoid materialising the full matrix.
    std::vector<cell_t> H(matrix_cells);
    CUDA_CHECK(cudaMemcpy(H.data(), dH, matrix_bytes, cudaMemcpyDeviceToHost));

    // Release device memory now that everything we need is back on the host.
    CUDA_CHECK(cudaFree(dH));
    CUDA_CHECK(cudaFree(dA));
    CUDA_CHECK(cudaFree(dB));

    // --- 6. Find the traceback start cell, then walk back -----------------
    int start_i, start_j;
    if (kind == AlignmentKind::Global) {
        // Global alignment always ends in the bottom-right corner.
        start_i = m;
        start_j = n;
    } else {
        // Local alignment starts traceback at the maximum-scoring cell. We scan
        // the host copy of the matrix to find it. (This could be done on the GPU
        // with a parallel reduction / atomicMax, but since we already copied the
        // matrix back for traceback, a simple host scan is clearer and the cost
        // is negligible next to the transfer.)
        cell_t best = 0;
        start_i = 0;
        start_j = 0;
        for (int i = 1; i <= m; ++i) {
            for (int j = 1; j <= n; ++j) {
                const cell_t v = H[(size_t)i * W + j];
                if (v > best) { best = v; start_i = i; start_j = j; }
            }
        }
    }

    // Reuse the EXACT same traceback the CPU reference uses, guaranteeing the
    // GPU returns an identical AlignmentResult whenever its matrix matches.
    return traceback_from_matrix(H.data(), m, n, a, b, scheme, kind, start_i, start_j);
}
