// ============================================================================
//  gpu_batched.cu
// ----------------------------------------------------------------------------
//  "Inter-sequence" GPU alignment: thousands of INDEPENDENT pairs, ONE thread
//  per pair, all scored at once. This is the throughput-oriented counterpart to
//  the wavefront kernel and is how GPUs are typically used to scan a query
//  against a large sequence database.
//
//  ========================================================================
//  WHY THIS IS THE "EASY" KIND OF GPU PARALLELISM
//  ========================================================================
//  The wavefront kernel had to wrestle with a data dependency (cells depend on
//  neighbours). Here there is NO dependency between pairs at all: aligning pair
//  #5 has nothing to do with aligning pair #6. That makes the problem
//  "embarrassingly parallel" -- we just hand each thread its own pair and let
//  them all run. No anti-diagonals, no inter-thread synchronisation, no clever
//  ordering. The interesting engineering shifts from "how do I find parallelism"
//  to "how do I feed memory to the GPU efficiently".
//
//  ========================================================================
//  TWO MEMORY-LAYOUT IDEAS YOU SHOULD TAKE AWAY
//  ========================================================================
//  (1) ROLLING ROWS (O(n) memory instead of O(m*n)):
//      A single thread cannot afford a full (m+1)x(n+1) matrix for its pair --
//      multiply that by thousands of threads and you'd exhaust GPU memory
//      instantly. So, exactly like cpu_score_only(), each thread keeps just TWO
//      rows ("prev" and "curr") and rolls forward. We get the score but lose the
//      ability to trace back -- the standard score-only trade-off.
//
//  (2) STRUCTURE-OF-ARRAYS / COALESCED LAYOUT:
//      The two rolling rows for ALL threads live in two big shared buffers. The
//      crucial choice is HOW we index them. We store the cell at column `col`
//      for thread `k` at:
//
//          buffer[ col * numPairs + k ]        <-- note: pair index is the FAST,
//                                                  contiguous dimension
//
//      Why? Threads k, k+1, k+2, ... in a warp all process the SAME column `col`
//      at the same time (when their sequences are equal length). With this
//      layout their addresses are buffer[col*numPairs + k], +(k+1), +(k+2) -- 32
//      consecutive 4-byte words. The hardware fuses those into a single wide
//      "coalesced" memory transaction instead of 32 separate ones. The naive
//      alternative (buffer[k * width + col], i.e. each thread's row contiguous)
//      would make warp-neighbours stride by `width` and shred memory bandwidth.
//      This layout choice is one of the single most important performance levers
//      in all of CUDA, which is why we go out of our way to demonstrate it.
//
//  ========================================================================
//  FLATTENING VARIABLE-LENGTH SEQUENCES FOR THE GPU
//  ========================================================================
//  A GPU kernel cannot take a std::vector<std::string>. So on the host we pack
//  all the A-sequences end-to-end into ONE char buffer, and record where each
//  one starts (an "offset" array) and how long it is (a "length" array). The
//  same for the B-sequences. Thread k then reads its A at seqA[offA[k] ..
//  offA[k]+lenA[k]). This "concatenate + offsets" idiom is everywhere in GPU
//  programming whenever you must ship ragged/variable-length data to the device.
// ============================================================================

#include "gpu_batched.cuh"
#include "cuda_utils.cuh"

#include <vector>
#include <string>
#include <algorithm>   // std::max
#include <cstdint>

// ----------------------------------------------------------------------------
//  device_max3 (local copy -- each .cu is its own translation unit)
// ----------------------------------------------------------------------------
__device__ inline cell_t bmax3(cell_t x, cell_t y, cell_t z) {
    cell_t m = x;
    if (y > m) m = y;
    if (z > m) m = z;
    return m;
}

// ----------------------------------------------------------------------------
//  KERNEL: batched_align_kernel
//  ----------------------------
//  One thread = one pair. The thread runs the full rolling-row DP for its pair
//  and writes a single score.
//
//  Parameters (all device pointers unless noted)
//    seqA/offA/lenA : flattened A sequences + per-pair offset and length.
//    seqB/offB/lenB : flattened B sequences + per-pair offset and length.
//    numPairs       : batch size (BY VALUE) -- also the stride of the row buffers.
//    rowBufP,rowBufC: the two big rolling-row scratch buffers, laid out
//                     [col * numPairs + pair] for coalescing (see header).
//    outScores      : output, one score per pair.
//    scheme         : scoring rulebook (BY VALUE).
//    is_local       : 1 = Smith-Waterman clamp-to-zero, 0 = Needleman-Wunsch.
// ----------------------------------------------------------------------------
__global__ void batched_align_kernel(const char* seqA, const int* offA, const int* lenA,
                                     const char* seqB, const int* offB, const int* lenB,
                                     int numPairs,
                                     cell_t* rowBufP, cell_t* rowBufC,
                                     cell_t* outScores,
                                     ScoringScheme scheme, int is_local) {
    const int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= numPairs) return;                 // surplus threads do nothing

    const int m   = lenA[k];                   // length of this pair's A
    const int n   = lenB[k];                   // length of this pair's B
    const char* A = seqA + offA[k];            // this pair's A characters
    const char* B = seqB + offB[k];            // this pair's B characters
    const cell_t gap = scheme.gap;

    // Local pointers we will SWAP each row. Because each thread only ever
    // touches buffer slots indexed by its own `k`, two threads never share a
    // slot -- so it is perfectly safe for different threads to have `prev` and
    // `curr` pointing at different physical buffers at the same instant (which
    // happens whenever their sequences have different lengths). No data race.
    cell_t* prev = rowBufP;
    cell_t* curr = rowBufC;

    // Helper lambdas would be nice, but to keep the indexing maximally explicit
    // (and beginner-readable) we spell out `col * numPairs + k` everywhere.

    // --- Initialise the "row 0" border into `prev` ------------------------
    for (int j = 0; j <= n; ++j) {
        prev[(size_t)j * numPairs + k] = is_local ? 0 : (cell_t)j * gap;
    }

    cell_t best = 0;   // running maximum, used only for local alignment

    // --- Fill rows 1..m, rolling prev/curr forward ------------------------
    for (int i = 1; i <= m; ++i) {
        // Left border of this row (column 0).
        curr[(size_t)0 * numPairs + k] = is_local ? 0 : (cell_t)i * gap;

        const char ai = A[i - 1];
        for (int j = 1; j <= n; ++j) {
            const char bj = B[j - 1];

            const cell_t diag = prev[(size_t)(j - 1) * numPairs + k] + substitution_score(ai, bj, scheme);
            const cell_t up   = prev[(size_t)(j    ) * numPairs + k] + gap;
            const cell_t left = curr[(size_t)(j - 1) * numPairs + k] + gap;

            cell_t v = bmax3(diag, up, left);
            if (is_local && v < 0) v = 0;

            curr[(size_t)j * numPairs + k] = v;
            if (is_local && v > best) best = v;
        }

        // Roll forward: the row we just finished becomes "prev" for the next i.
        cell_t* tmp = prev; prev = curr; curr = tmp;
    }

    // After the final swap, `prev` holds the last completed row (row m).
    //   * Global: the answer is the bottom-right cell, prev[n].
    //   * Local : the answer is the maximum we saw anywhere.
    cell_t result = is_local ? best : prev[(size_t)n * numPairs + k];
    outScores[k] = result;
}

// ----------------------------------------------------------------------------
//  gpu_batched_scores()  (host orchestration -- declared in the .cuh)
//  ---------------------------------------------------------------------------
//  Flatten the batch, ship it to the GPU, launch one thread per pair, bring the
//  scores home.
// ----------------------------------------------------------------------------
std::vector<int64_t> gpu_batched_scores(const std::vector<std::string>& as,
                                        const std::vector<std::string>& bs,
                                        const ScoringScheme& scheme,
                                        AlignmentKind kind,
                                        float* out_kernel_ms) {
    const int numPairs = static_cast<int>(as.size());
    std::vector<int64_t> scores(numPairs, 0);

    // Nothing to do for an empty batch (also avoids zero-size cudaMalloc).
    if (numPairs == 0) {
        if (out_kernel_ms) *out_kernel_ms = 0.0f;
        return scores;
    }
    // Defensive: the two arrays must describe the same pairs.
    if (bs.size() != as.size()) {
        // For a teaching tool we fail loudly rather than silently misbehaving.
        std::fprintf(stderr, "[gpu_batched_scores] as and bs must have equal size\n");
        std::exit(EXIT_FAILURE);
    }

    const int is_local = (kind == AlignmentKind::Local) ? 1 : 0;

    // --- 1. Flatten the ragged inputs into contiguous host buffers --------
    // PRECONDITION: the total concatenated length of all A-sequences (and of all
    // B-sequences) is assumed to fit in a signed 32-bit int -- i.e. < ~2.1 GB
    // per side. Offsets below are stored as `int` to keep the kernel's index
    // arithmetic cheap; for batches larger than that you would widen the offset
    // arrays (and the kernel's `offA`/`offB` parameters) to int64_t/size_t. The
    // per-sequence lengths stay `int` regardless, since individual sequences are
    // short. This limit is generous for any realistic teaching/demo batch.
    std::vector<int>  hOffA(numPairs), hLenA(numPairs);
    std::vector<int>  hOffB(numPairs), hLenB(numPairs);
    std::string       flatA, flatB;          // concatenated sequence characters
    int maxLenB = 0;                         // widest row buffer we'll need

    for (int k = 0; k < numPairs; ++k) {
        hOffA[k] = static_cast<int>(flatA.size());
        hLenA[k] = static_cast<int>(as[k].size());
        flatA += as[k];

        hOffB[k] = static_cast<int>(flatB.size());
        hLenB[k] = static_cast<int>(bs[k].size());
        flatB += bs[k];

        maxLenB = std::max(maxLenB, hLenB[k]);
    }
    const int maxColsP1 = maxLenB + 1;       // columns 0..maxLenB

    // --- 2. Allocate device buffers ---------------------------------------
    // Helper: never ask cudaMalloc for 0 bytes (it is allowed to return null);
    // round any zero size up to 1.
    auto nz = [](size_t s) { return s == 0 ? size_t(1) : s; };

    char *dSeqA=nullptr, *dSeqB=nullptr;
    int  *dOffA=nullptr, *dLenA=nullptr, *dOffB=nullptr, *dLenB=nullptr;
    cell_t *dRowP=nullptr, *dRowC=nullptr, *dOut=nullptr;

    CUDA_CHECK(cudaMalloc(&dSeqA, nz(flatA.size())));
    CUDA_CHECK(cudaMalloc(&dSeqB, nz(flatB.size())));
    CUDA_CHECK(cudaMalloc(&dOffA, nz(numPairs * sizeof(int))));
    CUDA_CHECK(cudaMalloc(&dLenA, nz(numPairs * sizeof(int))));
    CUDA_CHECK(cudaMalloc(&dOffB, nz(numPairs * sizeof(int))));
    CUDA_CHECK(cudaMalloc(&dLenB, nz(numPairs * sizeof(int))));
    CUDA_CHECK(cudaMalloc(&dOut,  nz(numPairs * sizeof(cell_t))));

    // The two rolling-row buffers: maxColsP1 columns x numPairs threads.
    const size_t rowCells = static_cast<size_t>(maxColsP1) * static_cast<size_t>(numPairs);
    CUDA_CHECK(cudaMalloc(&dRowP, nz(rowCells * sizeof(cell_t))));
    CUDA_CHECK(cudaMalloc(&dRowC, nz(rowCells * sizeof(cell_t))));

    // --- 3. Upload everything (host -> device) ----------------------------
    if (!flatA.empty()) CUDA_CHECK(cudaMemcpy(dSeqA, flatA.data(), flatA.size(), cudaMemcpyHostToDevice));
    if (!flatB.empty()) CUDA_CHECK(cudaMemcpy(dSeqB, flatB.data(), flatB.size(), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dOffA, hOffA.data(), numPairs * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dLenA, hLenA.data(), numPairs * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dOffB, hOffB.data(), numPairs * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dLenB, hLenB.data(), numPairs * sizeof(int), cudaMemcpyHostToDevice));

    // --- 4. Launch: one thread per pair -----------------------------------
    const int BLOCK = 128;                              // 128 threads/block
    const int grid  = (numPairs + BLOCK - 1) / BLOCK;   // enough blocks to cover

    GpuTimer timer;
    timer.start();
    batched_align_kernel<<<grid, BLOCK>>>(dSeqA, dOffA, dLenA,
                                          dSeqB, dOffB, dLenB,
                                          numPairs, dRowP, dRowC, dOut,
                                          scheme, is_local);
    CUDA_CHECK_KERNEL();   // launch error check + synchronise (safe: one launch)
    timer.stop();
    if (out_kernel_ms) *out_kernel_ms = timer.elapsed_ms();

    // --- 5. Copy scores back and widen cell_t -> int64_t ------------------
    std::vector<cell_t> tmp(numPairs);
    CUDA_CHECK(cudaMemcpy(tmp.data(), dOut, numPairs * sizeof(cell_t), cudaMemcpyDeviceToHost));
    for (int k = 0; k < numPairs; ++k) scores[k] = static_cast<int64_t>(tmp[k]);

    // --- 6. Free device memory --------------------------------------------
    CUDA_CHECK(cudaFree(dSeqA)); CUDA_CHECK(cudaFree(dSeqB));
    CUDA_CHECK(cudaFree(dOffA)); CUDA_CHECK(cudaFree(dLenA));
    CUDA_CHECK(cudaFree(dOffB)); CUDA_CHECK(cudaFree(dLenB));
    CUDA_CHECK(cudaFree(dRowP)); CUDA_CHECK(cudaFree(dRowC));
    CUDA_CHECK(cudaFree(dOut));

    return scores;
}
