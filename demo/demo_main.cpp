// ============================================================================
//  demo_main.cpp
// ----------------------------------------------------------------------------
//  THE SHOWCASE. Build and run this one program to see the whole project in
//  action. It is intentionally written as a guided tour: each section prints a
//  heading, explains what it is about to do, does it, and verifies the result.
//
//  What it demonstrates, in order:
//    1. Which GPU you are running on.
//    2. A small GLOBAL alignment (Needleman-Wunsch) with the full DP matrix
//       printed, so you can literally see the recurrence's output.
//    3. A small LOCAL alignment (Smith-Waterman), same treatment.
//    4. For BOTH, it runs the CPU reference AND the GPU wavefront kernel and
//       asserts they produced identical scores and identical gapped strings.
//    5. A larger, randomly-generated alignment timed CPU vs GPU.
//    6. A big BATCH of thousands of independent alignments scored on the GPU
//       (inter-sequence parallelism), verified against the CPU.
//
//  This file is plain host C++ -- it never includes a CUDA header. It talks to
//  the GPU only through the clean function declarations in the project headers,
//  whose implementations are compiled separately by nvcc. That separation is a
//  deliberate teaching point: your application code does NOT have to be CUDA
//  code to USE CUDA code.
// ============================================================================

#include <cstdio>
#include <cstdint>
#include <string>
#include <vector>
#include <chrono>     // std::chrono -- portable high-resolution CPU timing

#include "alignment_common.h"
#include "cpu_reference.h"
#include "gpu_wavefront.cuh"   // pure-C++ declarations (no CUDA headers leak in)
#include "gpu_batched.cuh"

// ----------------------------------------------------------------------------
//  Small console helpers (purely cosmetic, but they make the tour readable).
// ----------------------------------------------------------------------------

// A bold section divider.
static void banner(const char* title) {
    std::printf("\n");
    std::printf("================================================================\n");
    std::printf("  %s\n", title);
    std::printf("================================================================\n");
}

// Print the active scoring scheme so results are interpretable.
static void print_scheme(const ScoringScheme& s) {
    std::printf("  Scoring: match=%+d  mismatch=%+d  gap=%+d  (linear gap model)\n",
                s.match, s.mismatch, s.gap);
}

// ----------------------------------------------------------------------------
//  print_matrix()
//  --------------
//  Pretty-print the full DP score matrix for SMALL inputs. Seeing the actual
//  numbers -- with the B sequence across the top and the A sequence down the
//  side -- is the single most effective way to understand how the recurrence
//  fills the table and where the optimal path lives. We refuse to print large
//  matrices (they would flood the console).
// ----------------------------------------------------------------------------
static void print_matrix(const std::string& a, const std::string& b,
                         const ScoringScheme& scheme, AlignmentKind kind) {
    const int m = static_cast<int>(a.size());
    const int n = static_cast<int>(b.size());
    if (m > 16 || n > 24) {
        std::printf("  (matrix too large to display: %dx%d)\n", m + 1, n + 1);
        return;
    }

    // Recompute the matrix via the shared CPU routine (single source of truth).
    std::vector<cell_t> H = cpu_compute_matrix(a, b, scheme, kind);
    const int W = n + 1;

    // Header row: a leading blank, then a '*' for the empty-prefix column, then
    // each character of B.
    std::printf("        *");
    for (int j = 0; j < n; ++j) std::printf("%4c", b[j]);
    std::printf("\n");

    for (int i = 0; i <= m; ++i) {
        // Row label: '*' for the empty-prefix row, else A's character.
        char label = (i == 0) ? '*' : a[i - 1];
        std::printf("   %c ", label);
        for (int j = 0; j <= n; ++j) {
            std::printf("%4d", H[(size_t)i * W + j]);
        }
        std::printf("\n");
    }
}

// ----------------------------------------------------------------------------
//  run_pair()
//  ----------
//  Align one pair on BOTH the CPU and the GPU, print everything, and assert the
//  two agree. Returns true on success (used to set the program exit code).
// ----------------------------------------------------------------------------
static bool run_pair(const char* label,
                     const std::string& a, const std::string& b,
                     const ScoringScheme& scheme, AlignmentKind kind,
                     bool show_matrix) {
    std::printf("\n--- %s ---\n", label);
    std::printf("  A = \"%s\"  (length %zu)\n", a.c_str(), a.size());
    std::printf("  B = \"%s\"  (length %zu)\n", b.c_str(), b.size());
    std::printf("  Algorithm: %s\n", AlignmentKindName(kind));

    if (show_matrix) {
        std::printf("\n  DP score matrix (B across the top, A down the side):\n");
        print_matrix(a, b, scheme, kind);
    }

    // CPU reference (the "golden" answer).
    AlignmentResult cpu = cpu_align(a, b, scheme, kind);

    // GPU wavefront. We also capture the kernel-only time for interest.
    float gpu_ms = 0.0f;
    AlignmentResult gpu = gpu_wavefront_align(a, b, scheme, kind, &gpu_ms);

    // Show the CPU alignment (the GPU one is identical when the check passes).
    std::printf("\n  Optimal score: %lld\n", static_cast<long long>(cpu.score));
    std::printf("  Aligned region A[%d..%d) vs B[%d..%d):\n",
                cpu.begin_i, cpu.end_i, cpu.begin_j, cpu.end_j);
    std::printf("%s\n", format_alignment(cpu, scheme).c_str());

    // The crucial correctness check: do CPU and GPU agree on EVERYTHING?
    const bool ok = (cpu.score == gpu.score) &&
                    (cpu.aligned_a == gpu.aligned_a) &&
                    (cpu.aligned_b == gpu.aligned_b);

    std::printf("\n  GPU wavefront kernel time: %.3f ms\n", gpu_ms);
    if (ok) {
        std::printf("  [VERIFY] CPU and GPU agree exactly.  OK\n");
    } else {
        std::printf("  [VERIFY] MISMATCH between CPU and GPU!\n");
        std::printf("    CPU score=%lld  GPU score=%lld\n",
                    static_cast<long long>(cpu.score),
                    static_cast<long long>(gpu.score));
        std::printf("    CPU A: %s\n    GPU A: %s\n", cpu.aligned_a.c_str(), gpu.aligned_a.c_str());
        std::printf("    CPU B: %s\n    GPU B: %s\n", cpu.aligned_b.c_str(), gpu.aligned_b.c_str());
    }
    return ok;
}

// ----------------------------------------------------------------------------
//  timing_demo()
//  -------------
//  Align one LARGER random pair and time CPU vs GPU. The point here is honesty:
//  for a single moderate alignment the GPU wavefront is often NOT faster than
//  the CPU, because of the per-anti-diagonal launch overhead discussed at length
//  in gpu_wavefront.cu. We print both numbers and let the data speak.
// ----------------------------------------------------------------------------
static bool timing_demo(const ScoringScheme& scheme, AlignmentKind kind) {
    const int len = 2000;                 // 2000 x 2000 matrix = 4,002,001 cells
    uint32_t seed = 12345u;               // fixed seed -> reproducible
    std::string a = random_sequence(len, seed);
    std::string b = random_sequence(len, seed);

    std::printf("\n  Aligning two random length-%d sequences (%s):\n",
                len, AlignmentKindName(kind));

    // --- CPU timing ---
    auto t0 = std::chrono::high_resolution_clock::now();
    AlignmentResult cpu = cpu_align(a, b, scheme, kind);
    auto t1 = std::chrono::high_resolution_clock::now();
    double cpu_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

    // --- GPU timing (kernel only) ---
    float gpu_ms = 0.0f;
    AlignmentResult gpu = gpu_wavefront_align(a, b, scheme, kind, &gpu_ms);

    const bool ok = (cpu.score == gpu.score) &&
                    (cpu.aligned_a == gpu.aligned_a) &&
                    (cpu.aligned_b == gpu.aligned_b);

    std::printf("    optimal score      : %lld\n", static_cast<long long>(cpu.score));
    std::printf("    CPU time           : %8.2f ms\n", cpu_ms);
    std::printf("    GPU wavefront time : %8.2f ms (kernels only)\n", gpu_ms);
    std::printf("    correctness        : %s\n", ok ? "OK (CPU==GPU)" : "MISMATCH!");
    std::printf("    note: for ONE alignment the GPU's many tiny per-diagonal\n");
    std::printf("          launches often lose to the cache-friendly CPU. The GPU\n");
    std::printf("          shines on the BATCHED workload below.\n");
    return ok;
}

// ----------------------------------------------------------------------------
//  batched_demo()
//  --------------
//  The workload the GPU is actually GOOD at: a huge pile of independent pairs.
//  We score them all on the GPU (one thread per pair) and on the CPU, then check
//  every single score matches. We also report throughput (alignments/second).
// ----------------------------------------------------------------------------
static bool batched_demo(const ScoringScheme& scheme, AlignmentKind kind) {
    const int numPairs = 20000;           // twenty thousand independent pairs
    const int minLen = 60, maxLen = 120;  // each sequence is 60..120 bases

    std::printf("\n  Scoring %d independent random pairs (%s),\n",
                numPairs, AlignmentKindName(kind));
    std::printf("  each sequence %d..%d bases long, one GPU thread per pair.\n",
                minLen, maxLen);

    // Build the batch with a fixed seed for reproducibility.
    uint32_t seed = 2024u;
    std::vector<std::string> as(numPairs), bs(numPairs);
    for (int k = 0; k < numPairs; ++k) {
        // Vary lengths a little so the batch is "ragged" (realistic) and the
        // offset/length flattening machinery is genuinely exercised.
        int la = minLen + (int)(seed % (uint32_t)(maxLen - minLen + 1)); seed = seed * 1664525u + 1013904223u;
        int lb = minLen + (int)(seed % (uint32_t)(maxLen - minLen + 1)); seed = seed * 1664525u + 1013904223u;
        as[k] = random_sequence(la, seed);
        bs[k] = random_sequence(lb, seed);
    }

    // --- CPU: score every pair with the rolling-row routine ---
    auto t0 = std::chrono::high_resolution_clock::now();
    std::vector<int64_t> cpu_scores(numPairs);
    for (int k = 0; k < numPairs; ++k)
        cpu_scores[k] = cpu_score_only(as[k], bs[k], scheme, kind);
    auto t1 = std::chrono::high_resolution_clock::now();
    double cpu_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

    // --- GPU: score the whole batch in one launch ---
    float gpu_ms = 0.0f;
    std::vector<int64_t> gpu_scores = gpu_batched_scores(as, bs, scheme, kind, &gpu_ms);

    // --- Verify every score matches ---
    int mismatches = 0;
    for (int k = 0; k < numPairs; ++k)
        if (cpu_scores[k] != gpu_scores[k]) ++mismatches;

    std::printf("    CPU time           : %8.2f ms  (%.0f alignments/sec)\n",
                cpu_ms, numPairs / (cpu_ms / 1000.0));
    std::printf("    GPU time           : %8.2f ms  (%.0f alignments/sec, kernel only)\n",
                gpu_ms, numPairs / (gpu_ms / 1000.0));
    if (mismatches == 0) {
        std::printf("    correctness        : OK (all %d scores match)\n", numPairs);
    } else {
        std::printf("    correctness        : MISMATCH in %d / %d pairs!\n", mismatches, numPairs);
    }
    if (gpu_ms > 0.0f)
        std::printf("    GPU speedup        : %.1fx over single-threaded CPU\n", cpu_ms / gpu_ms);
    return mismatches == 0;
}

// ============================================================================
//  main()
// ============================================================================
int main() {
    bool all_ok = true;

    banner("CUDA Pairwise Alignment -- Smith-Waterman / Needleman-Wunsch");
    std::printf("  A didactic CUDA C++ study project.\n");

    banner("1. GPU device");
    gpu_print_device_info();

    const ScoringScheme scheme = DefaultDnaScheme();

    banner("2. Global alignment (Needleman-Wunsch), small example");
    print_scheme(scheme);
    // The classic textbook pair (the same one used on the Needleman-Wunsch
    // Wikipedia page) -- short enough to print the whole matrix.
    all_ok &= run_pair("NW: GCATGCU vs GATTACA",
                       "GCATGCU", "GATTACA", scheme,
                       AlignmentKind::Global, /*show_matrix=*/true);

    banner("3. Local alignment (Smith-Waterman), small example");
    print_scheme(scheme);
    // Two sequences that share the embedded motif "GTTGAC"; local alignment
    // should lock onto that shared region and ignore the unrelated flanks.
    all_ok &= run_pair("SW: shared motif",
                       "TTTGTTGACAAA", "CCGTTGACGG", scheme,
                       AlignmentKind::Local, /*show_matrix=*/true);

    banner("4. Timing: one larger alignment, CPU vs GPU wavefront");
    print_scheme(scheme);
    all_ok &= timing_demo(scheme, AlignmentKind::Global);

    banner("5. Batched throughput: thousands of pairs at once (GPU's strength)");
    print_scheme(scheme);
    all_ok &= batched_demo(scheme, AlignmentKind::Local);
    all_ok &= batched_demo(scheme, AlignmentKind::Global);

    banner("Summary");
    if (all_ok) {
        std::printf("  All correctness checks PASSED. \n");
        std::printf("  Read the heavily-commented sources in src/ and the guides in\n");
        std::printf("  docs/ to understand exactly how each piece works.\n");
    } else {
        std::printf("  SOME CHECKS FAILED -- see the [VERIFY]/MISMATCH lines above.\n");
    }
    std::printf("\n");

    // Conventional exit code: 0 = success, 1 = a check failed. Lets scripts /
    // CI detect failure automatically.
    return all_ok ? 0 : 1;
}
