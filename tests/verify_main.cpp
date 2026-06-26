// ============================================================================
//  verify_main.cpp
// ----------------------------------------------------------------------------
//  A standalone, exhaustive CORRECTNESS test harness -- the project's safety
//  net. The demo shows things working on a few inputs; THIS program tries to
//  BREAK the implementations across hundreds of cases and reports a single
//  pass/fail verdict (and a nonzero exit code on any failure, so CI can catch
//  regressions automatically).
//
//  It checks four independent layers, each guarding a different kind of bug:
//
//    LAYER 1 -- ALGEBRAIC INVARIANTS of the CPU reference itself.
//      The GPU is validated against the CPU, so if the CPU were subtly wrong,
//      the GPU could "agree" with a wrong answer. To stop that, we assert
//      properties that MUST hold for any correct aligner regardless of
//      implementation (identical-sequence scores, empty-input scores, symmetry,
//      score bounds). These pin down the oracle without circular reasoning.
//
//    LAYER 2 -- HAND-COMPUTED cases with known answers.
//
//    LAYER 3 -- CPU vs GPU WAVEFRONT agreement on random inputs (score AND the
//      full gapped alignment strings), across several scoring schemes and both
//      algorithms, including nasty edge cases (empty, length 1, identical).
//
//    LAYER 4 -- CPU vs GPU BATCHED agreement: the rolling-row score-only path on
//      both devices must produce identical scores for a whole random batch.
//
//  This file is plain host C++; like the demo it touches the GPU only through
//  the project's clean function declarations.
// ============================================================================

#include <cstdio>
#include <cstdint>
#include <string>
#include <vector>
#include <algorithm>   // std::min

#include "alignment_common.h"
#include "cpu_reference.h"
#include "gpu_wavefront.cuh"
#include "gpu_batched.cuh"

// ----------------------------------------------------------------------------
//  Tiny test framework: a global pass/fail tally and a CHECK helper.
// ----------------------------------------------------------------------------
static int g_pass = 0;
static int g_fail = 0;

// Record one boolean assertion. On failure, print a clear message so the exact
// failing case is identifiable.
static void check(bool cond, const char* what) {
    if (cond) {
        ++g_pass;
    } else {
        ++g_fail;
        std::printf("  [FAIL] %s\n", what);
    }
}

// Convenience: assert two int64 values are equal, printing both on mismatch.
static void check_eq(int64_t got, int64_t want, const char* what) {
    if (got == want) {
        ++g_pass;
    } else {
        ++g_fail;
        std::printf("  [FAIL] %s  (got %lld, want %lld)\n",
                    what, static_cast<long long>(got), static_cast<long long>(want));
    }
}

// ----------------------------------------------------------------------------
//  LAYER 1: algebraic invariants of the CPU reference.
// ----------------------------------------------------------------------------
static void test_invariants() {
    std::printf("Layer 1: CPU reference invariants\n");
    ScoringScheme s = DefaultDnaScheme();   // match=+2 mismatch=-1 gap=-2

    // (a) Aligning a sequence with ITSELF.
    //     Global: every column is a match -> score = len * match, no gaps.
    //     Local : the whole thing is the best local region -> same score.
    {
        std::string x = "ACGTACGTAC";   // length 10
        int64_t expect = (int64_t)x.size() * s.match;  // 10 * 2 = 20

        AlignmentResult g = cpu_align(x, x, s, AlignmentKind::Global);
        check_eq(g.score, expect, "self-align NW score == len*match");
        check(g.aligned_a == x && g.aligned_b == x, "self-align NW has no gaps");

        AlignmentResult l = cpu_align(x, x, s, AlignmentKind::Local);
        check_eq(l.score, expect, "self-align SW score == len*match");
    }

    // (b) EMPTY inputs.
    //     Global: aligning length-L against empty forces L gaps -> L * gap.
    //     Local : best you can do with nothing is the empty alignment -> 0.
    {
        std::string x = "ACGTA";          // length 5
        int64_t lenL = (int64_t)x.size();

        check_eq(cpu_align(x, "", s, AlignmentKind::Global).score, lenL * s.gap,
                 "NW(x, empty) == len*gap");
        check_eq(cpu_align("", x, s, AlignmentKind::Global).score, lenL * s.gap,
                 "NW(empty, x) == len*gap");
        check_eq(cpu_align(x, "", s, AlignmentKind::Local).score, 0,
                 "SW(x, empty) == 0");
        check_eq(cpu_align("", "", s, AlignmentKind::Global).score, 0,
                 "NW(empty, empty) == 0");
    }

    // (c) SYMMETRY: swapping the two sequences cannot change the optimal score.
    {
        std::string a = "GGATCGA", b = "GAATTCAGTTA";
        check_eq(cpu_align(a, b, s, AlignmentKind::Global).score,
                 cpu_align(b, a, s, AlignmentKind::Global).score,
                 "NW score is symmetric");
        check_eq(cpu_align(a, b, s, AlignmentKind::Local).score,
                 cpu_align(b, a, s, AlignmentKind::Local).score,
                 "SW score is symmetric");
    }

    // (d) BOUNDS: SW score is never negative; NW score never exceeds the
    //     all-match upper bound min(m,n)*match.
    {
        std::string a = "ACGAT", b = "TTTTT";   // deliberately dissimilar
        AlignmentResult sw = cpu_align(a, b, s, AlignmentKind::Local);
        check(sw.score >= 0, "SW score >= 0");
        AlignmentResult nw = cpu_align(a, b, s, AlignmentKind::Global);
        int64_t ub = (int64_t)std::min(a.size(), b.size()) * s.match;
        check(nw.score <= ub, "NW score <= min(m,n)*match");
    }

    // (e) ROLLING-ROW score equals FULL-MATRIX score (validates cpu_score_only,
    //     which is the template for the GPU batched kernel).
    {
        std::string a = "ACGGCTANNGT", b = "ACGTCATG";  // ('N' is just a letter)
        for (AlignmentKind k : {AlignmentKind::Global, AlignmentKind::Local}) {
            check_eq(cpu_score_only(a, b, s, k), cpu_align(a, b, s, k).score,
                     "rolling-row score == full-matrix score");
        }
    }
}

// ----------------------------------------------------------------------------
//  LAYER 2: hand-computed cases.
// ----------------------------------------------------------------------------
static void test_hand_cases() {
    std::printf("Layer 2: hand-computed cases\n");
    ScoringScheme s = DefaultDnaScheme();   // match=+2 mismatch=-1 gap=-2

    // "AGT" vs "AAGT": one extra 'A' in B forces a single gap. Two optimal
    // layouts ("A-GT" or "-AGT") both score: match + gap + match + match
    //                                       = 2  + (-2) + 2     + 2   = 4.
    check_eq(cpu_align("AGT", "AAGT", s, AlignmentKind::Global).score, 4,
             "NW(AGT, AAGT) == 4");

    // "AC" vs "AC": two matches -> 4.
    check_eq(cpu_align("AC", "AC", s, AlignmentKind::Global).score, 4,
             "NW(AC, AC) == 4");

    // Single mismatch: "A" vs "C" globally -> one mismatch column -> -1.
    check_eq(cpu_align("A", "C", s, AlignmentKind::Global).score, -1,
             "NW(A, C) == mismatch");

    // Same pair locally: a single negative column is worse than the empty
    // alignment, so SW reports 0.
    check_eq(cpu_align("A", "C", s, AlignmentKind::Local).score, 0,
             "SW(A, C) == 0");

    // A clear local motif: both contain "CGTACG"; SW should recover exactly that
    // 6-letter all-match region -> 6 * 2 = 12.
    check_eq(cpu_align("TTCGTACGTT", "AACGTACGAA", s, AlignmentKind::Local).score, 12,
             "SW shared 6-mer motif == 12");
}

// ----------------------------------------------------------------------------
//  LAYER 3: CPU vs GPU wavefront on random + edge inputs.
// ----------------------------------------------------------------------------
static void test_cpu_vs_gpu_wavefront() {
    std::printf("Layer 3: CPU vs GPU wavefront (random + edges)\n");

    // A few scoring schemes so we don't accidentally only test one set of signs.
    std::vector<ScoringScheme> schemes = {
        {  2, -1, -2 },   // default DNA
        {  1, -1, -1 },   // classic unit costs
        {  5, -4, -3 },   // BLAST-like nucleotide scores
        {  3, -2, -5 },   // expensive gaps -> favours mismatches
    };

    // Deterministic seed so a failure is reproducible.
    uint32_t seed = 777u;

    int cases = 0;
    for (const ScoringScheme& s : schemes) {
        for (AlignmentKind kind : {AlignmentKind::Global, AlignmentKind::Local}) {
            // Random lengths from 0 up to 60 -- deliberately including 0 and 1 to
            // exercise the empty/degenerate paths in BOTH implementations.
            for (int trial = 0; trial < 40; ++trial) {
                int la = (int)(seed % 61u); seed = seed * 1103515245u + 12345u;
                int lb = (int)(seed % 61u); seed = seed * 1103515245u + 12345u;
                std::string a = random_sequence(la, seed);
                std::string b = random_sequence(lb, seed);

                AlignmentResult cpu = cpu_align(a, b, s, kind);
                AlignmentResult gpu = gpu_wavefront_align(a, b, s, kind, nullptr);

                // Scores must match...
                bool score_ok = (cpu.score == gpu.score);
                // ...and so must the actual gapped alignment (same traceback).
                bool align_ok = (cpu.aligned_a == gpu.aligned_a) &&
                                (cpu.aligned_b == gpu.aligned_b);
                // ...and the score-only path must agree with the full score too.
                bool sonly_ok = (cpu_score_only(a, b, s, kind) == cpu.score);

                if (!(score_ok && align_ok && sonly_ok)) {
                    std::printf("    [case] scheme{%d,%d,%d} %s la=%d lb=%d\n",
                                s.match, s.mismatch, s.gap,
                                (kind == AlignmentKind::Global ? "NW" : "SW"), la, lb);
                    std::printf("      a=%s\n      b=%s\n", a.c_str(), b.c_str());
                }
                check(score_ok, "wavefront score matches CPU");
                check(align_ok, "wavefront alignment matches CPU");
                check(sonly_ok, "score-only matches full score");
                ++cases;
            }
        }
    }
    std::printf("  (ran %d random wavefront cases)\n", cases);
}

// ----------------------------------------------------------------------------
//  LAYER 4: CPU vs GPU batched scoring on a random batch.
// ----------------------------------------------------------------------------
static void test_cpu_vs_gpu_batched() {
    std::printf("Layer 4: CPU vs GPU batched scoring\n");
    ScoringScheme s = DefaultDnaScheme();
    uint32_t seed = 9090u;

    for (AlignmentKind kind : {AlignmentKind::Global, AlignmentKind::Local}) {
        const int numPairs = 5000;
        std::vector<std::string> as(numPairs), bs(numPairs);
        for (int k = 0; k < numPairs; ++k) {
            int la = (int)(seed % 81u); seed = seed * 1103515245u + 12345u; // 0..80
            int lb = (int)(seed % 81u); seed = seed * 1103515245u + 12345u;
            as[k] = random_sequence(la, seed);
            bs[k] = random_sequence(lb, seed);
        }

        std::vector<int64_t> gpu = gpu_batched_scores(as, bs, s, kind, nullptr);

        int mism = 0;
        for (int k = 0; k < numPairs; ++k) {
            int64_t want = cpu_score_only(as[k], bs[k], s, kind);
            if (gpu[k] != want) {
                if (mism < 5)  // print only the first few to avoid spam
                    std::printf("    [FAIL] pair %d: gpu=%lld cpu=%lld (la=%zu lb=%zu)\n",
                                k, (long long)gpu[k], (long long)want,
                                as[k].size(), bs[k].size());
                ++mism;
            }
        }
        check(mism == 0, (kind == AlignmentKind::Global)
                             ? "batched NW scores all match CPU"
                             : "batched SW scores all match CPU");
    }
}

// ============================================================================
//  main()
// ============================================================================
int main() {
    std::printf("=== Pairwise Alignment correctness tests ===\n\n");

    test_invariants();
    test_hand_cases();
    test_cpu_vs_gpu_wavefront();
    test_cpu_vs_gpu_batched();

    std::printf("\n=== RESULT: %d passed, %d failed ===\n", g_pass, g_fail);
    if (g_fail == 0) {
        std::printf("ALL TESTS PASSED.\n");
        return 0;
    }
    std::printf("THERE WERE FAILURES.\n");
    return 1;
}
