// ============================================================================
//  gpu_batched.cuh
// ----------------------------------------------------------------------------
//  Declares the SECOND GPU strategy: "inter-sequence" (a.k.a. "embarrassingly
//  parallel" or "database-scan") alignment.
//
//  THE CONTRAST WITH gpu_wavefront
//  -------------------------------
//    * gpu_wavefront: ONE big alignment, MANY threads cooperating on its single
//      DP matrix. Good when you have one or a few large alignments.
//    * gpu_batched (this file): MANY small alignments, ONE thread each, all
//      running independently and simultaneously. Good when you have a HUGE pile
//      of pairs to score -- e.g. comparing one query against an entire sequence
//      database (this is essentially how tools like CUDASW++ achieve their
//      throughput).
//
//  Because each thread owns a whole alignment by itself and there is zero
//  communication between threads, this is the easier and usually FASTER way to
//  saturate a GPU -- as long as you have enough independent pairs to fill it.
//  The trade-off: we compute SCORES ONLY (no traceback), which is the common
//  real-world pattern -- score everything fast, then re-align just the top hits
//  in full using the wavefront path.
// ============================================================================

#ifndef GPU_BATCHED_CUH
#define GPU_BATCHED_CUH

#include <string>
#include <vector>
#include <cstdint>
#include "alignment_common.h"

// ----------------------------------------------------------------------------
//  gpu_batched_scores()
//  --------------------
//  Score a whole batch of independent pairs in one GPU launch.
//
//  Parameters
//    as, bs         : parallel arrays of sequences. Pair k is (as[k], bs[k]).
//                     The two vectors MUST have the same length (== batch size).
//    scheme         : scoring rulebook (shared by every pair).
//    kind           : Global or Local -- applied to every pair in the batch.
//    out_kernel_ms  : optional; receives kernel-only GPU time (excludes copies).
//
//  Returns
//    A vector<int64_t> of length batch, where element k is the optimal score of
//    pair k -- matching cpu_score_only(as[k], bs[k], scheme, kind) exactly.
//
//  Defined in gpu_batched.cu.
// ----------------------------------------------------------------------------
std::vector<int64_t> gpu_batched_scores(const std::vector<std::string>& as,
                                        const std::vector<std::string>& bs,
                                        const ScoringScheme& scheme,
                                        AlignmentKind kind,
                                        float* out_kernel_ms = nullptr);

#endif // GPU_BATCHED_CUH
