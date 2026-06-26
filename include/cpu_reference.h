// ============================================================================
//  cpu_reference.h
// ----------------------------------------------------------------------------
//  The SERIAL, single-threaded, textbook implementation of Needleman-Wunsch and
//  Smith-Waterman. This is our "golden model": it is small, obviously correct,
//  and easy to read, and EVERY GPU result in this project is checked against it.
//
//  WHY HAVE A CPU VERSION AT ALL IN A "CUDA" PROJECT?
//    1. Correctness oracle. GPUs are unforgiving: an off-by-one in an index or
//       a missing synchronisation produces *plausible but wrong* numbers. The
//       only sane way to trust a kernel is to compare it against an
//       implementation you can verify by hand. That is this file's main job.
//    2. Performance baseline. "10x faster than what?" is only meaningful if you
//       have a sequential baseline to divide by. We time this code too.
//    3. Pedagogy. Reading the serial DP first makes the parallel version far
//       easier to understand -- the GPU is "the same recurrence, reorganised".
//
//  The two public functions below are intentionally given the SAME signature so
//  that the demo can call them interchangeably and the only thing that changes
//  is the AlignmentKind argument.
// ============================================================================

#ifndef CPU_REFERENCE_H
#define CPU_REFERENCE_H

#include <string>
#include <vector>
#include "alignment_common.h"   // ScoringScheme, AlignmentKind, AlignmentResult

// ----------------------------------------------------------------------------
//  cpu_align()
//  -----------
//  Compute the optimal alignment of `a` against `b` on the CPU.
//
//  Parameters
//    a, b   : the two input sequences (ungapped). Treated as opaque character
//             strings -- the code never assumes a particular alphabet.
//    scheme : the match/mismatch/gap rulebook (see alignment_common.h).
//    kind   : Global (Needleman-Wunsch) or Local (Smith-Waterman).
//
//  Returns
//    A fully populated AlignmentResult: the optimal score, the two gapped rows,
//    and the begin/end coordinates of the aligned region.
//
//  Complexity
//    Time  O(m*n)  -- one constant-work cell per (prefix of a, prefix of b).
//    Space O(m*n)  -- we keep the WHOLE score matrix so we can trace back the
//                     actual alignment afterwards. (If you only need the score,
//                     cpu_score_only() below drops this to O(min(m,n)) with two
//                     rolling rows; the GPU "batched" path uses the same idea.)
// ----------------------------------------------------------------------------
AlignmentResult cpu_align(const std::string& a,
                          const std::string& b,
                          const ScoringScheme& scheme,
                          AlignmentKind kind);

// ----------------------------------------------------------------------------
//  cpu_score_only()
//  ----------------
//  Compute JUST the optimal score (no traceback, no gapped strings), using only
//  two rolling rows of the DP matrix. It orients the shorter sequence onto the
//  columns, so the memory is O(min(m,n)). The GPU "batched / inter-sequence"
//  kernel uses the SAME rolling-row idea (there, sized to the B-sequence length,
//  i.e. O(n) per thread, since a whole batch shares one fixed orientation). This
//  function exists so the test harness can cross-check that the memory-frugal
//  rolling-row formulation gives the same score as the full-matrix version. If
//  the two ever disagree, the
//  bug is in the recurrence, not the traceback.
// ----------------------------------------------------------------------------
int64_t cpu_score_only(const std::string& a,
                       const std::string& b,
                       const ScoringScheme& scheme,
                       AlignmentKind kind);

// ----------------------------------------------------------------------------
//  traceback_from_matrix()
//  -----------------------
//  Reconstruct the actual gapped alignment by walking BACKWARDS through an
//  already-filled DP score matrix `H`. This routine is deliberately separated
//  from the matrix *filling* because the two GPU kernels fill the matrix in
//  parallel but reuse THIS EXACT host-side traceback -- guaranteeing the GPU
//  and CPU return byte-identical gapped strings whenever their matrices match.
//
//  Traceback is inherently SEQUENTIAL (each step depends on where the previous
//  step landed) and cheap -- O(m+n) steps -- so running it on the host even for
//  a GPU-filled matrix is the standard, sensible choice. The expensive
//  O(m*n) matrix fill is what we parallelise; the cheap O(m+n) walk we do not.
//
//  Parameters
//    H            : flat (m+1) x (n+1) matrix, row-major: H[i*(n+1)+j].
//    m, n         : lengths of sequences a and b respectively.
//    a, b         : the original ungapped sequences (needed to emit letters).
//    scheme       : scoring rulebook (needed to recognise which neighbour we
//                   came from by recomputing the diagonal substitution score).
//    kind         : Global stops at (0,0); Local stops at the first zero cell.
//    start_i,start_j : the cell to start tracing from. For Global this is
//                   (m, n) (bottom-right). For Local it is the cell holding the
//                   maximum score (found by the caller).
//
//  Returns a fully populated AlignmentResult (score field is copied from
//  H[start_i][start_j]).
// ----------------------------------------------------------------------------
AlignmentResult traceback_from_matrix(const cell_t* H,
                                      int m, int n,
                                      const std::string& a,
                                      const std::string& b,
                                      const ScoringScheme& scheme,
                                      AlignmentKind kind,
                                      int start_i, int start_j);

// ----------------------------------------------------------------------------
//  cpu_compute_matrix()
//  --------------------
//  Convenience wrapper that returns the FULL filled DP matrix as a flat
//  (m+1)*(n+1) row-major vector. Used by the demo to pretty-print the matrix for
//  small inputs (a great way to SEE the recurrence at work) and by tests that
//  want to compare the CPU and GPU matrices cell-by-cell, not just their scores.
// ----------------------------------------------------------------------------
std::vector<cell_t> cpu_compute_matrix(const std::string& a,
                                       const std::string& b,
                                       const ScoringScheme& scheme,
                                       AlignmentKind kind);

#endif // CPU_REFERENCE_H
