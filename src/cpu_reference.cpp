// ============================================================================
//  cpu_reference.cpp
// ----------------------------------------------------------------------------
//  Serial, textbook Needleman-Wunsch (global) and Smith-Waterman (local).
//  This is the GOLDEN MODEL the GPU is validated against, so clarity and
//  obvious correctness matter far more here than speed.
//
//  THE ONE BIG IDEA (dynamic programming)
//  --------------------------------------
//  Define H[i][j] = the best score of aligning the first i characters of A with
//  the first j characters of B. The key insight is that the LAST column of any
//  optimal alignment of those prefixes must be one of exactly three things:
//
//      (1) A[i-1] aligned with B[j-1]   (a "diagonal" move)
//      (2) A[i-1] aligned with a gap    (an "up" move: we used a char of A only)
//      (3) a gap  aligned with B[j-1]   (a "left" move: we used a char of B only)
//
//  Whichever of those three we pick, the REST of the alignment (everything to
//  its left) is itself an optimal alignment of smaller prefixes -- whose score
//  we have ALREADY computed and stored. So:
//
//      H[i][j] = max( H[i-1][j-1] + substitution(A[i-1], B[j-1]),   // diagonal
//                     H[i-1][j]   + gap,                            // up
//                     H[i][j-1]   + gap )                           // left
//
//  Smith-Waterman adds one twist: a fourth option, "start fresh here", worth 0,
//  so H[i][j] is also never allowed to drop below 0:
//
//      H[i][j] = max( 0, <the three options above> )
//
//  That single `max(0, ...)` is the entire mathematical difference between
//  global and local alignment. Everything else (border setup, where traceback
//  begins and ends) follows from it.
//
//  INDEXING CONVENTION (memorise this -- it is the #1 source of bugs)
//  -----------------------------------------------------------------
//    * The matrix has (m+1) rows and (n+1) columns: an extra row 0 and column 0
//      represent "the empty prefix". So matrix index i runs 0..m, j runs 0..n.
//    * But the SEQUENCES are 0-indexed: A's characters are A[0..m-1].
//    * Therefore matrix cell (i, j) compares A's i-th character A[i-1] with B's
//      j-th character B[j-1]. The "-1" appears every time we look up a letter.
//    * The matrix is stored FLAT (one std::vector) in row-major order, so cell
//      (i, j) lives at linear index i*(n+1) + j. We wrap that in idx().
// ============================================================================

#include "cpu_reference.h"

#include <vector>
#include <algorithm>   // std::max
#include <string>

namespace {  // anonymous namespace: these helpers are private to this file.

// idx(): map a 2-D matrix coordinate to a 1-D array offset (row-major).
//   stride = number of columns = (n+1). Marking it `inline` lets the compiler
//   fold this trivial multiply-add directly into the caller.
inline int idx(int i, int j, int n_plus_1) {
    return i * n_plus_1 + j;
}

} // anonymous namespace

// ----------------------------------------------------------------------------
//  fill_matrix() (file-local helper)
//  ---------------------------------
//  Fills the DP matrix H according to `kind`, and reports back WHERE traceback
//  should begin via best_i/best_j and the best score.
//
//    * Global: the answer is always at the bottom-right corner (m, n), because a
//      global alignment must consume both whole sequences. best score = H[m][n].
//    * Local: the answer is the single LARGEST cell anywhere in the matrix, so
//      we track the running maximum as we fill.
//
//  H must already be sized to (m+1)*(n+1). We fill borders first, then sweep
//  the interior row by row.
// ----------------------------------------------------------------------------
static void fill_matrix(const std::string& a,
                        const std::string& b,
                        const ScoringScheme& scheme,
                        AlignmentKind kind,
                        std::vector<cell_t>& H,
                        int& best_i, int& best_j, int64_t& best_score) {
    const int m  = static_cast<int>(a.size());
    const int n  = static_cast<int>(b.size());
    const int W  = n + 1;                 // row width (number of columns)
    const cell_t gap = scheme.gap;        // shorthand; gap is already negative

    // --- Border initialisation -------------------------------------------
    // Cell (0,0): aligning two empty prefixes costs nothing.
    H[idx(0, 0, W)] = 0;

    if (kind == AlignmentKind::Global) {
        // GLOBAL: aligning i characters of A against an EMPTY B forces i gaps,
        // costing i*gap. Likewise across the top row. These growing penalties
        // are what FORCE a global alignment to span the full length -- there is
        // no "free" way to skip the ends.
        for (int i = 1; i <= m; ++i) H[idx(i, 0, W)] = static_cast<cell_t>(i) * gap;
        for (int j = 1; j <= n; ++j) H[idx(0, j, W)] = static_cast<cell_t>(j) * gap;
    } else {
        // LOCAL: an alignment may start anywhere, so "align nothing yet" is
        // always available for free. The entire first row and column are 0.
        for (int i = 1; i <= m; ++i) H[idx(i, 0, W)] = 0;
        for (int j = 1; j <= n; ++j) H[idx(0, j, W)] = 0;
    }

    // For local alignment we hunt for the global maximum. Seed it with the
    // best border value so far (0), located at the origin.
    best_score = 0;
    best_i = 0;
    best_j = 0;

    // --- Interior fill ----------------------------------------------------
    // Sweep rows top-to-bottom, columns left-to-right. By the time we reach
    // cell (i, j), its three predecessors -- (i-1,j-1), (i-1,j), (i,j-1) -- are
    // all already final. THIS ordering is exactly what the GPU must respect
    // too; it just discovers a different, parallel-friendly order to visit the
    // cells in (see gpu_wavefront.cu).
    for (int i = 1; i <= m; ++i) {
        const char ai = a[i - 1];         // the i-th letter of A (0-indexed!)
        for (int j = 1; j <= n; ++j) {
            const char bj = b[j - 1];     // the j-th letter of B

            // The three candidate moves (see the header comment for the math):
            const cell_t diag = H[idx(i - 1, j - 1, W)] + substitution_score(ai, bj, scheme);
            const cell_t up   = H[idx(i - 1, j,     W)] + gap;  // gap in B
            const cell_t left = H[idx(i,     j - 1, W)] + gap;  // gap in A

            cell_t best = diag;
            if (up   > best) best = up;
            if (left > best) best = left;

            // The ONE difference between the two algorithms:
            if (kind == AlignmentKind::Local && best < 0) {
                best = 0;                 // "start a new local alignment here"
            }

            H[idx(i, j, W)] = best;

            // Track the maximum cell for local traceback.
            if (kind == AlignmentKind::Local && best > best_score) {
                best_score = best;
                best_i = i;
                best_j = j;
            }
        }
    }

    // For global alignment the answer is fixed at the corner.
    if (kind == AlignmentKind::Global) {
        best_i = m;
        best_j = n;
        best_score = H[idx(m, n, W)];
    }
}

// ----------------------------------------------------------------------------
//  traceback_from_matrix()  (PUBLIC -- declared in cpu_reference.h)
//  ---------------------------------------------------------------------------
//  Walk backwards from (start_i, start_j) to reconstruct the gapped alignment.
//  At each cell we ask: "which of the three moves produced this value?" and undo
//  it, prepending the corresponding characters. Because we build the strings
//  back-to-front, we reverse them at the end.
//
//  TIE-BREAKING POLICY (important for reproducibility): when more than one move
//  could have produced the current cell's value, we always prefer
//  diagonal > up > left, in that order. The GPU traceback reuses THIS function,
//  so both agree on which optimal alignment to report when several are tied.
// ----------------------------------------------------------------------------
AlignmentResult traceback_from_matrix(const cell_t* H,
                                      int m, int n,
                                      const std::string& a,
                                      const std::string& b,
                                      const ScoringScheme& scheme,
                                      AlignmentKind kind,
                                      int start_i, int start_j) {
    const int W = n + 1;

    AlignmentResult r;
    r.score = H[idx(start_i, start_j, W)];
    r.end_i = start_i;   // traceback ENDS (in forward terms) at the start cell
    r.end_j = start_j;

    std::string ra;      // reversed aligned A (built right-to-left)
    std::string rb;      // reversed aligned B
    ra.reserve(static_cast<size_t>(m + n));
    rb.reserve(static_cast<size_t>(m + n));

    int i = start_i;
    int j = start_j;

    // Loop until we hit the stopping condition:
    //   * Global: the origin (0,0) -- the whole alignment is consumed.
    //   * Local : a cell whose score is 0 -- that marks where this local
    //             alignment began (the "start fresh" option was taken).
    while (true) {
        if (kind == AlignmentKind::Global) {
            if (i == 0 && j == 0) break;
        } else { // Local
            if (H[idx(i, j, W)] == 0) break;
        }

        const cell_t here = H[idx(i, j, W)];

        // Try DIAGONAL first (consume one char from each sequence).
        if (i > 0 && j > 0 &&
            here == H[idx(i - 1, j - 1, W)] + substitution_score(a[i - 1], b[j - 1], scheme)) {
            ra.push_back(a[i - 1]);
            rb.push_back(b[j - 1]);
            --i; --j;
        }
        // Then UP (consume a char from A, gap in B).
        else if (i > 0 &&
                 here == H[idx(i - 1, j, W)] + scheme.gap) {
            ra.push_back(a[i - 1]);
            rb.push_back('-');
            --i;
        }
        // Otherwise it must be LEFT (gap in A, consume a char from B).
        // The `j > 0` guard is DEFENSIVE. By this matrix's construction the LEFT
        // branch is never actually reached at j == 0 (in a global matrix the UP
        // predecessor always matches at column 0 because H[i][0] = i*gap =
        // H[i-1][0] + gap; in a local matrix every border cell is 0 so the STOP
        // condition fires first). The guard therefore costs nothing today, but it
        // guarantees we can never read b[-1] or drive j negative if this shared
        // routine is ever reused on a matrix whose borders differ. If we somehow
        // stand at column 0, the only legal remaining move is UP.
        else if (j > 0) {
            ra.push_back('-');
            rb.push_back(b[j - 1]);
            --j;
        }
        else {
            ra.push_back(a[i - 1]);
            rb.push_back('-');
            --i;
        }
    }

    // (i, j) now sits at the START of the aligned region (the corner reached).
    r.begin_i = i;
    r.begin_j = j;

    // We built the rows backwards; reverse them into forward reading order.
    r.aligned_a.assign(ra.rbegin(), ra.rend());
    r.aligned_b.assign(rb.rbegin(), rb.rend());
    return r;
}

// ----------------------------------------------------------------------------
//  cpu_align()  (PUBLIC)
//  ---------------------
//  The full pipeline: allocate the matrix, fill it, then trace back. This is
//  the function the demo calls to get a complete, human-readable alignment.
// ----------------------------------------------------------------------------
AlignmentResult cpu_align(const std::string& a,
                          const std::string& b,
                          const ScoringScheme& scheme,
                          AlignmentKind kind) {
    const int m = static_cast<int>(a.size());
    const int n = static_cast<int>(b.size());

    // Allocate (m+1)*(n+1) cells, zero-initialised. (Borders are overwritten by
    // fill_matrix; the zero-init also makes the empty-input edge cases trivial.)
    std::vector<cell_t> H(static_cast<size_t>(m + 1) * static_cast<size_t>(n + 1), 0);

    int best_i = 0, best_j = 0;
    int64_t best_score = 0;
    fill_matrix(a, b, scheme, kind, H, best_i, best_j, best_score);

    // Hand the filled matrix to the shared traceback. H.data() is a contiguous
    // cell_t* exactly matching the layout traceback_from_matrix expects.
    return traceback_from_matrix(H.data(), m, n, a, b, scheme, kind, best_i, best_j);
}

// ----------------------------------------------------------------------------
//  cpu_score_only()  (PUBLIC)
//  --------------------------
//  Computes ONLY the optimal score, using the classic two-rolling-rows trick to
//  drop memory from O(m*n) to O(n). The insight: to compute row i we only ever
//  read row i-1 and the cell immediately to our left in row i. So we never need
//  the whole matrix -- just "the previous row" and "the current row".
//
//  This is exactly the strategy the GPU "batched / inter-sequence" kernel uses
//  for each of its thousands of independent pairs, where keeping a full matrix
//  per pair would blow the memory budget. We provide the CPU twin here so the
//  test harness can confirm the rolling-row recurrence agrees with the
//  full-matrix one (and hence that the GPU's frugal version is trustworthy).
//
//  NOTE: with only two rows we CANNOT trace back (we threw the history away).
//  Score-only is the right tool when you are scanning a database and only care
//  *which* sequences score highly; you re-align just the winners in full later.
// ----------------------------------------------------------------------------
int64_t cpu_score_only(const std::string& a,
                       const std::string& b,
                       const ScoringScheme& scheme,
                       AlignmentKind kind) {
    // Orient so the SHORTER sequence sits on the COLUMNS. The two rolling rows
    // are sized to the number of columns, so putting the shorter sequence there
    // makes the memory O(min(m, n)) rather than O(len(b)). Swapping the two
    // inputs cannot change the optimal SCORE -- the alignment score is symmetric
    // (verify_main.cpp asserts exactly this) -- and because score-only does no
    // traceback, the orientation is otherwise invisible to the caller.
    const std::string& A = (a.size() >= b.size()) ? a : b;   // longer  -> rows
    const std::string& B = (a.size() >= b.size()) ? b : a;   // shorter -> columns
    const int m = static_cast<int>(A.size());                // # rows    (>= n)
    const int n = static_cast<int>(B.size());                // # columns (= min)
    const cell_t gap = scheme.gap;

    // Two rows of width (n+1). `prev` = row i-1, `curr` = row i. We swap their
    // roles after finishing each row instead of copying.
    std::vector<cell_t> prev(n + 1, 0);
    std::vector<cell_t> curr(n + 1, 0);

    // Initialise row 0 (the "i = 0" border) exactly like fill_matrix.
    if (kind == AlignmentKind::Global) {
        for (int j = 0; j <= n; ++j) prev[j] = static_cast<cell_t>(j) * gap;
    } else {
        for (int j = 0; j <= n; ++j) prev[j] = 0;
    }

    int64_t best = 0; // local alignment: running maximum seen so far

    for (int i = 1; i <= m; ++i) {
        const char ai = A[i - 1];

        // Left border of the current row (the "j = 0" column).
        curr[0] = (kind == AlignmentKind::Global) ? static_cast<cell_t>(i) * gap : 0;

        for (int j = 1; j <= n; ++j) {
            const char bj = B[j - 1];
            const cell_t diag = prev[j - 1] + substitution_score(ai, bj, scheme);
            const cell_t up   = prev[j]     + gap;
            const cell_t left = curr[j - 1] + gap;   // already-updated left cell

            cell_t v = diag;
            if (up   > v) v = up;
            if (left > v) v = left;
            if (kind == AlignmentKind::Local && v < 0) v = 0;

            curr[j] = v;
            if (kind == AlignmentKind::Local && v > best) best = v;
        }

        prev.swap(curr);   // O(1): the just-finished `curr` becomes next `prev`
    }

    // Global answer is the bottom-right cell, which -- after the final swap --
    // lives in prev[n]. Local answer is the running maximum.
    if (kind == AlignmentKind::Global) return static_cast<int64_t>(prev[n]);
    return best;
}

// ----------------------------------------------------------------------------
//  cpu_compute_matrix()  (PUBLIC)
//  ------------------------------
//  Thin wrapper around fill_matrix() that hands back the whole matrix so callers
//  can inspect or print it. (The best_i/best_j/best_score it computes are
//  discarded here; callers that need them use cpu_align instead.)
// ----------------------------------------------------------------------------
std::vector<cell_t> cpu_compute_matrix(const std::string& a,
                                       const std::string& b,
                                       const ScoringScheme& scheme,
                                       AlignmentKind kind) {
    const int m = static_cast<int>(a.size());
    const int n = static_cast<int>(b.size());
    std::vector<cell_t> H(static_cast<size_t>(m + 1) * static_cast<size_t>(n + 1), 0);
    int bi = 0, bj = 0; int64_t bs = 0;
    fill_matrix(a, b, scheme, kind, H, bi, bj, bs);
    return H;
}
