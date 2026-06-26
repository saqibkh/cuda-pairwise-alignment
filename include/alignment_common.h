// ============================================================================
//  alignment_common.h
// ----------------------------------------------------------------------------
//  PURPOSE
//  -------
//  This header is the *shared vocabulary* of the entire project. Every other
//  translation unit -- the serial CPU reference, the two CUDA implementations,
//  the demo, and the test harness -- includes this file so that they all agree
//  on:
//
//    * how an alignment is *scored*        (struct ScoringScheme)
//    * which *flavour* of alignment we run (enum  AlignmentKind)
//    * what a finished alignment *looks like* (struct AlignmentResult)
//    * a handful of tiny helper functions that are needed everywhere.
//
//  Keeping these definitions in ONE place is a deliberate design choice: if the
//  CPU and the GPU disagreed even slightly about what "+2 for a match" means,
//  our correctness checks (which compare GPU output against the CPU "golden"
//  output) would be meaningless. A single source of truth removes that risk.
//
//  BIOLOGICAL BACKGROUND (the 30-second version)
//  ---------------------------------------------
//  "Pairwise sequence alignment" asks: given two strings of letters (think DNA
//  bases A/C/G/T, or amino acids), what is the *best* way to line them up,
//  position by position, allowing us to insert "gaps" ('-') so that similar
//  regions stack on top of each other?  Two classic dynamic-programming (DP)
//  algorithms answer this:
//
//    * Needleman-Wunsch (1970) -> GLOBAL alignment: align the *entire* length
//      of both sequences end-to-end. Good when the two sequences are roughly
//      the same thing (e.g. the same gene in two species).
//
//    * Smith-Waterman (1981)   -> LOCAL alignment: find the single best-scoring
//      *sub-region* shared by the two sequences, ignoring the unrelated ends.
//      Good for finding a small motif inside a large sequence.
//
//  Both algorithms share ~90% of their machinery; they differ in only three
//  small but important details (border initialisation, a "clamp to zero" rule,
//  and where traceback starts/stops). We exploit that shared structure heavily.
//
//  This header is plain C++ (no CUDA): it is compiled by BOTH the host C++
//  compiler (cl.exe / g++) and by nvcc, so it must contain nothing that is
//  GPU-specific. CUDA-only helpers live in cuda_utils.cuh instead.
// ============================================================================

#ifndef ALIGNMENT_COMMON_H            // classic include guard: prevents this
#define ALIGNMENT_COMMON_H            // header being textually pasted twice into
                                      // the same translation unit, which would
                                      // cause "redefinition" compile errors.

#include <string>                     // std::string -- we store sequences and
                                      // the gapped alignment rows as strings.
#include <vector>                     // std::vector -- used by callers for
                                      // batches of sequences.
#include <cstdint>                    // fixed-width integer types (int64_t etc.)

// ----------------------------------------------------------------------------
//  cell_t -- the type of ONE cell in the dynamic-programming score matrix.
//  ----------------------------------------------------------------------------
//  We use a plain 32-bit `int`. Why 32 bits and not 64?
//    * MEMORY. The DP matrix has (m+1)*(n+1) cells. At 4 bytes/cell a
//      4096 x 4096 alignment already needs ~67 MB; at 8 bytes it would be
//      ~134 MB. On the GPU, where memory bandwidth is the bottleneck, halving
//      the matrix size directly speeds the kernel up. 4 bytes is the right
//      default.
//    * RANGE. The largest possible score is bounded by (length * match). With
//      match=+2, a sequence of ~1 billion characters would be needed to
//      overflow a signed 32-bit int. Our sequences are far shorter, so int is
//      safe. (The PUBLIC AlignmentResult::score is still int64_t for headroom.)
//
//  Defining this as a single typedef means the CPU matrix, the GPU matrix, and
//  the shared traceback routine all agree on the element type -- which lets us
//  memcpy a GPU-filled matrix straight into the CPU traceback with no
//  conversion.
// ----------------------------------------------------------------------------
using cell_t = int;

// ----------------------------------------------------------------------------
//  ALN_HD -- portable "this function runs on BOTH host and device" marker.
//  ----------------------------------------------------------------------------
//  A handful of tiny helpers (notably substitution_score()) are called from
//  THREE places: ordinary C++ on the CPU, AND inside CUDA kernels on the GPU.
//  CUDA requires any function called from device code to be tagged with the
//  `__device__` (or `__host__ __device__`) execution-space qualifier. But those
//  keywords are ONLY understood by nvcc -- the plain host compiler (cl.exe / g++)
//  that builds the .cpp files would choke on them.
//
//  nvcc predefines the macro `__CUDACC__` whenever it is the compiler. So we
//  define ALN_HD to expand to `__host__ __device__` under nvcc, and to NOTHING
//  under a normal C++ compiler. Marking a function `ALN_HD inline` then lets the
//  SAME source line compile correctly in both worlds, and -- importantly --
//  guarantees the CPU and GPU use byte-identical scoring logic.
// ----------------------------------------------------------------------------
#if defined(__CUDACC__)
  #define ALN_HD __host__ __device__
#else
  #define ALN_HD
#endif

// A sentinel meaning "minus infinity" for the recurrence. We use a value far
// below any reachable score but safely away from INT_MIN so that adding a
// (negative) gap penalty to it cannot underflow/wrap around. Used to disable
// impossible moves at the matrix borders.
static const cell_t NEG_INF = -1000000000; // -1e9, ~half of INT_MIN's magnitude

// ----------------------------------------------------------------------------
//  ScoringScheme
//  -------------
//  A "scoring scheme" is the rulebook that turns a *column* of an alignment
//  into a number. We use the simplest widely-taught model: a LINEAR gap model
//  with a single substitution reward/penalty pair.
//
//  For any aligned column there are exactly three possibilities:
//    1. two identical letters line up      -> add `match`    (a reward, > 0)
//    2. two different letters line up       -> add `mismatch` (a penalty, < 0)
//    3. a letter lines up against a gap '-' -> add `gap`      (a penalty, < 0)
//
//  The total alignment score is just the sum over all columns. The algorithms'
//  job is to choose the column layout (where to put the gaps) that MAXIMISES
//  this sum.
//
//  SIGN CONVENTION (read this carefully -- it bites everyone once):
//    We store `mismatch` and `gap` as the *actual* signed values we add. So a
//    typical DNA scheme is { match=+2, mismatch=-1, gap=-2 }. The code never
//    negates them again; it always *adds* them. Storing the sign in the data
//    (rather than writing `- gap` everywhere) keeps the recurrence equations
//    looking exactly like the textbook formulas.
//
//  WHY A SEPARATE LINEAR-GAP MODEL?  Real bioinformatics tools usually use an
//  "affine" gap model (a big penalty to *open* a gap, a smaller penalty to
//  *extend* it -- the Gotoh algorithm). Affine gaps need three DP matrices
//  instead of one and roughly triple the bookkeeping. We deliberately start
//  with the linear model because it makes the *parallelisation* ideas crystal
//  clear; affine gaps are noted throughout as an "extension exercise".
// ----------------------------------------------------------------------------
struct ScoringScheme {
    int match;     // score ADDED when two letters are equal      (e.g. +2)
    int mismatch;  // score ADDED when two letters differ         (e.g. -1)
    int gap;       // score ADDED for each gap position '-'       (e.g. -2)
};

// A sensible default DNA scoring scheme, handy for demos and tests. Declared
// `inline` so the single definition can live in a header without violating the
// One Definition Rule when many .cpp/.cu files include it.
inline ScoringScheme DefaultDnaScheme() {
    ScoringScheme s;
    s.match    =  2;   // reward identical bases
    s.mismatch = -1;   // mild penalty for a substitution
    s.gap      = -2;   // stronger penalty for an insertion/deletion (indel)
    return s;
}

// ----------------------------------------------------------------------------
//  AlignmentKind
//  -------------
//  Selects which of the two algorithms we run. We pass this around as a single
//  enum instead of duplicating every function, because (as noted above) the two
//  algorithms share almost all of their code. The few places that must behave
//  differently simply branch on this value.
//
//  Using `enum class` (a "scoped enumeration") rather than a bare `enum` means
//  you must write `AlignmentKind::Global`, not just `Global`. That avoids name
//  clashes and makes call sites self-documenting.
// ----------------------------------------------------------------------------
enum class AlignmentKind {
    Global,  // Needleman-Wunsch: align the full length of both sequences.
    Local    // Smith-Waterman:   find the best-scoring local sub-alignment.
};

// Human-readable name, used when printing results. `inline` for the same ODR
// reason as above.
inline const char* AlignmentKindName(AlignmentKind k) {
    return (k == AlignmentKind::Global) ? "Needleman-Wunsch (global)"
                                        : "Smith-Waterman (local)";
}

// ----------------------------------------------------------------------------
//  substitution_score()
//  ---------------------
//  The tiny but central helper that scores a single (letter, letter) pair.
//  It is called once per cell of the DP matrix, i.e. O(m*n) times, so it is
//  written to be trivially inlinable.
//
//  NOTE: real protein aligners replace this with a *substitution matrix*
//  (BLOSUM62, PAM250, ...) -- a 20x20 lookup table where, say, swapping
//  leucine for isoleucine costs less than swapping leucine for proline. Our
//  match/mismatch pair is the degenerate 2-value version of that idea. Pulling
//  it behind this one function means upgrading to BLOSUM later touches exactly
//  one place.
// ----------------------------------------------------------------------------
ALN_HD inline int substitution_score(char a, char b, const ScoringScheme& s) {
    return (a == b) ? s.match : s.mismatch;
}

// ----------------------------------------------------------------------------
//  AlignmentResult
//  ---------------
//  Everything a caller wants to know after an alignment finishes.
//
//  `score` is the optimal score (the number the DP maximised).
//
//  `aligned_a` / `aligned_b` are the two input sequences rewritten WITH gap
//  characters '-' inserted, and -- crucially -- they always have equal length.
//  Reading them top-to-bottom, column by column, reconstructs the alignment:
//
//        aligned_a:  G C A T G - C U
//        aligned_b:  G - A T T A C A
//                    | . | | . . | .   <- match(|) / mismatch(.) / gap
//
//  For a LOCAL (Smith-Waterman) alignment these strings hold only the matched
//  sub-region, and the begin_/end_ coordinates below tell you *where* in the
//  original sequences that region sits.
//
//  The begin_/end_ fields use 0-based, half-open-ish coordinates into the
//  ORIGINAL (ungapped) sequences:
//      begin_i, end_i  -> range covered in sequence A
//      begin_j, end_j  -> range covered in sequence B
//  For a global alignment these are simply 0..lenA and 0..lenB.
// ----------------------------------------------------------------------------
struct AlignmentResult {
    int64_t     score;      // optimal alignment score (can be negative for NW)
    std::string aligned_a;  // sequence A with '-' gaps inserted
    std::string aligned_b;  // sequence B with '-' gaps inserted
    int begin_i;            // start index in A (inclusive)
    int begin_j;            // start index in B (inclusive)
    int end_i;              // end   index in A (exclusive)
    int end_j;              // end   index in B (exclusive)
};

// ----------------------------------------------------------------------------
//  format_alignment()
//  -------------------
//  Produces the pretty three-line block shown above (sequence A, a "match bar",
//  sequence B) as a single printable std::string. Implemented in
//  alignment_common.cpp so that this header stays free of heavy <sstream> code.
//
//  `scheme` is needed so the middle "bar" can mark matches vs mismatches using
//  the SAME equality rule the scoring used.
// ----------------------------------------------------------------------------
std::string format_alignment(const AlignmentResult& r, const ScoringScheme& scheme);

// ----------------------------------------------------------------------------
//  random_sequence()
//  -----------------
//  Generates a random DNA string of the given length over the alphabet ACGT,
//  using a caller-supplied 32-bit seed. We implement our OWN tiny pseudo-random
//  generator (a "xorshift") rather than <random> so that the EXACT same numbers
//  come out on every platform and compiler -- reproducibility matters when a
//  test fails and you want to recreate it. The seed is advanced by reference so
//  successive calls produce different sequences.
// ----------------------------------------------------------------------------
std::string random_sequence(int length, uint32_t& seed_inout);

#endif // ALIGNMENT_COMMON_H
