// ============================================================================
//  alignment_common.cpp
// ----------------------------------------------------------------------------
//  Implements the small, GPU-independent helpers declared in
//  include/alignment_common.h:
//
//      * format_alignment() -- turn an AlignmentResult into a pretty 3-line block
//      * random_sequence()  -- deterministic random DNA generator for tests
//
//  There is nothing CUDA-specific here, so this file is compiled by the plain
//  host C++ compiler. Keeping these utilities out of the .cu files also keeps
//  nvcc's job (and its error messages) smaller.
// ============================================================================

#include "alignment_common.h"

#include <sstream>   // std::ostringstream -- build the formatted block piece by piece
#include <cstdint>   // uint32_t

// ----------------------------------------------------------------------------
//  format_alignment()
//  ------------------
//  Given the two gapped rows (aligned_a / aligned_b), produce three text lines:
//
//      line 1: aligned_a
//      line 2: a "bar" that marks each column as match '|', mismatch '.', or
//              gap ' '
//      line 3: aligned_b
//
//  The two gapped rows are GUARANTEED by construction to have equal length
//  (every alignment step appends exactly one character -- possibly '-' -- to
//  each row), so a single index `i` can walk both simultaneously.
// ----------------------------------------------------------------------------
std::string format_alignment(const AlignmentResult& r, const ScoringScheme& scheme) {
    // `scheme` is currently unused for the *symbols* (we decide match/mismatch
    // purely by character equality), but we accept it so that, if you later
    // switch to a substitution matrix where "similar but not identical" letters
    // deserve their own symbol (the classic BLAST '+'), you already have it in
    // hand. Cast to void to silence "unused parameter" warnings for now.
    (void)scheme;

    std::ostringstream top;   // accumulates the top sequence row
    std::ostringstream mid;   // accumulates the match/mismatch/gap bar
    std::ostringstream bot;   // accumulates the bottom sequence row

    // Walk every column of the alignment exactly once.
    for (size_t i = 0; i < r.aligned_a.size(); ++i) {
        const char a = r.aligned_a[i];
        const char b = r.aligned_b[i];

        top << a;             // top row is just sequence A as-is
        bot << b;             // bottom row is just sequence B as-is

        // Decide the connector symbol for this column:
        if (a == '-' || b == '-') {
            mid << ' ';       // a gap column: nothing "matches", leave blank
        } else if (a == b) {
            mid << '|';       // identical letters: a solid match bar
        } else {
            mid << '.';       // different letters: a mismatch dot
        }
    }

    // Glue the three lines together with newlines into one returnable string.
    std::ostringstream out;
    out << top.str() << '\n'
        << mid.str() << '\n'
        << bot.str();
    return out.str();
}

// ----------------------------------------------------------------------------
//  random_sequence()
//  -----------------
//  A self-contained, fully deterministic DNA generator.
//
//  WHY NOT <random>?  The C++ <random> distributions are NOT guaranteed to
//  produce identical output across compilers/standard-library versions. For a
//  STUDY project where "run the test, it fails, paste the seed, reproduce
//  exactly" is a core workflow, we want bit-for-bit reproducibility. A 5-line
//  xorshift PRNG gives us that with zero dependencies.
//
//  THE GENERATOR: "xorshift32" by George Marsaglia. It maintains a single
//  32-bit state and, on each step, XORs the state with shifted copies of
//  itself. It is NOT cryptographically secure and NOT the highest-quality PRNG,
//  but it is more than random enough to stress-test an alignment kernel, and it
//  is famous precisely for being tiny and portable.
//
//  The state is passed by reference (`seed_inout`) and advanced in place, so a
//  caller can keep calling random_sequence() and keep getting fresh data.
// ----------------------------------------------------------------------------
std::string random_sequence(int length, uint32_t& seed_inout) {
    // Guard against a zero seed: xorshift has a fixed point at 0 (0 XOR 0 == 0),
    // so a zero state would emit nothing but zeros forever. Nudge it to a
    // nonzero constant if needed.
    if (seed_inout == 0) seed_inout = 0x9E3779B9u; // (a chunk of the golden ratio)

    static const char ALPHABET[4] = {'A', 'C', 'G', 'T'};

    std::string out;
    out.reserve(static_cast<size_t>(length)); // avoid repeated reallocation

    uint32_t x = seed_inout;                   // pull the running state local
    for (int i = 0; i < length; ++i) {
        // --- one xorshift32 step -------------------------------------------
        // Each line mixes the bits a little more. The shift amounts (13,17,5)
        // are Marsaglia's published constants that give the generator its full
        // period of 2^32 - 1.
        x ^= x << 13;
        x ^= x >> 17;
        x ^= x << 5;
        // -------------------------------------------------------------------

        // Map the 32 random bits down to one of four bases. We take the low two
        // bits; for this generator they are well-mixed enough for our purposes.
        out.push_back(ALPHABET[x & 0x3u]);
    }

    seed_inout = x;   // write the advanced state back so the next call differs
    return out;
}
