// ============================================================================
//  gpu_wavefront.cuh
// ----------------------------------------------------------------------------
//  Declares the GPU implementation that parallelises a SINGLE pairwise
//  alignment using the "anti-diagonal wavefront" technique. This is the
//  intellectual centrepiece of the project, so read gpu_wavefront.cu's long
//  header comment for the full explanation; here we just expose the entry
//  point.
//
//  "INTRA-SEQUENCE" PARALLELISM: we take ONE alignment of two (possibly long)
//  sequences and spread the work of filling its DP matrix across thousands of
//  GPU threads. Contrast with gpu_batched.cuh, which runs MANY small alignments
//  at once with one thread each ("inter-sequence" parallelism). They are two
//  fundamentally different ways to use a GPU for the same algorithm, and seeing
//  both side by side is the whole point of this study project.
// ============================================================================

#ifndef GPU_WAVEFRONT_CUH
#define GPU_WAVEFRONT_CUH

#include <string>
#include "alignment_common.h"   // ScoringScheme, AlignmentKind, AlignmentResult

// ----------------------------------------------------------------------------
//  gpu_wavefront_align()
//  ---------------------
//  Compute the optimal alignment of `a` vs `b` on the GPU and return the SAME
//  fully-populated AlignmentResult that cpu_align() would. The result is
//  intended to be bit-for-bit identical to the CPU reference -- that equality
//  is exactly what the test harness asserts.
//
//  Pipeline (all orchestrated on the host inside this one call):
//    1. Upload the two sequences to device global memory.
//    2. Launch an init kernel to set the matrix borders.
//    3. Launch ONE kernel per anti-diagonal to fill the interior in parallel.
//    4. Copy the finished matrix back to the host.
//    5. Run the shared, serial traceback_from_matrix() on the host.
//
//  Parameters
//    a, b           : input sequences.
//    scheme         : match/mismatch/gap rulebook.
//    kind           : Global (NW) or Local (SW).
//    out_kernel_ms  : optional; if non-null, receives the measured GPU time
//                     spent in the fill kernels (excludes host<->device copies),
//                     so callers can report a fair "compute-only" timing.
//
//  This function is declared here (host-callable C++), but DEFINED in
//  gpu_wavefront.cu where the actual __global__ kernels live.
// ----------------------------------------------------------------------------
AlignmentResult gpu_wavefront_align(const std::string& a,
                                    const std::string& b,
                                    const ScoringScheme& scheme,
                                    AlignmentKind kind,
                                    float* out_kernel_ms = nullptr);

// ----------------------------------------------------------------------------
//  gpu_print_device_info()
//  -----------------------
//  Prints a short summary of GPU 0 (name, compute capability, memory, SM count).
//  Declared here -- a host-safe header that pulls in NO CUDA headers -- so that
//  the pure-C++ demo can call it without itself depending on <cuda_runtime.h>.
//  The implementation (which DOES use the CUDA runtime) lives in
//  gpu_wavefront.cu, compiled by nvcc.
// ----------------------------------------------------------------------------
void gpu_print_device_info();

#endif // GPU_WAVEFRONT_CUH
