// ============================================================================
//  cuda_utils.cuh
// ----------------------------------------------------------------------------
//  Small, reusable CUDA host-side utilities shared by the GPU implementations:
//
//      * CUDA_CHECK(...)       -- wrap any CUDA runtime call so errors are
//                                 caught at the exact line they happen.
//      * CUDA_CHECK_KERNEL()   -- check the *most recent kernel launch*.
//      * GpuTimer              -- a RAII-ish stopwatch built on CUDA events that
//                                 measures real on-GPU elapsed time.
//
//  This file is included ONLY by .cu translation units (compiled by nvcc),
//  because it pulls in <cuda_runtime.h>. Plain host .cpp files must not include
//  it. The ".cuh" extension is the conventional marker for "CUDA header".
//
//  WHY ALL THIS CEREMONY FOR ERROR CHECKING?
//  -----------------------------------------
//  CUDA runtime functions do NOT throw and do NOT abort on failure -- they
//  return a cudaError_t code that is silently ignored unless you check it. Even
//  worse, KERNEL launches are asynchronous: a launch can appear to "succeed"
//  and the actual error (illegal memory access, too many resources, ...) only
//  surfaces later, on the next synchronising call. Beginners therefore lose
//  hours to bugs that a one-line check would have pinpointed instantly. The
//  macros below make checking effortless, so there is no excuse to skip it.
// ============================================================================

#ifndef CUDA_UTILS_CUH
#define CUDA_UTILS_CUH

#include <cuda_runtime.h>   // cudaError_t, cudaMalloc, cudaEvent_t, ...
#include <cstdio>           // std::fprintf
#include <cstdlib>          // std::exit, EXIT_FAILURE

// ----------------------------------------------------------------------------
//  cuda_check_impl()
//  -----------------
//  The function the CUDA_CHECK macro expands into. Kept as a real function (not
//  buried in the macro) so the logic is debuggable and only the file/line
//  capture needs to live in the macro.
//
//    err  : the cudaError_t returned by the call we are checking.
//    expr : the literal source text of that call (for the message).
//    file : __FILE__ captured at the call site.
//    line : __LINE__ captured at the call site.
//
//  On any error we print a precise, copy-pasteable diagnostic and hard-exit.
//  For a teaching/demo tool, failing loudly and immediately is the right
//  behaviour; a production library would instead propagate the error upward.
// ----------------------------------------------------------------------------
inline void cuda_check_impl(cudaError_t err, const char* expr,
                            const char* file, int line) {
    if (err != cudaSuccess) {
        std::fprintf(stderr,
                     "\n[CUDA ERROR] %s\n"
                     "   at %s:%d\n"
                     "   call: %s\n",
                     cudaGetErrorString(err), file, line, expr);
        std::exit(EXIT_FAILURE);
    }
}

// CUDA_CHECK(call): wrap any cuda* call that returns cudaError_t.
//   Usage:  CUDA_CHECK(cudaMalloc(&p, bytes));
//   The do/while(0) idiom makes the macro behave like a single statement so it
//   is safe to use inside an unbraced if/else. # stringises the argument so the
//   error message can show the exact call text.
#define CUDA_CHECK(call)                                              \
    do {                                                             \
        cudaError_t _err = (call);                                   \
        cuda_check_impl(_err, #call, __FILE__, __LINE__);            \
    } while (0)

// CUDA_CHECK_KERNEL(): call immediately AFTER a <<<...>>> kernel launch.
//   It checks two distinct things:
//     1. cudaGetLastError()      -- did the LAUNCH itself fail (e.g. bad config,
//                                   too many threads)? This is synchronous.
//     2. cudaDeviceSynchronize() -- wait for the kernel to finish and surface
//                                   any error that occurred DURING execution
//                                   (e.g. an out-of-bounds access). This is what
//                                   makes asynchronous kernel bugs visible.
//   In release-tuned code you would skip the synchronize for performance; in a
//   learning tool, catching the bug is worth the stall.
#define CUDA_CHECK_KERNEL()                                          \
    do {                                                             \
        cuda_check_impl(cudaGetLastError(),  "kernel launch",        \
                        __FILE__, __LINE__);                         \
        cuda_check_impl(cudaDeviceSynchronize(), "cudaDeviceSynchronize after kernel", \
                        __FILE__, __LINE__);                         \
    } while (0)

// ----------------------------------------------------------------------------
//  GpuTimer
//  --------
//  Measures elapsed GPU time using CUDA EVENTS. Why not just read the host
//  clock around a launch? Because kernel launches are asynchronous -- the host
//  call returns almost immediately, long before the GPU finishes. CUDA events
//  are timestamps recorded IN the GPU's command stream, so the measured
//  interval reflects real device-side work.
//
//  Usage:
//      GpuTimer t;
//      t.start();
//      my_kernel<<<...>>>(...);
//      t.stop();                       // records stop event + synchronises
//      float ms = t.elapsed_ms();
// ----------------------------------------------------------------------------
struct GpuTimer {
    cudaEvent_t beg;   // event marking the start
    cudaEvent_t end;   // event marking the stop

    GpuTimer() {
        // Creating events is cheap; we do it once per timer.
        CUDA_CHECK(cudaEventCreate(&beg));
        CUDA_CHECK(cudaEventCreate(&end));
    }

    ~GpuTimer() {
        // Events are a device resource; destroy them so we do not leak. We do
        // NOT use CUDA_CHECK here because throwing/exiting from a destructor is
        // poor form -- best-effort cleanup is enough.
        cudaEventDestroy(beg);
        cudaEventDestroy(end);
    }

    // Record the "start" timestamp into the default stream (stream 0). All
    // subsequent launches in that stream will be timed.
    void start() { CUDA_CHECK(cudaEventRecord(beg, 0)); }

    // Record the "stop" timestamp, then BLOCK the host until the GPU actually
    // reaches it -- otherwise elapsed_ms() below would read an event that has
    // not happened yet.
    void stop() {
        CUDA_CHECK(cudaEventRecord(end, 0));
        CUDA_CHECK(cudaEventSynchronize(end));
    }

    // Return the milliseconds of GPU time between the start and stop events.
    // cudaEventElapsedTime has ~0.5 microsecond resolution.
    float elapsed_ms() {
        float ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&ms, beg, end));
        return ms;
    }
};

#endif // CUDA_UTILS_CUH
