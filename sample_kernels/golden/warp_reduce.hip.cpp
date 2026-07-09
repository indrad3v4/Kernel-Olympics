// GOLDEN reference port — warp_reduce.cu → correct wavefront-64 HIP.
//
// Regression oracle for the deterministic warp rewriter (P1.1) and the
// verifier. This is what a *correct* CUDA→ROCm port of the classic
// shuffle-reduction looks like on AMD, where a wavefront is 64 lanes, not 32.
//
// Three things had to change vs. the NVIDIA source, and they split cleanly
// into "mechanical" (the rewriter does these) and "algorithmic" (it must not):
//
//   1. shared[32] → shared[64]   — the block is one 64-lane wavefront, so
//      shared[tid] with tid up to 63 indexes out of a 32-element array. This
//      is the silent out-of-bounds that SIGSEGVs the naive port. [algorithmic:
//      depends on the launch width, left to the porter]
//   2. add the offset-32 step     — a 64-lane reduction needs 6 shuffle steps
//      (32,16,8,4,2,1), not the 5 (16,8,4,2,1) that sum a 32-lane warp.
//      Missing it sums only half the wavefront. [algorithmic — the rewriter
//      deliberately does NOT touch reduction step counts]
//   3. 0xffffffff → maskless __shfl_down  — the 32-bit participation mask
//      cannot address lanes 32-63. [mechanical — the rewriter widens the mask;
//      here we use HIP's idiomatic maskless __shfl_down, equivalent on AMD]
//
// NOTE: certified-by-inspection only. To CERTIFY, compile + run on an AMD GPU
// (hipcc) and diff against sample_kernels/reference/warp_reduce_output.txt.

#include <hip/hip_runtime.h>

__global__ void warp_reduce_kernel(const float* input, float* output, int n) {
    __shared__ float shared[64];  // one wavefront = 64 lanes on AMD

    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    shared[tid] = (idx < n) ? input[idx] : 0.0f;
    __syncthreads();

    float val = shared[tid];

    // 64-lane shuffle reduction: 6 steps. The offset-32 step is the one the
    // 32-lane NVIDIA source omits.
    val += __shfl_down(val, 32);
    val += __shfl_down(val, 16);
    val += __shfl_down(val, 8);
    val += __shfl_down(val, 4);
    val += __shfl_down(val, 2);
    val += __shfl_down(val, 1);

    if (tid == 0) {
        output[blockIdx.x] = val;
    }
}
