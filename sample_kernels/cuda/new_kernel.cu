// Sample CUDA kernel: warp-level inclusive prefix sum (Hillis-Steele scan)
// Demonstrates __shfl_up_sync (upward lane shift) and __activemask()
// Both assume a 32-lane warp — silently wrong on AMD's 64-lane wavefront
//
// This kernel exercises the "shfl_up_sync" and "activemask" danger patterns.

__global__ void warp_scan_kernel(const float* input, float* output, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int lane = threadIdx.x % 32;  // BUG: hardcoded 32 — not wavefront-size-aware

    float val = (idx < n) ? input[idx] : 0.0f;

    // BUG: __activemask() returns a 32-bit lane mask — cannot represent all 64
    // lanes of an AMD wavefront, so lanes 32-63 are silently dropped from the mask
    unsigned mask = __activemask();

    // Inclusive prefix sum via upward shuffle (Hillis-Steele scan)
    // BUG: __shfl_up_sync offsets assume a 32-lane warp; on wavefront64 the
    // shuffle boundary is wrong and the scan produces incorrect partial sums
    for (int offset = 1; offset < 32; offset *= 2) {
        float n_val = __shfl_up_sync(mask, val, offset);
        if (lane >= offset) {
            val += n_val;
        }
    }

    if (idx < n) {
        output[idx] = val;
    }
}
