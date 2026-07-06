// Sample CUDA kernel: warp-level histogram with atomic operations
// Demonstrates __shfl_xor_sync (butterfly) for warp-level atomics
// and warp-size-dependent shared memory patterns
//
// This kernel would be used for the "pattern memory speeds up" demo
// because it shares structural similarity with warp_reduce.cu

__global__ void histogram_kernel(const float* input, int* histogram,
                                  int n, int num_bins) {
    extern __shared__ int shared_hist[];
    
    // Initialize shared histogram to zero
    int tid = threadIdx.x;
    if (tid < 256) {
        shared_hist[tid] = 0;
    }
    __syncthreads();
    
    // Each thread processes its assigned elements
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;
    
    while (idx < n) {
        int bin = (int)(input[idx] * num_bins);
        bin = min(max(bin, 0), num_bins - 1);
        
        // Warp-level atomic using shuffle (butterfly pattern)
        // DANGER: __shfl_xor_sync assumes 32-lane warp
        int lane_mask = __ballot_sync(0xffffffff, 1);  // Get active lanes
        
        // Butterfly reduction within warp
        // BUG: XOR shuffle on wavefront64 exchanges lanes across the wrong boundary
        int val = 1;
        val += __shfl_xor_sync(0xffffffff, val, 16);  // BUG: wavefront64 = 64 lanes
        val += __shfl_xor_sync(0xffffffff, val, 8);
        val += __shfl_xor_sync(0xffffffff, val, 4);
        val += __shfl_xor_sync(0xffffffff, val, 2);
        val += __shfl_xor_sync(0xffffffff, val, 1);
        
        // Warp leader atomically adds to shared histogram
        if ((tid & 0x1f) == 0) {  // BUG: uses 0x1f (32) as warp mask
            atomicAdd(&shared_hist[bin], val);
        }
        
        idx += stride;
    }
    
    __syncthreads();
    
    // Warp leader writes to global histogram
    // BUG: hardcoded warp size check
    if ((tid & 0x1f) == 0) {
        // Warp leader atomically adds warp-local result to global
        const int WARP_SIZE = 32;  // BUG: should be wavefront size on AMD
        for (int i = 0; i < num_bins; i += warpSize) {
            if (tid + i < 256) {
                atomicAdd(&histogram[tid + i], shared_hist[tid + i]);
            }
        }
    }
}
