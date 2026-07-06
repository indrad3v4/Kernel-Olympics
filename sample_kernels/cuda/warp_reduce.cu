// Sample CUDA kernel with warp divergence — a shuffle-based reduction
// This is the CLASSIC example of warp(32) → wavefront(64) breakage
// On AMD GPUs: wavefront = 64 threads, not 32
// __shfl_down_sync(0xffffffff, val, 16) moves data by 16 — but in wavefront 64,
// this skips half the lanes. The reduction produces WRONG results silently.

__global__ void warp_reduce_kernel(const float* input, float* output, int n) {
    __shared__ float shared[32];  // BUG: hardcoded to 32 (warp size on NVIDIA)
    
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Load data
    shared[tid] = (idx < n) ? input[idx] : 0.0f;
    __syncthreads();
    
    // Warp-level reduction using shuffle
    float val = shared[tid];
    
    // DANGER: these assume warp size = 32
    val += __shfl_down_sync(0xffffffff, val, 16);  // BUG: on wavefront64, moves by 16 not 32
    val += __shfl_down_sync(0xffffffff, val, 8);   // OK: 8 < 32
    val += __shfl_down_sync(0xffffffff, val, 4);
    val += __shfl_down_sync(0xffffffff, val, 2);
    val += __shfl_down_sync(0xffffffff, val, 1);
    
    if (tid == 0) {
        output[blockIdx.x] = val;
    }
}
