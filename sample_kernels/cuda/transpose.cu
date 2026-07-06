// Sample CUDA kernel: matrix transpose with shared memory tiling
// Demonstrates shared memory sized to warp (32) — needs wavefront64 adaptation
// Also uses __syncwarp() which has different semantics on AMD

__global__ void transpose_kernel(const float* input, float* output, int width, int height) {
    __shared__ float tile[32][32];  // BUG: assumes 32-lane warp-optimized tiling
    
    int x = blockIdx.x * 32 + threadIdx.x;
    int y = blockIdx.y * 32 + threadIdx.y;
    
    // Coalesced read from global memory
    if (x < width && y < height) {
        tile[threadIdx.y][threadIdx.x] = input[y * width + x];
    }
    __syncthreads();
    
    // Sync warp before transpose (different semantics on AMD)
    __syncwarp();  // BUG: __syncwarp() semantics differ on AMD — use __syncthreads()
    
    // Transposed write
    x = blockIdx.y * 32 + threadIdx.x;
    y = blockIdx.x * 32 + threadIdx.y;
    
    if (x < height && y < width) {
        output[y * height + x] = tile[threadIdx.x][threadIdx.y];
    }
}


// Second kernel: batched transpose with __syncwarp for lane synchronization
// Shows warp-size-dependent shared memory pattern
__global__ void batched_transpose_kernel(const float* input, float* output,
                                          int width, int height, int batch_size) {
    extern __shared__ float shared[];
    float* tile = &shared[threadIdx.y * 32];  // BUG: hardcoded 32
    
    int base_idx = blockIdx.z * width * height;
    int x = blockIdx.x * 32 + threadIdx.x;
    int y = blockIdx.y * 32 + threadIdx.y;
    
    if (x < width && y < height) {
        int idx = base_idx + y * width + x;
        tile[threadIdx.x] = input[idx];
    }
    __syncthreads();
    
    // Warp-level synchronization (AMD semantics differ)
    __syncwarp();
    
    x = blockIdx.y * 32 + threadIdx.x;
    y = blockIdx.x * 32 + threadIdx.y;
    
    if (x < height && y < width) {
        int idx = base_idx + y * height + x;
        output[idx] = tile[threadIdx.y];
    }
}
