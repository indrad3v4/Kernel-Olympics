// Sample CUDA kernel: 2D convolution with warp-level tiling
// Demonstrates:
//   - Shared memory tiling sized to 32 (warp size dependency)
//   - Warp-level synchronization with __syncwarp()
//   - Hardcoded 32 in loop bounds and offset arithmetic
//   - Cooperative warp-level load/store patterns
//
// On AMD GPUs: wavefront = 64 threads → tiling breaks silently

#define TILE_SIZE 32  // BUG: assumes 32-lane warp for optimal tiling
#define FILTER_RADIUS 3

__global__ void conv2d_kernel(const float* input, float* output,
                               int width, int height, int channels) {
    // BUG: shared memory tile assumes 32×32 warp-aligned access
    __shared__ float tile[TILE_SIZE + 2 * FILTER_RADIUS][TILE_SIZE + 2 * FILTER_RADIUS];

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int bx = blockIdx.x * TILE_SIZE;  // BUG: 32 instead of wavefront size
    int by = blockIdx.y * TILE_SIZE;

    // Cooperative load: each thread loads one element into shared memory
    int gx = bx + tx - FILTER_RADIUS;
    int gy = by + ty - FILTER_RADIUS;

    // BUG: uses 0x1f (32) as warp mask — assumes 32-lane warp
    const int WARP_MASK = 0x1f;

    if (gx >= 0 && gx < width && gy >= 0 && gy < height) {
        tile[ty + FILTER_RADIUS][tx + FILTER_RADIUS] = input[gy * width + gx];
    } else {
        tile[ty + FILTER_RADIUS][tx + FILTER_RADIUS] = 0.0f;
    }
    __syncthreads();

    // BUG: __syncwarp() semantics differ on AMD GPUs
    // Used here for warp-fused filter application
    if (tx < TILE_SIZE && ty < TILE_SIZE) {
        float sum = 0.0f;
        for (int dy = -FILTER_RADIUS; dy <= FILTER_RADIUS; dy++) {
            for (int dx = -FILTER_RADIUS; dx <= FILTER_RADIUS; dx++) {
                sum += tile[ty + FILTER_RADIUS + dy][tx + FILTER_RADIUS + dx];
            }
        }

        // Warp-level reduction across batch: bug-prone on wavefront64
        // BUG: __shfl_xor_sync assumes 32-lane warp
        sum += __shfl_xor_sync(WARP_MASK, sum, 16);  // BUG: 16 ≠ 32 for wavefront
        sum += __shfl_xor_sync(WARP_MASK, sum, 8);
        sum += __shfl_xor_sync(WARP_MASK, sum, 1);
        __syncwarp();  // BUG: __syncwarp instead of __syncthreads

        int ox = bx + tx;
        int oy = by + ty;
        if (ox < width && oy < height) {
            output[oy * width + ox] = sum;
        }
    }
}
