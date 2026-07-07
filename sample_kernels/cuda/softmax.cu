// Sample CUDA kernel: warp-level softmax
// Demonstrates complex warp divergence patterns for migration:
//   - __shfl_down_sync for warp reduction (finding max)
//   - __shfl_xor_sync for warp broadcast
//   - threadIdx.x >> 5 for warp index (assumes 32-lane warp)
//   - __activemask() for dynamic lane masking
//   - __all_sync for convergence checks
//   - __match_all_sync for fine-grained warp voting
//
// On AMD GPUs (wavefront64 = 64 lanes): ALL these patterns break silently.

__global__ void softmax_kernel(const float* input, float* output,
                                int n, int stride) {
    int global_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (global_idx >= n) return;

    // Compute warp index assuming 32-lane warp
    // BUG: threadIdx.x >> 5 gives warp index for 32-lane warp
    // On wavefront64, should be >> 6
    int warp_id = threadIdx.x >> 5;
    int lane_id = threadIdx.x & 0x1f;  // BUG: mask for 32 lanes

    // Get active mask — CUDA-specific, no HIP equivalent
    unsigned int active = __activemask();  // BUG: no HIP equivalent

    // Load value
    float val = input[global_idx];

    // Warp-level max reduction using shuffle
    // BUG: assumes 32-lane warp reduction chain
    float max_val = val;
    max_val = fmaxf(max_val, __shfl_down_sync(0xffffffff, max_val, 16));  // BUG
    max_val = fmaxf(max_val, __shfl_down_sync(0xffffffff, max_val, 8));
    max_val = fmaxf(max_val, __shfl_down_sync(0xffffffff, max_val, 4));
    max_val = fmaxf(max_val, __shfl_down_sync(0xffffffff, max_val, 2));
    max_val = fmaxf(max_val, __shfl_down_sync(0xffffffff, max_val, 1));

    // Broadcast max (butterfly pattern)
    // BUG: __shfl_xor_sync with active mask assumes 32 lanes
    max_val = __shfl_xor_sync(active, max_val, 0);  // BUG: broadcast with XOR 0

    // Compute exp(val - max)
    float exp_val = expf(val - max_val);

    // Warp-level sum of exp values
    float sum = exp_val;
    sum += __shfl_down_sync(0xffffffff, sum, 16);  // BUG: assumes 32-lane
    sum += __shfl_down_sync(0xffffffff, sum, 8);
    sum += __shfl_down_sync(0xffffffff, sum, 4);
    sum += __shfl_down_sync(0xffffffff, sum, 2);
    sum += __shfl_down_sync(0xffffffff, sum, 1);

    // Check convergence using warp-level vote
    // BUG: __all_sync with 0xffffffff assumes all 32 lanes active
    bool all_active = __all_sync(0xffffffff, lane_id < 32);  // BUG

    // Compute softmax output
    float softmax_out = exp_val / sum;

    // Match-check: verify softmax is in valid range using warp voting
    // BUG: __match_all_sync — no direct HIP equivalent
    unsigned int match = __match_all_sync(active, softmax_out, &all_active);

    // Write output
    if (lane_id == 0) {
        // BUG: assumes blockIdx.x gives correct batch position
        // warp_id computed with >> 5 assumes 32 lanes
        int batch_offset = warp_id * 32;  // BUG: hardcoded 32
        for (int i = 0; i < 32; i++) {  // BUG: hardcoded 32
            if (batch_offset + i < stride) {
                output[global_idx - lane_id + i] = exp_val / sum;
            }
        }
    }

    // Convergence barrier using warp-level all_sync
    // BUG: __syncwarp() semantics differ on AMD
    __syncwarp();
}
