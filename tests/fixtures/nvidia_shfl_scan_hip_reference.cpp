// Reference DEVICE_SUBSET HIP port of nvidia_shfl_scan.cu.
//
// The original CUDA source is a full NVIDIA sample program whose host
// driver depends on unvendored ``helper_cuda.h``, ``helper_functions.h``,
// and ``shfl_integral_image.cuh``. Under port_mode == DEVICE_SUBSET only
// the __global__ / __device__ code is ported here; the test harness is
// synthesized by ``verifier._harness_from_spec`` from
// ``src/verification/specs/nvidia_shfl_scan.json``.
//
// Correctness on AMD wavefront64:
//   * ``__shfl_up_sync(mask, value, i, width)`` becomes ``__shfl_up(value, i,
//     width)`` — HIP's non-``_sync`` intrinsics do the same thing, and the
//     CUDA mask arg is a no-op on ROCm (all lanes participate).
//   * ``warpSize`` is 64 on gfx targets, so the shared-memory scratchpad
//     ``extern __shared__ int sums[]`` is sized by 64, not by a hard-coded
//     32. The block is launched with ``block.x == 64`` in the spec so
//     ``blockDim.x / warpSize == 1``, which means the inter-warp reduction
//     is a no-op — correct for a single-wavefront block.
//   * ``width`` in ``__shfl_up`` is preserved — silently swapping 32 → 64
//     would change which lanes contribute to each partial and break the
//     scan-tree invariant.
//   * No host code (main, shuffle_simple_test, shuffle_integral_image_test,
//     CPUverify, iDivUp) — the verifier's synthesized harness calls
//     ``shfl_scan_test`` directly and reads back ``partial_sums``.

#include <hip/hip_runtime.h>

__global__ void shfl_scan_test(int* data, int width, int* partial_sums)
{
    extern __shared__ int sums[];
    int id = ((blockIdx.x * blockDim.x) + threadIdx.x);
    int lane_id = id % warpSize;
    int warp_id = threadIdx.x / warpSize;

    int value = data[id];

    for (int i = 1; i <= width; i *= 2) {
        int n = __shfl_up(value, i, width);
        if (lane_id >= i) {
            value += n;
        }
    }

    if (threadIdx.x % warpSize == warpSize - 1) {
        sums[warp_id] = value;
    }

    __syncthreads();

    if (warp_id == 0 && lane_id < (blockDim.x / warpSize)) {
        int warp_sum = sums[lane_id];
        int cap = (blockDim.x / warpSize);
        for (int i = 1; i <= cap; i *= 2) {
            int n = __shfl_up(warp_sum, i, cap);
            if (lane_id >= i) {
                warp_sum += n;
            }
        }
        sums[lane_id] = warp_sum;
    }

    __syncthreads();

    int blockSum = 0;
    if (warp_id > 0) {
        blockSum = sums[warp_id - 1];
    }
    value += blockSum;

    data[id] = value;

    if (partial_sums != nullptr && threadIdx.x == blockDim.x - 1) {
        partial_sums[blockIdx.x] = value;
    }
}

__global__ void uniform_add(int* data, int* partial_sums, int len)
{
    __shared__ int buf;
    int id = ((blockIdx.x * blockDim.x) + threadIdx.x);

    if (id > len) {
        return;
    }

    if (threadIdx.x == 0) {
        buf = partial_sums[blockIdx.x];
    }

    __syncthreads();
    data[id] += buf;
}
