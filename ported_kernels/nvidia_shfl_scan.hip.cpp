#include <iostream>
#include <hip/hip_runtime.h>
#include <cmath>
#include <vector>
#include <cstdio>

#define HIP_CHECK(cmd) do { hipError_t e = cmd; if(e != hipSuccess) { printf("HIP ERR %s:%d: %s\n", __FILE__, __LINE__, hipGetErrorString(e)); return 1; } } while(0)

// Scan using shfl - takes log2(n) steps
// Uses warpSize so it works on both wave32 (RDNA3) and wave64 (CDNA).

__global__ void shfl_scan_test(int *data, int width, int *partial_sums = NULL)
{
    extern __shared__ int sums[];
    int id      = ((blockIdx.x * blockDim.x) + threadIdx.x);
    int lane_id = threadIdx.x % warpSize;
    int warp_id = threadIdx.x / warpSize;

    int value = data[id];

    int scan_width = (width < warpSize) ? width : warpSize;

#pragma unroll
    for (int i = 1; i <= scan_width; i *= 2) {
        int n = __shfl_up(value, i, scan_width);
        if (lane_id >= i)
            value += n;
    }

    if (threadIdx.x % warpSize == warpSize - 1) {
        sums[warp_id] = value;
    }

    __syncthreads();

    int n_warps = blockDim.x / warpSize;
    if (warp_id == 0 && lane_id < n_warps) {
        int warp_sum = sums[lane_id];

        for (int i = 1; i <= n_warps; i *= 2) {
            int n = __shfl_up(warp_sum, i, n_warps);
            if (lane_id >= i)
                warp_sum += n;
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

    if (partial_sums != NULL && threadIdx.x == blockDim.x - 1) {
        partial_sums[blockIdx.x] = value;
    }
}

__global__ void uniform_add(int *data, int *partial_sums, int len)
{
    __shared__ int buf;
    int id = ((blockIdx.x * blockDim.x) + threadIdx.x);
    if (id > len) return;
    if (threadIdx.x == 0) buf = partial_sums[blockIdx.x];
    __syncthreads();
    data[id] += buf;
}

int main() {
    const int n_elements = 65536;
    int *h_data = NULL, *h_partial_sums = NULL, *h_result = NULL;
    int *d_data = NULL, *d_partial_sums = NULL;

    int sz = sizeof(int) * n_elements;
    int blockSize = 256;
    int gridSize = n_elements / blockSize;
    int nWarps = blockSize / 32;  // conservative (RDNA3 wave32)
    int shmem_sz = nWarps * sizeof(int);
    int n_partialSums = n_elements / blockSize;
    int partial_sz = n_partialSums * sizeof(int);

    hipDeviceProp_t prop;
    HIP_CHECK(hipGetDeviceProperties(&prop, 0));
    int ws = prop.warpSize;  // 32 on RDNA3, 64 on CDNA
    printf("Starting nvidia_shfl_scan on AMD ROCm\n");
    printf("Device: %s\n", prop.name);
    printf("warpSize: %d\n", ws);
    printf("gridSize: %d blockSize: %d nWarps: %d\n", gridSize, blockSize, nWarps);
    printf("n_elements: %d shmem_sz: %d\n", n_elements, shmem_sz);

    HIP_CHECK(hipHostMalloc(&h_data, sz));
    HIP_CHECK(hipHostMalloc(&h_result, sz));
    HIP_CHECK(hipHostMalloc(&h_partial_sums, partial_sz));

    for (int i = 0; i < n_elements; i++) h_data[i] = 1;

    HIP_CHECK(hipMalloc(&d_data, sz));
    HIP_CHECK(hipMalloc(&d_partial_sums, partial_sz));
    HIP_CHECK(hipMemset(d_partial_sums, 0, partial_sz));
    HIP_CHECK(hipMemcpy(d_data, h_data, sz, hipMemcpyHostToDevice));

    hipEvent_t start, stop;
    HIP_CHECK(hipEventCreate(&start));
    HIP_CHECK(hipEventCreate(&stop));
    float et = 0;

    HIP_CHECK(hipEventRecord(start, 0));
    printf("Launch 1: shfl_scan_test grid=%d block=%d shmem=%d\n", gridSize, blockSize, shmem_sz);
    fflush(stdout);
    hipLaunchKernelGGL(shfl_scan_test, dim3(gridSize), dim3(blockSize), shmem_sz, 0, d_data, 32, d_partial_sums);
    HIP_CHECK(hipGetLastError());
    HIP_CHECK(hipDeviceSynchronize());
    printf("Launch 1 OK\n");

    int p_blockSize = (n_partialSums < blockSize) ? n_partialSums : blockSize;
    int p_gridSize = (n_partialSums + p_blockSize - 1) / p_blockSize;
    printf("Launch 2: shfl_scan_test grid=%d block=%d shmem=%d\n", p_gridSize, p_blockSize, shmem_sz);
    fflush(stdout);
    hipLaunchKernelGGL(shfl_scan_test, dim3(p_gridSize), dim3(p_blockSize), shmem_sz, 0, d_partial_sums, 32);
    HIP_CHECK(hipGetLastError());
    HIP_CHECK(hipDeviceSynchronize());
    printf("Launch 2 OK\n");

    printf("Launch 3: uniform_add grid=%d block=%d\n", gridSize - 1, blockSize);
    fflush(stdout);
    hipLaunchKernelGGL(uniform_add, dim3(gridSize - 1), dim3(blockSize), 0, 0, d_data + blockSize, d_partial_sums, n_elements);
    HIP_CHECK(hipGetLastError());
    HIP_CHECK(hipDeviceSynchronize());
    printf("Launch 3 OK\n");

    HIP_CHECK(hipEventRecord(stop, 0));
    HIP_CHECK(hipEventSynchronize(stop));
    HIP_CHECK(hipEventElapsedTime(&et, start, stop));

    HIP_CHECK(hipMemcpy(h_result, d_data, sz, hipMemcpyDeviceToHost));
    HIP_CHECK(hipMemcpy(h_partial_sums, d_partial_sums, partial_sz, hipMemcpyDeviceToHost));

    printf("Test Sum: %d\n", h_partial_sums[n_partialSums - 1]);
    printf("Time (ms): %f\n", et);

    // CPU verify
    for (int i = 0; i < n_elements - 1; i++) {
        h_data[i + 1] = h_data[i] + h_data[i + 1];
    }
    long long diff = 0;
    for (int i = 0; i < n_elements; i++) {
        diff += h_data[i] - h_result[i];
    }
    printf("CPU verify result diff (GPUvsCPU) = %lld\n", diff);
    bool pass = (diff == 0);
    printf("TEST: %s\n", pass ? "PASSED" : "FAILED");

    HIP_CHECK(hipHostFree(h_data));
    HIP_CHECK(hipHostFree(h_result));
    HIP_CHECK(hipHostFree(h_partial_sums));
    HIP_CHECK(hipFree(d_data));
    HIP_CHECK(hipFree(d_partial_sums));
    HIP_CHECK(hipEventDestroy(start));
    HIP_CHECK(hipEventDestroy(stop));

    return pass ? 0 : 1;
}
