#include <cstdio>
#include <cstdlib>
#include <cstring>

int main() {
    // Reference for nvidia_simple_atomics test
    //
    // The CUDA sample launches testKernel<<<64, 256>>>(dOData) with 11 ints.
    // The kernel uses global-memory atomics (atomicAdd, atomicSub, atomicExch,
    // atomicMin, atomicMax, atomicAnd, atomicOr, atomicXor) on different
    // elements of the output buffer.
    //
    // hOData is initialized to 0, except hOData[8] = hOData[10] = 0xff.
    //
    // Kernel testKernel does the following (from simpleAtomicIntrinsics_kernel.cuh):
    //   tid = threadIdx.x + blockIdx.x * blockDim.x
    //   atomicAdd(&gpuData[0], 1)           —  64*256 = 16384 increments
    //   atomicSub(&gpuData[1], 1)           —  16384 decrements → -16384
    //   atomicExch(&gpuData[2], tid)        —  last thread wins: tid = 16383
    //   atomicMin(&gpuData[3], tid)         —  min = 0
    //   atomicMax(&gpuData[4], tid)         —  max = 16383
    //   atomicAnd(&gpuData[5], tid)         —  starts 0→tid=0 only matches, result 0
    //   atomicOr(&gpuData[6], tid)          —  starts 0, OR accumulates, result = 16383
    //   atomicXor(&gpuData[7], tid)         —  starts 0, XOR accumulates
    //   atomicAnd(&gpuData[8], tid)         —  starts 0xff = 255, AND with tid → 0
    //   atomicOr(&gpuData[9], tid)          —  starts 0, OR accumulates
    //   atomicXor(&gpuData[10], tid)        —  starts 0xff, XOR accumulates

    const int numData = 11;
    int gpuData[11];
    memset(gpuData, 0, sizeof(gpuData));
    gpuData[8] = 0xff;
    gpuData[10] = 0xff;

    const int numThreads = 256;
    const int numBlocks = 64;
    const int totalThreads = numThreads * numBlocks;

    for (int tid = 0; tid < totalThreads; tid++) {
        // atomicAdd
        gpuData[0]++;
        // atomicSub
        gpuData[1]--;
        // atomicExch — last writer wins
        gpuData[2] = tid;
        // atomicMin
        if (tid < gpuData[3]) gpuData[3] = tid;
        // atomicMax
        if (tid > gpuData[4]) gpuData[4] = tid;
        // atomicAnd
        gpuData[5] &= tid;
        // atomicOr
        gpuData[6] |= tid;
        // atomicXor
        gpuData[7] ^= tid;
        // atomicAnd (started 0xff)
        gpuData[8] &= tid;
        // atomicOr (started 0)
        gpuData[9] |= tid;
        // atomicXor (started 0xff)
        gpuData[10] ^= tid;
    }

    for (int i = 0; i < numData; i++)
        printf("%d\n", gpuData[i]);

    return 0;
}
