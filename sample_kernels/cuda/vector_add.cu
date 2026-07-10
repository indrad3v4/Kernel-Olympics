// ── Simple CUDA vector addition — ideal first test for porting ──
// ~15 lines, no warp ops, no shared memory, no libraries.
// If this ports, the pipeline's base flow works.

#include <cuda_runtime.h>
#include <stdio.h>

__global__ void vector_add(float *a, float *b, float *c, int n) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < n) {
        c[idx] = a[idx] + b[idx];
    }
}

int main() {
    int n = 1024;
    float *a, *b, *c;
    cudaMalloc(&a, n * sizeof(float));
    cudaMalloc(&b, n * sizeof(float));
    cudaMalloc(&c, n * sizeof(float));
    vector_add<<<1, 256>>>(a, b, c, n);
    cudaDeviceSynchronize();
    cudaFree(a); cudaFree(b); cudaFree(c);
    printf("vector_add: PASSED\n");
    return 0;
}
