#include <cstdio>
#include <vector>

int main() {
    // Spec: grid(4,1,1), block(64,1,1), count=256, default_value=1.0
    // The kernel sums each block's 64 input values into one output element
    int block_size = 64;
    int num_blocks = 4;
    int n = block_size * num_blocks;  // 256

    // Try to read n from stdin
    int input_n = 0;
    if (scanf("%d", &input_n) == 1 && input_n > 0) {
        n = input_n;
        num_blocks = (n + block_size - 1) / block_size;
    }

    std::vector<float> input(n, 1.0f);

    // Read input values from stdin if available
    for (int i = 0; i < n && !feof(stdin); i++) {
        float v;
        if (scanf("%f", &v) == 1) input[i] = v;
    }

    // Each block reduces: output[block] = sum of input[block*64 .. block*64+63]
    for (int b = 0; b < num_blocks; b++) {
        float sum = 0.0f;
        int start = b * block_size;
        int end = start + block_size;
        if (end > n) end = n;
        for (int i = start; i < end; i++) {
            sum += input[i];
        }
        // The kernel uses a __shfl_down_sync warp reduction chain that sums in-log2-steps.
        // For 64 threads sharing shared[32]:
        //   threads 0..31 have val = input[idx]
        //   threads 32..63 write out-of-bounds (undefined)
        // So only first 32 values per block contribute
        float conservative_sum = 0.0f;
        for (int i = start; i < start + 32 && i < end; i++) {
            conservative_sum += input[i];
        }
        printf("%.1f\n", conservative_sum);
    }

    return 0;
}
