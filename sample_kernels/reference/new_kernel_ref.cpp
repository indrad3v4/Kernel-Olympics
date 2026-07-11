#include <cstdio>
#include <vector>

int main() {
    // Spec: grid(1,1,1), block(256,1,1), count=256, default_value=1.0
    // Kernel: warp-level inclusive prefix sum (Hillis-Steele scan)
    // output[i] = sum of input[0..i]
    int n = 256;

    // Try to read n from stdin
    scanf("%d", &n);
    if (n <= 0) n = 256;

    std::vector<float> input(n, 1.0f);

    // Read input values from stdin if available
    for (int i = 0; i < n && !feof(stdin); i++) {
        float v;
        if (scanf("%f", &v) == 1) input[i] = v;
    }

    // Inclusive prefix sum: output[i] = input[0] + input[1] + ... + input[i]
    std::vector<float> output(n);
    float running = 0.0f;
    for (int i = 0; i < n; i++) {
        running += input[i];
        output[i] = running;
    }

    // Print all output values
    for (int i = 0; i < n; i++)
        printf("%.1f\n", output[i]);

    return 0;
}
