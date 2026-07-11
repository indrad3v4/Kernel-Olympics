#include <cstdio>
#include <vector>
#include <algorithm>

int main() {
    // Spec: grid(1,1,1), block(256,1,1), count=256, default_value=1.0, num_bins=256
    int n = 256;
    int num_bins = 256;

    // Try to read n and num_bins from stdin
    scanf("%d %d", &n, &num_bins);
    if (n <= 0) n = 256;
    if (num_bins <= 0) num_bins = 256;

    std::vector<float> input(n, 1.0f);

    // Read input values from stdin if available
    for (int i = 0; i < n && !feof(stdin); i++) {
        float v;
        if (scanf("%f", &v) == 1) input[i] = v;
    }

    // Compute histogram: bin = clamp((int)(input[i] * num_bins), 0, num_bins-1)
    std::vector<int> histogram(num_bins, 0);
    for (int i = 0; i < n; i++) {
        int bin = static_cast<int>(input[i] * num_bins);
        bin = std::max(0, std::min(bin, num_bins - 1));
        histogram[bin]++;
    }

    // Print all bins
    for (int i = 0; i < num_bins; i++)
        printf("%d\n", histogram[i]);

    return 0;
}
