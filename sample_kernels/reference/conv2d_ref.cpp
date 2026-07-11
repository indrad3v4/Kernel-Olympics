#include <cstdio>
#include <vector>
#include <algorithm>

int main() {
    // Spec: grid(8,8,1), block(38,38,1), count=65536, default_value=1.0,
    //        kernel_args_override="256, 256, 1" → width=256, height=256, channels=1
    int width = 256, height = 256, channels = 1;
    const int FILTER_RADIUS = 3;
    const int FILTER_SIZE = 2 * FILTER_RADIUS + 1;  // 7

    // Try to read dimensions from stdin
    scanf("%d %d %d", &width, &height, &channels);
    if (width <= 0) width = 256;
    if (height <= 0) height = 256;
    if (channels <= 0) channels = 1;

    int n = width * height * channels;
    std::vector<float> input(n, 1.0f);

    // Read input values from stdin if available
    for (int i = 0; i < n && !feof(stdin); i++) {
        float v;
        if (scanf("%f", &v) == 1) input[i] = v;
    }

    // Simple box-filter convolution: for each output pixel, sum over a
    // (2*R+1) x (2*R+1) window of the input (symmetric padding with 0)
    // The kernel's conv2d uses TILE_SIZE=32, FILTER_RADIUS=3 with a naive
    // 7x7 sum — no separate filter kernel, just a box filter.
    std::vector<float> output(n, 0.0f);

    for (int c = 0; c < channels; c++) {
        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                float sum = 0.0f;
                for (int dy = -FILTER_RADIUS; dy <= FILTER_RADIUS; dy++) {
                    for (int dx = -FILTER_RADIUS; dx <= FILTER_RADIUS; dx++) {
                        int sx = x + dx;
                        int sy = y + dy;
                        if (sx >= 0 && sx < width && sy >= 0 && sy < height) {
                            sum += input[(c * height + sy) * width + sx];
                        }
                    }
                }
                output[(c * height + y) * width + x] = sum;
            }
        }
    }

    // Print all output values
    for (int i = 0; i < n; i++)
        printf("%.1f\n", output[i]);

    return 0;
}
