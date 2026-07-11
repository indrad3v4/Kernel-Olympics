#include <cstdio>
#include <vector>

int main() {
    int width = 3072, height = 3072;  // default from spec

    // Try to read dimensions from stdin
    if (scanf("%d %d", &width, &height) != 2) {
        width = 3072;
        height = 3072;
    }
    if (width <= 0) width = 3072;
    if (height <= 0) height = 3072;

    int n = width * height;
    std::vector<float> mat(n, 1.0f);

    // Try to read matrix data from stdin
    for (int i = 0; i < n && !feof(stdin); i++) {
        float v;
        if (scanf("%f", &v) == 1) mat[i] = v;
    }

    // Transpose: output[j * height + i] = input[i * width + j]
    std::vector<float> trans(n);
    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {
            trans[j * height + i] = mat[i * width + j];
        }
    }

    // Print all transposed values
    for (int i = 0; i < n; i++)
        printf("%.1f\n", trans[i]);

    return 0;
}
