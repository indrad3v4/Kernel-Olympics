#include <cstdio>
#include <vector>

int main() {
    int n = 256;  // default from legacy harness
    scanf("%d", &n);
    if (n <= 0) n = 256;

    std::vector<float> a(n), b(n), c(n);
    for (int i = 0; i < n; i++) a[i] = static_cast<float>(i);
    for (int i = 0; i < n; i++) b[i] = static_cast<float>(n - i);

    for (int i = 0; i < n; i++) c[i] = a[i] + b[i];

    for (int i = 0; i < n; i++)
        printf("%.1f\n", c[i]);

    return 0;
}
