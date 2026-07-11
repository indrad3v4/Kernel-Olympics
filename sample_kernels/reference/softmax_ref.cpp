#include <cstdio>
#include <cmath>
#include <vector>

int main() {
    int n = 256;   // default from spec
    float default_val = 1.0f;

    // Try to read seed data from stdin
    std::vector<float> values;
    float v;
    while (scanf("%f", &v) == 1) {
        values.push_back(v);
    }

    if (values.empty()) {
        // No seed data — use spec defaults
        values.assign(n, default_val);
    }
    n = static_cast<int>(values.size());

    // Compute softmax: exp(x_i) / sum(exp(x_j))
    double max_val = values[0];
    for (int i = 1; i < n; i++)
        if (values[i] > max_val) max_val = values[i];

    double sum = 0.0;
    std::vector<double> exps(n);
    for (int i = 0; i < n; i++) {
        exps[i] = std::exp(static_cast<double>(values[i]) - max_val);
        sum += exps[i];
    }

    for (int i = 0; i < n; i++)
        printf("%.10f\n", exps[i] / sum);

    return 0;
}
