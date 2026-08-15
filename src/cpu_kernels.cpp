#include "kernels.h"

#include <algorithm>
#include <cmath>
#include <cstring>

void cpu_matmul(const float* a, const float* b, float* c,
                int m, int n, int k) {
  // a is m x k, b is k x n, c is m x n
  for (int i = 0; i < m; ++i) {
    for (int j = 0; j < n; ++j) {
      float sum = 0.0f;
      for (int t = 0; t < k; ++t) {
        sum += a[i * k + t] * b[t * n + j];
      }
      c[i * n + j] = sum;
    }
  }
}

void cpu_add(const float* a, const float* b, float* out, int n) {
  for (int i = 0; i < n; ++i) out[i] = a[i] + b[i];
}

void cpu_relu(float* data, int n) {
  for (int i = 0; i < n; ++i) {
    if (data[i] < 0.0f) data[i] = 0.0f;
  }
}

void cpu_softmax(float* data, int n) {
  float maxv = data[0];
  for (int i = 1; i < n; ++i) maxv = std::max(maxv, data[i]);
  float sum = 0.0f;
  for (int i = 0; i < n; ++i) {
    data[i] = std::exp(data[i] - maxv);
    sum += data[i];
  }
  for (int i = 0; i < n; ++i) data[i] /= sum;
}

void cpu_fused_matmul_relu(const float* a, const float* b, float* c,
                           int m, int n, int k) {
  cpu_matmul(a, b, c, m, n, k);
  cpu_relu(c, m * n);
}

#if !defined(MLIR_HAS_CUDA)

bool cuda_available() { return false; }

void cuda_matmul(const float*, const float*, float*, int, int, int) {}
void cuda_fused_matmul_relu(const float*, const float*, float*, int, int, int) {}
void cuda_relu(float*, int) {}
void cuda_add(const float*, const float*, float*, int) {}

#endif
