#pragma once

#include <vector>

// cpu fallbacks used when cuda is missing
void cpu_matmul(const float* a, const float* b, float* c,
                int m, int n, int k);

void cpu_add(const float* a, const float* b, float* out, int n);

void cpu_relu(float* data, int n);

void cpu_softmax(float* data, int n);

void cpu_fused_matmul_relu(const float* a, const float* b, float* c,
                           int m, int n, int k);

// cuda entry points (no-op stubs if built without cuda)
bool cuda_available();

void cuda_matmul(const float* a, const float* b, float* c,
                 int m, int n, int k);

void cuda_fused_matmul_relu(const float* a, const float* b, float* c,
                            int m, int n, int k);

void cuda_relu(float* data, int n);

void cuda_add(const float* a, const float* b, float* out, int n);
