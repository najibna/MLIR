#include "kernels.h"

#include <cuda_runtime.h>
#include <cstdio>

// basic matmul kernel, one thread per output cell
__global__ void matmul_kernel(const float* a, const float* b, float* c,
                              int m, int n, int k) {
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;
  if (row >= m || col >= n) return;

  float sum = 0.0f;
  for (int t = 0; t < k; ++t) {
    sum += a[row * k + t] * b[t * n + col];
  }
  c[row * n + col] = sum;
}

// same as matmul but clamps negatives to zero
__global__ void fused_matmul_relu_kernel(const float* a, const float* b, float* c,
                                         int m, int n, int k) {
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;
  if (row >= m || col >= n) return;

  float sum = 0.0f;
  for (int t = 0; t < k; ++t) {
    sum += a[row * k + t] * b[t * n + col];
  }
  c[row * n + col] = sum > 0.0f ? sum : 0.0f;
}

__global__ void relu_kernel(float* data, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n && data[i] < 0.0f) data[i] = 0.0f;
}

__global__ void add_kernel(const float* a, const float* b, float* out, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) out[i] = a[i] + b[i];
}

static void check_cuda(cudaError_t err, const char* what) {
  if (err != cudaSuccess) {
    std::fprintf(stderr, "cuda error in %s: %s\n", what, cudaGetErrorString(err));
  }
}

bool cuda_available() {
  int count = 0;
  if (cudaGetDeviceCount(&count) != cudaSuccess) return false;
  return count > 0;
}

void cuda_matmul(const float* a, const float* b, float* c,
                 int m, int n, int k) {
  float *da = nullptr, *db = nullptr, *dc = nullptr;
  size_t sa = sizeof(float) * m * k;
  size_t sb = sizeof(float) * k * n;
  size_t sc = sizeof(float) * m * n;

  check_cuda(cudaMalloc(&da, sa), "malloc a");
  check_cuda(cudaMalloc(&db, sb), "malloc b");
  check_cuda(cudaMalloc(&dc, sc), "malloc c");
  check_cuda(cudaMemcpy(da, a, sa, cudaMemcpyHostToDevice), "copy a");
  check_cuda(cudaMemcpy(db, b, sb, cudaMemcpyHostToDevice), "copy b");

  dim3 block(16, 16);
  dim3 grid((n + 15) / 16, (m + 15) / 16);
  matmul_kernel<<<grid, block>>>(da, db, dc, m, n, k);
  check_cuda(cudaDeviceSynchronize(), "sync matmul");
  check_cuda(cudaMemcpy(c, dc, sc, cudaMemcpyDeviceToHost), "copy c");

  cudaFree(da);
  cudaFree(db);
  cudaFree(dc);
}

void cuda_fused_matmul_relu(const float* a, const float* b, float* c,
                            int m, int n, int k) {
  float *da = nullptr, *db = nullptr, *dc = nullptr;
  size_t sa = sizeof(float) * m * k;
  size_t sb = sizeof(float) * k * n;
  size_t sc = sizeof(float) * m * n;

  check_cuda(cudaMalloc(&da, sa), "malloc a");
  check_cuda(cudaMalloc(&db, sb), "malloc b");
  check_cuda(cudaMalloc(&dc, sc), "malloc c");
  check_cuda(cudaMemcpy(da, a, sa, cudaMemcpyHostToDevice), "copy a");
  check_cuda(cudaMemcpy(db, b, sb, cudaMemcpyHostToDevice), "copy b");

  dim3 block(16, 16);
  dim3 grid((n + 15) / 16, (m + 15) / 16);
  fused_matmul_relu_kernel<<<grid, block>>>(da, db, dc, m, n, k);
  check_cuda(cudaDeviceSynchronize(), "sync fused");
  check_cuda(cudaMemcpy(c, dc, sc, cudaMemcpyDeviceToHost), "copy c");

  cudaFree(da);
  cudaFree(db);
  cudaFree(dc);
}

void cuda_relu(float* data, int n) {
  float* d = nullptr;
  size_t bytes = sizeof(float) * n;
  check_cuda(cudaMalloc(&d, bytes), "malloc relu");
  check_cuda(cudaMemcpy(d, data, bytes, cudaMemcpyHostToDevice), "copy relu");
  int threads = 256;
  int blocks = (n + threads - 1) / threads;
  relu_kernel<<<blocks, threads>>>(d, n);
  check_cuda(cudaDeviceSynchronize(), "sync relu");
  check_cuda(cudaMemcpy(data, d, bytes, cudaMemcpyDeviceToHost), "copy back");
  cudaFree(d);
}

void cuda_add(const float* a, const float* b, float* out, int n) {
  float *da = nullptr, *db = nullptr, *dout = nullptr;
  size_t bytes = sizeof(float) * n;
  check_cuda(cudaMalloc(&da, bytes), "malloc add a");
  check_cuda(cudaMalloc(&db, bytes), "malloc add b");
  check_cuda(cudaMalloc(&dout, bytes), "malloc add out");
  check_cuda(cudaMemcpy(da, a, bytes, cudaMemcpyHostToDevice), "copy add a");
  check_cuda(cudaMemcpy(db, b, bytes, cudaMemcpyHostToDevice), "copy add b");
  int threads = 256;
  int blocks = (n + threads - 1) / threads;
  add_kernel<<<blocks, threads>>>(da, db, dout, n);
  check_cuda(cudaDeviceSynchronize(), "sync add");
  check_cuda(cudaMemcpy(out, dout, bytes, cudaMemcpyDeviceToHost), "copy add out");
  cudaFree(da);
  cudaFree(db);
  cudaFree(dout);
}
