#include "naive_simplepir.h"

#include <cuda_runtime.h>
#include <new>
#include <stdint.h>
#include <stdlib.h>

struct NaiveGpuCtx {
  uint32_t L = 0, M = 0, N = 0;
  uint32_t *d_db = nullptr;
  uint32_t *d_A = nullptr;
  uint32_t *d_H = nullptr;
  uint32_t *d_s = nullptr;
  uint32_t *d_e = nullptr;
  uint32_t *d_q = nullptr;
  uint32_t *d_ans = nullptr;
};

static bool cuda_ok(cudaError_t e) { return e == cudaSuccess; }

namespace {

__global__ void k_gemm_ln(const uint32_t *A, const uint32_t *B, uint32_t *C, int L, int M, int N) {
  int col = blockIdx.x * blockDim.x + threadIdx.x;
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  if (row >= L || col >= N) {
    return;
  }
  uint32_t acc = 0;
  for (int m = 0; m < M; ++m) {
    acc += A[row * M + m] * B[m * N + col];
  }
  C[row * N + col] = acc;
}

__global__ void k_gemv_A_s_e(uint32_t *q, const uint32_t *A, const uint32_t *s, const uint32_t *e, int M,
                             int N) {
  int row = blockIdx.x * blockDim.x + threadIdx.x;
  if (row >= M) {
    return;
  }
  uint32_t acc = e[row];
  for (int n = 0; n < N; ++n) {
    acc += A[row * N + n] * s[n];
  }
  q[row] = acc;
}

__global__ void k_add_delta(uint32_t *q, uint32_t idx, uint32_t delta) {
  if (threadIdx.x == 0 && blockIdx.x == 0) {
    q[idx] += delta;
  }
}

__global__ void k_gemv_db_q(uint32_t *ans, const uint32_t *db, const uint32_t *q, int L, int M) {
  int row = blockIdx.x * blockDim.x + threadIdx.x;
  if (row >= L) {
    return;
  }
  uint32_t acc = 0;
  for (int m = 0; m < M; ++m) {
    acc += db[row * M + m] * q[m];
  }
  ans[row] = acc;
}

static void launch_setup_gemm(NaiveGpuCtx *ctx, int L, int M, int N) {
  dim3 tpb16(16, 16);
  dim3 blocks_gemm((N + 15) / 16, (L + 15) / 16);
  k_gemm_ln<<<blocks_gemm, tpb16>>>(ctx->d_db, ctx->d_A, ctx->d_H, L, M, N);
}

static void launch_query_gemv_with_delta(NaiveGpuCtx *ctx, int M, int N, uint32_t delta_row, uint32_t delta) {
  int tpb = 256;
  int grid = (M + tpb - 1) / tpb;
  k_gemv_A_s_e<<<grid, tpb>>>(ctx->d_q, ctx->d_A, ctx->d_s, ctx->d_e, M, N);
  k_add_delta<<<1, 1>>>(ctx->d_q, delta_row, delta);
}

static void launch_answer_gemv(NaiveGpuCtx *ctx, int L, int M) {
  int tpb = 256;
  k_gemv_db_q<<<((L + tpb - 1) / tpb), tpb>>>(ctx->d_ans, ctx->d_db, ctx->d_q, L, M);
}

}  // namespace

NaiveGpuSimplepirHandle naive_gpu_ctx_create(uint32_t L, uint32_t M, uint32_t N) {
  NaiveGpuCtx *ctx = new (std::nothrow) NaiveGpuCtx();
  if (!ctx) {
    return nullptr;
  }
  ctx->L = L;
  ctx->M = M;
  ctx->N = N;
  size_t sz_db = static_cast<size_t>(L) * M * sizeof(uint32_t);
  size_t sz_A = static_cast<size_t>(M) * N * sizeof(uint32_t);
  size_t sz_H = static_cast<size_t>(L) * N * sizeof(uint32_t);
  size_t sz_M = static_cast<size_t>(M) * sizeof(uint32_t);
  size_t sz_N = static_cast<size_t>(N) * sizeof(uint32_t);
  size_t sz_L = static_cast<size_t>(L) * sizeof(uint32_t);
  if (!cuda_ok(cudaMalloc(&ctx->d_db, sz_db)) || !cuda_ok(cudaMalloc(&ctx->d_A, sz_A)) ||
      !cuda_ok(cudaMalloc(&ctx->d_H, sz_H)) || !cuda_ok(cudaMalloc(&ctx->d_s, sz_N)) ||
      !cuda_ok(cudaMalloc(&ctx->d_e, sz_M)) || !cuda_ok(cudaMalloc(&ctx->d_q, sz_M)) ||
      !cuda_ok(cudaMalloc(&ctx->d_ans, sz_L))) {
    naive_gpu_ctx_destroy(static_cast<NaiveGpuSimplepirHandle>(ctx));
    return nullptr;
  }
  return static_cast<NaiveGpuSimplepirHandle>(ctx);
}

void naive_gpu_ctx_destroy(NaiveGpuSimplepirHandle h) {
  auto *ctx = static_cast<NaiveGpuCtx *>(h);
  if (!ctx) {
    return;
  }
  cudaFree(ctx->d_db);
  cudaFree(ctx->d_A);
  cudaFree(ctx->d_H);
  cudaFree(ctx->d_s);
  cudaFree(ctx->d_e);
  cudaFree(ctx->d_q);
  cudaFree(ctx->d_ans);
  delete ctx;
}

int naive_gpu_ctx_upload_db_A(NaiveGpuSimplepirHandle h, const uint32_t *host_db, const uint32_t *host_A) {
  auto *ctx = static_cast<NaiveGpuCtx *>(h);
  if (!ctx || !host_db || !host_A) {
    return 1;
  }
  const int L = static_cast<int>(ctx->L);
  const int M = static_cast<int>(ctx->M);
  const int N = static_cast<int>(ctx->N);
  size_t sz_db = static_cast<size_t>(L) * M * sizeof(uint32_t);
  size_t sz_A = static_cast<size_t>(M) * N * sizeof(uint32_t);
  if (!cuda_ok(cudaMemcpy(ctx->d_db, host_db, sz_db, cudaMemcpyHostToDevice)) ||
      !cuda_ok(cudaMemcpy(ctx->d_A, host_A, sz_A, cudaMemcpyHostToDevice))) {
    return 2;
  }
  return 0;
}

static double elapsed_ms(cudaEvent_t start, cudaEvent_t stop) {
  cudaEventSynchronize(stop);
  float ms = 0.0f;
  cudaEventElapsedTime(&ms, start, stop);
  return static_cast<double>(ms);
}

int naive_gpu_ctx_run_e2e(NaiveGpuSimplepirHandle h, const uint32_t *host_db, const uint32_t *host_A,
                          const uint32_t *host_secret, const uint32_t *host_err, uint32_t delta_row,
                          uint32_t delta, double *out_ms_setup, double *out_ms_query, double *out_ms_answer,
                          double *out_ms_total) {
  auto *ctx = static_cast<NaiveGpuCtx *>(h);
  if (!ctx || !host_db || !host_A || !host_secret || !host_err || !out_ms_setup || !out_ms_query ||
      !out_ms_answer || !out_ms_total) {
    return 1;
  }

  const int L = static_cast<int>(ctx->L);
  const int M = static_cast<int>(ctx->M);
  const int N = static_cast<int>(ctx->N);
  if (delta_row >= static_cast<uint32_t>(M)) {
    return 2;
  }

  cudaEvent_t ev0, ev1, ev2, ev3, ev4;
  cudaEventCreate(&ev0);
  cudaEventCreate(&ev1);
  cudaEventCreate(&ev2);
  cudaEventCreate(&ev3);
  cudaEventCreate(&ev4);

  size_t sz_db = static_cast<size_t>(L) * M * sizeof(uint32_t);
  size_t sz_A = static_cast<size_t>(M) * N * sizeof(uint32_t);

  cudaEventRecord(ev0);
  if (!cuda_ok(cudaMemcpy(ctx->d_db, host_db, sz_db, cudaMemcpyHostToDevice)) ||
      !cuda_ok(cudaMemcpy(ctx->d_A, host_A, sz_A, cudaMemcpyHostToDevice))) {
    cudaEventDestroy(ev0);
    cudaEventDestroy(ev1);
    cudaEventDestroy(ev2);
    cudaEventDestroy(ev3);
    cudaEventDestroy(ev4);
    return 3;
  }

  launch_setup_gemm(ctx, L, M, N);
  cudaEventRecord(ev1);

  if (!cuda_ok(cudaMemcpy(ctx->d_s, host_secret, static_cast<size_t>(N) * sizeof(uint32_t),
                     cudaMemcpyHostToDevice)) ||
      !cuda_ok(cudaMemcpy(ctx->d_e, host_err, static_cast<size_t>(M) * sizeof(uint32_t),
                     cudaMemcpyHostToDevice))) {
    cudaEventDestroy(ev0);
    cudaEventDestroy(ev1);
    cudaEventDestroy(ev2);
    cudaEventDestroy(ev3);
    cudaEventDestroy(ev4);
    return 4;
  }

  launch_query_gemv_with_delta(ctx, M, N, delta_row, delta);
  cudaEventRecord(ev2);

  launch_answer_gemv(ctx, L, M);
  cudaEventRecord(ev3);

  uint32_t pin = 0;
  if (!cuda_ok(cudaMemcpy(&pin, ctx->d_ans, sizeof(uint32_t), cudaMemcpyDeviceToHost))) {
    cudaEventDestroy(ev0);
    cudaEventDestroy(ev1);
    cudaEventDestroy(ev2);
    cudaEventDestroy(ev3);
    cudaEventDestroy(ev4);
    return 5;
  }

  cudaEventRecord(ev4);
  cudaDeviceSynchronize();

  *out_ms_setup = elapsed_ms(ev0, ev1);
  *out_ms_query = elapsed_ms(ev1, ev2);
  *out_ms_answer = elapsed_ms(ev2, ev3);
  *out_ms_total = elapsed_ms(ev0, ev4);

  cudaEventDestroy(ev0);
  cudaEventDestroy(ev1);
  cudaEventDestroy(ev2);
  cudaEventDestroy(ev3);
  cudaEventDestroy(ev4);

  (void)pin;
  return 0;
}

int naive_gpu_ctx_run_e2e_device_resident(NaiveGpuSimplepirHandle h, const uint32_t *host_secret,
                                          const uint32_t *host_err, uint32_t delta_row, uint32_t delta,
                                          double *out_ms_setup, double *out_ms_query, double *out_ms_answer,
                                          double *out_ms_total) {
  auto *ctx = static_cast<NaiveGpuCtx *>(h);
  if (!ctx || !host_secret || !host_err || !out_ms_setup || !out_ms_query || !out_ms_answer || !out_ms_total) {
    return 1;
  }

  const int L = static_cast<int>(ctx->L);
  const int M = static_cast<int>(ctx->M);
  const int N = static_cast<int>(ctx->N);
  if (delta_row >= static_cast<uint32_t>(M)) {
    return 2;
  }

  cudaEvent_t ev0, ev1, ev2, ev3, ev4;
  cudaEventCreate(&ev0);
  cudaEventCreate(&ev1);
  cudaEventCreate(&ev2);
  cudaEventCreate(&ev3);
  cudaEventCreate(&ev4);

  cudaEventRecord(ev0);
  launch_setup_gemm(ctx, L, M, N);
  cudaEventRecord(ev1);

  if (!cuda_ok(cudaMemcpy(ctx->d_s, host_secret, static_cast<size_t>(N) * sizeof(uint32_t),
                     cudaMemcpyHostToDevice)) ||
      !cuda_ok(cudaMemcpy(ctx->d_e, host_err, static_cast<size_t>(M) * sizeof(uint32_t),
                     cudaMemcpyHostToDevice))) {
    cudaEventDestroy(ev0);
    cudaEventDestroy(ev1);
    cudaEventDestroy(ev2);
    cudaEventDestroy(ev3);
    cudaEventDestroy(ev4);
    return 4;
  }

  launch_query_gemv_with_delta(ctx, M, N, delta_row, delta);
  cudaEventRecord(ev2);

  launch_answer_gemv(ctx, L, M);
  cudaEventRecord(ev3);

  uint32_t pin = 0;
  if (!cuda_ok(cudaMemcpy(&pin, ctx->d_ans, sizeof(uint32_t), cudaMemcpyDeviceToHost))) {
    cudaEventDestroy(ev0);
    cudaEventDestroy(ev1);
    cudaEventDestroy(ev2);
    cudaEventDestroy(ev3);
    cudaEventDestroy(ev4);
    return 5;
  }

  cudaEventRecord(ev4);
  cudaDeviceSynchronize();

  *out_ms_setup = elapsed_ms(ev0, ev1);
  *out_ms_query = elapsed_ms(ev1, ev2);
  *out_ms_answer = elapsed_ms(ev2, ev3);
  *out_ms_total = elapsed_ms(ev0, ev4);

  cudaEventDestroy(ev0);
  cudaEventDestroy(ev1);
  cudaEventDestroy(ev2);
  cudaEventDestroy(ev3);
  cudaEventDestroy(ev4);

  (void)pin;
  return 0;
}
