#include "matmul_packed_gpu.h"

#include <chrono>
#include <cuda_runtime.h>
#include <mutex>
#include <stdint.h>

namespace {

constexpr int kBasis = 10;
constexpr int kBasis2 = 20;
constexpr uint32_t kMask = (1u << kBasis) - 1u;
constexpr int kCompression = 3;

bool ok(cudaError_t e) { return e == cudaSuccess; }

std::mutex g_mu;
uint32_t *g_d_a = nullptr;
uint32_t *g_d_b = nullptr;
uint32_t *g_d_out = nullptr;
size_t cap_a = 0, cap_b = 0, cap_out = 0;

uint32_t *g_d_db = nullptr;
size_t cap_db = 0;
size_t pers_rows = 0, pers_cols = 0;

cudaStream_t g_packed_stream = nullptr;
cudaEvent_t g_ev_query_ready = nullptr;
cudaEvent_t g_ev_kernel_done = nullptr;

bool ensure(uint32_t **p, size_t *cap, size_t need_elems) {
  if (*cap >= need_elems) {
    return true;
  }
  cudaFree(*p);
  *p = nullptr;
  *cap = 0;
  uint32_t *n = nullptr;
  if (!ok(cudaMalloc(reinterpret_cast<void **>(&n), need_elems * sizeof(uint32_t)))) {
    return false;
  }
  *p = n;
  *cap = need_elems;
  return true;
}

bool ensure_packed_stream() {
  if (g_packed_stream) {
    return true;
  }
  return ok(cudaStreamCreate(&g_packed_stream));
}

bool ensure_timing_events() {
  if (g_ev_query_ready) {
    return true;
  }
  if (!ok(cudaEventCreate(&g_ev_query_ready))) {
    return false;
  }
  if (!ok(cudaEventCreate(&g_ev_kernel_done))) {
    cudaEventDestroy(g_ev_query_ready);
    g_ev_query_ready = nullptr;
    return false;
  }
  return true;
}

__global__ void k_matmul_vec_packed_u32(uint32_t *out, const uint32_t *a, const uint32_t *b, size_t a_rows,
                                        size_t a_cols) {
  const size_t i0 = static_cast<size_t>(blockIdx.x) * 8u;
  if (i0 >= a_rows) {
    return;
  }

  const size_t rem = a_rows - i0;
  const int n = rem < 8u ? static_cast<int>(rem) : 8;
  uint32_t t[8] = {0, 0, 0, 0, 0, 0, 0, 0};

  for (size_t j = 0; j < a_cols; ++j) {
    const size_t i2 = j * static_cast<size_t>(kCompression);
    for (int r = 0; r < n; ++r) {
      const uint32_t db = a[(i0 + static_cast<size_t>(r)) * a_cols + j];
      t[r] += (db & kMask) * b[i2];
      t[r] += ((db >> kBasis) & kMask) * b[i2 + 1];
      t[r] += ((db >> kBasis2) & kMask) * b[i2 + 2];
    }
  }

  for (int r = 0; r < n; ++r) {
    out[i0 + static_cast<size_t>(r)] += t[r];
  }
}

int resident_packed_launch(uint32_t *out, size_t row_off, size_t a_rows, size_t a_cols, const uint32_t *b,
                           float *kernel_ms_out, float *e2e_ms_out) {
  if (row_off > pers_rows || row_off + a_rows > pers_rows || a_cols != pers_cols) {
    return 6;
  }
  const size_t b_elems = a_cols * static_cast<size_t>(kCompression);
  const size_t out_elems = a_rows;

  if (!g_d_db) {
    return 7;
  }

  if (!ensure(&g_d_b, &cap_b, b_elems) || !ensure(&g_d_out, &cap_out, out_elems)) {
    return 2;
  }

  if (!ensure_packed_stream()) {
    return 8;
  }

  const bool want_kernel_ms = (kernel_ms_out != nullptr);
  if (want_kernel_ms && !ensure_timing_events()) {
    return 9;
  }

  using clock = std::chrono::steady_clock;
  clock::time_point t0{};
  if (e2e_ms_out) {
    t0 = clock::now();
  }

  cudaStream_t stream = g_packed_stream;

  if (!ok(cudaMemcpyAsync(g_d_out, out, out_elems * sizeof(uint32_t), cudaMemcpyHostToDevice, stream))) {
    return 3;
  }
  if (!ok(cudaMemcpyAsync(g_d_b, b, b_elems * sizeof(uint32_t), cudaMemcpyHostToDevice, stream))) {
    return 3;
  }

  // Query Q (and seeded out buffer) are queued; next event marks when those H2Ds complete.
  if (want_kernel_ms) {
    if (!ok(cudaEventRecord(g_ev_query_ready, stream))) {
      return 10;
    }
  }

  const uint32_t *d_a = g_d_db + row_off * a_cols;
  const int blocks = static_cast<int>((a_rows + 7u) / 8u);
  k_matmul_vec_packed_u32<<<blocks, 1, 0, stream>>>(g_d_out, d_a, g_d_b, a_rows, a_cols);

  if (!ok(cudaGetLastError())) {
    return 4;
  }

  if (want_kernel_ms) {
    if (!ok(cudaEventRecord(g_ev_kernel_done, stream))) {
      return 10;
    }
  }

  if (!ok(cudaMemcpyAsync(out, g_d_out, out_elems * sizeof(uint32_t), cudaMemcpyDeviceToHost, stream))) {
    return 5;
  }

  if (!ok(cudaStreamSynchronize(stream))) {
    return 5;
  }

  if (e2e_ms_out) {
    const auto t1 = clock::now();
    *e2e_ms_out =
        std::chrono::duration<float, std::milli>(t1 - t0).count();
  }

  if (want_kernel_ms) {
    if (!ok(cudaEventElapsedTime(kernel_ms_out, g_ev_query_ready, g_ev_kernel_done))) {
      return 11;
    }
  }

  return 0;
}

}  // namespace

extern "C" int simplepir_matmul_vec_packed_gpu(uint32_t *out, const uint32_t *a, const uint32_t *b,
                                               size_t a_rows, size_t a_cols) {
  if (!out || !a || !b || a_rows == 0 || a_cols == 0) {
    return 1;
  }
  const size_t a_elems = a_rows * a_cols;
  const size_t b_elems = a_cols * static_cast<size_t>(kCompression);
  const size_t out_elems = a_rows;

  std::lock_guard<std::mutex> lock(g_mu);

  if (!ensure(&g_d_a, &cap_a, a_elems) || !ensure(&g_d_b, &cap_b, b_elems) ||
      !ensure(&g_d_out, &cap_out, out_elems)) {
    return 2;
  }

  if (!ok(cudaMemcpy(g_d_out, out, out_elems * sizeof(uint32_t), cudaMemcpyHostToDevice))) {
    return 3;
  }
  if (!ok(cudaMemcpy(g_d_a, a, a_elems * sizeof(uint32_t), cudaMemcpyHostToDevice)) ||
      !ok(cudaMemcpy(g_d_b, b, b_elems * sizeof(uint32_t), cudaMemcpyHostToDevice))) {
    return 3;
  }

  const int blocks = static_cast<int>((a_rows + 7u) / 8u);
  k_matmul_vec_packed_u32<<<blocks, 1>>>(g_d_out, g_d_a, g_d_b, a_rows, a_cols);

  if (!ok(cudaGetLastError())) {
    return 4;
  }
  if (!ok(cudaDeviceSynchronize()) ||
      !ok(cudaMemcpy(out, g_d_out, out_elems * sizeof(uint32_t), cudaMemcpyDeviceToHost))) {
    return 5;
  }
  return 0;
}

extern "C" void simplepir_packed_db_release(void) {
  std::lock_guard<std::mutex> lock(g_mu);
  cudaFree(g_d_db);
  g_d_db = nullptr;
  cap_db = 0;
  pers_rows = pers_cols = 0;
}

extern "C" int simplepir_packed_db_upload(const uint32_t *host, size_t rows, size_t cols) {
  if (!host || rows == 0 || cols == 0) {
    return 1;
  }
  const size_t elems = rows * cols;
  std::lock_guard<std::mutex> lock(g_mu);
  if (!ensure(&g_d_db, &cap_db, elems)) {
    return 2;
  }
  if (!ok(cudaMemcpy(g_d_db, host, elems * sizeof(uint32_t), cudaMemcpyHostToDevice))) {
    return 3;
  }
  pers_rows = rows;
  pers_cols = cols;
  return 0;
}

extern "C" int simplepir_matmul_vec_packed_gpu_resident(uint32_t *out, size_t row_off, size_t a_rows,
                                                        size_t a_cols, const uint32_t *b) {
  if (!out || !b || a_rows == 0 || a_cols == 0) {
    return 1;
  }
  std::lock_guard<std::mutex> lock(g_mu);
  return resident_packed_launch(out, row_off, a_rows, a_cols, b, nullptr, nullptr);
}

extern "C" int simplepir_matmul_vec_packed_gpu_resident_timed(uint32_t *out, size_t row_off, size_t a_rows,
                                                                size_t a_cols, const uint32_t *b,
                                                                float *kernel_ms_out, float *e2e_ms_out) {
  if (!out || !b || a_rows == 0 || a_cols == 0) {
    return 1;
  }
  std::lock_guard<std::mutex> lock(g_mu);
  return resident_packed_launch(out, row_off, a_rows, a_cols, b, kernel_ms_out, e2e_ms_out);
}
