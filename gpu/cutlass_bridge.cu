#include "cutlass_bridge.h"

#include <limits>
#include <mutex>

#include <cuda_runtime.h>

#include "cutlass/cutlass.h"
#include "cutlass/gemm/device/gemm.h"
#include "cutlass/layout/matrix.h"

namespace {

bool checked_cuda(cudaError_t err) {
  return err == cudaSuccess;
}

std::mutex g_pool_mu;
uint32_t *g_d_a = nullptr;
uint32_t *g_d_b = nullptr;
uint32_t *g_d_c = nullptr;
size_t g_cap_a = 0;
size_t g_cap_b = 0;
size_t g_cap_c = 0;

bool ensure_device_buffer(uint32_t **ptr, size_t *cap_elems, size_t need_elems) {
  if (*cap_elems >= need_elems) {
    return true;
  }
  if (*ptr != nullptr) {
    cudaFree(*ptr);
    *ptr = nullptr;
    *cap_elems = 0;
  }
  uint32_t *p = nullptr;
  if (!checked_cuda(cudaMalloc(reinterpret_cast<void **>(&p), need_elems * sizeof(uint32_t)))) {
    return false;
  }
  *ptr = p;
  *cap_elems = need_elems;
  return true;
}

}  // namespace

extern "C" int simplepir_matmul_cutlass_u32(uint32_t *out, const uint32_t *a, const uint32_t *b,
                                            size_t a_rows, size_t a_cols, size_t b_cols) {
  if (out == nullptr || a == nullptr || b == nullptr) {
    return 1;
  }

  if (a_rows == 0 || a_cols == 0 || b_cols == 0) {
    return 1;
  }

  if (a_rows > static_cast<size_t>(std::numeric_limits<int>::max()) ||
      a_cols > static_cast<size_t>(std::numeric_limits<int>::max()) ||
      b_cols > static_cast<size_t>(std::numeric_limits<int>::max())) {
    return 1;
  }

  const size_t a_elems = a_rows * a_cols;
  const size_t b_elems = a_cols * b_cols;
  const size_t c_elems = a_rows * b_cols;

  std::lock_guard<std::mutex> lock(g_pool_mu);

  if (!ensure_device_buffer(&g_d_a, &g_cap_a, a_elems)) {
    return 2;
  }
  if (!ensure_device_buffer(&g_d_b, &g_cap_b, b_elems)) {
    return 2;
  }
  if (!ensure_device_buffer(&g_d_c, &g_cap_c, c_elems)) {
    return 2;
  }

  if (!checked_cuda(cudaMemcpy(g_d_a, a, a_elems * sizeof(uint32_t), cudaMemcpyHostToDevice)) ||
      !checked_cuda(cudaMemcpy(g_d_b, b, b_elems * sizeof(uint32_t), cudaMemcpyHostToDevice))) {
    return 3;
  }

  using Gemm = cutlass::gemm::device::Gemm<uint32_t, cutlass::layout::RowMajor, uint32_t,
                                           cutlass::layout::RowMajor, uint32_t,
                                           cutlass::layout::RowMajor, uint32_t,
                                           cutlass::arch::OpClassSimt, cutlass::arch::Sm50>;

  Gemm gemm_op;
  Gemm::Arguments args(
      {static_cast<int>(a_rows), static_cast<int>(b_cols), static_cast<int>(a_cols)},
      {g_d_a, static_cast<int>(a_cols)}, {g_d_b, static_cast<int>(b_cols)},
      {g_d_c, static_cast<int>(b_cols)}, {g_d_c, static_cast<int>(b_cols)}, {1, 0});

  cutlass::Status status = gemm_op(args);
  if (status != cutlass::Status::kSuccess) {
    return 4;
  }

  if (!checked_cuda(cudaDeviceSynchronize()) ||
      !checked_cuda(cudaMemcpy(out, g_d_c, c_elems * sizeof(uint32_t), cudaMemcpyDeviceToHost))) {
    return 5;
  }

  return 0;
}

namespace {

using CutlassGemmU32 =
    cutlass::gemm::device::Gemm<uint32_t, cutlass::layout::RowMajor, uint32_t, cutlass::layout::RowMajor,
                                uint32_t, cutlass::layout::RowMajor, uint32_t, cutlass::arch::OpClassSimt,
                                cutlass::arch::Sm50>;

bool run_cutlass_gemm_u32(uint32_t *d_a, int lda, uint32_t *d_b, int ldb, uint32_t *d_c, int ldc, int m, int n,
                          int k) {
  CutlassGemmU32 gemm_op;
  CutlassGemmU32::Arguments args({m, n, k}, {d_a, lda}, {d_b, ldb}, {d_c, ldc}, {d_c, ldc}, {1, 0});
  return gemm_op(args) == cutlass::Status::kSuccess;
}

struct CutlassPersistCtx {
  std::mutex mu;
  uint32_t *d_a = nullptr;
  uint32_t *d_b = nullptr;
  uint32_t *d_c = nullptr;
  size_t cap_a = 0;
  size_t cap_b = 0;
  size_t cap_c = 0;
  size_t lhs_rows = 0;
  size_t lhs_cols = 0;
  size_t b_cols_stored = 0;
  bool have_lhs_only = false;
  bool have_ab = false;

  void free_all() {
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    d_a = d_b = d_c = nullptr;
    cap_a = cap_b = cap_c = 0;
    have_lhs_only = have_ab = false;
  }

  ~CutlassPersistCtx() { free_all(); }
};

bool persist_ensure(CutlassPersistCtx *ctx, uint32_t **ptr, size_t *cap, size_t need_elems) {
  if (*cap >= need_elems) {
    return true;
  }
  cudaFree(*ptr);
  *ptr = nullptr;
  *cap = 0;
  uint32_t *p = nullptr;
  if (!checked_cuda(cudaMalloc(reinterpret_cast<void **>(&p), need_elems * sizeof(uint32_t)))) {
    return false;
  }
  *ptr = p;
  *cap = need_elems;
  return true;
}

}  // namespace

extern "C" SimplepirCutlassPersistHandle simplepir_cutlass_persist_create(void) {
  auto *ctx = new (std::nothrow) CutlassPersistCtx();
  return static_cast<SimplepirCutlassPersistHandle>(ctx);
}

extern "C" void simplepir_cutlass_persist_destroy(SimplepirCutlassPersistHandle h) {
  auto *ctx = static_cast<CutlassPersistCtx *>(h);
  delete ctx;
}

extern "C" int simplepir_cutlass_persist_upload_lhs(SimplepirCutlassPersistHandle h, const uint32_t *host_a,
                                                    size_t a_rows, size_t a_cols) {
  auto *ctx = static_cast<CutlassPersistCtx *>(h);
  if (!ctx || !host_a || a_rows == 0 || a_cols == 0) {
    return 1;
  }
  std::lock_guard<std::mutex> lock(ctx->mu);
  const size_t elems = a_rows * a_cols;
  if (!persist_ensure(ctx, &ctx->d_a, &ctx->cap_a, elems)) {
    return 2;
  }
  if (!checked_cuda(cudaMemcpy(ctx->d_a, host_a, elems * sizeof(uint32_t), cudaMemcpyHostToDevice))) {
    return 3;
  }
  ctx->lhs_rows = a_rows;
  ctx->lhs_cols = a_cols;
  ctx->have_lhs_only = true;
  ctx->have_ab = false;
  return 0;
}

extern "C" int simplepir_cutlass_persist_gemm_rhs_from_host(SimplepirCutlassPersistHandle h,
                                                              const uint32_t *host_b, size_t b_cols,
                                                              uint32_t *host_c) {
  auto *ctx = static_cast<CutlassPersistCtx *>(h);
  if (!ctx || !host_b || !host_c || b_cols == 0) {
    return 1;
  }
  std::lock_guard<std::mutex> lock(ctx->mu);
  if (!ctx->have_lhs_only || ctx->have_ab) {
    return 6;
  }
  const int m = static_cast<int>(ctx->lhs_rows);
  const int n = static_cast<int>(b_cols);
  const int k = static_cast<int>(ctx->lhs_cols);
  const size_t b_elems = static_cast<size_t>(k) * b_cols;
  const size_t c_elems = static_cast<size_t>(m) * b_cols;
  if (!persist_ensure(ctx, &ctx->d_b, &ctx->cap_b, b_elems)) {
    return 2;
  }
  if (!persist_ensure(ctx, &ctx->d_c, &ctx->cap_c, c_elems)) {
    return 2;
  }
  if (!checked_cuda(cudaMemcpy(ctx->d_b, host_b, b_elems * sizeof(uint32_t), cudaMemcpyHostToDevice))) {
    return 3;
  }
  if (!run_cutlass_gemm_u32(ctx->d_a, k, ctx->d_b, n, ctx->d_c, n, m, n, k)) {
    return 4;
  }
  if (!checked_cuda(cudaDeviceSynchronize()) ||
      !checked_cuda(cudaMemcpy(host_c, ctx->d_c, c_elems * sizeof(uint32_t), cudaMemcpyDeviceToHost))) {
    return 5;
  }
  return 0;
}

extern "C" int simplepir_cutlass_persist_upload_ab(SimplepirCutlassPersistHandle h, const uint32_t *host_a,
                                                    size_t a_rows, size_t a_cols, const uint32_t *host_b,
                                                    size_t b_cols) {
  auto *ctx = static_cast<CutlassPersistCtx *>(h);
  if (!ctx || !host_a || !host_b || a_rows == 0 || a_cols == 0 || b_cols == 0) {
    return 1;
  }
  std::lock_guard<std::mutex> lock(ctx->mu);
  const size_t a_elems = a_rows * a_cols;
  const size_t b_elems = a_cols * b_cols;
  if (!persist_ensure(ctx, &ctx->d_a, &ctx->cap_a, a_elems)) {
    return 2;
  }
  if (!persist_ensure(ctx, &ctx->d_b, &ctx->cap_b, b_elems)) {
    return 2;
  }
  if (!checked_cuda(cudaMemcpy(ctx->d_a, host_a, a_elems * sizeof(uint32_t), cudaMemcpyHostToDevice)) ||
      !checked_cuda(cudaMemcpy(ctx->d_b, host_b, b_elems * sizeof(uint32_t), cudaMemcpyHostToDevice))) {
    return 3;
  }
  ctx->lhs_rows = a_rows;
  ctx->lhs_cols = a_cols;
  ctx->b_cols_stored = b_cols;
  ctx->have_ab = true;
  ctx->have_lhs_only = false;
  return 0;
}

extern "C" int simplepir_cutlass_persist_gemm_ab_resident(SimplepirCutlassPersistHandle h, uint32_t *host_c,
                                                           size_t a_rows, size_t a_cols, size_t b_cols) {
  auto *ctx = static_cast<CutlassPersistCtx *>(h);
  if (!ctx || !host_c) {
    return 1;
  }
  std::lock_guard<std::mutex> lock(ctx->mu);
  if (!ctx->have_ab || ctx->lhs_rows != a_rows || ctx->lhs_cols != a_cols || ctx->b_cols_stored != b_cols) {
    return 6;
  }
  const int m = static_cast<int>(a_rows);
  const int n = static_cast<int>(b_cols);
  const int k = static_cast<int>(a_cols);
  const size_t c_elems = static_cast<size_t>(m) * n;
  if (!persist_ensure(ctx, &ctx->d_c, &ctx->cap_c, c_elems)) {
    return 2;
  }
  if (!run_cutlass_gemm_u32(ctx->d_a, k, ctx->d_b, n, ctx->d_c, n, m, n, k)) {
    return 4;
  }
  if (!checked_cuda(cudaDeviceSynchronize()) ||
      !checked_cuda(cudaMemcpy(host_c, ctx->d_c, c_elems * sizeof(uint32_t), cudaMemcpyDeviceToHost))) {
    return 5;
  }
  return 0;
}
