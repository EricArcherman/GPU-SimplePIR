#include "cutlass_bridge.h"

#include <limits>

#include <cuda_runtime.h>

#include "cutlass/cutlass.h"
#include "cutlass/gemm/device/gemm.h"
#include "cutlass/layout/matrix.h"

namespace {

bool checked_cuda(cudaError_t err) {
  return err == cudaSuccess;
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

  uint32_t *d_a = nullptr;
  uint32_t *d_b = nullptr;
  uint32_t *d_c = nullptr;

  if (!checked_cuda(cudaMalloc(reinterpret_cast<void **>(&d_a), a_elems * sizeof(uint32_t)))) {
    return 2;
  }
  if (!checked_cuda(cudaMalloc(reinterpret_cast<void **>(&d_b), b_elems * sizeof(uint32_t)))) {
    cudaFree(d_a);
    return 2;
  }
  if (!checked_cuda(cudaMalloc(reinterpret_cast<void **>(&d_c), c_elems * sizeof(uint32_t)))) {
    cudaFree(d_a);
    cudaFree(d_b);
    return 2;
  }

  if (!checked_cuda(cudaMemcpy(d_a, a, a_elems * sizeof(uint32_t), cudaMemcpyHostToDevice)) ||
      !checked_cuda(cudaMemcpy(d_b, b, b_elems * sizeof(uint32_t), cudaMemcpyHostToDevice))) {
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    return 3;
  }

  using Gemm = cutlass::gemm::device::Gemm<uint32_t, cutlass::layout::RowMajor, uint32_t,
                                           cutlass::layout::RowMajor, uint32_t,
                                           cutlass::layout::RowMajor, uint32_t,
                                           cutlass::arch::OpClassSimt, cutlass::arch::Sm50>;

  Gemm gemm_op;
  Gemm::Arguments args(
      {static_cast<int>(a_rows), static_cast<int>(b_cols), static_cast<int>(a_cols)},
      {d_a, static_cast<int>(a_cols)}, {d_b, static_cast<int>(b_cols)},
      {d_c, static_cast<int>(b_cols)}, {d_c, static_cast<int>(b_cols)}, {1, 0});

  cutlass::Status status = gemm_op(args);
  if (status != cutlass::Status::kSuccess) {
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    return 4;
  }

  if (!checked_cuda(cudaDeviceSynchronize()) ||
      !checked_cuda(cudaMemcpy(out, d_c, c_elems * sizeof(uint32_t), cudaMemcpyDeviceToHost))) {
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    return 5;
  }

  cudaFree(d_a);
  cudaFree(d_b);
  cudaFree(d_c);
  return 0;
}
