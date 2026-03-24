#pragma once

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// Returns 0 on success; non-zero on failure.
int simplepir_matmul_cutlass_u32(uint32_t *out, const uint32_t *a, const uint32_t *b,
                                 size_t a_rows, size_t a_cols, size_t b_cols);

// ---- Device-resident GEMM (CUTLASS): upload large factor(s) once, reuse for many RHS/out cycles.

typedef void *SimplepirCutlassPersistHandle;

SimplepirCutlassPersistHandle simplepir_cutlass_persist_create(void);
void simplepir_cutlass_persist_destroy(SimplepirCutlassPersistHandle h);

// Upload LHS A (a_rows × a_cols) to device; used by simplepir_cutlass_persist_gemm_rhs_from_host.
int simplepir_cutlass_persist_upload_lhs(SimplepirCutlassPersistHandle h, const uint32_t *host_a,
                                         size_t a_rows, size_t a_cols);

// C = A * B with A resident; B is (a_cols × b_cols) row-major on host; C is (a_rows × b_cols) to host.
int simplepir_cutlass_persist_gemm_rhs_from_host(SimplepirCutlassPersistHandle h, const uint32_t *host_b,
                                                 size_t b_cols, uint32_t *host_c);

// Upload both factors once: A is (a_rows × a_cols), B is (a_cols × b_cols) row-major.
int simplepir_cutlass_persist_upload_ab(SimplepirCutlassPersistHandle h, const uint32_t *host_a,
                                        size_t a_rows, size_t a_cols, const uint32_t *host_b, size_t b_cols);

// C = A * B with both on device (dimensions must match last upload_ab). Copies C to host_c only.
int simplepir_cutlass_persist_gemm_ab_resident(SimplepirCutlassPersistHandle h, uint32_t *host_c,
                                                size_t a_rows, size_t a_cols, size_t b_cols);

#ifdef __cplusplus
}
#endif
