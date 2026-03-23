#pragma once

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// Returns 0 on success; non-zero on failure.
int simplepir_matmul_cutlass_u32(uint32_t *out, const uint32_t *a, const uint32_t *b,
                                 size_t a_rows, size_t a_cols, size_t b_cols);

#ifdef __cplusplus
}
#endif
