#pragma once

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// Host pointers; uint32 matches pir Elem. basis=10, compression=3 only (same as pir.c).
// Computes out[i] += ... same as matMulVecPacked for i in [0,a_rows) (any positive a_rows).
int simplepir_matmul_vec_packed_gpu(uint32_t *out, const uint32_t *a, const uint32_t *b, size_t a_rows,
                                    size_t a_cols);

// Full squished DB on device (one upload after Squish). row_off + a_rows must fit uploaded rows;
// a_cols must match uploaded cols.
int simplepir_packed_db_upload(const uint32_t *host, size_t rows, size_t cols);
void simplepir_packed_db_release(void);
int simplepir_matmul_vec_packed_gpu_resident(uint32_t *out, size_t row_off, size_t a_rows, size_t a_cols,
                                            const uint32_t *b);

// Same math as simplepir_matmul_vec_packed_gpu_resident on a dedicated stream with optional timings:
// - kernel_ms_out (CUDA events): milliseconds on the GPU timeline from after H2D for Q (and initial
//   out buffer) completes until the packed kernel completes. Excludes D2H of the result.
// - e2e_ms_out (host steady_clock): wall time for the entire call including all copies and sync.
// Pass NULL for either pointer to skip that measurement.
int simplepir_matmul_vec_packed_gpu_resident_timed(uint32_t *out, size_t row_off, size_t a_rows,
                                                   size_t a_cols, const uint32_t *b, float *kernel_ms_out,
                                                   float *e2e_ms_out);

#ifdef __cplusplus
}
#endif
