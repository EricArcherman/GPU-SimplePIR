#pragma once

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// Opaque handle (implemented in .cu as NaiveGpuCtx*).
typedef void *NaiveGpuSimplepirHandle;

NaiveGpuSimplepirHandle naive_gpu_ctx_create(uint32_t L, uint32_t M, uint32_t N);

void naive_gpu_ctx_destroy(NaiveGpuSimplepirHandle ctx);

// Copy uncompressed DB (L×M) and A (M×N) to device once. Call before naive_gpu_ctx_run_e2e_device_resident.
int naive_gpu_ctx_upload_db_A(NaiveGpuSimplepirHandle ctx, const uint32_t *host_db, const uint32_t *host_A);

// Full E2E: H2D(DB,A) + setup + H2D(s,e) + query + answer + minimal D2H (each call).
int naive_gpu_ctx_run_e2e(NaiveGpuSimplepirHandle ctx, const uint32_t *host_db, const uint32_t *host_A,
                          const uint32_t *host_secret, const uint32_t *host_err, uint32_t delta_row,
                          uint32_t delta, double *out_ms_setup, double *out_ms_query, double *out_ms_answer,
                          double *out_ms_total);

// E2E assuming DB and A already on device (use naive_gpu_ctx_upload_db_A). Only H2D(s,e) per call + kernels + pin D2H.
// setup_ms = device GEMM only; query_ms = H2D(s,e) + query kernels; answer_ms = answer kernel.
int naive_gpu_ctx_run_e2e_device_resident(NaiveGpuSimplepirHandle ctx, const uint32_t *host_secret,
                                          const uint32_t *host_err, uint32_t delta_row, uint32_t delta,
                                          double *out_ms_setup, double *out_ms_query, double *out_ms_answer,
                                          double *out_ms_total);

#ifdef __cplusplus
}
#endif
