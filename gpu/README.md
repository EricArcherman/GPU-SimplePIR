# CUTLASS GEMM Bridge

This directory provides an initial CUDA/CUTLASS backend for SimplePIR's
32-bit matrix multiplication.

The exported C ABI includes:

```
int simplepir_matmul_cutlass_u32(uint32_t* out, const uint32_t* a, const uint32_t* b,
                                 size_t a_rows, size_t a_cols, size_t b_cols);
```

and a **device-resident** handle (`SimplepirCutlassPersistHandle`) so you can upload `A` (or `DB`+`A`) once and run many GEMMs with only the smaller matrix (and output) crossing the bus—see `simplepir_cutlass_persist_*` in `cutlass_bridge.h`. Go wrappers live in `pir/cutlass_persist.go` (`//go:build cutlass`).

with row-major semantics:

```
out[a_rows x b_cols] = a[a_rows x a_cols] * b[a_cols x b_cols]
```

The function returns:
- `0` on success
- non-zero on failure (allocation/copy/kernel/runtime errors)

## Build

```
git clone https://github.com/NVIDIA/cutlass.git third_party/cutlass
cmake -S gpu -B gpu/build -DCUTLASS_PATH="$(pwd)/third_party/cutlass" -DCMAKE_CUDA_ARCHITECTURES=90
cmake --build gpu/build -j
```

Set `CMAKE_CUDA_ARCHITECTURES` for your GPU.

The same build also produces `libsimplepir_naive.a`: a **naive CUDA mock** of an end-to-end SimplePIR-shaped pipeline (uncompressed `H = DB·A`, `q = A·s + e`, `ans = DB·q`) using per-element GEMM/GEMV kernels. It is **not** protocol-identical to squished `MatrixMulVecPacked` in the Go code; it exists to compare rough GPU vs CPU timings.

`naive_gpu_ctx_upload_db_A` + `naive_gpu_ctx_run_e2e_device_resident` keep **DB and A on the GPU** between queries (only `s` and `e` are copied each iteration).

Benchmark vs real `SimplePIR.Answer` (requires Go 1.17+, CUDA, and `libsimplepir_naive` built as above):

```
export CGO_ENABLED=1
export LD_LIBRARY_PATH="/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-}"
cd pir/
LOG_N=20 D=256 go test -tags cuda_e2e -bench BenchmarkNaiveGPUVsSimplePIR -run ^$ -benchtime 5x
```

## Full online server (Query on GPU + packed Answer on GPU)

With CUTLASS for `MatrixMulVec` (Query) and the CUDA packed-Answer kernel for `MatrixMulVecPacked`, one server iteration is **one `Query` + one `Answer`**—compare CPU vs GPU with:

```
export CGO_ENABLED=1
export LD_LIBRARY_PATH="/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-}"
export SIMPLEPIR_USE_CUTLASS=1
export SIMPLEPIR_CUTLASS_RESIDENT_A=1
export SIMPLEPIR_GPU_PACKED_ANSWER=1
export SIMPLEPIR_GPU_PACKED_DB_RESIDENT=1
cd pir/
LOG_N=18 D=256 go test -tags cutlass -bench BenchmarkSimplePIROnlineServer -run '^$'
```

The benchmark GPU sub-test sets `SIMPLEPIR_GPU_PACKED_DB_RESIDENT=1` so the **full squished `DB.Data`** is uploaded once after `Squish` (`FakeSetup` / `Setup`) and each `Answer` only moves the query vector and result (no per-answer H2D for the DB slice). `Reset` / `Unsquish` releases the device copy. DoublePIR still uses the non-resident packed path for `H1` / intermediate matrices that are not row slices of `DB.Data`.

Without `SIMPLEPIR_GPU_PACKED_DB_RESIDENT=1`, the packed kernel uploads the selected DB rows each call. Resident `A` (`SIMPLEPIR_CUTLASS_RESIDENT_A=1`) still applies to Query.

**Packed Answer online-only benchmark** (time only `MatrixMulVecPacked`, i.e. `D·Q`): compares **CPU**, **GPU with H2D(D) every iteration**, and **GPU with resident D**. `go test -tags cutlass -bench BenchmarkSimplePIRPackedAnswerOnlineOnly -run '^$'`. Uses **`matmul_packed_gpu.cu`**, not CUTLASS.

**CUDA event timing (resident D):** `simplepir_matmul_vec_packed_gpu_resident_timed` uses a dedicated stream, `cudaMemcpyAsync` for Q and the initial out buffer, `cudaEventRecord` after those H2Ds complete (query + out resident on device), then the packed kernel, then `cudaEventRecord` after the kernel. `cudaEventElapsedTime` yields **kernel_ms** (device timeline from “inputs ready” through kernel end; excludes result D2H). A host **steady_clock** span over the whole call reports **e2e_ms** (all copies + `cudaStreamSynchronize`). No extra atomics: one kernel launch; the completion event waits for all blocks. Benchmark: `go test -tags cutlass -bench BenchmarkPackedAnswerResidentCUDAEvents -run '^$'`. Go wrapper: `pir.MatmulPackedResidentTimed` in `packed_gpu_timed.go`.
