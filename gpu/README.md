# CUTLASS GEMM Bridge

This directory provides an initial CUDA/CUTLASS backend for SimplePIR's
32-bit matrix multiplication.

The exported C ABI is:

```
int simplepir_matmul_cutlass_u32(uint32_t* out, const uint32_t* a, const uint32_t* b,
                                 size_t a_rows, size_t a_cols, size_t b_cols);
```

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
