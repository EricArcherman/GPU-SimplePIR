//go:build cutlass

package pir

/*
#cgo CFLAGS: -I${SRCDIR}/../gpu
#cgo LDFLAGS: -L${SRCDIR}/../gpu/build -lsimplepir_cutlass -lstdc++ -lcudart
#include "matmul_packed_gpu.h"
*/
import "C"

import "unsafe"

// MatmulPackedResidentTimed runs the resident-D packed kernel with optional CUDA event + host timing.
// out must hold at least aRows uint32s at the start of out.Data for H2D/D2H (same as MatrixMulVecPacked).
// Returns rc 0 on success; kernelMs is GPU time from Q (+initial out) on-device until kernel completes;
// e2eMs is host wall time for the full call (H2D + kernel + D2H + sync).
func MatmulPackedResidentTimed(out *Matrix, rowOff, aRows, aCols uint64, b *Matrix, basis, compression uint64) (
	kernelMs, e2eMs float32, rc int) {
	if basis != 10 || compression != 3 {
		return 0, 0, -1
	}
	var km, em C.float
	r := C.simplepir_matmul_vec_packed_gpu_resident_timed(
		(*C.uint32_t)(unsafe.Pointer(&out.Data[0])),
		C.size_t(rowOff),
		C.size_t(aRows),
		C.size_t(aCols),
		(*C.uint32_t)(unsafe.Pointer(&b.Data[0])),
		(*C.float)(&km),
		(*C.float)(&em),
	)
	return float32(km), float32(em), int(r)
}
