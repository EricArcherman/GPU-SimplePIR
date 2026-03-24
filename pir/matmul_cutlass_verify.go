//go:build cutlass

package pir

/*
#cgo CFLAGS: -I${SRCDIR}/../gpu
#include "pir.h"
#include "../gpu/cutlass_bridge.h"
*/
import "C"

import "unsafe"

// Helpers for TestMatrixMulCUTLASSMatchesCPU (cgo is not allowed in *_test.go).

func matMulCPUReference(a, b *Matrix) *Matrix {
	out := MatrixZeros(a.Rows, b.Cols)
	C.matMul(
		(*C.Elem)(unsafe.Pointer(&out.Data[0])),
		(*C.Elem)(unsafe.Pointer(&a.Data[0])),
		(*C.Elem)(unsafe.Pointer(&b.Data[0])),
		C.size_t(a.Rows),
		C.size_t(a.Cols),
		C.size_t(b.Cols),
	)
	return out
}

func matMulCUTLASSDirect(a, b *Matrix) (*Matrix, int) {
	out := MatrixZeros(a.Rows, b.Cols)
	status := C.simplepir_matmul_cutlass_u32(
		(*C.uint32_t)(unsafe.Pointer(&out.Data[0])),
		(*C.uint32_t)(unsafe.Pointer(&a.Data[0])),
		(*C.uint32_t)(unsafe.Pointer(&b.Data[0])),
		C.size_t(a.Rows),
		C.size_t(a.Cols),
		C.size_t(b.Cols),
	)
	return out, int(status)
}
