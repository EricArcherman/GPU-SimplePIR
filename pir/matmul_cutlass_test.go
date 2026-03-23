//go:build cutlass

package pir

/*
#cgo CFLAGS: -I${SRCDIR}/../gpu
#include "pir.h"
#include "../gpu/cutlass_bridge.h"
*/
import "C"

import (
	"fmt"
	"testing"
	"unsafe"
)

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

func matMulCUTLASSDirect(a, b *Matrix) (*Matrix, C.int) {
	out := MatrixZeros(a.Rows, b.Cols)
	status := C.simplepir_matmul_cutlass_u32(
		(*C.uint32_t)(unsafe.Pointer(&out.Data[0])),
		(*C.uint32_t)(unsafe.Pointer(&a.Data[0])),
		(*C.uint32_t)(unsafe.Pointer(&b.Data[0])),
		C.size_t(a.Rows),
		C.size_t(a.Cols),
		C.size_t(b.Cols),
	)
	return out, status
}

func assertEqualMatrix(t *testing.T, want, got *Matrix, label string) {
	t.Helper()
	if want.Rows != got.Rows || want.Cols != got.Cols {
		t.Fatalf("%s dims mismatch: want %dx%d got %dx%d", label, want.Rows, want.Cols, got.Rows, got.Cols)
	}

	for i := 0; i < len(want.Data); i++ {
		if want.Data[i] != got.Data[i] {
			row := uint64(i) / want.Cols
			col := uint64(i) % want.Cols
			t.Fatalf("%s mismatch at (%d,%d): want=%d got=%d", label, row, col, want.Data[i], got.Data[i])
		}
	}
}

func TestMatrixMulCUTLASSMatchesCPU(t *testing.T) {
	t.Setenv("SIMPLEPIR_USE_CUTLASS", "1")

	// Keep these above the current backend threshold (1,000,000 mul-add ops)
	// so MatrixMul() actually exercises the GPU backend path.
	cases := []struct {
		aRows uint64
		aCols uint64
		bCols uint64
	}{
		{128, 128, 128},
		{96, 160, 96},
		{64, 512, 64},
	}

	for _, tc := range cases {
		name := fmt.Sprintf("%dx%d_x_%dx%d", tc.aRows, tc.aCols, tc.aCols, tc.bCols)
		t.Run(name, func(t *testing.T) {
			a := MatrixRand(tc.aRows, tc.aCols, 16, 0)
			b := MatrixRand(tc.aCols, tc.bCols, 16, 0)

			cpu := matMulCPUReference(a, b)

			directGPU, status := matMulCUTLASSDirect(a, b)
			if status != 0 {
				t.Skipf("CUTLASS matmul unavailable on this machine (status=%d)", int(status))
			}
			assertEqualMatrix(t, cpu, directGPU, "cpu_vs_cutlass_direct")

			backendGPU := MatrixMul(a, b)
			assertEqualMatrix(t, cpu, backendGPU, "cpu_vs_matrixmul_backend")
		})
	}
}
