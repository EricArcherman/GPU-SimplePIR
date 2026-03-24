//go:build cutlass

package pir

import (
	"fmt"
	"testing"
)

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

	// Includes small GEMMs: default SIMPLEPIR_CUTLASS_MIN_OPS is 0 (always try GPU).
	cases := []struct {
		aRows uint64
		aCols uint64
		bCols uint64
	}{
		{32, 32, 32},
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
				t.Skipf("CUTLASS matmul unavailable on this machine (status=%d)", status)
			}
			assertEqualMatrix(t, cpu, directGPU, "cpu_vs_cutlass_direct")

			backendGPU := MatrixMul(a, b)
			assertEqualMatrix(t, cpu, backendGPU, "cpu_vs_matrixmul_backend")
		})
	}
}

func TestMatrixMulVecCUTLASSMatchesCPU(t *testing.T) {
	t.Setenv("SIMPLEPIR_USE_CUTLASS", "1")

	cases := []struct {
		aRows uint64
		aCols uint64
	}{
		{48, 48},
		{256, 128},
		{512, 512},
	}

	for _, tc := range cases {
		name := fmt.Sprintf("%dx%d_vec", tc.aRows, tc.aCols)
		t.Run(name, func(t *testing.T) {
			a := MatrixRand(tc.aRows, tc.aCols, 16, 0)
			b := MatrixRand(tc.aCols, 1, 16, 0)

			cpu := matMulCPUReference(a, b)

			directGPU, status := matMulCUTLASSDirect(a, b)
			if status != 0 {
				t.Skipf("CUTLASS matmul unavailable on this machine (status=%d)", status)
			}
			assertEqualMatrix(t, cpu, directGPU, "cpu_vs_cutlass_vec_direct")

			backendOut := MatrixMulVec(a, b)
			assertEqualMatrix(t, cpu, backendOut, "cpu_vs_matrixmulvec_backend")
		})
	}
}
