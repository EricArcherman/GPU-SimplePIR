//go:build cutlass

package pir

import (
	"testing"
)

func TestMatmulVecPackedGPUMatchesCPU(t *testing.T) {
	aRows := uint64(128)
	aCols := uint64(64)
	a := MatrixRand(aRows, aCols, 16, 0)
	b := MatrixRand(aCols*3, 1, 16, 0)

	t.Setenv("SIMPLEPIR_GPU_PACKED_ANSWER", "0")
	cpu := MatrixMulVecPacked(a, b, 10, 3)

	t.Setenv("SIMPLEPIR_GPU_PACKED_ANSWER", "1")
	gpu := MatrixMulVecPacked(a, b, 10, 3)

	if cpu.Rows != gpu.Rows || cpu.Cols != gpu.Cols {
		t.Fatalf("dims mismatch")
	}
	for i := range cpu.Data {
		if cpu.Data[i] != gpu.Data[i] {
			t.Fatalf("mismatch at %d want=%d got=%d", i, cpu.Data[i], gpu.Data[i])
		}
	}
}

func TestMatmulVecPackedGPUResidentSubsliceMatchesCPU(t *testing.T) {
	t.Setenv("SIMPLEPIR_GPU_PACKED_DB_RESIDENT", "1")
	aRows := uint64(128)
	aCols := uint64(64)
	full := MatrixRand(aRows, aCols, 16, 0)
	b := MatrixRand(aCols*3, 1, 16, 0)
	sub := full.SelectRows(16, 32)

	DB := &Database{Data: full, Info: DBinfo{}}
	if packedGPUDBSyncHook != nil {
		packedGPUDBSyncHook(DB)
	}
	t.Cleanup(func() {
		if packedGPUDBClearHook != nil {
			packedGPUDBClearHook()
		}
	})

	t.Setenv("SIMPLEPIR_GPU_PACKED_ANSWER", "0")
	cpu := MatrixMulVecPacked(sub, b, 10, 3)

	t.Setenv("SIMPLEPIR_GPU_PACKED_ANSWER", "1")
	gpu := MatrixMulVecPacked(sub, b, 10, 3)

	if cpu.Rows != gpu.Rows || cpu.Cols != gpu.Cols {
		t.Fatalf("dims mismatch")
	}
	for i := range cpu.Data {
		if cpu.Data[i] != gpu.Data[i] {
			t.Fatalf("mismatch at %d want=%d got=%d", i, cpu.Data[i], gpu.Data[i])
		}
	}
}
