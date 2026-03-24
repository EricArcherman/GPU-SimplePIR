//go:build cutlass

package pir

import (
	"os"
	"strconv"
	"testing"
)

func TestCutlassPersistQueryMatchesHost(t *testing.T) {
	t.Setenv("SIMPLEPIR_USE_CUTLASS", "0")
	M, N := uint64(64), uint64(48)
	A := MatrixRand(M, N, 16, 0)
	s := MatrixRand(N, 1, 16, 0)
	want := MatrixMul(A, s)

	t.Setenv("SIMPLEPIR_USE_CUTLASS", "1")
	gotFull := MatrixZeros(M, 1)
	if err := CutlassMatmulHostFull(gotFull, A, s); err != nil {
		t.Fatal(err)
	}
	assertMatrixEqual(t, want, gotFull, "cutlass_full")

	p, err := NewCutlassPersist()
	if err != nil {
		t.Fatal(err)
	}
	defer p.Destroy()
	if err := p.UploadLHS(A); err != nil {
		t.Fatal(err)
	}
	gotRes := MatrixZeros(M, 1)
	if err := p.GemmRHSFromHost(s, gotRes); err != nil {
		t.Fatal(err)
	}
	assertMatrixEqual(t, want, gotRes, "cutlass_resident")
}

func TestCutlassPersistSetupMatchesHost(t *testing.T) {
	t.Setenv("SIMPLEPIR_USE_CUTLASS", "0")
	L, Mdim, Ndim := uint64(32), uint64(40), uint64(24)
	DB := MatrixRand(L, Mdim, 16, 0)
	A := MatrixRand(Mdim, Ndim, 16, 0)
	want := MatrixMul(DB, A)

	t.Setenv("SIMPLEPIR_USE_CUTLASS", "1")
	gotFull := MatrixZeros(L, Ndim)
	if err := CutlassMatmulHostFull(gotFull, DB, A); err != nil {
		t.Fatal(err)
	}
	assertMatrixEqual(t, want, gotFull, "setup_full")

	p, err := NewCutlassPersist()
	if err != nil {
		t.Fatal(err)
	}
	defer p.Destroy()
	if err := p.UploadAB(DB, A); err != nil {
		t.Fatal(err)
	}
	gotRes := MatrixZeros(L, Ndim)
	if err := p.GemmABResident(L, Mdim, Ndim, gotRes); err != nil {
		t.Fatal(err)
	}
	assertMatrixEqual(t, want, gotRes, "setup_resident")
}

func assertMatrixEqual(t *testing.T, want, got *Matrix, label string) {
	t.Helper()
	if want.Rows != got.Rows || want.Cols != got.Cols {
		t.Fatalf("%s: dims %dx%d vs %dx%d", label, want.Rows, want.Cols, got.Rows, got.Cols)
	}
	for i := range want.Data {
		if want.Data[i] != got.Data[i] {
			t.Fatalf("%s: mismatch at %d want=%d got=%d", label, i, want.Data[i], got.Data[i])
		}
	}
}

// BenchmarkCutlassResidentGEMM measures Query-shaped (A·s) and Setup-shaped (DB·A) GEMMs:
// full H2D each iteration vs device-resident factors.
//
// Optional env: CUTLASS_BENCH_M, CUTLASS_BENCH_N (Query: A is M×N, s is N×1).
// CUTLASS_BENCH_L, CUTLASS_BENCH_M2, CUTLASS_BENCH_N2 (Setup: DB is L×M2, A is M2×N2).
func BenchmarkCutlassResidentGEMM(b *testing.B) {
	if os.Getenv("SIMPLEPIR_USE_CUTLASS") != "1" {
		b.Skip("set SIMPLEPIR_USE_CUTLASS=1")
	}

	M := uint64(4096)
	N := uint64(1024)
	if v, err := strconv.ParseUint(os.Getenv("CUTLASS_BENCH_M"), 10, 64); err == nil && v > 0 {
		M = v
	}
	if v, err := strconv.ParseUint(os.Getenv("CUTLASS_BENCH_N"), 10, 64); err == nil && v > 0 {
		N = v
	}

	A := MatrixRand(M, N, 16, 0)
	s := MatrixRand(N, 1, 16, 0)
	out := MatrixZeros(M, 1)

	b.Run("QueryShape_H2D_AB_each_iter", func(b *testing.B) {
		for i := 0; i < b.N; i++ {
			if err := CutlassMatmulHostFull(out, A, s); err != nil {
				b.Fatal(err)
			}
		}
	})

	p, err := NewCutlassPersist()
	if err != nil {
		b.Fatal(err)
	}
	defer p.Destroy()
	if err := p.UploadLHS(A); err != nil {
		b.Fatal(err)
	}
	if err := p.GemmRHSFromHost(s, out); err != nil {
		b.Fatal(err)
	}

	b.Run("QueryShape_resident_A_H2D_s_only", func(b *testing.B) {
		for i := 0; i < b.N; i++ {
			if err := p.GemmRHSFromHost(s, out); err != nil {
				b.Fatal(err)
			}
		}
	})

	L := uint64(2048)
	M2 := uint64(2048)
	N2 := uint64(512)
	if v, err := strconv.ParseUint(os.Getenv("CUTLASS_BENCH_L"), 10, 64); err == nil && v > 0 {
		L = v
	}
	if v, err := strconv.ParseUint(os.Getenv("CUTLASS_BENCH_M2"), 10, 64); err == nil && v > 0 {
		M2 = v
	}
	if v, err := strconv.ParseUint(os.Getenv("CUTLASS_BENCH_N2"), 10, 64); err == nil && v > 0 {
		N2 = v
	}

	DB := MatrixRand(L, M2, 16, 0)
	A2 := MatrixRand(M2, N2, 16, 0)
	H := MatrixZeros(L, N2)

	b.Run("SetupShape_H2D_AB_each_iter", func(b *testing.B) {
		for i := 0; i < b.N; i++ {
			if err := CutlassMatmulHostFull(H, DB, A2); err != nil {
				b.Fatal(err)
			}
		}
	})

	p2, err := NewCutlassPersist()
	if err != nil {
		b.Fatal(err)
	}
	defer p2.Destroy()
	if err := p2.UploadAB(DB, A2); err != nil {
		b.Fatal(err)
	}
	if err := p2.GemmABResident(L, M2, N2, H); err != nil {
		b.Fatal(err)
	}

	b.Run("SetupShape_resident_DB_A", func(b *testing.B) {
		for i := 0; i < b.N; i++ {
			if err := p2.GemmABResident(L, M2, N2, H); err != nil {
				b.Fatal(err)
			}
		}
	})
}
