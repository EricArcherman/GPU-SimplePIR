//go:build cutlass

package pir

import (
	"os"
	"strconv"
	"testing"
)

// BenchmarkSimplePIRServerQuery compares SimplePIR.Query (includes A·secret via MatrixMulVec)
// with CUTLASS H2D each call vs device-resident A (SIMPLEPIR_CUTLASS_RESIDENT_A=1).
//
//	go test -tags cutlass -bench BenchmarkSimplePIRServerQuery -run ^$ -benchtime 20x
//	LOG_N=18 D=256 ... optional
func BenchmarkSimplePIRServerQuery(b *testing.B) {
	_ = os.Setenv("SIMPLEPIR_USE_CUTLASS", "1")

	numEntries := uint64(1 << 16)
	d := uint64(256)
	if v, err := strconv.Atoi(os.Getenv("LOG_N")); err == nil && v != 0 {
		numEntries = uint64(1 << v)
	}
	if v, err := strconv.Atoi(os.Getenv("D")); err == nil && v != 0 {
		d = uint64(v)
	}

	pi := SimplePIR{}
	p := pi.PickParams(numEntries, d, SEC_PARAM, LOGQ)
	i := uint64(0)
	if i >= p.L*p.M {
		b.Fatalf("bad index")
	}
	DB := MakeRandomDB(numEntries, d, &p)
	shared := pi.Init(DB.Info, p)
	_, _ = pi.FakeSetup(DB, p)

	b.Run("Query_cutlass_H2D_A_each", func(b *testing.B) {
		_ = os.Setenv("SIMPLEPIR_CUTLASS_RESIDENT_A", "0")
		ResetCutlassResidentServerForBench()
		b.ResetTimer()
		for k := 0; k < b.N; k++ {
			_, _ = pi.Query(i, shared, p, DB.Info)
		}
	})

	b.Run("Query_cutlass_resident_A", func(b *testing.B) {
		_ = os.Setenv("SIMPLEPIR_CUTLASS_RESIDENT_A", "1")
		ResetCutlassResidentServerForBench()
		_, _ = pi.Query(i, shared, p, DB.Info)
		b.ResetTimer()
		for k := 0; k < b.N; k++ {
			_, _ = pi.Query(i, shared, p, DB.Info)
		}
	})

	pi.Reset(DB, p)
}
