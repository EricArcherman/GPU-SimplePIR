//go:build cutlass

package pir

import (
	"os"
	"strconv"
	"testing"
)

// BenchmarkSimplePIROnlineServer compares full server online work per iteration:
// one SimplePIR.Query + one SimplePIR.Answer (single-query batch).
//
// CPU: C matmul / packed only (CUTLASS and GPU packed off).
// GPU: SIMPLEPIR_USE_CUTLASS=1, SIMPLEPIR_CUTLASS_RESIDENT_A=1, SIMPLEPIR_GPU_PACKED_ANSWER=1,
// SIMPLEPIR_GPU_PACKED_DB_RESIDENT=1 (full squished DB on device; no per-Answer H2D for DB slice).
//
//	LOG_N=18 D=256 go test -tags cutlass -bench BenchmarkSimplePIROnlineServer -run ^$ -benchtime 10x
func BenchmarkSimplePIROnlineServer(b *testing.B) {
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

	b.Run("CPU_online_Query_plus_Answer", func(b *testing.B) {
		_ = os.Setenv("SIMPLEPIR_USE_CUTLASS", "0")
		_ = os.Setenv("SIMPLEPIR_CUTLASS_RESIDENT_A", "0")
		_ = os.Setenv("SIMPLEPIR_GPU_PACKED_ANSWER", "0")
		_ = os.Setenv("SIMPLEPIR_GPU_PACKED_DB_RESIDENT", "0")
		ResetCutlassResidentServerForBench()

		DB := MakeRandomDB(numEntries, d, &p)
		shared := pi.Init(DB.Info, p)
		serverState, _ := pi.FakeSetup(DB, p)

		b.ResetTimer()
		for k := 0; k < b.N; k++ {
			_, q := pi.Query(i, shared, p, DB.Info)
			qs := MakeMsgSlice(q)
			_ = pi.Answer(DB, qs, serverState, shared, p)
		}
		b.StopTimer()
		pi.Reset(DB, p)
	})

	b.Run("GPU_online_Query_plus_Answer", func(b *testing.B) {
		_ = os.Setenv("SIMPLEPIR_USE_CUTLASS", "1")
		_ = os.Setenv("SIMPLEPIR_CUTLASS_RESIDENT_A", "1")
		_ = os.Setenv("SIMPLEPIR_GPU_PACKED_ANSWER", "1")
		_ = os.Setenv("SIMPLEPIR_GPU_PACKED_DB_RESIDENT", "1")
		ResetCutlassResidentServerForBench()

		DB := MakeRandomDB(numEntries, d, &p)
		shared := pi.Init(DB.Info, p)
		serverState, _ := pi.FakeSetup(DB, p)

		b.ResetTimer()
		for k := 0; k < b.N; k++ {
			_, q := pi.Query(i, shared, p, DB.Info)
			qs := MakeMsgSlice(q)
			_ = pi.Answer(DB, qs, serverState, shared, p)
		}
		b.StopTimer()
		pi.Reset(DB, p)
	})
}
