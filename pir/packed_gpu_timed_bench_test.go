//go:build cutlass

package pir

import (
	"os"
	"strconv"
	"testing"
)

// BenchmarkPackedAnswerResidentCUDAEvents measures resident-D packed Answer with CUDA event timing:
//   - kernel_ms: cudaEventElapsedTime from after H2D(Q)+H2D(initial out) until kernel end (device timeline).
//   - e2e_ms: host steady_clock for the full C call (H2D + kernel + D2H + sync).
//
// DB upload and squish are outside the timed loop (preload).
//
//	LOG_N=20 D=256 go test -tags cutlass -bench BenchmarkPackedAnswerResidentCUDAEvents -run '^$' -benchtime 30x
func BenchmarkPackedAnswerResidentCUDAEvents(b *testing.B) {
	if os.Getenv("SIMPLEPIR_GPU_PACKED_DB_RESIDENT") != "1" {
		_ = os.Setenv("SIMPLEPIR_GPU_PACKED_DB_RESIDENT", "1")
	}
	_ = os.Setenv("SIMPLEPIR_GPU_PACKED_ANSWER", "1")

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

	ResetCutlassResidentServerForBench()
	DB := MakeRandomDB(numEntries, d, &p)
	shared := pi.Init(DB.Info, p)
	_, _ = pi.FakeSetup(DB, p)
	_, qMsg := pi.Query(i, shared, p, DB.Info)
	q := qMsg.Data[0]

	out := MatrixNew(DB.Data.Rows+8, 1)
	rowOff, ok := packedSubmatrixRowOffset(DB.Data, DB.Data)
	if !ok {
		b.Fatalf("DB.Data should cover full matrix")
	}

	b.ResetTimer()
	var sumKernel, sumE2E float64
	for k := 0; k < b.N; k++ {
		b.StopTimer()
		for j := range out.Data {
			out.Data[j] = 0
		}
		b.StartTimer()
		km, em, rc := MatmulPackedResidentTimed(out, rowOff, DB.Data.Rows, DB.Data.Cols, q, 10, 3)
		if rc != 0 {
			b.Fatalf("MatmulPackedResidentTimed rc=%d", rc)
		}
		sumKernel += float64(km)
		sumE2E += float64(em)
	}
	b.StopTimer()
	pi.Reset(DB, p)

	n := float64(b.N)
	b.ReportMetric(sumKernel/n, "kernel_ms") // GPU: Q ready → kernel done
	b.ReportMetric(sumE2E/n, "e2e_ms")       // host wall for full copy+compute+copy
}
